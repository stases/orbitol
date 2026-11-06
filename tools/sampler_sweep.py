"""Sampler-setting sweep under the training evaluation protocol, with timings.

Each setting overrides TrainConfig's sampler fields (sample_num_steps, S_churn, sigma_max, aa_rho_coords, ...). All
settings share the checkpoint's EMA weights, the skeleton draws (same seed) and the in-training evaluator: every
sample relaxed (OEQ NequIP, the checkpoint's relax steps), SUN/MSUN over all attempts, one-shot and relaxation
metrics, plausibility diagnostics and the symmetry panel. Timings split sampling into network calls (CUDA-synchronized)
and decode, and evaluation into relaxation and scoring; compilation is excluded by a warm-up. One JSON and one sample
file per setting; finished settings are skipped, so a job can be resubmitted.

  python -m tools.sampler_sweep --checkpoint CKPT --out DIR --settings '{"name": "steps20", "sample_num_steps": 20}' ...
"""
import argparse
from dataclasses import replace
import functools
import json
from pathlib import Path
import time

import torch

from orbitol import contracts as C, eval_adapter, sample as sample_module, train
from orbitol.eval_diagnostics import variant_metrics
from orbitol.metric_names import log_view

SAMPLER_FIELDS = ("sample_num_steps", "rho", "S_churn", "S_min", "S_max", "S_noise", "sigma_min", "sigma_max",
                  "aa_rho_coords", "aa_rho_lattice", "sample_token_budget", "compile_trunk", "decode_impl")


def timed(owner, name, timers, key, synchronize=False):
    """Accumulate wall-clock seconds of owner.name into timers[key]."""
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        if synchronize:
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            if synchronize:
                torch.cuda.synchronize()
            timers[key] += time.perf_counter() - started
    setattr(owner, name, wrapper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--settings", nargs="+", type=json.loads, required=True,
                        help='JSON objects: a "name" plus TrainConfig sampler fields to override')
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=2048, help="samples drawn once before timing (compilation)")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = replace(C.TrainConfig(**C.normalize_historical_config(checkpoint["cfg"])), sample_count=args.samples,
                  sun_k=0, nequip_eval=True, eval_mode="sync", val_every=0)
    run = train.Trainer(cfg, run_name="trainer", out_root=args.out, device="cuda", no_wandb=True)
    run.model.load_state_dict(checkpoint["model"])
    run.ema.load_state_dict(checkpoint["ema"])
    step = checkpoint["step"]
    del checkpoint
    timers = dict.fromkeys(("network", "decode", "relax", "score"), 0.0)
    timed(sample_module, "orbit_edm_sampler", timers, "network", synchronize=True)
    timed(eval_adapter, "relax_all", timers, "relax")
    timed(eval_adapter, "run_in_child", timers, "score")

    def draw(sampler_cfg, n):
        return sample_module.sample_pool(run.model, run.ema, sampler_cfg, run.tables, run.tb, run.encoding,
                                         run.allowed_mask, run.bank, n=n, seed=args.seed, device=run.device)

    warmed, timed_decoders = set(), set()
    for setting in args.settings:
        name, overrides = setting["name"], {k: v for k, v in setting.items() if k != "name"}
        target = args.out / f"{name}.json"
        if target.exists():
            continue
        sampler_cfg = replace(cfg, **overrides)
        if sampler_cfg.decode_impl not in timed_decoders:
            owner = sample_module
            if sampler_cfg.decode_impl == "fast":
                from orbitol import decode_fast as owner
            timed(owner, "decode_batch", timers, "decode")
            timed_decoders.add(sampler_cfg.decode_impl)
        if sampler_cfg.decode_impl not in warmed:
            draw(replace(sampler_cfg, sample_num_steps=2), args.warmup)
            warmed.add(sampler_cfg.decode_impl)
        timers.update(dict.fromkeys(timers, 0.0))
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = draw(sampler_cfg, args.samples)
        sample_seconds = time.perf_counter() - started
        torch.save(dict(rows=result["rows"], requested_G=result["requested_G"], failures=result["failures"]),
                   args.out / f"{name}.samples.pt")
        started = time.perf_counter()
        metrics = run.evaluator.evaluate(result["rows"], step=step, requested_G=result["requested_G"])
        metrics.update(variant_metrics(run.evaluator.relaxation_diagnostics(),
                                       checked_count=metrics.get("joint/checked_count", args.samples)))
        eval_seconds = time.perf_counter() - started
        steps = sampler_cfg.sample_num_steps
        per_1k = 1000 / args.samples
        timing = dict(sample_seconds=sample_seconds, network_seconds=timers["network"],
                      decode_seconds=timers["decode"], eval_seconds=eval_seconds, relax_seconds=timers["relax"],
                      score_seconds=timers["score"], nfe=2 * steps - 1,  # Heun: no correction on the last step
                      sample_s_per_1k=sample_seconds * per_1k, network_s_per_1k=timers["network"] * per_1k,
                      decode_s_per_1k=timers["decode"] * per_1k,
                      network_ms_per_nfe_per_1k=1000 * timers["network"] * per_1k / (2 * steps - 1))
        report = dict(name=name, overrides=overrides, sampler={k: getattr(sampler_cfg, k) for k in SAMPLER_FIELDS},
                      checkpoint=str(args.checkpoint), checkpoint_step=step, samples=args.samples, seed=args.seed,
                      gpu=torch.cuda.get_device_name(), timing=timing, failure_counts=result["failure_counts"],
                      contains_a_rate=float(result["contains_a"].float().mean()), metrics=metrics,
                      logged=log_view(metrics))
        target.write_text(json.dumps(report, indent=1, default=float) + "\n")
        print(name, json.dumps({k: metrics.get(k) for k in ("sample_ema/MSUN", "sample_ema/MSUN_oneshot",
                                                            "sample_ema/relax/rms_disp_median",
                                                            "sample_ema/relax/energy_drop_meV_median")}),
              json.dumps({k: round(v, 3) for k, v in timing.items()}), flush=True)


if __name__ == "__main__":
    main()
