"""Sample a saved EMA snapshot and use the shared evaluation protocol."""
from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import torch

import orbitol.core as core
from orbitol.contracts import TrainConfig, resolve_dataset_paths
from orbitol.eval_adapter import DNGEvaluator, ThermoSettings, _sample_items
from orbitol.eval_diagnostics import variant_metrics
from orbitol import contracts as C
from orbitol.metric_names import log_view
from orbitol.posthoc_io import (
    SNAPSHOT_NAME, atomic_writer, identity, read_json, sha256 as _sha256,
    verify_pool, write_json as _write_json,
)


def _positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["crystalite", "orbitol"], required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "sample", "evaluate"], default="all")
    parser.add_argument("--sample-count", type=_positive, default=argparse.SUPPRESS)
    parser.add_argument("--sun-k", type=int, default=argparse.SUPPRESS,
                        help="positive: explicit legacy raw-UN subset estimator")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--relax-mode", choices=["batch", "sequential"], default=argparse.SUPPRESS)
    parser.add_argument("--nequip-backend", choices=["oeq", "torchscript"], default=argparse.SUPPRESS)
    parser.add_argument("--relax-steps", type=_positive, default=argparse.SUPPRESS)
    parser.add_argument("--nequip-oeq-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--nequip-ts-path", type=Path)
    parser.add_argument("--ppd-path", type=Path)
    parser.add_argument("--relax-batch-size", type=_positive, default=argparse.SUPPRESS)
    parser.add_argument("--symmetry-reference", type=Path)
    parser.add_argument("--wandb-project", help="Default: the pool's dataset project")
    parser.add_argument("--wandb-name")
    parser.add_argument("--sampler-overrides", type=json.loads, default={},
                        help="Orbitol only: JSON object of orbitol.sample.SAMPLER_OVERRIDES, e.g. steps, churn, sigma")
    args = parser.parse_args(argv)
    if args.sampler_overrides:
        from orbitol.sample import SAMPLER_OVERRIDES
        if args.kind != "orbitol" or not set(args.sampler_overrides) <= set(SAMPLER_OVERRIDES):
            parser.error(f"--sampler-overrides is Orbitol-only and limited to {SAMPLER_OVERRIDES}")
    args.evaluation_overrides = set(vars(args))
    resolved = resolve_snapshot_paths(args)
    for name in ("sample_count", "sun_k", "seed", "relax_mode", "relax_batch_size", "nequip_backend", "relax_steps"):
        setattr(args, name, getattr(resolved, name))
    if args.sun_k < 0 or args.seed < 0:
        parser.error("sun-k and seed must be nonnegative")
    if args.stage != "evaluate" and args.snapshot is None:
        parser.error("--snapshot is required for sample/all stages")
    return args


def _snapshot_info(snapshot, kind="crystalite"):
    # These trusted local checkpoints are large. Read metadata without copying
    # all model/EMA tensors into RAM; the sampler subprocess loads the weights.
    checkpoint = torch.load(snapshot, map_location="cpu", weights_only=False, mmap=True)
    if not checkpoint.get("ema_state_dict" if kind == "crystalite" else "ema"):
        raise ValueError("Snapshot has no EMA weights; regular-weight fallback is forbidden")
    step = int(checkpoint["step"])
    if step < 0:
        raise ValueError("Snapshot step must be nonnegative")
    match = SNAPSHOT_NAME.fullmatch(Path(snapshot).name)
    if not match or int(match[1]) != step:
        raise ValueError("Snapshot step does not match step_XXXXXXX.pt filename")
    cfg = dict(checkpoint["model_args" if kind == "crystalite" else "cfg"])
    if kind == "orbitol":
        from orbitol.contracts import normalize_historical_config
        cfg = normalize_historical_config(cfg)
    return step, cfg


def _sampler_settings(model_args, args):
    args = resolve_snapshot_paths(args, model_args)
    # The core CLI reads these directly from model_args, with these defaults.
    defaults = dict(
        sample_num_steps=100, nmax=20, lattice_repr="y1", sigma_min=0.002,
        sigma_max=80.0, rho=7.0, S_churn=20.0, S_min=0.0, S_max=999.0,
        S_noise=1.0, sigma_data_type=1.0, sigma_data_coord=0.25,
        sigma_data_lattice=1.0, aa_frac_max_scale=0.0, aa_rho_types=0.0,
        aa_rho_coords=0.0, aa_rho_lattice=0.0,
    )
    settings = {key: model_args.get(key, value) for key, value in defaults.items()}
    expected = dict(sample_num_steps=150, S_churn=60, S_noise=1.003,
                    aa_rho_types=0, aa_rho_coords=0, aa_rho_lattice=0, nmax=20)
    for key, value in expected.items():
        if settings[key] != value:
            raise ValueError(f"The paper sampler requires {key}={value}; snapshot has {settings[key]}")
    chunk_size = int(model_args.get("sample_chunk_size", 256))
    settings.update(
        sample_mode="ema", atom_count_strategy="empirical",
        dataset_name=model_args.get("dataset_name", "mp20"),
        data_root=str(args.data_root.resolve()), sample_seed=args.seed,
        num_samples=args.sample_count, sample_chunk_size=max(1, chunk_size or args.sample_count),
        bf16=bool(model_args.get("bf16", False)), device=args.device,
        allowed_elements="train_split_support",
    )
    return settings


def sample_crystalite(snapshot, out, settings):
    """Invoke the pinned sampler unchanged, with explicit EMA and count policy."""
    data_root = Path(settings["data_root"])
    # The core constructor can preprocess/download missing data. Prevent that
    # by checking every input needed by its empirical-count path in advance.
    for relative in ("raw/train.csv", "raw/val.csv", "processed/mp20_tokens_train.pt",
                     "processed/mp20_tokens_val.pt"):
        path = data_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required read-only dataset input missing: {path}")
    command = [
        sys.executable, "-c",
        "import orbitol.core; import runpy; "
        "runpy.run_module('src.sample_crystalite_ckpt', run_name='__main__')",
        "--checkpoint", str(snapshot), "--output_dir", str(out),
        "--device", settings["device"], "--sample_mode", "ema",
        "--num_samples", str(settings["num_samples"]),
        "--sample_seed", str(settings["sample_seed"]),
        "--sample_num_steps", str(settings["sample_num_steps"]),
        "--sample_chunk_size", str(settings["sample_chunk_size"]),
        "--nmax", str(settings["nmax"]), "--dataset_name", settings["dataset_name"],
        "--data_root", str(data_root), "--atom_count_strategy", "empirical",
        "--save_pt", "--no-save_extxyz", "--no-save_cifs",
    ]
    if settings["bf16"]:
        command.append("--bf16")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
           "ORBITOL_CORE_ROOT": core.CORE_ROOT, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONHASHSEED": str(settings["sample_seed"])}
    subprocess.run(command, cwd=out, env=env, check=True)
    return torch.load(out / "samples.pt", map_location="cpu", weights_only=False)


def sample_orbitol(snapshot, *, n, seed, device, chunk=256, overrides=None):
    from orbitol.sample import sample_from_checkpoint
    try:
        return sample_from_checkpoint(snapshot, n=n, seed=seed, device=device, chunk=chunk, overrides=overrides)
    finally:
        # sample_from_checkpoint returns CPU tensors. Release the model and its
        # CUDA allocator cache before the long-lived evaluator is used again.
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def _batch_samples(items):
    if isinstance(items, dict):
        rows = items
    else:
        if not items:
            raise ValueError("Sampler returned no samples")
        rows = {key: torch.stack([item[key] for item in items])
                for key in ("A0", "F1", "Y1", "pad_mask")}
    _sample_items(rows)  # validate the actual adapter contract before saving
    return rows


def wandb_project(args, pool):
    """The explicit --wandb-project, else the project of the dataset the pool was sampled from."""
    if args.wandb_project:
        return args.wandb_project
    sampler = pool["sampler"]
    cfg = (sampler.get("cfg", {}) if pool["kind"] == "orbitol"
           else {"dataset": sampler.get("dataset_name", "mp20")})
    return cfg.get("wandb_project") or resolve_dataset_paths(cfg).wandb_project


def _log_wandb(args, metrics, step, settings, pool):
    if not args.wandb_name:
        return
    import wandb

    project = wandb_project(args, pool)

    # Display names are not identities in W&B. Use a stable ID across output
    # directories/snapshots; a new evaluation series needs a different name.
    run_id = hashlib.sha256(
        f"{project}\0{args.wandb_name}".encode()
    ).hexdigest()[:24]
    run = wandb.init(
        entity=C.local_value("ORBITOL_WANDB_ENTITY"),
        project=project, name=args.wandb_name, id=run_id, resume="allow",
        dir=str(args.out), config={"kind": args.kind, "thermo": settings["thermo"],
                                  "sample_count": settings.get("sample_count", args.sample_count),
                                  "sun_k": settings.get("sun_k", args.sun_k),
                                  "seed": settings.get("sample_seed", args.seed),
                                  "sampler": pool["sampler"]},
    )
    try:
        wandb.define_metric("train_step")
        wandb.define_metric("*", step_metric="train_step")
        # Do not use log(step=step): W&B's history index must keep increasing
        # independently of the snapshot's training step when resuming.
        wandb.log(log_view({**metrics, "train_step": step},
                           legacy_subset=(settings.get("sun_k", args.sun_k) or 0) > 0))
    finally:
        run.finish()


def resolve_snapshot_paths(args, model_args=None):
    """CLI flags win over snapshot cfg.extra, then snapshot dataset defaults.

    Return a copy: queue arguments can be shared by runs on different datasets.
    Crystalite stores its dataset under model_args.dataset_name.
    """
    cfg = model_args if model_args is not None else getattr(args, "dataset_config", {})
    if getattr(args, "kind", "orbitol") == "crystalite":
        cfg = {"dataset": cfg.get("dataset_name", cfg.get("dataset", "mp20"))}
    paths = resolve_dataset_paths(cfg, overrides=dict(
        data_root=getattr(args, "data_root", None), phase_diagram=getattr(args, "ppd_path", None),
        symmetry_reference=getattr(args, "symmetry_reference", None),
    ))
    defaults = dict(sample_count=cfg.get("sample_count", 10000), sun_k=cfg.get("sun_k", 0),
                    seed=cfg.get("extra", {}).get("sample_seed", 123),
                    relax_mode=cfg.get("nequip_relax_mode", "batch"),
                    relax_batch_size=cfg.get("nequip_relax_batch_size", 128),
                    nequip_backend=cfg.get("nequip_backend", "oeq"),
                    relax_steps=cfg.get("thermo_relax_steps", 500))
    explicit = getattr(args, "evaluation_overrides", vars(args))
    defaults.update({key: getattr(args, key) for key in defaults if key in explicit})
    return SimpleNamespace(**{**vars(args), **defaults, "dataset_config": cfg,
                             "data_root": Path(paths.data_root),
                             "ppd_path": Path(paths.phase_diagram),
                             "symmetry_reference": Path(paths.symmetry_reference)})


def thermo_settings(args):
    args = resolve_snapshot_paths(args)
    cfg = getattr(args, "dataset_config", {})
    # Manual posthoc evaluation requests thermo even if the historical trainer
    # disabled its in-process MLIP monitoring.
    model = cfg.get("extra", {}).get("nequip_model")
    ts_path = getattr(args, "nequip_ts_path", None) or (model if args.nequip_backend == "torchscript" else None)
    oeq_path = getattr(args, "nequip_oeq_path", None) or (model if args.nequip_backend == "oeq" else None)
    return ThermoSettings(
        nequip_backend=args.nequip_backend, thermo_relax_steps=args.relax_steps,
        nequip_model_kind="torchscript", nequip_ts_path=str(Path(ts_path or C.local_path("ORBITOL_NEQUIP_TS")).resolve()),
        nequip_oeq_path=str(Path(oeq_path).resolve()) if oeq_path else "",
        nequip_relax_mode=args.relax_mode, thermo_stability_device=args.device,
        thermo_ppd_mp=str(args.ppd_path.resolve()), thermo_stability_batch=args.relax_batch_size,
    )


def evaluation_settings(args):
    args = resolve_snapshot_paths(args)
    reference = args.symmetry_reference.expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(f"Prebuilt symmetry reference required: {reference}")
    thermo = thermo_settings(args)
    model_path = thermo.model_path if thermo is not None else None
    return dict(
        core_root=core.CORE_ROOT, data_root=str(args.data_root.resolve()),
        device=args.device, sample_count=args.sample_count, sun_k=args.sun_k,
        tag="sample_ema", no_thermo_before_steps=0, thermo=asdict(thermo) if thermo else None,
        mlip_model_path=model_path, mlip_model_sha256=_sha256(model_path) if model_path else None,
        phase_diagram_sha256=_sha256(thermo.thermo_ppd_mp) if thermo else None,
        symmetry_reference=str(reference),
        sample_seed=args.seed,
        protocol="legacy_raw_un_subset" if args.sun_k > 0 else "relaxed_endpoints_v1",
        joint_subset="raw_UN" if args.sun_k > 0 else "all_attempts",
    )


def saved_thermo(settings):
    """Old pool manifests explicitly describe the historical TorchScript backend."""
    if settings["thermo"] is None:
        return None
    return ThermoSettings(**{"nequip_backend": "torchscript", "thermo_relax_steps": 200, **settings["thermo"]})


def build_evaluator(settings):
    # Validate with the same rule used by training, then pin the validated
    # histograms in memory. The lazy panel can never rebuild the reference.
    from orbitol.train import verify_symmetry_reference
    reference = settings["symmetry_reference"]
    verify_symmetry_reference(settings["data_root"], reference)
    saved = read_json(reference)
    evaluator = DNGEvaluator(
        settings["data_root"], device=settings["device"], sample_count=settings["sample_count"],
        sun_k=settings["sun_k"], no_thermo_before_steps=0,
        thermo=saved_thermo(settings), symmetry_cache=reference,
    )
    evaluator.symmetry_panel._reference = saved["histograms"]
    evaluator.sample_seed = settings["sample_seed"]
    return evaluator


def score_samples(rows, records, *, evaluator, step, requested_G=None):
    """CPU-only scoring, shared by posthoc pools and completed GPU stages."""
    from orbitol.eval_protocol import score
    from src.eval.stability import load_phase_diagram

    return score(
        _sample_items(rows), records,
        references=dict(ref_structs=evaluator.ref_structs, novelty_ref_structs=evaluator.novelty_ref_structs,
                        symmetry_panel=evaluator.symmetry_panel),
        ppd=load_phase_diagram(evaluator.thermo.thermo_ppd_mp) if evaluator.thermo is not None else None,
        settings=dict(thermo=asdict(evaluator.thermo) if evaluator.thermo is not None else None,
                      sample_count=evaluator.sample_count, sample_seed=evaluator.sample_seed,
                      sun_k=evaluator.sun_k, no_thermo_before_steps=0),
        tag=evaluator.tag, step=step, requested_G=requested_G,
    )


def sample_metrics(result):
    """Preserve generator diagnostics alongside protocol metrics in every stage."""
    from orbitol.train import DECODE_FAILURES

    metrics = {}
    if "contains_a" in result:
        metrics["orbit/contains_a_rate"] = float(result["contains_a"].float().mean())
    if "failure_counts" in result:
        reasons = set(DECODE_FAILURES) | result["failure_counts"].keys()
        metrics.update({f"orbit/failures/{key}": result["failure_counts"].get(key, 0) for key in reasons})
    return metrics


def sample_stage(args):
    """Commit an independently evaluable pool, with pool.json as commit marker."""
    start = time.perf_counter()
    args.snapshot = args.snapshot.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    # Resolve controls from the snapshot before comparing an existing pool.
    # After pruning, the committed sampler config supplies the same defaults.
    existing = verify_pool(args.out) if (args.out / "pool.json").exists() else None
    if existing is not None:
        sampler = existing["sampler"]
        model_args = sampler.get("cfg", {}) if args.kind == "orbitol" else sampler
        args = resolve_snapshot_paths(args, model_args)
        pool = existing
        if (pool["kind"], pool["snapshot"], pool["n"], pool["seed"], sampler.get("overrides", {})) != (
                args.kind, str(args.snapshot), args.sample_count, args.seed, args.sampler_overrides):
            raise ValueError("Existing pool has different sampling provenance")
        if args.snapshot.exists() and any(pool[key] != value for key, value in identity(args.snapshot).items()):
            raise ValueError("Snapshot changed after this pool was committed")
        return pool
    core.verify_core()
    before = identity(args.snapshot)
    step, model_args = _snapshot_info(args.snapshot, args.kind)
    args = resolve_snapshot_paths(args, model_args)
    settings = evaluation_settings(args)
    snapshot_hash = _sha256(args.snapshot)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.kind == "crystalite":
        sampler = _sampler_settings(model_args, args)
        # The unchanged core CLI writes non-atomically. Keep all its output in
        # a private staging directory; only our durable commit is published.
        with tempfile.TemporaryDirectory(prefix=".sampling-", dir=args.out) as staging:
            result = _batch_samples(sample_crystalite(args.snapshot, Path(staging), sampler))
        rows = result
    else:
        overrides = dict(args.sampler_overrides)
        resolved_cfg = replace(TrainConfig(**model_args), **overrides)
        model_args = {**model_args, **overrides, "compile_trunk": resolved_cfg.compile_trunk,
                      "sample_token_budget": resolved_cfg.sample_token_budget}
        chunk = int(model_args.get("extra", {}).get("sample_chunk_size", 256))
        sampler = dict(cfg=model_args, overrides=overrides, sample_mode="ema", sample_seed=args.seed,
                       num_samples=args.sample_count, sample_chunk_size=chunk, device=args.device)
        result = sample_orbitol(args.snapshot, n=args.sample_count, seed=args.seed,
                                device=args.device, chunk=chunk, overrides=overrides)
        rows = _batch_samples(result["rows"])
    if len(rows["A0"]) != args.sample_count:
        raise ValueError("Sampler returned a different sample count than requested")
    if identity(args.snapshot) != before:
        raise RuntimeError("Snapshot changed during sampling; pool was not committed")
    settings.update(kind=args.kind, snapshot=str(args.snapshot), snapshot_sha256=snapshot_hash,
                    step=step, sampler=sampler,
                    seeds=dict(sampler=args.seed, evaluator=args.seed, sun_base=args.seed + step,
                               sun_selection=args.seed + step + sum(map(ord, "sample_ema"))))
    with atomic_writer(args.out / "samples.pt") as stream:
        torch.save(result, stream)
    _write_json(args.out / "evaluation_settings.json", settings)
    pool = dict(kind=args.kind, snapshot=str(args.snapshot), snapshot_sha256=snapshot_hash,
                **before, step=step, n=args.sample_count, seed=args.seed, sampler=sampler,
                samples_sha256=_sha256(args.out / "samples.pt"), status="complete",
                sample_seconds=time.perf_counter() - start)
    _write_json(args.out / "pool.json", pool)  # publish LAST
    return pool


def _load_pool_rows(out, settings, samples_sha256=None):
    data = (Path(out) / "samples.pt").read_bytes()
    if samples_sha256 is not None and hashlib.sha256(data).hexdigest() != samples_sha256:
        raise ValueError(f"Sample hash mismatch: {out}")
    result = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    rows = result["rows"] if settings["kind"] == "orbitol" else result
    # "all_relaxed_v1" is a legacy name for the same protocol.
    if settings["kind"] == "orbitol" and settings.get("protocol") not in {"all_relaxed_v1", "relaxed_endpoints_v1"}:
        # Pools committed before Orbitol emitted Niggli cells carry skewed primitive
        # bases, which the pinned evaluator rejects by angle; reducing is idempotent.
        from orbitol.decode import reduce_rows
        rows = reduce_rows(rows)
    items = _sample_items(rows)
    if len(rows["A0"]) != settings["sample_count"]:
        raise ValueError("Pool sample count differs from its manifest")
    return result, rows, items


def _score_pool_worker(connection, out, settings, step, samples_sha256):
    """Load and score a pool in a fresh CPU process, receiving only its endpoints."""
    try:
        from orbitol.eval_protocol import complete_score, score_raw
        from src.eval.stability import load_phase_diagram

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.setsid()
        affinity = sorted(os.sched_getaffinity(0))
        if len(affinity) > 1:
            os.sched_setaffinity(0, affinity[1:])
        result, _, items = _load_pool_rows(out, settings, samples_sha256)
        evaluator = build_evaluator(settings)
        references = evaluator.references
        scoring_settings = {**evaluator.settings, "no_thermo_before_steps": 0}
        ppd = load_phase_diagram(evaluator.thermo.thermo_ppd_mp)
        prepared = score_raw(items, references=references, settings=scoring_settings, tag=evaluator.tag,
                             requested_G=result.get("requested_G"))
        connection.send(("ready", None))
        try:
            relaxed = connection.recv()
        except EOFError:
            return
        os.sched_setaffinity(0, affinity)
        scored = complete_score(prepared, relaxed, references=references, ppd=ppd,
                                settings=scoring_settings, tag=evaluator.tag, step=step)
        connection.send(("ok", scored))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def _overlap_score(out, settings, step, samples_sha256, relax):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe()
    process = context.Process(target=_score_pool_worker,
                              args=(sender, str(out), settings, step, samples_sha256))
    reaped = False

    def reap(terminate=False):
        nonlocal reaped
        if process.pid is None or reaped:
            return
        if terminate:
            try:
                # Check ownership without reaping: the group ID must remain reserved.
                os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    os.kill(process.pid, signal.SIGKILL)
            except (ChildProcessError, ProcessLookupError):
                pass
        process.join()
        reaped = True

    def receive(expected):
        status, payload = receiver.recv()
        if status == "error":
            raise RuntimeError(f"scoring failed in the child process:\n{payload}")
        if status != expected:
            raise RuntimeError(f"Unexpected scoring response: {status}")
        return payload

    try:
        process.start()
        sender.close()
        relaxed = relax()
        try:
            receive("ready")
            receiver.send(relaxed)
            scored = receive("ok")
        except (EOFError, BrokenPipeError, ConnectionResetError):
            reap(terminate=True)
            raise RuntimeError(f"scoring process died with exit code {process.exitcode}") from None
        reap()
        if process.exitcode != 0:
            raise RuntimeError(f"scoring process died with exit code {process.exitcode}")
        return relaxed, scored
    finally:
        receiver.close()
        sender.close()
        reap(terminate=True)


def evaluate_stage(out, *, evaluator=None, symmetry_reference=None):
    """Evaluate a verified pool without ever opening its original snapshot."""
    start = time.perf_counter()
    out = Path(out).resolve()
    pool = verify_pool(out)
    settings = read_json(out / "evaluation_settings.json")
    original_reference = settings["symmetry_reference"]
    if symmetry_reference is not None:
        settings["symmetry_reference"] = str(Path(symmetry_reference).expanduser().resolve())
    if not Path(settings["symmetry_reference"]).is_file():
        raise FileNotFoundError(f"Prebuilt symmetry reference required: {settings['symmetry_reference']}")
    if (settings["kind"], settings["step"], settings["sample_count"], settings["sample_seed"]) != (
            pool["kind"], pool["step"], pool["n"], pool["seed"]):
        raise ValueError("Pool/evaluation settings mismatch")
    if settings["mlip_model_path"] is not None and _sha256(settings["mlip_model_path"]) != settings["mlip_model_sha256"]:
        raise ValueError("NequIP model hash differs from the pool settings")
    if settings.get("phase_diagram_sha256") is not None and (
            _sha256(settings["thermo"]["thermo_ppd_mp"]) != settings["phase_diagram_sha256"]):
        raise ValueError("Phase-diagram hash differs from the pool settings")
    result, rows, items = _load_pool_rows(out, settings)
    if settings["sun_k"] == 0 and settings["thermo"] is not None:
        from orbitol.eval_protocol import relax_all
        from orbitol.eval_score import RELAXED_FIELDS, write_csv

        thermo = saved_thermo(settings)

        def relax():
            relaxer = (evaluator.relaxer if evaluator is not None and evaluator.relaxer is not None
                       else thermo.make_relaxer())
            return relax_all(items, relaxer, thermo.thermo_relax_steps)

        if evaluator is None:
            relaxed, (metrics, records) = _overlap_score(out, settings, pool["step"], pool["samples_sha256"], relax)
        else:
            relaxed = relax()
            metrics, records = score_samples(rows, relaxed, evaluator=evaluator, step=pool["step"],
                                             requested_G=result.get("requested_G"))
        write_csv(out / "relaxed.csv.gz", relaxed, fields=RELAXED_FIELDS)
        write_csv(out / "per_sample.csv.gz", records)
    else:
        # The explicitly selected historical subset retains its core selection rule.
        evaluator = evaluator if evaluator is not None else build_evaluator(settings)
        metrics = evaluator.evaluate(rows, step=pool["step"], requested_G=result.get("requested_G"))
        records = evaluator.relaxation_diagnostics() if hasattr(evaluator, "relaxation_diagnostics") else []
    # Diagnostics are observations of the completed pinned evaluation. Commit
    # records before metrics, so a complete pool evaluation includes both.
    with atomic_writer(out / "relaxed_records.jsonl") as stream:
        for record in records:
            stream.write((json.dumps(record, allow_nan=False) + "\n").encode())
    if settings["sun_k"] > 0 or settings["thermo"] is None:
        metrics.update(variant_metrics(records, checked_count=metrics.get("joint/checked_count", len(records))))
    if pool["kind"] == "orbitol":
        metrics.update(sample_metrics(result))
    metrics["posthoc/sample_seconds"] = pool["sample_seconds"]
    metrics["posthoc/evaluate_seconds"] = time.perf_counter() - start
    metrics = {key: float(value) if math.isfinite(value) else None for key, value in metrics.items()}
    if settings["symmetry_reference"] != original_reference:
        _write_json(out / "evaluation_settings.json", settings)
    _write_json(out / "metrics.json", metrics)
    return metrics


def main(argv=None):
    args = parse_args(argv)
    args.out = args.out.expanduser().resolve()
    if args.stage in {"all", "sample"}:
        pool = sample_stage(args)
        if args.stage == "sample":
            print(args.out / "pool.json")
            return pool
    pool = verify_pool(args.out)
    if pool["kind"] != args.kind:
        raise ValueError("--kind differs from the saved pool")
    metrics = evaluate_stage(args.out, symmetry_reference=args.symmetry_reference)
    _log_wandb(args, metrics, pool["step"], read_json(args.out / "evaluation_settings.json"), pool)
    print(args.out / "metrics.json")
    return metrics


if __name__ == "__main__":
    main()
