"""Sample EMA weights, relax every attempt, then submit CPU scoring."""
from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess
import time
from types import SimpleNamespace

from orbitol.eval_score import RELAXED_FIELDS, parse_args, stage_directory, write_csv
from orbitol.posthoc_io import atomic_writer, fsync_dir, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def versions():
    result = {"python": platform.python_version()}
    for package in ("torch", "nequip", "openequivariance", "torch-sim-atomistic"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def submit_score(run_dir, step):
    command = [str(ROOT / "slurm/submit"), "--parsable", str(ROOT / "slurm/eval_score.sbatch"),
               "--run-dir", str(Path(run_dir).resolve()), "--step", str(step)]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True,
                            env={**os.environ, "ORBITOL_REPO_ROOT": str(ROOT)}, timeout=60)
    return result.stdout.strip().split(";")[0]


def run_stage(run_dir, step):
    directory = stage_directory(run_dir, step)
    done = directory / "gpu.done.json"
    if done.is_file():
        return read_json(done)

    import torch
    from orbitol.contracts import TrainConfig
    from orbitol.eval_adapter import _sample_items
    from orbitol.eval_protocol import relax_all
    from orbitol.eval_snapshot import load_eval_snapshot
    from orbitol.posthoc_eval import evaluation_settings, sample_orbitol, saved_thermo

    start = time.perf_counter()
    snapshot = Path(run_dir).resolve() / "eval_snapshots" / f"step_{step:07d}.pt"
    checkpoint = load_eval_snapshot(snapshot)
    if int(checkpoint["step"]) != step:
        raise ValueError("Snapshot step disagrees with the requested step")
    cfg = TrainConfig(**checkpoint["cfg"])
    del checkpoint
    if cfg.sun_k > 0:
        raise ValueError("Async stages require sun_k=0; use sync or posthoc for the legacy subset")
    settings = evaluation_settings(SimpleNamespace(
        kind="orbitol", dataset_config=asdict(cfg), device="cuda", evaluation_overrides=set(),
    ))
    chunk = int(cfg.extra.get("sample_chunk_size", 256))
    samples = sample_orbitol(snapshot, n=cfg.sample_count, seed=settings["sample_seed"], device="cuda", chunk=chunk)
    items = _sample_items(samples["rows"])
    if len(items) != cfg.sample_count:
        raise ValueError("Sampler returned a different sample count than requested")
    sample_seconds = time.perf_counter() - start
    settings.update(step=step, sampler=dict(cfg=asdict(cfg), sample_mode="ema", sample_seed=settings["sample_seed"],
                                           num_samples=cfg.sample_count, sample_chunk_size=chunk, device="cuda"))
    samples["evaluation_settings"] = settings
    with atomic_writer(directory / "samples.pt") as stream:
        torch.save(samples, stream)
    thermo = saved_thermo(settings)
    if thermo is None:
        raise ValueError("GPU evaluation requires thermodynamics")
    relax_start = time.perf_counter()
    records = relax_all(items, thermo.make_relaxer(), cfg.thermo_relax_steps)
    if [record["index"] for record in records] != list(range(cfg.sample_count)):
        raise ValueError("Relaxation must return one ordered record per attempt")
    write_csv(directory / "relaxed.csv.gz", records, fields=RELAXED_FIELDS)
    receipt = dict(step=step, job_id=os.environ.get("SLURM_JOB_ID"), versions=versions(),
                   sample_seconds=sample_seconds, relax_seconds=time.perf_counter() - relax_start,
                   seconds=time.perf_counter() - start)
    write_json(done, receipt)
    snapshot.unlink()
    fsync_dir(snapshot.parent)
    submit_score(run_dir, step)
    return receipt


def main(argv=None):
    args = parse_args(argv, description=__doc__)
    return run_stage(args.run_dir, args.step)


if __name__ == "__main__":
    main()
