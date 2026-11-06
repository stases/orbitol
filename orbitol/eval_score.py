"""Score a completed GPU evaluation and log its training-step metrics."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from orbitol.posthoc_io import atomic_writer, read_json, write_json

RELAXED_FIELDS = ("index", "constructed", "relax_status", "reason", "final_structure", "energy_eV", "nsteps",
                  "initial_energy_eV", "initial_fmax", "final_fmax", "final_max_force")


def parse_args(argv=None, *, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args(argv)
    if args.step < 0:
        parser.error("step must be nonnegative")
    args.run_dir = args.run_dir.expanduser().resolve()
    return args


def stage_directory(run_dir, step):
    return Path(run_dir).expanduser().resolve() / "eval" / f"step_{step:07d}"


def write_csv(path, records, *, fields=None):
    """Publish a gzip CSV only after every row and the gzip trailer are durable."""
    fields = fields or list(dict.fromkeys(key for record in records for key in record))
    with atomic_writer(path) as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for record in records:
                    row = dict(record)
                    for key, value in row.items():
                        if hasattr(value, "as_dict"):
                            value = value.as_dict()
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, allow_nan=False)
                        row[key] = value
                    writer.writerow(row)


def read_relaxed(path):
    from pymatgen.core import Structure

    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        records = []
        for row in csv.DictReader(stream):
            row["index"] = int(row["index"])
            row["constructed"] = row["constructed"] == "True"
            row["final_structure"] = Structure.from_dict(json.loads(row["final_structure"])) if row["final_structure"] else None
            row["energy_eV"] = float(row["energy_eV"]) if row["energy_eV"] else None
            for key in ("initial_energy_eV", "initial_fmax", "final_fmax", "final_max_force"):
                row[key] = float(row[key]) if row.get(key) else None
            row["nsteps"] = int(row["nsteps"]) if row["nsteps"] else None
            row["reason"] = row["reason"] or None
            records.append(row)
    return records


def clean_metrics(metrics):
    return {key: float(value) if value is not None and math.isfinite(value) else None
            for key, value in metrics.items()}


def run_stage(run_dir, step):
    directory = stage_directory(run_dir, step)
    done = directory / "score.done.json"
    if done.is_file():
        return read_json(done)
    if not (directory / "gpu.done.json").is_file():
        raise FileNotFoundError(f"GPU stage is incomplete: {directory}")

    import torch
    from orbitol.posthoc_eval import build_evaluator, score_samples, sample_metrics, _log_wandb

    start = time.perf_counter()
    samples = torch.load(directory / "samples.pt", map_location="cpu", weights_only=False)
    settings = {**samples["evaluation_settings"], "device": "cpu"}
    if settings["step"] != step:
        raise ValueError("Sample settings disagree with the requested step")
    records = read_relaxed(directory / "relaxed.csv.gz")
    metrics, per_sample = score_samples(
        samples["rows"], records, evaluator=build_evaluator(settings), step=step,
        requested_G=samples.get("requested_G"),
    )
    metrics.update(sample_metrics(samples))
    metrics = clean_metrics(metrics)
    write_csv(directory / "per_sample.csv.gz", per_sample)
    write_json(directory / "metrics.json", metrics)
    cfg = settings["sampler"]["cfg"]
    args = SimpleNamespace(
        out=directory, kind="orbitol", wandb_project=cfg.get("wandb_project"),
        wandb_name=f"{cfg.get('wandb_name') or Path(run_dir).name}-eval",
        sample_count=settings["sample_count"], sun_k=settings["sun_k"], seed=settings["sample_seed"],
    )
    _log_wandb(args, metrics, step, settings, dict(kind="orbitol", sampler=settings["sampler"]))
    receipt = dict(step=step, job_id=os.environ.get("SLURM_JOB_ID"), seconds=time.perf_counter() - start)
    write_json(done, receipt)
    return receipt


def main(argv=None):
    args = parse_args(argv)
    return run_stage(args.run_dir, args.step)


if __name__ == "__main__":
    main()
