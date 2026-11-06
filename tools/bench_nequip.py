"""GPU-only NequIP relaxer benchmark; no sampling, training or core edits."""
from __future__ import annotations

import argparse
import cProfile
from contextlib import nullcontext
import json
from pathlib import Path
import pstats
import time

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Lattice, Structure

import orbitol.core as core
from orbitol import contracts as C
from orbitol.nequip_relax import (
    RobustRelaxer, make_batch_relaxer, make_sequential_relaxer,
)
from orbitol.posthoc_eval import _sha256
from src.utils.stability_logger import StabilityLogger


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _relax_indexed(relaxer, structures, steps):
    """Use the logger's offender protocol, retaining order for comparison."""
    try:
        return relaxer.relax_many(structures, steps=steps), {}
    except RuntimeError as exc:
        indices = StabilityLogger._extract_batched_failure_indices(exc, len(structures))
        if len(structures) == 1:
            return [None], {0: str(exc)}
        if indices:
            failed = set(indices)
            keep = [i for i in range(len(structures)) if i not in failed]
            results, errors = _relax_indexed(relaxer, [structures[i] for i in keep], steps) if keep else ([], {})
            output = [None] * len(structures)
            for i, result in zip(keep, results, strict=True):
                output[i] = result
            return output, {**{i: str(exc) for i in failed}, **{keep[i]: msg for i, msg in errors.items()}}
        middle = len(structures) // 2
        left, le = _relax_indexed(relaxer, structures[:middle], steps)
        right, re = _relax_indexed(relaxer, structures[middle:], steps)
        return left + right, {**le, **{i + middle: msg for i, msg in re.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path,
                        default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="Write cProfile data and CPU time breakdowns")
    parser.add_argument("--include-garbage", action="store_true")
    parser.add_argument("--reference-filter", action="store_true",
                        help="Use the original uncached ASE Frechet filter for comparison")
    parser.add_argument("--jit-optimization", choices=["default", "off"], default="default",
                        help="Diagnostic only: compare a separate process with JIT optimization off")
    args = parser.parse_args(argv)
    args.model = args.model or Path(C.local_path("ORBITOL_NEQUIP_TS"))
    args.val_csv = args.val_csv or Path(C.dataset_spec("mp20").data_root) / "raw/val.csv"
    if min(args.n, args.batch_size, args.repeats) <= 0 or args.steps < 0:
        parser.error("n, batch-size and repeats must be positive; steps must be nonnegative")
    core.verify_core()
    torch.set_num_threads(1)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.val_csv, nrows=args.n)
    structures = [Structure.from_str(cif, fmt="cif") for cif in frame["cif"]]
    if not structures:
        raise ValueError("Validation CSV contains no structures")
    if args.include_garbage:
        structures.append(Structure(Lattice.cubic(2.2), ["O"] * 20,
                                    np.random.default_rng(0).uniform(0, 1, (20, 3))))
    report = dict(
        model=str(args.model.resolve()), model_sha256=_sha256(args.model),
        val_csv=str(args.val_csv.resolve()), n=len(structures), steps=args.steps,
        batch_size=args.batch_size, repeats=args.repeats, device=args.device,
        jit_optimization=args.jit_optimization, profiled=args.profile,
        reference_filter=args.reference_filter, fmax=.005, max_force_abort=1e6,
        optimizer="FIRE", cell_filter="frechet", paths={},
    )
    execution = torch.jit.optimized_execution(False) if args.jit_optimization == "off" else nullcontext()
    with execution:
        for mode, factory in (("sequential", make_sequential_relaxer), ("batch", make_batch_relaxer)):
            _sync(args.device)
            start = time.perf_counter()
            backend = factory(args.model, args.device)
            if mode == "sequential" and args.reference_filter:
                from ase.filters import FrechetCellFilter
                backend.filter_cls = FrechetCellFilter
            relaxer = RobustRelaxer(backend)
            _sync(args.device)
            load_seconds = time.perf_counter() - start
            runs = []
            profiler = cProfile.Profile() if args.profile else None
            if profiler:
                profiler.enable()
            for repeat in range(args.repeats):
                energies, errors, calls = [], {}, []
                chunk_size = 1 if mode == "sequential" else args.batch_size
                for offset in range(0, len(structures), chunk_size):
                    group = structures[offset:offset + chunk_size]
                    _sync(args.device)
                    start = time.perf_counter()
                    results, failures = _relax_indexed(relaxer, group, args.steps)
                    _sync(args.device)
                    elapsed = time.perf_counter() - start
                    calls.append(dict(offset=offset, count=len(group), seconds=elapsed))
                    energies.extend(None if result is None else float(result["trajectory"].energies[-1])
                                    for result in results)
                    errors.update({i + offset: message for i, message in failures.items()})
                seconds = sum(call["seconds"] for call in calls)
                runs.append(dict(repeat=repeat, seconds=seconds, seconds_per_structure=seconds / len(structures),
                                 energies_eV=energies, failure_count=len(errors), failures=errors, calls=calls))
                print(f"{mode} repeat={repeat}: {seconds / len(structures):.4f} s/structure, failures={len(errors)}", flush=True)
            if profiler:
                profiler.disable()
                profiler.dump_stats(str(args.out / f"{mode}.prof"))
                with (args.out / f"{mode}_profile.txt").open("w") as stream:
                    stats = pstats.Stats(profiler, stream=stream)
                    stats.sort_stats("cumulative").print_stats(45)
                    stats.sort_stats("tottime").print_stats(30)
            report["paths"][mode] = dict(load_seconds=load_seconds, runs=runs)
            del backend, relaxer
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    agreements = []
    for seq, batch in zip(report["paths"]["sequential"]["runs"], report["paths"]["batch"]["runs"], strict=True):
        pairs = [(i, a, b) for i, (a, b) in enumerate(zip(seq["energies_eV"], batch["energies_eV"], strict=True))
                 if a is not None and b is not None]
        deltas = [abs(a - b) for _, a, b in pairs]
        per_atom = [abs(a - b) / len(structures[i]) for i, a, b in pairs]
        agreements.append(dict(repeat=seq["repeat"], compared=len(pairs),
                               max_abs_eV=max(deltas, default=None),
                               mean_abs_eV=float(np.mean(deltas)) if deltas else None,
                               max_abs_eV_per_atom=max(per_atom, default=None)))
    report["energy_agreement"] = agreements
    (args.out / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(agreements, indent=2))
    print(args.out / "benchmark.json")


if __name__ == "__main__":
    main()
