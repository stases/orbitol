"""spglib symmetry distributions, with failures outside the SG histogram.

Shares and distances condition on successful detection. Counts and failure rates
always use all input indices. JSD uses natural logarithms (range 0..ln(2)); the
effective group count is exp(entropy). The reference is the dataset's *token cache*,
decoded in exactly the same way as generated samples.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache
import json
import multiprocessing
import os
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import spglib

import orbitol.core  # noqa: F401  bind the pin before importing src
from orbitol.data import file_sha256
from orbitol.spglib_worker import detect_cells
from src.data.mp20_tokens import tokens_to_structure


SYMPRECS = (0.1, 0.01)
ANGLE_TOLERANCE = 5.0
SYSTEMS = (
    "triclinic", "monoclinic", "orthorhombic", "tetragonal",
    "trigonal", "hexagonal", "cubic",
)


def crystal_system(group: int) -> int:
    """Return the seven-system index for an international SG number."""
    if not 1 <= group <= 230:
        raise ValueError(f"Invalid space group: {group}")
    return int(np.searchsorted([2, 15, 74, 142, 167, 194, 230], group))


def detect_group(structure, symprec: float = 0.1) -> int:
    """Zero denotes a failed detection, never P1."""
    if structure is None:
        return 0
    try:
        cell = (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers)
        if not all(np.isfinite(x).all() for x in cell) or len(structure) == 0:
            return 0
        dataset = spglib.get_symmetry_dataset(
            cell, symprec=symprec, angle_tolerance=ANGLE_TOLERANCE,
        )
        return int(dataset.number) if dataset is not None else 0
    except Exception:
        return 0


def _cell(structure):
    if structure is None or len(structure) == 0:
        return None
    cell = (np.asarray(structure.lattice.matrix, dtype=float), np.asarray(structure.frac_coords, dtype=float),
            np.asarray(structure.atomic_numbers, dtype=int))
    return cell if all(np.isfinite(x).all() for x in cell) else None


def detect_many(structures: Iterable, symprecs=SYMPRECS, *, workers=None, chunk=128,
                function=detect_cells) -> np.ndarray:
    """Space groups (n, len(symprecs)), 0 for a failed detection, with spglib in spawned workers.

    spglib can corrupt the heap on pathological cells and abort the whole process, which no
    Python exception handler can catch. A crashed worker loses only its chunk, which is split and
    retried until the offending cell is isolated and recorded as a failed detection.
    """
    cells = [_cell(s) for s in structures]
    groups = np.zeros((len(cells), len(symprecs)), dtype=np.int16)
    if not any(cell is not None for cell in cells):
        return groups
    context = multiprocessing.get_context("spawn")

    def run(batches, n_workers):
        """Fill groups for each batch that completes; return the batches lost to a broken pool."""
        lost = []
        with ProcessPoolExecutor(n_workers, mp_context=context) as pool:
            futures = {pool.submit(function, [cells[i] for i in indices], tuple(symprecs), ANGLE_TOLERANCE): indices
                       for indices in batches}
            for future in as_completed(futures):
                try:
                    groups[futures[future]] = future.result()
                except BrokenProcessPool:
                    lost.append(futures[future])
        return lost

    # A crash breaks the whole pool, so every unfinished batch is lost with the culprit:
    # split lost batches until single cells remain, then rerun each single cell alone.
    pending = [list(range(i, min(i + chunk, len(cells)))) for i in range(0, len(cells), chunk)]
    workers = workers or min(8, len(os.sched_getaffinity(0)))
    singles = []
    while pending:
        lost = run(pending, workers)
        singles += [indices for indices in lost if len(indices) == 1]
        pending = [half for indices in lost if len(indices) > 1
                   for half in (indices[:len(indices) // 2], indices[len(indices) // 2:])]
    for indices in singles:
        run([indices], 1)  # a cell that crashes on its own keeps its zero (failed detection)
    return groups


def group_histogram(groups) -> np.ndarray:
    """231 bins: bin zero is a failure count, bins 1..230 are SG counts."""
    groups = np.asarray(groups, dtype=int)
    if np.any((groups < 0) | (groups > 230)):
        raise ValueError("Detected groups must be between 0 and 230")
    return np.bincount(groups, minlength=231)


def _probabilities(histogram) -> np.ndarray:
    counts = np.asarray(histogram, dtype=float)[1:]
    return counts / counts.sum() if counts.sum() else np.full(230, np.nan)


def _system_probabilities(probabilities) -> np.ndarray:
    return np.add.reduceat(np.asarray(probabilities), [0, 2, 15, 74, 142, 167, 194])


def crystal_system_tv(groups, reference_histogram) -> float:
    p = _system_probabilities(_probabilities(group_histogram(groups)))
    q = _system_probabilities(_probabilities(reference_histogram))
    return float(0.5 * np.abs(p - q).sum())


def distribution_metrics(groups, reference_histogram) -> dict[str, float]:
    counts = group_histogram(groups)
    n = int(counts.sum())
    p, q = _probabilities(counts), _probabilities(reference_histogram)
    systems = _system_probabilities(p)
    mid = (p + q) / 2
    if np.isfinite(mid).all():
        jsd = 0.0
        for prob in (p, q):
            keep = prob > 0
            jsd += 0.5 * float(np.sum(prob[keep] * np.log(prob[keep] / mid[keep])))
    else:
        jsd = float("nan")
    positive = p[p > 0]
    effective = float(np.exp(-np.sum(positive * np.log(positive)))) if positive.size else float("nan")
    return {
        "count": float(n),
        "detected_count": float(n - counts[0]),
        "failure_count": float(counts[0]),
        "failure_rate": float(counts[0] / n) if n else float("nan"),
        "p1_share": float(p[0]),
        "crystal_system_tv": crystal_system_tv(groups, reference_histogram),
        "space_group_jsd": jsd,
        "effective_groups": effective,
        **{f"crystal_system/{name}": float(systems[i]) for i, name in enumerate(SYSTEMS)},
    }


@lru_cache(maxsize=1)
def supergroup_closure() -> tuple[frozenset[int], ...]:
    """Transitive inverse of pyxtal's maximal t- and k-subgroup tables.

    This is containment between space-group *types*, not a certificate for an
    embedding, origin, or orientation in a particular cell.
    """
    from pyxtal.symmetry import Group

    parents = [set() for _ in range(231)]
    for number in range(1, 231):
        group = Group(number)
        for table in (group.get_max_t_subgroup(), group.get_max_k_subgroup()):
            for subgroup in table["subgroup"]:
                parents[int(subgroup)].add(number)
    closure = [frozenset()]
    for number in range(1, 231):
        seen, todo = {number}, [number]
        while todo:
            for parent in parents[todo.pop()]:
                if parent not in seen:
                    seen.add(parent)
                    todo.append(parent)
        closure.append(frozenset(seen))
    return tuple(closure)


def bootstrap_ci(values_fn: Callable[[np.ndarray], float], n: int,
                 reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval, resampling n sample indices with replacement.

    ``values_fn(indices)`` must recompute the statistic on those indices (e.g.
    TV or the mean of fixed per-sample joint flags). Undefined replicates make
    the interval undefined; they are not silently dropped.
    """
    if n < 0 or reps <= 0:
        raise ValueError("n must be nonnegative and reps must be positive")
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = [values_fn(rng.integers(0, n, size=n)) for _ in range(reps)]
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def reference_valid(saved, expected, train_cache):
    """Exact legacy metadata, or the same content at another path/mtime."""
    meta = saved.get("metadata", {})
    if {k: v for k, v in meta.items() if k != "train_cache_sha256"} == expected:
        return True
    loose = {k: v for k, v in expected.items() if k not in ("train_cache", "mtime_ns")}
    return ("train_cache_sha256" in meta
            and {k: v for k, v in meta.items()
                 if k not in ("train_cache", "mtime_ns", "train_cache_sha256")} == loose
            and meta["train_cache_sha256"] == file_sha256(train_cache))


class SymmetryPanel:
    """Lazy train reference, persisted next to the run, never in processed/."""

    def __init__(self, train_cache: str | Path, train_items: list,
                 cache_path: str | Path):
        self.train_cache = Path(train_cache).resolve()
        self.train_items = train_items
        self.cache_path = Path(cache_path).resolve()
        if self.cache_path.is_relative_to(self.train_cache.parent):
            raise ValueError("Symmetry reference must be outside the read-only processed directory")
        self._reference: dict[str, list[int]] | None = None

    @property
    def reference(self) -> dict[str, list[int]]:
        if self._reference is not None:
            return self._reference
        stat = self.train_cache.stat()
        metadata = {
            "schema": 1, "train_cache": str(self.train_cache),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "count": len(self.train_items), "spglib": spglib.__version__,
            "symprecs": list(SYMPRECS), "angle_tolerance": ANGLE_TOLERANCE,
            "decoder": "crystalite-03603fe:tokens_to_structure",
        }
        if self.cache_path.exists():
            saved = json.loads(self.cache_path.read_text())
            histograms = saved.get("histograms", {})
            valid = all(
                len(histograms.get(str(p), [])) == 231
                and sum(histograms[str(p)]) == len(self.train_items)
                and all(isinstance(c, int) and c >= 0 for c in histograms[str(p)])
                for p in SYMPRECS
            )
            if reference_valid(saved, metadata, self.train_cache) and valid:
                self._reference = histograms
                return histograms
        histograms = {str(p): np.zeros(231, dtype=np.int64) for p in SYMPRECS}
        for item in self.train_items:
            try:
                structure = tokens_to_structure(item)
            except Exception:
                structure = None
            for p in SYMPRECS:
                histograms[str(p)][detect_group(structure, p)] += 1
        self._reference = {key: value.tolist() for key, value in histograms.items()}
        metadata["train_cache_sha256"] = file_sha256(self.train_cache)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps({
            "metadata": metadata, "histograms": self._reference,
        }, indent=2) + "\n")
        return self._reference

    def evaluate(self, raw_structures: list, relaxed_records: list, *, requested_G=None,
                 precomputed_groups=None) -> dict[str, float]:
        metrics = {}
        relaxed = [record.final_structure for record in relaxed_records]
        for label, structures in (("raw", raw_structures), ("relaxed", relaxed)):
            detected = (detect_many(structures) if precomputed_groups is None
                        else np.asarray(precomputed_groups[label]))
            if detected.shape != (len(structures), len(SYMPRECS)):
                raise ValueError(f"Invalid {label} symmetry groups shape: {detected.shape}")
            for j, p in enumerate(SYMPRECS):
                groups = detected[:, j]
                prefix = f"sym/{label}/{p:g}"
                metrics.update({
                    f"{prefix}/{key}": value
                    for key, value in distribution_metrics(groups, self.reference[str(p)]).items()
                })
                if label == "relaxed" and p == 0.1 and requested_G is not None:
                    requested = np.asarray(requested_G, dtype=int)
                    if requested.shape != (len(raw_structures),) or np.any((requested < 1) | (requested > 230)):
                        raise ValueError("requested_G must contain one space group (1..230) per sample")
                    targets = requested[[r.sample_index for r in relaxed_records]]
                    closure = supergroup_closure()
                    retained = sum(int(g) in closure[int(target)] for g, target in zip(groups, targets))
                    system_retained = sum(
                        g > 0 and crystal_system(int(g)) == crystal_system(int(target))
                        for g, target in zip(groups, targets)
                    )
                    n = len(groups)
                    metrics.update({
                        f"{prefix}/retention_count": float(retained),
                        f"{prefix}/retention_total": float(n),
                        f"{prefix}/retention": retained / n if n else float("nan"),
                        f"{prefix}/crystal_system_retention": system_retained / n if n else float("nan"),
                    })
        return metrics


def main(argv=None):
    """Build the same train reference as DNGEvaluator.symmetry_panel.reference."""
    import argparse
    import torch
    from orbitol.contracts import DATASET_NAMES, resolve_dataset_paths

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--dataset", choices=DATASET_NAMES, default="mp20")
    build.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    paths = resolve_dataset_paths(args.dataset)
    source = Path(paths.data_root) / "processed/mp20_tokens_train.pt"
    # Invoke the panel directly: DNGEvaluator also loads validation and constructs
    # novelty references, neither of which contributes to this train histogram.
    items = torch.load(source, map_location="cpu", weights_only=False)
    panel = SymmetryPanel(source, items, args.out or paths.symmetry_reference)
    reference = panel.reference
    print(json.dumps(dict(path=str(panel.cache_path), count=len(items),
                          failures={p: hist[0] for p, hist in reference.items()})))
    return reference


if __name__ == "__main__":
    main()
