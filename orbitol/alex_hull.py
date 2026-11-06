"""Fixed Alex-MP reference hulls with independent reference corrections.

Only composition amounts and corrected eV/atom are retained, grouped by chemical
system. Pandas is needed to build a reference, but not to unpickle or score it.
The pickle retains source provenance and counts, including both assets' hashes.

The baseline launcher slurm/crystalite_baseline.sbatch exposes only the
pinned core on PYTHONPATH; it must also expose Orbitol to load these pickles.
"""

import warnings

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from scipy.optimize import linprog


FLAGS = frozenset({"known_bad", "recorrect", "potcar_align"})


class AlexHull:
    """Hull energies on a fixed reference, with cached entry arrays for each queried element set."""

    def __init__(self, entries, manifest):
        self._entries = entries
        self.manifest = manifest
        self._cache = {}

    @classmethod
    def from_table(cls, table_path, flags: frozenset[str]):
        """Read a single Parquet file and the manifest.json next to it.

        ``potcar_shift`` is a total-energy offset to *subtract*. Corrections
        are total energies too; division by ``natoms`` happens last. As in the
        released reference scorer, NaN-energy entries do not define the hull.
        """
        import hashlib
        import json
        from pathlib import Path

        import pandas as pd

        flags = frozenset(flags)
        if unknown := flags - FLAGS:
            raise ValueError(f"Unknown AlexHull flags: {', '.join(sorted(unknown))}")
        path = Path(table_path)
        if not path.is_file():
            raise ValueError(f"Expected a single Parquet file: {path.resolve()}")
        manifest_bytes = (path.parent / "manifest.json").read_bytes()
        table_manifest = json.loads(manifest_bytes)
        try:
            asset_sha256 = {
                "known_bad": table_manifest["inputs"]["known_bad"]["sha256"],
                "potcar_shifts": table_manifest["inputs"]["shifts"]["sha256"],
            }
        except (KeyError, TypeError) as exc:
            raise ValueError("Table manifest must include inputs.known_bad and inputs.shifts SHA256s") from exc
        table = pd.read_parquet(path)
        required = {
            "key", "chemsys", "elements", "amounts", "natoms", "raw_energy",
            "stored_correction", "recorrected_correction", "potcar_shift",
            "unlisted_nonlegacy", "known_bad",
        }
        if missing := required - set(table.columns):
            raise ValueError(f"Missing AlexHull table columns: {', '.join(sorted(missing))}")
        keep = np.ones(len(table), dtype=bool)
        if "known_bad" in flags:
            actions = table["known_bad"]
            if not (actions.isna() | actions.isin(["exclude", "retain_flagged"])).all():
                raise ValueError("Expected known_bad actions exclude, retain_flagged, or null")
            keep &= ~actions.isin(["exclude"]).to_numpy()
        if "potcar_align" in flags:
            unlisted = table["unlisted_nonlegacy"]
            if unlisted.isna().any() or not unlisted.isin([True, False]).all():
                raise ValueError("Expected boolean values in unlisted_nonlegacy")
            keep &= ~unlisted.to_numpy(dtype=bool)
        table = table.loc[keep].copy()
        energy = table["raw_energy"].to_numpy(dtype=np.float64, copy=True)
        if "potcar_align" in flags:
            energy -= table["potcar_shift"].to_numpy(dtype=np.float64)
        correction = "recorrected_correction" if "recorrect" in flags else "stored_correction"
        energy += table[correction].to_numpy(dtype=np.float64)
        if np.isinf(energy).any():
            raise ValueError("Infinite corrected reference energy")
        is_nan = np.isnan(energy)
        n_dropped_nan = int(is_nan.sum())
        if n_dropped_nan and table_manifest["counts"]["recorrect_failures"] == 0:
            raise ValueError(f"Found {n_dropped_nan} NaN reference energies but counts.recorrect_failures is 0")
        table["_energy"] = energy
        table = table.loc[~is_nan]
        entries = {}
        for chemsys, group in table.groupby("chemsys", sort=True):
            symbols = tuple(chemsys.split("-"))
            amounts = np.empty((len(group), len(symbols)), dtype="<f8")
            for i, (elements, counts) in enumerate(zip(group["elements"], group["amounts"])):
                elements = tuple(elements)
                if len(elements) != len(symbols) or set(elements) != set(symbols):
                    raise ValueError(f"Elements do not match chemical system {chemsys}")
                if len(counts) != len(elements):
                    raise ValueError(f"Amounts do not match elements in {chemsys}")
                amounts[i] = [counts[elements.index(symbol)] for symbol in symbols]
            natoms = group["natoms"].to_numpy(dtype=np.float64)
            if not (amounts.sum(axis=1) == natoms).all():
                raise ValueError(f"Invalid composition amounts or natoms in {chemsys}")
            energy_per_atom = np.ascontiguousarray(group["_energy"].to_numpy() / natoms, dtype="<f8")
            entries[symbols] = (amounts, energy_per_atom)

        digest = hashlib.sha256()
        for symbols, arrays in entries.items():
            for name, array in zip(("amounts", "energy_per_atom"), arrays):
                header = json.dumps([symbols, name, array.shape, array.dtype.str]).encode()
                digest.update(len(header).to_bytes(8, "little"))
                digest.update(header)
                digest.update(memoryview(array))
        variant = "original" if not flags else "v1" if flags == FLAGS else "+".join(sorted(flags))
        manifest = {
            "format_version": 1,
            "variant": variant,
            "flags": sorted(flags),
            "table_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "table_manifest": {key: table_manifest[key] for key in (
                "inputs", "counts", "git_commit", "pymatgen_version", "table_sha256",
            )},
            "asset_sha256": asset_sha256,
            "arrays_sha256": digest.hexdigest(),
            "n_entries": len(table),
            "n_chemsys": len(entries),
            "n_dropped_nan": n_dropped_nan,
            "missing_terminals": sorted({symbol for symbols in entries for symbol in symbols
                                         if (symbol,) not in entries}),
        }
        return cls(entries, manifest)

    def _system(self, comp):
        """Element order, fractional compositions (n, k) and eV/atom of every entry in comp's element set."""
        from itertools import combinations

        symbols = tuple(sorted(el.symbol for el in comp.elements))
        missing = [symbol for symbol in symbols if (symbol,) not in self._entries]
        if missing:
            raise ValueError(
                "Unsupported elements in AlexHull (no terminal reference): " + ", ".join(missing)
            )
        if symbols not in self._cache:
            fractions, energies = [], []
            for size in range(1, len(symbols) + 1):
                for subsystem in combinations(symbols, size):
                    if subsystem not in self._entries:
                        continue
                    amounts, energy_per_atom = self._entries[subsystem]
                    block = np.zeros((len(amounts), len(symbols)))
                    block[:, [symbols.index(symbol) for symbol in subsystem]] = (
                        amounts / amounts.sum(axis=1, keepdims=True))
                    fractions.append(block)
                    energies.append(energy_per_atom)
            self._cache[symbols] = (np.concatenate(fractions), np.concatenate(energies))
            if len(self._cache) > 4096:
                del self._cache[next(iter(self._cache))]
        return symbols, *self._cache[symbols]

    def get_hull_energy_per_atom(self, comp):
        """Return equilibrium energy in eV/atom on the fixed reference.

        The lower hull at comp is the cheapest mixture of reference entries with comp's
        atomic fractions: min e.x subject to F^T x = f(comp), x >= 0. This linear program
        gives pymatgen's PhaseDiagram value without building the convex hull, which made
        scoring 10k structures across thousands of chemical systems take an hour.
        """
        symbols, fractions, energies = self._system(comp)
        target = [comp.get_atomic_fraction(symbol) for symbol in symbols]
        result = linprog(energies, A_eq=fractions.T, b_eq=target, bounds=(0, None), method="highs",
                         options={"primal_feasibility_tolerance": 1e-10,
                                  "dual_feasibility_tolerance": 1e-10})
        if result.status != 0:
            raise ValueError(f"Hull linear program failed for {comp.reduced_formula}: {result.message}")
        return float(result.fun)

    def get_e_above_hull(self, entry, allow_negative=False, on_error="raise", **kwargs):
        """Score an already corrected candidate without inserting it into the hull.

        Follows pymatgen's PhaseDiagram.get_e_above_hull below the hull: on_error is
        "raise", "warn" or "ignore"; other keywords (check_stable) do not change the value.
        """
        e_above = entry.energy_per_atom - self.get_hull_energy_per_atom(entry.composition)
        if e_above < -PhaseDiagram.numerical_tol and not allow_negative:
            message = f"No valid decomposition found for {entry}! (e_h: {e_above})"
            if on_error == "raise":
                raise ValueError(message)
            if on_error == "warn":
                warnings.warn(message)
            return None
        return e_above

    def __getstate__(self):
        return {"_entries": self._entries, "manifest": self.manifest}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = {}


def main():
    import argparse
    import json
    from pathlib import Path
    import pickle

    # The canonical class name keeps `python -m` pickles independent of __main__.
    from orbitol.alex_hull import AlexHull

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True, help="Parquet file with an adjacent manifest.json")
    parser.add_argument("--variant", required=True, help="original, v1, or flags joined by +")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.variant == "v1":
        flags = FLAGS
    elif args.variant == "original":
        flags = frozenset()
    else:
        flags = frozenset(args.variant.split("+"))
    if unknown := flags - FLAGS:
        parser.error(f"Unknown AlexHull flags: {', '.join(sorted(unknown))}")
    hull = AlexHull.from_table(args.table, flags)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        pickle.dump(hull, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(hull.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
