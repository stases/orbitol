"""Build the common, provenanced table for the Alex reference fix-sets."""
from __future__ import annotations

import argparse
import atexit
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
import gzip
import hashlib
from importlib.metadata import version
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import subprocess
import time
import warnings

import orbitol.core  # Bind the pinned core before importing its candidate helper.


REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "orbitol/assets/alex_reference"
TOLERANCE = 1e-6
CHUNK_SIZE = 512
_ENV = None
_PROCESSOR = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_inputs(files):
    inputs = {}
    for name, path in files.items():
        started = time.monotonic()
        logging.info("Hashing %s", path)
        inputs[name] = {"path": str(path), "sha256": sha256(path)}
        logging.info("Hashed %s in %.2f s", path, time.monotonic() - started)
    return inputs


def _read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def _potcar_element(titel):
    if not isinstance(titel, str) or len(titel.split()) < 2:
        return None
    return titel.split()[1].split("_", 1)[0]


def _load_legacy(path):
    references = _read_json(path)
    references = references.get("mp2023", references)
    legacy = {}
    for element, entry in sorted(references.items()):
        specs = [spec for spec in entry["parameters"]["potcar_spec"]
                 if _potcar_element(spec.get("titel")) == element]
        hashes = {spec.get("hash") for spec in specs}
        if len(hashes) != 1 or not all(isinstance(h, str) and h for h in hashes):
            raise ValueError(f"Missing or ambiguous legacy POTCAR for {element}: {path}")
        legacy[element] = {"titel": min(spec["titel"] for spec in specs), "hash": hashes.pop()}
    if not legacy:
        raise ValueError(f"No elemental references in {path}")
    return legacy


def _load_shifts(path, legacy):
    asset = _read_json(path)
    if asset.get("schema") != 1 or not isinstance(asset.get("elements"), dict):
        raise ValueError(f"Expected shifts schema 1 with an elements map: {path}")
    for element, record in asset["elements"].items():
        old_hash = record.get("legacy", {}).get("hash")
        new_hash = record.get("new", {}).get("hash")
        if element not in legacy or old_hash != legacy[element]["hash"]:
            raise ValueError(f"Shift asset legacy hash disagrees with MP for {element}")
        if not isinstance(new_hash, str) or not new_hash or new_hash == old_hash:
            raise ValueError(f"Invalid new POTCAR hash for {element}")
        if record.get("target") not in {"legacy", "new"}:
            raise ValueError(f"Invalid shift target for {element}")
        shift = float(record["shift"])
        if not math.isfinite(shift) or record.get("status") not in {"applied", "not_applied"}:
            raise ValueError(f"Invalid shift or status for {element}")
        if record["status"] == "not_applied" and shift != 0:
            raise ValueError(f"not_applied shift must be zero for {element}")
    return asset["elements"]


def _load_known_bad(path):
    actions = {}
    for record in _read_json(path)["entries"]:
        key = record.get("key")
        if key is None:
            continue
        if key in actions or not isinstance(record.get("action"), str):
            raise ValueError(f"Duplicate key or missing action in {path}: {key}")
        actions[key] = record["action"]
    return actions


def _potcar_status(element, potcar_hash, legacy, shifts):
    if element in legacy and potcar_hash == legacy[element]["hash"]:
        return "legacy"
    # A new-file hash is usable only with a fitted shift; unknown offsets are excluded under potcar_align.
    if element in shifts and shifts[element]["status"] == "applied" and potcar_hash == shifts[element]["new"]["hash"]:
        return "listed"
    return "unlisted"


def _potcars(parameters, elements, amounts, legacy, shifts):
    triples = Counter()
    hashes = defaultdict(set)
    specs = parameters.get("potcar_spec")
    unlisted = not isinstance(specs, list) or not specs
    for spec in specs if isinstance(specs, list) else []:
        spec = spec if isinstance(spec, dict) else {}
        titel, potcar_hash = spec.get("titel"), spec.get("hash")
        titel = titel if isinstance(titel, str) else None
        potcar_hash = potcar_hash if isinstance(potcar_hash, str) and potcar_hash else None
        element = _potcar_element(titel)
        triples[element, titel, potcar_hash] += 1
        hashes[element].add(potcar_hash)
        unlisted |= _potcar_status(element, potcar_hash, legacy, shifts) == "unlisted"
    unlisted |= set(hashes) != set(elements)
    nonlegacy, contributions = [], {}
    for element, amount in zip(elements, amounts, strict=True):
        observed = hashes.get(element, set())
        if any(h is not None and h != legacy.get(element, {}).get("hash") for h in observed):
            nonlegacy.append(element)
        if len(observed) != 1 or None in observed:
            unlisted = True
            continue
        potcar_hash = next(iter(observed))
        shift = shifts.get(element)
        if shift is None:
            continue
        # Corrected raw energy is raw - potcar_shift for either target scale.
        if shift["target"] == "legacy" and potcar_hash == shift["new"]["hash"]:
            contributions[element] = amount * float(shift["shift"])
        elif shift["target"] == "new" and potcar_hash == legacy[element]["hash"]:
            contributions[element] = -amount * float(shift["shift"])
    return nonlegacy, math.fsum(contributions.values()), bool(unlisted), triples, contributions


def _nonzero_hubbards(parameters):
    return {(element, float(value)) for element, value in (parameters.get("hubbards") or {}).items()
            if float(value) != 0}


class _EntryProcessor:
    def __init__(self, legacy, shifts, known_bad):
        from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
        from src.utils.sample_stats import _build_mp2020_parameters

        self.compat = MaterialsProject2020Compatibility(check_potcar=False)
        self.parameters = _build_mp2020_parameters
        self.legacy, self.shifts, self.known_bad = legacy, shifts, known_bad

    def correction(self, entry):
        try:
            corrected = self.compat.process_entry(entry.copy(), on_error="raise")
            if corrected is None:
                return math.nan, "mp2020_removed"
            correction = float(corrected.correction)
            if not math.isfinite(correction):
                return math.nan, "mp2020_nonfinite_correction"
            return correction, None
        except Exception as exc:
            return math.nan, f"{type(exc).__name__}: {exc}"

    def process(self, key, record):
        from pymatgen.analysis.structure_analyzer import oxide_type
        from pymatgen.core import Composition, Structure
        from pymatgen.entries.computed_entries import ComputedStructureEntry

        comp = Composition(record["composition"]).element_composition
        elements = sorted(comp.get_el_amt_dict())
        amounts = [float(comp[element]) for element in elements]
        structure = Structure.from_dict(record["structure"])
        energy, stored_correction = float(record["energy"]), float(record.get("correction", 0))
        entry_id = record.get("entry_id")
        entry_id = str(entry_id) if entry_id is not None else None
        parameters = record.get("parameters") or {}
        data = copy.deepcopy(record.get("data") or {})
        stored_oxide = data.get("oxide_type")
        data.setdefault("oxidation_states", {})
        oxidation_states_nonempty = bool(data["oxidation_states"])
        stored = ComputedStructureEntry(structure, energy, composition=comp, entry_id=entry_id,
                                        parameters=copy.deepcopy(parameters), data=data)
        reproduced, stored_error = self.correction(stored)
        synth = self.parameters(composition=structure.composition, mp2020_compat=self.compat)
        candidate = ComputedStructureEntry(structure=structure, composition=structure.composition,
                                           energy=energy, parameters=synth, entry_id=entry_id)
        recorrected, recorrect_error = self.correction(candidate)
        nonlegacy, shift, unlisted, triples, contributions = _potcars(
            parameters, elements, amounts, self.legacy, self.shifts)
        row = {
            "key": key, "entry_id": entry_id,
            "source": entry_id.split("-", 1)[0] if entry_id else "other",
            "chemsys": "-".join(elements), "elements": elements, "amounts": amounts,
            "natoms": float(comp.num_atoms), "raw_energy": energy,
            "stored_correction": stored_correction,
            "stored_reproduced": abs(reproduced - stored_correction) <= TOLERANCE,
            "stored_error": stored_error, "recorrected_correction": recorrected,
            "recorrect_error": recorrect_error,
            "synth_params_mismatch": (synth["run_type"] != parameters.get("run_type")
                                      or _nonzero_hubbards(synth) != _nonzero_hubbards(parameters)),
            "oxide_type_changed": stored_oxide != oxide_type(structure, 1.05),
            "oxidation_states_nonempty": oxidation_states_nonempty,
            "nonlegacy_elements": nonlegacy, "potcar_shift": shift,
            "unlisted_nonlegacy": unlisted, "known_bad": self.known_bad.get(key),
        }
        return row, triples, contributions


def _entry_key(key):
    prefix, separator, index = key.rpartition(b".")
    return bool(separator and b"." in prefix and index.isdigit())


def _open_lmdb(path):
    import lmdb

    return lmdb.open(str(path), subdir=Path(path).is_dir(), readonly=True, lock=False, readahead=False)


def _key_ranges(path, limit):
    """Inspect keys only; each entry value is read once, by its assigned worker."""
    ranges, start, end, count = [], None, None, 0
    with _open_lmdb(path) as env, env.begin() as txn:
        for key in txn.cursor().iternext(values=False):
            if not _entry_key(key):
                continue
            if start is None:
                start = key
            end = key
            count += 1
            if count % CHUNK_SIZE == 0:
                ranges.append((start, end))
                start = None
            if limit is not None and count >= limit:
                break
    if start is not None:
        ranges.append((start, end))
    return ranges


def _init_worker(path, legacy, shifts, known_bad):
    global _ENV, _PROCESSOR
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    warnings.filterwarnings("ignore", message="Failed to guess oxidation states for Entry.*")
    _PROCESSOR = _EntryProcessor(legacy, shifts, known_bad)
    _ENV = _open_lmdb(path)
    atexit.register(_ENV.close)


def _process_range(bounds):
    start, end = bounds
    rows, triples, shifted = [], Counter(), Counter()
    with _ENV.begin() as txn:
        cursor = txn.cursor()
        valid = cursor.set_range(start)
        while valid and cursor.key() <= end:
            key = cursor.key()
            if _entry_key(key):
                try:
                    record = pickle.loads(cursor.value())
                    if isinstance(record, dict) and "composition" in record and "energy" in record:
                        row, inventory, contributions = _PROCESSOR.process(key.decode(), record)
                        rows.append(row)
                        triples.update(inventory)
                        shifted.update(element for element, value in contributions.items() if value != 0)
                except Exception as exc:
                    raise RuntimeError(f"Cannot build Alex entry {key!r}: {exc}") from exc
            valid = cursor.next()
    return rows, triples, shifted


def _batches(ranges, workers, initargs):
    if workers == 1:
        _init_worker(*initargs)
        try:
            for bounds in ranges:
                yield _process_range(bounds)
        finally:
            _ENV.close()
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=initargs) as executor:
        # Bound outstanding results while preserving key order and fixed Parquet row groups.
        ranges = iter(ranges)
        pending = deque()
        for _ in range(2 * workers):
            bounds = next(ranges, None)
            if bounds is not None:
                pending.append(executor.submit(_process_range, bounds))
        while pending:
            yield pending.popleft().result()
            bounds = next(ranges, None)
            if bounds is not None:
                pending.append(executor.submit(_process_range, bounds))


def _schema():
    import pyarrow as pa

    return pa.schema([
        ("key", pa.string()), ("entry_id", pa.string()), ("source", pa.string()),
        ("chemsys", pa.string()), ("elements", pa.list_(pa.string())),
        ("amounts", pa.list_(pa.float64())), ("natoms", pa.float64()),
        ("raw_energy", pa.float64()), ("stored_correction", pa.float64()),
        ("stored_reproduced", pa.bool_()), ("stored_error", pa.string()),
        ("recorrected_correction", pa.float64()), ("recorrect_error", pa.string()),
        ("synth_params_mismatch", pa.bool_()), ("oxide_type_changed", pa.bool_()),
        ("oxidation_states_nonempty", pa.bool_()),
        ("nonlegacy_elements", pa.list_(pa.string())), ("potcar_shift", pa.float64()),
        ("unlisted_nonlegacy", pa.bool_()), ("known_bad", pa.string()),
    ])


def build(lmdb, mp_elemental, known_bad, shifts, out, *, workers=1, limit=None, source_gz=None):
    """Write deterministic rows and provenance; limit selects at most N entry keys."""
    started = time.monotonic()
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    paths = {name: Path(path).resolve() for name, path in
             dict(lmdb=lmdb, mp_elemental=mp_elemental, known_bad=known_bad, shifts=shifts).items()}
    if source_gz is not None:
        paths["source_gz"] = Path(source_gz).resolve()
    for name, path in paths.items():
        if not path.exists():
            hint = "; supply an existing asset with --shifts" if name == "shifts" else ""
            raise FileNotFoundError(f"Missing {name} input: {path}{hint}")
    legacy = _load_legacy(paths["mp_elemental"])
    shift_map = _load_shifts(paths["shifts"], legacy)
    bad = _load_known_bad(paths["known_bad"])
    files = {name: path / "data.mdb" if name == "lmdb" and path.is_dir() else path
             for name, path in paths.items()}
    out = Path(out).resolve()
    table, manifest_path = out / "entries.parquet", out / "manifest.json"
    diagnostics_path = out / "diagnostics.jsonl"
    if table.exists() or manifest_path.exists() or diagnostics_path.exists():
        raise FileExistsError(f"Build outputs already exist in {out}; use a new output directory")
    identities = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in files.items()}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import pyarrow as pa
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    temporary = out / ".entries.parquet.tmp"
    diagnostics_temporary = out / ".diagnostics.jsonl.tmp"
    counts = dict(entries=0, stored_reproduced_false=0, recorrect_changed=0, recorrect_failures=0,
                  synth_params_mismatch=0, oxide_type_changed=0, oxidation_states_nonempty=0,
                  unlisted_nonlegacy=0)
    sources, actions, inventory, shifted = Counter(), Counter(), Counter(), Counter()
    diagnostic_flags = ("synth_params_mismatch", "oxide_type_changed", "oxidation_states_nonempty")
    schema = _schema()
    initargs = (str(paths["lmdb"]), legacy, shift_map, bad)
    try:
        with ThreadPoolExecutor(max_workers=1) as hashing:
            # Hashing the full archive also runs for a limited smoke build.
            inputs_future = hashing.submit(_hash_inputs, files)
            logging.info("Partitioning entry keys from %s", paths["lmdb"])
            ranges = _key_ranges(paths["lmdb"], limit)
            logging.info("Processing %d key ranges with %d workers", len(ranges), workers)
            with (
                pq.ParquetWriter(temporary, schema, compression="zstd") as writer,
                diagnostics_temporary.open("w", encoding="utf-8") as diagnostics,
            ):
                for rows, triples, shift_counts in _batches(ranges, workers, initargs):
                    if rows:
                        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    inventory.update(triples)
                    shifted.update(shift_counts)
                    for row in rows:
                        counts["entries"] += 1
                        sources[row["source"]] += 1
                        counts["stored_reproduced_false"] += not row["stored_reproduced"]
                        counts["recorrect_failures"] += row["recorrect_error"] is not None
                        counts["recorrect_changed"] += (abs(row["recorrected_correction"]
                                                           - row["stored_correction"]) > TOLERANCE)
                        counts["unlisted_nonlegacy"] += row["unlisted_nonlegacy"]
                        flags = []
                        for name in diagnostic_flags:
                            counts[name] += row[name]
                            if row[name]:
                                flags.append(name)
                        if not row["stored_reproduced"]:
                            flags.append("stored_reproduced_false")
                        if row["recorrect_error"] is not None:
                            flags.append("recorrect_failure")
                        if flags:
                            diagnostic = {"key": row["key"], "flags": sorted(flags)}
                            if row["recorrect_error"] is not None:
                                diagnostic["recorrect_error"] = row["recorrect_error"]
                            diagnostics.write(json.dumps(diagnostic, sort_keys=True, allow_nan=False) + "\n")
                        if row["known_bad"] is not None:
                            actions[row["known_bad"]] += 1
            logging.info("Processed %d entries; waiting for input hashes", counts["entries"])
            inputs = inputs_future.result()
        for name, path in files.items():
            if identities[name] != (path.stat().st_size, path.stat().st_mtime_ns):
                raise RuntimeError(f"Input changed during build: {path}")
        counts.update(per_source=dict(sorted(sources.items())), known_bad_by_action=dict(sorted(actions.items())),
                      nonzero_shift_by_element=dict(sorted(shifted.items())))
        core_helper = Path(orbitol.core.CORE_ROOT) / "src/utils/sample_stats.py"
        manifest = {
            "schema": 1, "inputs": inputs, "legacy_potcars": legacy,
            "potcar_inventory": [dict(element=element, titel=titel, hash=potcar_hash, count=count,
                                      status=_potcar_status(element, potcar_hash, legacy, shift_map))
                                 for (element, titel, potcar_hash), count in sorted(
                                     inventory.items(), key=lambda item: tuple(x or "" for x in item[0]))],
            "pymatgen_version": version("pymatgen"),
            "git_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
            "builder": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
            "core_helper": {"path": str(core_helper), "sha256": sha256(core_helper)},
            "limit": limit, "selection": "lexicographic entry keys", "correction_tolerance_eV": TOLERANCE,
            "counts": counts, "diagnostics_sha256": sha256(diagnostics_temporary),
            "table_sha256": sha256(temporary),
            "elapsed_seconds": time.monotonic() - started,
        }
        temporary.replace(table)
        diagnostics_temporary.replace(diagnostics_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    finally:
        temporary.unlink(missing_ok=True)
        diagnostics_temporary.unlink(missing_ok=True)
    return manifest


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--mp-elemental", type=Path, required=True)
    parser.add_argument("--known-bad", type=Path, default=ASSETS / "known_bad.json")
    parser.add_argument("--shifts", type=Path, default=ASSETS / "potcar_shifts.json")
    parser.add_argument("--source-gz", type=Path, help="Optional compressed archive, hashed separately")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Smoke only: first N entry keys in LMDB order")
    args = parser.parse_args(argv)
    try:
        manifest = build(**vars(args))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({"counts": manifest["counts"], "elapsed_seconds": manifest["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
