"""Dataset-selected orbit/atom caches, deterministic augmentation and batch resume.

Orbit-view geometric certificates use float64 BEFORE float32 cache quantisation.
Atom view starts from Crystalite's float32 tokens and caches its exact training
latent. Cache manifests record each view's round-trip precision and tolerances.

Use ResumableSampler as DataLoader's ``batch_sampler``. Call ``mark_consumed()``
only after an optimiser step, and checkpoint its state. Iteration/prefetch never
advances the saved cursor. Set the epoch on both sampler and dataset each epoch.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
from importlib.metadata import version
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import re
import time
import warnings

import numpy as np
from pymatgen.core import Lattice, Structure
import spglib
import torch
from torch.utils.data import Dataset, Sampler

import orbitol.core  # noqa: F401 -- bind the pinned core before importing src
from orbitol import contracts as C
from orbitol.standardise import (
    ANGLE_TOLERANCE, FRAC_TOL, SYMPREC, Standardised, fractional_cost,
    species_permutation, standardise,
)
from orbitol.tables import TableSet, load_tables
from src.data.mp20_tokens import decode_Y1
from src.models.lattice_repr import y1_to_lattice_latent

K_CACHE = 20
OrbitBatch = dict[str, torch.Tensor | list[str]]


def family_ltri(lat, family):
    """Fill dependent entries in the input dtype, without clipping clean cells."""
    lat = np.array(lat, copy=True)
    name = C.FAMILIES[int(family)]
    if name != "triclinic":
        lat[[1, 4]] = 0
    if name not in ("triclinic", "monoclinic"):
        lat[3] = 0
    if name in ("tetragonal", "cubic"):
        lat[2] = lat[0]
    if name == "cubic":
        lat[5] = lat[0]
    if name == "hexagonal":
        lat[1] = -0.5 * np.exp(lat[0])
        lat[2] = lat[0] + np.asarray(np.log(np.sqrt(3) / 2), dtype=lat.dtype)
    return lat


def ltri_to_lattice(lat):
    p = np.asarray(lat, dtype=np.float64)
    return np.array([[np.exp(p[0]), 0, 0], [p[1], np.exp(p[2]), 0],
                     [p[3], p[4], np.exp(p[5])]], dtype=np.float64)


def lattice_to_ltri(lattice, family):
    lattice = np.asarray(lattice, dtype=np.float64)
    gram = lattice @ lattice.T
    L = np.linalg.cholesky(gram)
    lat = family_ltri(np.array([np.log(L[0, 0]), L[1, 0], np.log(L[1, 1]),
                               L[2, 0], L[2, 1], np.log(L[2, 2])]), family)
    tied = ltri_to_lattice(lat)
    if np.linalg.norm(tied @ tied.T - gram) > 1e-8 * np.linalg.norm(gram):
        raise ValueError("Idealised lattice violates its family's ltri ties")
    return lat


@lru_cache(maxsize=None)
def _selector(matrix, dof):
    """Recover E^T from E^T A=I, using only the frozen integer chart."""
    A = np.asarray(matrix).reshape(3, 3)[:, :dof]
    axes = []
    for e in np.eye(dof):
        candidates = np.flatnonzero(np.all(A == e, axis=1))
        if not len(candidates):
            raise ValueError("Chart has no free-axis selector")
        axes.append(int(candidates[0]))
    return tuple(axes)


def expand_tokens(row, A0, U, tables):
    """Expand conventional-cell members in float64, retaining species."""
    a = tables.arrays
    xyz, types = [], []
    if not (len(row) == len(A0) == len(U)):
        raise ValueError("Token lengths differ")
    for r, z, u in zip(row, A0, U):
        m = int(a["row_mult"][r])
        if r <= 0 or not m:
            raise ValueError(f"Invalid real row {r}")
        xyz.append(a["row_W"][r, :m] @ np.asarray(u, dtype=np.float64) + a["row_v"][r, :m])
        types.extend([z] * m)
    return np.mod(np.concatenate(xyz), 1.0), np.asarray(types, dtype=np.int64)


def validate_round_trip(tokens, ideal: Standardised, tables):
    """Fatal species-bijective fractional and rotation-invariant lattice check."""
    xyz, types = expand_tokens(tokens["row"], tokens["A0"], tokens["U"], tables)
    permutation = species_permutation(xyz, types, ideal.positions, ideal.types)
    delta = xyz - ideal.positions[permutation]
    frac_error = float(np.max(np.abs(delta - np.rint(delta))))
    L = ltri_to_lattice(tokens["lat"])
    target = np.linalg.cholesky(ideal.lattice @ ideal.lattice.T)
    lattice_error = float(np.linalg.norm(L - target) / np.linalg.norm(target))
    if not np.isfinite(lattice_error) or lattice_error > 1e-8:
        raise ValueError(f"Lattice round trip relative error {lattice_error:g} exceeds 1e-8")
    return frac_error, lattice_error


def tokenise(structure, tables: TableSet, *, mp_id=""):
    """Tokenise an input or already-standardised cell; retain float64 until caching."""
    ideal = structure if isinstance(structure, Standardised) else standardise(structure, tables=tables)
    a, group = tables.arrays, ideal.G
    start, count = int(a["group_row_start"][group]), int(a["group_row_count"][group])
    nr = int(a["group_ops_count"][group])
    R, t = a["group_ops_R"][group, :nr], a["group_ops_t"][group, :nr]
    assigned = np.zeros(len(ideal.positions), dtype=bool)
    rows, elements, coords = [], [], []
    for j, x in enumerate(ideal.positions):
        if assigned[j]:
            continue
        same = np.flatnonzero(ideal.types == ideal.types[j])
        cost = fractional_cost(R @ x + t, ideal.positions[same])
        nearest = cost.argmin(axis=1)
        if np.any(cost[np.arange(nr), nearest] > FRAC_TOL):
            raise ValueError(f"Frozen orbit is missing atoms for {mp_id}")
        orbit = same[np.unique(nearest)]
        if assigned[orbit].any():
            raise ValueError(f"Overlapping frozen orbits for {mp_id}")
        candidates = np.arange(start, start + count)
        candidates = candidates[a["row_mult"][candidates] == len(orbit)]
        matches = []
        for r in candidates:
            d = int(a["row_dof"][r])
            axes = _selector(tuple(a["row_A"][r].ravel()), d)
            rhs = ideal.positions[orbit] - a["row_b"][r]
            us = np.zeros((len(orbit), 3), dtype=np.float64)
            us[:, :d] = np.mod(rhs[:, axes], 1.0)
            residual = us @ a["row_A"][r].T - rhs
            on_chart = np.max(np.abs(residual - np.rint(residual)), axis=1) <= FRAC_TOL
            for u in us[on_chart]:
                decoded = a["row_W"][r, :len(orbit)] @ u + a["row_v"][r, :len(orbit)]
                try:
                    species_permutation(decoded, ideal.types[orbit], ideal.positions[orbit], ideal.types[orbit])
                except ValueError:
                    continue
                matches.append((int(r), u))
                break
        if len(matches) != 1:
            raise ValueError(f"{mp_id}: orbit of size {len(orbit)} has {len(matches)} matching rows")
        r, u = matches[0]
        rows.append(r)
        elements.append(int(ideal.types[j]))
        coords.append(u)
        assigned[orbit] = True
    if not assigned.all() or not 1 <= len(rows) <= K_CACHE:
        raise ValueError(f"{mp_id}: incomplete cell or more than {K_CACHE} orbit tokens")
    family = int(a["group_family"][group])
    result = dict(
        mp_id=str(mp_id), G=group, hall=ideal.hall, family=family, n_orbits=len(rows),
        row=np.asarray(rows, dtype=np.int64), A0=np.asarray(elements, dtype=np.int64),
        U=np.asarray(coords, dtype=np.float64), lat=lattice_to_ltri(ideal.lattice, family),
        n_conv=len(ideal.positions), n_prim=int(a["row_mprim"][rows].sum()),
        disp_max=ideal.disp_max, disp_mean=ideal.disp_mean, strain=ideal.strain,
        transformation_matrix=ideal.transformation_matrix, origin_shift=ideal.origin_shift,
        rotation=ideal.rotation, frozen_shift=ideal.frozen_shift,
    )
    frac, lattice = validate_round_trip(result, ideal, tables)
    result.update(round_trip_ok=True, round_trip_frac_error=frac, round_trip_lattice_error=lattice)
    return result


@lru_cache(maxsize=None)
def _polar_shift_map(matrix, dof):
    """The frozen chart's pseudoinverse is independent of item and epoch."""
    return np.linalg.pinv(np.asarray(matrix).reshape(3, 3)[:, :dof])


def augment_tokens(row, U, lat, group, tables, generator):
    """Apply all three equivalences in float64; return rows, U, lat and applied P."""
    a, cs = tables.arrays, tables.cosets
    row = np.array(row, dtype=np.int64, copy=True)
    U, lat = np.array(U, dtype=np.float64, copy=True), np.array(lat, dtype=np.float64, copy=True)
    P = np.eye(3)
    # P1 has no table alternatives and a single identity chart image, but its polar space is all of
    # R^3: the polar shift is a uniform random translation, as Crystalite's own training applies
    # (train_crystalite.py, augment_translate=True). P1 structures train as G = 1 orbit items.
    if group == 1:
        U[:, :3] = np.mod(U[:, :3] + generator.random(3), 1.0)
        return row, U, lat, P
    ci, nc = int(cs.group_start[group]), int(cs.group_count[group])
    proper = np.arange(ci, ci + nc)[cs.proper[ci:ci + nc]]
    if not len(proper):
        raise ValueError(f"No proper normaliser cosets for group {group}")
    c = int(generator.choice(proper))
    P = cs.P[c].copy()
    if cs.det[c] != 1 or not np.isclose(np.linalg.det(P), 1, atol=1e-12, rtol=0):
        raise ValueError("Augmentation requires det P = +1")
    lo, hi = int(cs.map_start[c]), int(cs.map_start[c + 1])
    source = cs.source_row[lo:hi]
    lookup = {int(r): lo + i for i, r in enumerate(source)}
    maps = np.asarray([lookup[int(r)] for r in row])
    U = np.einsum("kij,kj->ki", cs.S[maps], U) + cs.s[maps]
    row = cs.target_row[maps].astype(np.int64)
    if cs.reexpress_lattice[c]:
        inverse = np.linalg.inv(P)
        # Rows L' = P^-T L, equivalently Gram' = P^-T Gram P^-1.
        lat = lattice_to_ltri(inverse.T @ ltri_to_lattice(lat), a["group_family"][group])
    shift = generator.random(3) * a["group_polar"][group]
    for k, r in enumerate(row):
        d = int(a["row_dof"][r])
        if d:
            U[k, :d] += _polar_shift_map(tuple(a["row_A"][r].ravel()), d) @ shift
        image = int(generator.integers(int(a["row_img_count"][r])))
        U[k] = a["row_img_S"][r, image] @ U[k] + a["row_img_s"][r, image]
        U[k, :d] = np.mod(U[k, :d], 1.0)
        U[k, d:] = 0
    return row, U, lat, P


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_content_sha256(npz_path):
    """Advisory digest of sorted, little-endian cache arrays, excluding manifest."""
    digest = hashlib.sha256()
    with np.load(npz_path, allow_pickle=False) as cache:
        for name in sorted(set(cache.files) - {"manifest"}):
            arr = cache[name]
            arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("<"), copy=False))
            header = json.dumps([name, list(arr.shape), arr.dtype.str], separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            digest.update(arr.tobytes())
    return digest.hexdigest()


def _normalise_symbol(symbol):
    return re.sub(r"[\s_]", "", symbol.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉−", "0123456789-")))


@lru_cache(maxsize=1)
def _space_group_symbols():
    # Use spglib's tabulated Hermann-Mauguin aliases for all 530 Hall settings,
    # not a symmetry redetection of the (possibly distorted) CSV structure.
    symbols = {}
    for hall in range(1, 531):
        group = spglib.get_spacegroup_type(hall)
        for alias in (group.international_short, group.international_full,
                      *group.international.split("=")):
            key = _normalise_symbol(alias)
            symbols.setdefault(key, set()).add(int(group.number))
            if group.choice:
                symbols.setdefault(f"{key}:{group.choice}", set()).add(int(group.number))
    return symbols


def space_group_number(symbol):
    """Resolve a tabulated HM symbol/setting; reject unknown/ambiguous labels."""
    numbers = _space_group_symbols().get(_normalise_symbol(symbol), set())
    if len(numbers) != 1:
        raise ValueError(f"Unknown or ambiguous Hermann-Mauguin symbol: {symbol!r}")
    return next(iter(numbers))


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _numpy_repr_values(value):
    """Parse numeric numpy-repr text without executing Python or accepting truncation."""
    residue = re.sub(_NUMBER, "", value)
    if re.sub(r"[\s,]", "", residue):
        raise ValueError(f"Invalid numeric array: {value!r}")
    values = np.asarray([float(v) for v in re.findall(_NUMBER, value)])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite array")
    return values


def _numpy_repr_rows(value):
    # MatterGen serializes a list of numpy vectors as [array([...])\n array([...])].
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError("Expected bracketed array rows")
    body = value[1:-1]
    pattern = r"array\(\s*\[[^\[\]]*\]\s*\)|\[[^\[\]]*\]"
    rows = [re.search(r"\[([^\[\]]*)\]", match[0])[1] for match in re.finditer(pattern, body)]
    if not rows or re.sub(r"[\s,]", "", re.sub(pattern, "", body)):
        raise ValueError("Invalid numpy-repr rows")
    result = np.asarray([_numpy_repr_values(row) for row in rows])
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("Expected three coordinates per array row")
    return result


def parse_mattergen_record(record):
    """Cartesian Angstrom positions + lattice rows, with the CSV's SG label."""
    cell = _numpy_repr_rows(record["cell"])
    positions = _numpy_repr_rows(record["positions"])
    text = record["atomic_numbers"].strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("Expected bracketed atomic numbers")
    numbers = _numpy_repr_values(text[1:-1])
    if (cell.shape != (3, 3) or not 1 <= len(numbers) <= K_CACHE
            or positions.shape != (len(numbers), 3)
            or np.any((numbers < 1) | (numbers > 118) | (numbers != np.floor(numbers)))
            or abs(np.linalg.det(cell)) < 1e-12):
        raise ValueError("Invalid MatterGen cell, positions or atomic numbers")
    return {"material_id": record["material_id"],
            "structure": Structure(cell, numbers.astype(int), positions, coords_are_cartesian=True),
            "spacegroup.number": space_group_number(record["space_group"])}


def read_csv_records(path, limit=None, *, csv_format="mp20"):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    required = {"mp20": {"cif", "material_id", "spacegroup.number"},
                "mattergen": {"positions", "cell", "atomic_numbers", "material_id", "space_group"}}
    if csv_format not in required:
        raise ValueError(f"Unknown CSV format: {csv_format}")
    csv.field_size_limit(max(csv.field_size_limit(), 10_000_000))
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not required[csv_format] <= set(reader.fieldnames or []):
            label = "MP-20" if csv_format == "mp20" else "MatterGen"
            raise ValueError(f"CSV lacks required {label} columns")
        for record in (reader if limit is None else itertools.islice(reader, limit)):
            if csv_format == "mattergen":
                try:
                    record = parse_mattergen_record(record)
                except Exception as exc:
                    # Carry bad rows through the same max-failures/manifest path as CIF failures.
                    record = {"material_id": record["material_id"],
                              "_parse_error": f"{type(exc).__name__}: {exc}"}
            yield record


def _process_record(record, tables):
    mp_id = record["material_id"]
    if "_parse_error" in record:
        return None, f"{mp_id}: {record['_parse_error']}"
    try:
        if "structure" in record:
            structure = record["structure"]
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")
                structure = Structure.from_str(record["cif"], fmt="cif")
        result = tokenise(structure, tables, mp_id=mp_id)
        result["csv_G"] = int(record["spacegroup.number"])
        return result, None
    except Exception as exc:
        return None, f"{mp_id}: {type(exc).__name__}: {exc}"


_WORKER_TABLES = None


def _init_worker(table_dir):
    global _WORKER_TABLES
    torch.set_num_threads(1)
    _WORKER_TABLES = load_tables(table_dir)


def _worker(record):
    return _process_record(record, _WORKER_TABLES)


def _pack_cache(records, manifest):
    if not records:
        raise ValueError("Cannot build an empty cache")
    n = len(records)
    arrays = {name: np.asarray([r[name] for r in records])
              for name in records[0] if name not in ("row", "A0", "U", "lat")}
    arrays.update(row=np.zeros((n, K_CACHE), dtype=np.int64),
                  A0=np.zeros((n, K_CACHE), dtype=np.int64),
                  U=np.zeros((n, K_CACHE, 3), dtype=np.float32),
                  lat=np.zeros((n, 6), dtype=np.float32))
    for i, r in enumerate(records):
        k = r["n_orbits"]
        arrays["row"][i, :k], arrays["A0"][i, :k] = r["row"], r["A0"]
        arrays["U"][i, :k] = np.mod(r["U"].astype(np.float32), np.float32(1))
        arrays["lat"][i] = family_ltri(r["lat"].astype(np.float32), r["family"])
    arrays["manifest"] = np.asarray(json.dumps(manifest, sort_keys=True))
    return arrays


def build_cache(split, tables_dir, out, *, limit=None, workers=1, csv_path=None, max_failures=0,
                dataset="mp20"):
    """Build in CSV order; report every failed mp_id.

    More than ``max_failures`` failures publish nothing. Up to that many are excluded and listed in the
    manifest (``excluded``): spglib cannot symmetrise a few MP-20 cells exactly (\"ssm_get_exact_positions
    failed\"), and their standardised positions then miss the 1e-6 round trip.
    """
    if split not in ("train", "val", "test") or workers < 1:
        raise ValueError("Invalid split or worker count")
    started = time.perf_counter()
    paths = C.resolve_dataset_paths(dataset)
    csv_path = Path(csv_path) if csv_path is not None else Path(paths.data_root) / "raw" / f"{split}.csv"
    tables = load_tables(tables_dir)
    source = read_csv_records(csv_path, limit, csv_format=paths.csv_format)
    records, failures = [], []

    def collect(results):
        for result, failure in results:
            if failure is not None:
                failures.append(failure)
            else:
                records.append(result)

    if workers == 1:
        collect(_process_record(record, tables) for record in source)
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                          initargs=(str(tables_dir),)) as pool:
            collect(pool.imap(_worker, source, chunksize=8))
    if len(failures) > max_failures:
        raise RuntimeError(f"Orbit cache failed for {len(failures)} mp_ids:\n" + "\n".join(failures))
    if failures:
        print(f"warning: excluding {len(failures)} structures from {split}:\n" + "\n".join(failures), flush=True)
    if not records:
        raise ValueError("No structures read from CSV")
    manifest = dict(
        format_version=1, split=split, table_sha256=tables.sha256,
        symprec=SYMPREC, angle_tolerance=ANGLE_TOLERANCE,
        csv_path=str(csv_path.resolve()), csv_sha256=file_sha256(csv_path),
        libraries={p: version(p) for p in ("numpy", "scipy", "spglib", "pyxtal", "pymatgen", "torch")},
        limit=limit, count=len(records), K=K_CACHE,
        round_trip_precision="float64 before float32 storage",
        fractional_tolerance=FRAC_TOL, lattice_relative_tolerance=1e-8,
        displacement_definition="internal displacement in ideal lattice after affine basis/origin transform, Angstrom",
        strain_definition="Frobenius norm of principal stretch minus identity",
        label_agreement=sum(r["G"] == r["csv_G"] for r in records) / len(records),
        excluded=list(failures), max_failures=max_failures,
    )
    return _write_cache(records, manifest, out, started)


def _write_cache(records, manifest, out, started):
    arrays = _pack_cache(records, manifest)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{manifest['split']}.npz"
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    manifest["cache_sha256"] = file_sha256(path)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["seconds_per_structure"] = manifest["elapsed_seconds"] / len(records)
    manifest_path = out / f"{manifest['split']}.manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return manifest


def build_atom_cache(split, tables_dir, out, *, limit=None, dataset="mp20"):
    """Keep Crystalite's stored cells and atom order, represented as P1.

    U is F1, without an offset: training centres it and decode adds 1/2 back.
    Cache the core's float32 Y1 -> ltri conversion once, so the unchanged trainer
    reads exactly Crystalite's clean latent.
    No symmetry search, cell reduction, or exclusions are performed. Malformed
    inputs abort publication. CSV metadata is unavailable; transforms are identity.
    """
    if split not in ("train", "val", "test"):
        raise ValueError("Invalid split")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    started = time.perf_counter()
    source = Path(C.resolve_dataset_paths(dataset).data_root) / "processed" / f"mp20_tokens_{split}.pt"
    tables = load_tables(tables_dir)
    row_id = tables.row_id(1, "a")
    family = C.FAMILIES.index("triclinic")
    items = torch.load(source, map_location="cpu", weights_only=False)
    items = items if limit is None else items[:limit]
    if not items:
        raise ValueError("No structures read from token file")
    # A single nonempty batch uses precisely the pinned training conversion.
    y1 = torch.stack([item["Y1"] for item in items]).float()
    latents = y1_to_lattice_latent(y1, "ltri").numpy()
    records = []
    for i, (item, lat) in enumerate(zip(items, latents, strict=True)):
        mp_id = str(item.get("mp_id", f"{split}:{i}"))
        pad = item["pad_mask"].numpy()
        atoms, coords = item["A0"].numpy(), item["F1"].numpy()
        if (pad.dtype != np.bool_ or pad.ndim != 1 or atoms.shape != pad.shape
                or coords.shape != (*pad.shape, 3)):
            raise ValueError(f"{mp_id}: invalid atom token shapes or mask")
        atoms, coords = atoms[~pad], coords[~pad]
        n = len(atoms)
        if (not 1 <= n <= K_CACHE or not np.isfinite(atoms).all()
                or np.any((atoms < 1) | (atoms > 118) | (atoms != np.floor(atoms)))
                or not np.isfinite(coords).all() or not np.isfinite(lat).all()):
            raise ValueError(f"{mp_id}: invalid atom count, species, coordinates or lattice")
        rows = np.full(n, row_id, dtype=np.int64)
        # Check the stored chart against the core's decoded cell. Its Y1 decoder
        # and the training latent are float32; the orbit path's 1e-8 certificate
        # for pre-quantisation float64 cells does not apply to these source items.
        xyz, species = expand_tokens(rows, atoms, coords, tables)
        delta = xyz - coords
        frac_error = float(np.max(np.abs(delta - np.rint(delta))))
        lengths, angles = decode_Y1(item["Y1"])
        lattice = Lattice.from_parameters(*lengths, *angles).matrix
        target = np.linalg.cholesky(lattice @ lattice.T)
        lattice_error = float(np.linalg.norm(ltri_to_lattice(lat) - target) / np.linalg.norm(target))
        if (not np.array_equal(species, atoms) or frac_error > FRAC_TOL
                or not np.isfinite(lattice_error) or lattice_error > 1e-6):
            raise ValueError(f"{mp_id}: atom-view round trip failed")
        records.append(dict(
            mp_id=mp_id, G=1, hall=int(tables.arrays["group_hall"][1]), family=family,
            n_orbits=n, row=rows, A0=atoms.astype(np.int64), U=coords, lat=lat,
            n_conv=n, n_prim=n, disp_max=0.0, disp_mean=0.0, strain=0.0,
            transformation_matrix=np.eye(3), origin_shift=np.zeros(3),
            rotation=np.eye(3), frozen_shift=np.zeros(3), csv_G=-1,
            round_trip_ok=True, round_trip_frac_error=frac_error,
            round_trip_lattice_error=lattice_error,
        ))
    manifest = dict(
        format_version=1, split=split, table_sha256=tables.sha256, view="atom",
        source_token_path=str(source.resolve()), source_token_sha256=file_sha256(source),
        symprec=SYMPREC, angle_tolerance=ANGLE_TOLERANCE,
        csv_path=None, csv_sha256=None,
        libraries={p: version(p) for p in ("numpy", "scipy", "spglib", "pyxtal", "pymatgen", "torch")},
        limit=limit, count=len(records), K=K_CACHE,
        round_trip_precision="float32 Crystalite source/latent; float64 expansion",
        fractional_tolerance=FRAC_TOL, lattice_relative_tolerance=1e-6,
        displacement_definition="zero: stored Crystalite cell, no standardisation",
        strain_definition="zero: stored Crystalite cell, no standardisation",
        label_agreement=None, excluded=[], max_failures=0,
    )
    return _write_cache(records, manifest, out, started)


class OrbitDataset(Dataset):
    def __init__(self, cache_npz, tables, *, augment: bool, aug_seed: int):
        self.tables = load_tables(tables) if isinstance(tables, (str, Path)) else tables
        with np.load(cache_npz, allow_pickle=False) as data:
            self.cache = {name: data[name] for name in data.files}
        if "manifest" not in self.cache:
            raise ValueError("Cache has no table hash manifest; rebuild it")
        manifest = json.loads(str(self.cache["manifest"].item()))
        if manifest.get("format_version") != 1 or manifest.get("table_sha256") != self.tables.sha256:
            raise ValueError("Cache/table sha256 or format mismatch")
        if manifest.get("symprec") != SYMPREC or manifest.get("angle_tolerance") != ANGLE_TOLERANCE:
            raise ValueError("Cache standardisation tolerances differ")
        if augment and manifest.get("split") != "train":
            raise ValueError("Augmentation is restricted to the train split")
        if not self.cache["round_trip_ok"].all():
            raise ValueError("Cache contains failed round trips")
        if aug_seed < 0:
            raise ValueError("aug_seed must be nonnegative")
        self.augment, self.aug_seed = augment, int(aug_seed)
        # A shared scalar also updates copies held by persistent DataLoader workers.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.cache["G"])

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self._epoch.fill_(int(epoch))

    def __getitem__(self, idx):
        idx = int(idx)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        c = self.cache
        k, group, family = int(c["n_orbits"][idx]), int(c["G"][idx]), int(c["family"][idx])
        row, U, lat = c["row"][idx, :k].copy(), c["U"][idx, :k].copy(), c["lat"][idx].copy()
        if self.augment:
            generator = np.random.default_rng((self.aug_seed, int(self._epoch.item()), idx))
            row, U, lat, _ = augment_tokens(row, U, lat, group, self.tables, generator)
        U = np.mod(U.astype(np.float32), np.float32(1))
        mask = np.arange(3) < self.tables.arrays["row_dof"][row, None]
        U[~mask] = 0
        return dict(mp_id=str(c["mp_id"][idx]), G=group, family=family,
                    row=row.astype(np.int64), A0=c["A0"][idx, :k].astype(np.int64, copy=True),
                    U=U, dof_mask=mask, pad_mask=np.zeros(k, dtype=bool),
                    lat=family_ltri(lat.astype(np.float32), family), n_prim=int(c["n_prim"][idx]))


def collate_orbits(items, K, pad_multiple=0) -> OrbitBatch:
    """Keep item order; optionally round the batch maximum up within capacity K."""
    if not items or K < 1:
        raise ValueError("Collation requires items and a positive K")
    if type(pad_multiple) is not int or pad_multiple < 0:
        raise ValueError("pad_multiple must be a nonnegative integer")
    for item in items:
        if item.keys() != C.ORBIT_BATCH_KEYS.keys():
            raise ValueError("Item keys differ from ORBIT_BATCH_KEYS")
    lengths = np.asarray([len(item["row"]) for item in items])
    invalid = lengths[(lengths < 1) | (lengths > K)]
    if len(invalid):
        raise ValueError(f"{invalid[0]} tokens do not fit K={K}")
    if pad_multiple:
        K = min(K, ((int(lengths.max()) + pad_multiple - 1) // pad_multiple) * pad_multiple)
    present = np.arange(K) < lengths[:, None]
    batch = dict(mp_id=[item["mp_id"] for item in items])
    for name in ("G", "family", "n_prim"):
        batch[name] = torch.tensor([item[name] for item in items], dtype=torch.int64)
    batch["lat"] = torch.as_tensor(np.stack([item["lat"] for item in items]), dtype=torch.float32)
    # One NumPy gather and one tensor wrapper per field, rather than per token item.
    for name, tail, dtype in (("row", (), np.int64), ("A0", (), np.int64),
                               ("U", (3,), np.float32), ("dof_mask", (3,), np.bool_),
                               ("pad_mask", (), np.bool_)):
        values = np.full((len(items), K, *tail), name == "pad_mask", dtype=dtype)
        values[present] = np.concatenate([item[name] for item in items])
        batch[name] = torch.from_numpy(values)
    return batch


class ResumableSampler(Sampler):
    """Batch sampler; optimiser acknowledgement is independent of prefetch."""
    def __init__(self, n, *, seed, batch_size, drop_last=True):
        if n < 0 or seed < 0 or batch_size < 1:
            raise ValueError("Invalid sampler size, seed or batch size")
        self.n, self.seed, self.batch_size = int(n), int(seed), int(batch_size)
        self.drop_last = bool(drop_last)
        self.epoch, self.cursor = 0, 0

    @property
    def total_batches(self):
        return (self.n // self.batch_size if self.drop_last
                else (self.n + self.batch_size - 1) // self.batch_size)

    def __len__(self):
        return self.total_batches - self.cursor

    def __iter__(self):
        permutation = np.random.default_rng((self.seed, self.epoch)).permutation(self.n)
        start = self.cursor
        return iter([permutation[i * self.batch_size:(i + 1) * self.batch_size].tolist()
                     for i in range(start, self.total_batches)])

    def mark_consumed(self, count=1):
        if count < 0 or int(count) != count or self.cursor + count > self.total_batches:
            raise ValueError("Consumed cursor exceeds the epoch")
        self.cursor += int(count)

    def set_epoch(self, epoch):
        if epoch < 0 or int(epoch) != epoch:
            raise ValueError("Invalid epoch")
        if epoch != self.epoch:
            self.epoch, self.cursor = int(epoch), 0

    def state_dict(self):
        return dict(n=self.n, seed=self.seed, batch_size=self.batch_size,
                    drop_last=self.drop_last, epoch=self.epoch, cursor=self.cursor)

    def load_state_dict(self, state):
        for key in ("n", "seed", "batch_size", "drop_last"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Sampler checkpoint differs in {key}")
        epoch, cursor = state["epoch"], state["cursor"]
        if epoch < 0 or int(epoch) != epoch or int(cursor) != cursor or not 0 <= cursor <= self.total_batches:
            raise ValueError("Invalid sampler checkpoint cursor/epoch")
        self.epoch, self.cursor = int(epoch), int(cursor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-cache")
    build.add_argument("--dataset", choices=C.DATASET_NAMES, default="mp20")
    build.add_argument("--view", choices=("orbit", "atom"), default="orbit")
    build.add_argument("--split", choices=("train", "val", "test"), required=True)
    build.add_argument("--tables", type=Path, default=None)
    build.add_argument("--out", type=Path,
                       help="defaults to the selected dataset's orbit or atom cache")
    build.add_argument("--limit", type=int)
    build.add_argument("--workers", type=int, default=1)
    build.add_argument("--max-failures", type=int, default=0,
                       help="exclude and record up to this many failed structures instead of aborting")
    args = parser.parse_args(argv)
    args.tables = args.tables or Path(C.TABLES_DIR)
    if args.out is None:
        paths = C.resolve_dataset_paths(args.dataset)
        args.out = Path(paths.orbit_cache_dir if args.view == "orbit" else paths.atom_cache_dir)
    if args.view == "atom":
        if args.workers != 1 or args.max_failures != 0:
            parser.error("atom view requires --workers 1 and --max-failures 0")
        result = build_atom_cache(args.split, args.tables, args.out, limit=args.limit, dataset=args.dataset)
    else:
        result = build_cache(args.split, args.tables, args.out, limit=args.limit, workers=args.workers,
                             max_failures=args.max_failures, dataset=args.dataset)
    print(json.dumps(dict(split=args.split, round_trips_ok=result["count"], count=result["count"],
                          elapsed_seconds=result["elapsed_seconds"],
                          seconds_per_structure=result["seconds_per_structure"],
                          cache=str((args.out / f"{args.split}.npz").resolve()))))


if __name__ == "__main__":
    main()
