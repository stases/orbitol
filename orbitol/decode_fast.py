"""Packed CPU decoding with certified small-radius geometry predicates.

Rows are the default output; primitive pymatgen objects are built only when
``structures=True``. Ambiguous geometry and uncertified reductions go through
the reference decoder. That fallback is in-process, not a crash boundary.
``STATS`` counts input structures, fallbacks, and ``fallback_<cause>`` totals.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from pymatgen.core import Structure

import orbitol.core  # noqa: F401 -- bind the pinned core before importing src
from orbitol import decode as reference
from orbitol.decode_reduce import reduce_lattices as _reduce
from orbitol.jit import njit
from orbitol.lattice import to_matrix
from src.crystalite.sampler import clamp_lattice_latent
from src.models.lattice_repr import gram_to_y1


STATS = Counter()
_TABLE_CACHE = OrderedDict()
_VALIDATOR_DTYPES = frozenset(np.dtype(name) for name in (
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64",
))
_ROUNDING_GAMMA = 32 * np.finfo(np.float64).eps / (1 - 32 * np.finfo(np.float64).eps)
_TABLE_NAMES = (
    "row_group", "row_dof", "row_member_mask", "row_W", "row_v", "row_prim_mask",
    "group_Pc", "group_ncent", "group_family", "group_ops_count", "group_ops_R", "group_ops_t",
)
_FAILURES = {
    1: "coinciding_orbits", 2: "collapsed_orbit", 3: "containment_failed",
    4: "non_finite", 5: "invalid_lattice",
}
_TOKEN_FAILURES = (
    None, "invalid_group", "invalid_shape", "empty_structure", "non_finite",
    "invalid_rows", "reused_fixed_row", "invalid_species",
)
_FALLBACK_CAUSES = {
    10: "radius_certificate", 11: "collision_boundary", 12: "containment_boundary",
    13: "metric_boundary", 14: "ambiguous_matching", 15: "geometry_arithmetic",
}


@dataclass
class _Tables:
    """Frozen real members and operations, cached by table contents."""

    arrays: dict
    member_offsets: np.ndarray
    member_W: np.ndarray
    member_v: np.ndarray
    member_primitive: np.ndarray
    inverse_P: np.ndarray
    op_offsets: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray


def _prepared(tables):
    """Share preparation across equal tables, including distinct objects."""
    arrays = {name: reference._array(tables, name) for name in _TABLE_NAMES}
    key = getattr(tables, "sha256", None)
    if key is None:
        digest = hashlib.sha256()
        for name, value in arrays.items():
            value = np.ascontiguousarray(value)
            digest.update(repr((name, value.shape, value.dtype.str)).encode())
            digest.update(memoryview(value).cast("B"))
        key = digest.hexdigest()
    if key in _TABLE_CACHE:
        _TABLE_CACHE.move_to_end(key)
        return _TABLE_CACHE[key]
    # A later in-place mutation must not change an entry under its old digest.
    arrays = {name: value.copy() for name, value in arrays.items()}
    mask = arrays["row_member_mask"]
    member_offsets = np.r_[0, np.cumsum(mask.sum(axis=1))].astype(np.int64)
    P = arrays["group_Pc"].astype(np.float64)
    inverse = np.broadcast_to(np.eye(3), P.shape).copy()
    counts = arrays["group_ops_count"].astype(np.int64)
    inverse[counts > 0] = np.linalg.inv(P[counts > 0])
    op_offsets = np.r_[0, np.cumsum(counts)]
    op_group = np.repeat(np.arange(len(counts)), counts)
    op_number = np.arange(op_offsets[-1]) - np.repeat(op_offsets[:-1], counts)
    rotations = arrays["group_ops_R"][op_group, op_number].astype(np.float64)
    translations = arrays["group_ops_t"][op_group, op_number].astype(np.float64)
    # Preserve contains_ops' multiplication association when changing basis.
    rotations = inverse[op_group] @ rotations @ P[op_group]
    translations = (translations[:, None] @ inverse[op_group].transpose(0, 2, 1))[:, 0]
    prepared = _Tables(
        arrays, member_offsets, arrays["row_W"][mask], arrays["row_v"][mask],
        arrays["row_prim_mask"][mask], inverse, op_offsets, rotations, translations,
    )
    _TABLE_CACHE[key] = prepared
    while len(_TABLE_CACHE) > 4:
        _TABLE_CACHE.popitem(last=False)
    return prepared


@njit
def _validate_tokens(group, rows, species, coords, op_counts, row_groups, row_dof):
    """Reference token checks and precedence on numeric, shaped tokens."""
    if group < 1 or group >= len(op_counts) or op_counts[group] == 0:
        return 1
    if not len(rows):
        return 3
    if not np.isfinite(rows).all() or not np.isfinite(species).all() or not np.isfinite(coords).all():
        return 4
    for row in rows:
        if row != np.floor(row) or row <= 0 or row >= len(row_groups):
            return 5
        if row_groups[int(row)] != group:
            return 5
    for i in range(len(rows)):
        if row_dof[int(rows[i])] == 0:
            for j in range(i):
                if rows[j] == rows[i]:
                    return 6
    for atom in species:
        if atom < 1 or atom > 118 or atom != np.floor(atom):
            return 7
    return 0


@njit
def _dot3(a, b):
    """Three-component dot product without reassociation or fastmath."""
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]


@njit
def _matmul3(a, b):
    """Small matrix product used only in guarded geometric predicates."""
    result = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            result[i, j] = _dot3(a[i], b[:, j])
    return result


@njit
def _inverse3(a):
    """Small inverse; non-finite results fail the radius certificate."""
    result = np.empty((3, 3), dtype=np.float64)
    result[0, 0] = a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]
    result[0, 1] = a[0, 2] * a[2, 1] - a[0, 1] * a[2, 2]
    result[0, 2] = a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1]
    result[1, 0] = a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2]
    result[1, 1] = a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0]
    result[1, 2] = a[0, 2] * a[1, 0] - a[0, 0] * a[1, 2]
    result[2, 0] = a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]
    result[2, 1] = a[0, 1] * a[2, 0] - a[0, 0] * a[2, 1]
    result[2, 2] = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    determinant = _dot3(a[0], result[:, 0])
    if determinant == 0 or not np.isfinite(determinant):
        result[:] = np.nan
        return result
    return result / determinant


@njit
def _radius_bounds(lattice, radius, frac):
    """Bound both image uniqueness and Cartesian roundoff per cell.

    A nearby image obeys |d_j-n_j| <= r ||L^-1[:,j]||. Bounds
    below 1/2 certify the rounded image as the only possible candidate
    inside the radius, even for skew lattices.

    The reference converts positions to Cartesian coordinates before subtracting;
    subtracting fractional coordinates first can lose a different number of ulps.
    A gamma_32 allowance times (1 + 2 max|f|) sum|L| bounds the Cartesian error
    budget for coordinate conversions, periodic shifts and subtraction. The 1
    covers an image shift, and the L1 bound also bounds the Euclidean error.
    If this error is e, a squared-distance comparison within 2 r e + e^2 of
    r^2 is ambiguous. Inflate the search radius by e as well: an early fractional
    rejection must not skip a pair that roundoff could put inside the cutoff.

    Retain the original small boundary margins and the 0.49 uniqueness cutoff.
    These are conservative floating-point guards, not an all-input IEEE proof.
    """
    inverse = _inverse3(lattice)
    largest_coordinate = 0.0
    for i in range(len(frac)):
        for j in range(3):
            largest_coordinate = max(largest_coordinate, abs(frac[i, j]))
    cartesian_error = _ROUNDING_GAMMA * (1.0 + 2.0 * largest_coordinate) * np.abs(lattice).sum()
    margin = max(1e-6 * radius * radius, cartesian_error * (2.0 * radius + cartesian_error))
    bounds = np.empty(3, dtype=np.float64)
    for j in range(3):
        bounds[j] = (radius + cartesian_error) * np.sqrt(_dot3(inverse[:, j], inverse[:, j])) + 1e-12
    certified = np.isfinite(margin) and np.isfinite(bounds).all() and (bounds < 0.49).all()
    return bounds, margin, certified


@njit
def _distance_squared(a, b, lattice, bounds):
    """Evaluate the only possible image, or reject by fractional bounds."""
    d0 = a[0] - b[0]
    d0 -= np.rint(d0)
    if abs(d0) > bounds[0]:
        return np.inf
    d1 = a[1] - b[1]
    d1 -= np.rint(d1)
    if abs(d1) > bounds[1]:
        return np.inf
    d2 = a[2] - b[2]
    d2 -= np.rint(d2)
    if abs(d2) > bounds[2]:
        return np.inf
    c0 = (d0 * lattice[0, 0] + d1 * lattice[1, 0]) + d2 * lattice[2, 0]
    c1 = (d0 * lattice[0, 1] + d1 * lattice[1, 1]) + d2 * lattice[2, 1]
    c2 = (d0 * lattice[0, 2] + d1 * lattice[1, 2]) + d2 * lattice[2, 2]
    return (c0 * c0 + c1 * c1) + c2 * c2


@njit
def _collisions(lattice, frac, orbits, bounds, margin):
    """Cross-token coincidences precede all within-token collapses."""
    cross = False
    collapsed = False
    for i in range(len(frac)):
        for j in range(i + 1, len(frac)):
            distance = _distance_squared(frac[i], frac[j], lattice, bounds)
            if np.isnan(distance):
                return 15
            if abs(distance - 1e-6) <= margin:
                return 11
            if distance < 1e-6:
                if orbits[i] != orbits[j]:
                    cross = True
                else:
                    collapsed = True
    if cross:
        return 1
    return 2 if collapsed else 0


@njit
def _containment(lattice, frac, species, rotations, translations, bounds, margin, cart_tol, metric_rtol):
    """Check every metric and species-preserving periodic bijection.

    Multiple candidates request the reference's general bipartite matching.
    Unique candidates must form a permutation, rather than just hit each site.
    """
    if not len(frac):
        return 3
    gram = _matmul3(lattice, lattice.T)
    limit = metric_rtol * np.sqrt(np.sum(gram * gram))
    cutoff = cart_tol * cart_tol
    if not np.isfinite(limit):
        return 15
    for op in range(len(rotations)):
        rotation = rotations[op]
        residual = _matmul3(_matmul3(rotation.T, gram), rotation) - gram
        error = np.sqrt(np.sum(residual * residual))
        if not np.isfinite(error):
            return 15
        if error > 1.1 * limit:
            return 3
        if error >= 0.9 * limit:
            return 13
        used = np.zeros(len(frac), dtype=np.bool_)
        for i in range(len(frac)):
            transformed = np.empty(3, dtype=np.float64)
            for axis in range(3):
                transformed[axis] = (_dot3(rotation[axis], frac[i]) + translations[op, axis]) % 1.0
            if not np.isfinite(transformed).all():
                return 15
            match = -1
            for j in range(len(frac)):
                if species[j] != species[i]:
                    continue
                distance = _distance_squared(frac[j], transformed, lattice, bounds)
                if np.isnan(distance):
                    return 15
                if abs(distance - cutoff) <= margin:
                    return 12
                if distance <= cutoff:
                    if match != -1:
                        return 14
                    match = j
            if match == -1 or used[match]:
                return 3
            used[match] = True
    return 0


@njit
def _geometry(lattices, conv_frac, conv_orbits, conv_offsets, primitive_lattices,
              primitive_frac, primitive_species, primitive_offsets, groups, op_offsets,
              rotations, translations, status, cart_tol, metric_rtol):
    """Check ragged cells without batch/operation/atom-pair allocations."""
    contains = np.zeros(len(groups), dtype=np.bool_)
    for i in range(len(groups)):
        if status[i]:
            continue
        start, end = conv_offsets[i], conv_offsets[i + 1]
        frac = conv_frac[start:end]
        if not np.isfinite(frac).all():
            status[i] = 4
            continue
        bounds, margin, certified = _radius_bounds(lattices[i], 1e-3, frac)
        if not certified:
            status[i] = 10
            continue
        status[i] = _collisions(lattices[i], frac, conv_orbits[start:end], bounds, margin)
        if status[i]:
            continue
        start, end = primitive_offsets[i], primitive_offsets[i + 1]
        frac = primitive_frac[start:end]
        lattice = primitive_lattices[i]
        if not np.isfinite(frac).all() or not np.isfinite(lattice).all():
            status[i] = 3
            continue
        bounds, margin, certified = _radius_bounds(lattice, cart_tol, frac)
        if not certified:
            status[i] = 10
            continue
        group = groups[i]
        op_start, op_end = op_offsets[group], op_offsets[group + 1]
        status[i] = _containment(
            lattice, frac, primitive_species[start:end], rotations[op_start:op_end],
            translations[op_start:op_end], bounds, margin, cart_tol, metric_rtol,
        )
        contains[i] = status[i] == 0
    return contains


@dataclass
class _ArrayResult:
    """Packed primitive data kept for optional object construction."""

    rows: reference.CrystaliteRows
    contains: np.ndarray
    fallback: dict
    lattices: np.ndarray
    frac: np.ndarray
    species: np.ndarray
    offsets: np.ndarray


def _write_rows(rows, ids, reduced, lattices, frac, species, offsets):
    """Preserve the Cartesian round trip, core Y1 clamp and both wraps."""
    if not len(ids):
        return
    L = torch.as_tensor(reduced, dtype=torch.float64)
    y1 = clamp_lattice_latent(gram_to_y1(L @ L.transpose(-1, -2)), "y1").float()
    good = torch.isfinite(y1).all(dim=1).numpy()
    for i in ids[~good]:
        rows.failures[int(i)] = "non_finite"
    ids, reduced, y1 = ids[good], reduced[good], y1[good]
    if not len(ids):
        return
    counts = np.diff(offsets)[ids]
    row_number = np.repeat(np.arange(len(ids)), counts)
    dest = ids[row_number]
    columns = np.arange(counts.sum()) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    source = offsets[dest] + columns
    cart = (frac[source, None] @ lattices[dest])[:, 0]
    fractional = (cart[:, None] @ np.linalg.inv(reduced)[row_number])[:, 0] % 1.0
    # Structure.get_reduced_structure retains the original sites for allclose cells.
    same = np.isclose(reduced, lattices[ids]).all(axis=(1, 2))[row_number]
    fractional[same] = frac[source[same]]
    rows["A0"].numpy()[dest, columns] = species[source]
    rows["F1"].numpy()[dest, columns] = (fractional % 1.0).astype(np.float32) % 1.0
    rows["Y1"].numpy()[ids] = y1.numpy()
    rows["pad_mask"].numpy()[dest, columns] = False


def _decode_arrays(groups, token_offsets, token_rows, token_species, token_coords, lattices, tables, n_max):
    """Expand, emit, check and serialize only real members in a batch."""
    count = len(groups)
    arrays = tables.arrays
    token_parent = np.repeat(np.arange(count), np.diff(token_offsets))
    multiplicities = np.diff(tables.member_offsets)[token_rows]
    owners = np.repeat(np.arange(len(token_rows)), multiplicities)
    local = np.arange(len(owners)) - np.repeat(np.r_[0, np.cumsum(multiplicities)[:-1]], multiplicities)
    members = tables.member_offsets[token_rows[owners]] + local
    parent = token_parent[owners]
    with np.errstate(over="ignore", invalid="ignore"):
        frac = (np.einsum("nij,nj->ni", tables.member_W[members], token_coords[owners] + 0.5)
                + tables.member_v[members]) % 1.0
        determinants = np.linalg.det(lattices)
        gram = lattices @ lattices.transpose(0, 2, 1)
    species = token_species[owners].astype(np.int64)
    conv_offsets = np.r_[0, np.cumsum(np.bincount(parent, minlength=count))]
    noncentred = arrays["group_ncent"][groups] == 1
    keep = tables.member_primitive[members] | noncentred[parent]
    primitive_parent = parent[keep]
    P = arrays["group_Pc"][groups].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        primitive_lattices = P.transpose(0, 2, 1) @ lattices
        primitive_frac = (frac[keep, None] @ tables.inverse_P[groups[primitive_parent]].transpose(0, 2, 1))[:, 0] % 1.0
    primitive_lattices[noncentred] = lattices[noncentred]
    primitive_frac[noncentred[primitive_parent]] = frac[keep][noncentred[primitive_parent]]
    primitive_species = species[keep]
    primitive_offsets = np.r_[0, np.cumsum(np.bincount(primitive_parent, minlength=count))]
    status = np.zeros(count, dtype=np.int8)
    status[determinants <= 0] = 5
    finite = np.isfinite(lattices).all(axis=(1, 2)) & np.isfinite(determinants) & np.isfinite(gram).all(axis=(1, 2))
    status[~finite] = 4
    contains = _geometry(
        lattices, frac, owners, conv_offsets, primitive_lattices, primitive_frac,
        primitive_species, primitive_offsets, groups, tables.op_offsets, tables.rotations,
        tables.translations, status, reference.CHECK_A["cart_tol"], reference.CHECK_A["metric_rtol"],
    )
    rows = reference.CrystaliteRows(count, n_max)
    rows.failures.update({i: _FAILURES[int(code)] for i, code in enumerate(status) if code in _FAILURES})
    fallback = {i: _FALLBACK_CAUSES[int(code)] for i, code in enumerate(status) if code in _FALLBACK_CAUSES}
    counts = np.diff(primitive_offsets)
    for i in np.flatnonzero((status == 0) & (counts > n_max)):
        rows.failures[int(i)] = "too_many_atoms"
    ids = np.flatnonzero((status == 0) & (counts <= n_max))
    if len(ids):
        reduced, codes = _reduce(primitive_lattices[ids])
        fallback.update({int(i): "reduction" for i in ids[codes == 2]})
        good = codes != 2
        _write_rows(rows, ids[good], reduced[good], primitive_lattices,
                    primitive_frac, primitive_species, primitive_offsets)
    return _ArrayResult(rows, contains, fallback, primitive_lattices, primitive_frac,
                        primitive_species, primitive_offsets)


@torch.no_grad()
def decode_batch(sampler_out, skel, tables, encoding, allowed_mask, *, n_max, structures=False) -> reference.DecodedBatch:
    """Decode like the reference, optionally building unreduced primitives."""
    types = torch.as_tensor(sampler_out["type"]).detach().cpu().float()
    u_c = reference._numpy(sampler_out["u"] if "u" in sampler_out else sampler_out["u_c"], np.float64)
    z = reference._numpy(sampler_out["z"], np.float64)
    if types.ndim != 3 or types.shape[-1] != 16:
        raise ValueError("type must have shape (B,K,16)")
    batch_size, token_count = types.shape[:2]
    if u_c.shape != (batch_size, token_count, 3) or z.shape != (batch_size, 6):
        raise ValueError("u must have shape (B,K,3) and z must have shape (B,6)")
    finite = reference._numpy(sampler_out.get("finite", np.ones(batch_size, dtype=bool)))
    skeletons, pad_mask, positions = reference._batch_skeletons(skel, batch_size, token_count)
    predicted = reference.decode_types(types, pad_mask, encoding, allowed_mask).numpy()
    finite_types = (torch.isfinite(types).all(dim=-1) | pad_mask).all(dim=-1).numpy()
    requested = np.array([s.G for s in skeletons], dtype=np.int64)
    if not isinstance(n_max, int) or n_max < 1:
        raise ValueError("n_max must be a positive integer")
    output = reference.CrystaliteRows(batch_size, n_max)
    emitted = [None] * batch_size if structures else None
    contains = np.zeros(batch_size, dtype=bool)
    STATS["structures"] += batch_size
    op_counts = reference._array(tables, "group_ops_count")
    row_groups = reference._array(tables, "row_group")
    row_dof = reference._array(tables, "row_dof")
    finite_z = np.isfinite(z).all(axis=1)
    ids, token_rows, token_species, token_coords = [], [], [], []
    for i, skeleton in enumerate(skeletons):
        rows = np.asarray(skeleton.rows)
        species = np.asarray(skeleton.A0) if skeleton.A0 is not None else predicted[i, positions[i]]
        coords = u_c[i, positions[i]]
        # Unusual public inputs retain the reference's shape/type error handling.
        regular = (isinstance(skeleton.G, (int, np.integer)) and rows.ndim == 1
                   and species.shape == rows.shape and coords.shape == (len(rows), 3)
                   and rows.dtype in _VALIDATOR_DTYPES and species.dtype in _VALIDATOR_DTYPES)
        if regular:
            code = _validate_tokens(skeleton.G, rows, species, coords, op_counts, row_groups, row_dof)
            reason = _TOKEN_FAILURES[code]
        else:
            reason = reference._validate_tokens(skeleton.G, rows, species, coords, tables)
        if reason is None and (not finite[i] or not finite_z[i]
                               or (skeleton.A0 is None and not finite_types[i])):
            reason = "non_finite"
        if reason:
            output.failures[i] = reason
            continue
        ids.append(i)
        token_rows.append(rows.astype(np.int64))
        token_species.append(species)
        token_coords.append(coords)
    if ids:
        ids = np.asarray(ids, dtype=np.int64)
        prepared = _prepared(tables)
        groups = requested[ids]
        families = torch.as_tensor(prepared.arrays["group_family"][groups], dtype=torch.int64)
        lattices = to_matrix(torch.as_tensor(z[ids], dtype=torch.float64), families).numpy()
        offsets = np.r_[0, np.cumsum([len(rows) for rows in token_rows])]
        try:
            result = _decode_arrays(
                groups, offsets, np.concatenate(token_rows), np.concatenate(token_species),
                np.concatenate(token_coords), lattices, prepared, n_max,
            )
        except Exception:
            # An unexpected fast-path error must not cost the batch: the reference isolates
            # per-structure failures as decode_error, so the fast decoder is never less robust.
            STATS["fallback"] += batch_size
            STATS["fallback_exception"] += batch_size
            decoded = reference.decode_batch(sampler_out, skel, tables, encoding, allowed_mask, n_max=n_max)
            if not structures:
                decoded.structures = None
            return decoded
        for key in output:
            output[key].numpy()[ids] = result.rows[key].numpy()
        contains[ids] = result.contains
        output.failures.update({int(ids[j]): reason for j, reason in result.rows.failures.items()})
        for j, cause in result.fallback.items():
            i = int(ids[j])
            STATS["fallback"] += 1
            STATS[f"fallback_{cause}"] += 1
            # Normalized tensors retain non-leading padding and fixed species.
            one = {"type": types[i:i + 1], "u": u_c[i:i + 1], "z": z[i:i + 1], "finite": finite[i:i + 1]}
            selected = {"G": requested[i:i + 1], "row": np.zeros((1, token_count), dtype=np.int64),
                        "pad_mask": pad_mask[i:i + 1]}
            selected["row"][0, positions[i]] = skeletons[i].rows
            if skeletons[i].A0 is not None:
                selected["fixed_A0"] = np.zeros((1, token_count), dtype=np.int64)
                selected["fixed_A0"][0, positions[i]] = skeletons[i].A0
            old = reference.decode_batch(one, selected, tables, encoding, allowed_mask, n_max=n_max)
            for key in output:
                output[key][i] = old.rows[key][0]
            contains[i] = old.contains_a[0]
            output.failures.pop(i, None)
            if 0 in old.failures:
                output.failures[i] = old.failures[0]
            if structures:
                emitted[i] = old.structures[0]
        if structures:
            for j, i in enumerate(ids):
                if i in output.failures or j in result.fallback:
                    continue
                start, end = result.offsets[j:j + 2]
                emitted[i] = Structure(result.lattices[j], result.species[start:end], result.frac[start:end])
    return reference.DecodedBatch(emitted, output, output.failures, contains, requested)
