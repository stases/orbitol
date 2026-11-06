"""Orbit decoding, exact primitive emission and Crystalite sample rows.

Fractional vectors are columns; lattice vectors are rows. Thus a primitive basis
P gives L_prim = P.T @ L and x_prim = inv(P) @ x. Only the frozen member flags
choose centring representatives; no geometric primitive-cell search is used.

Sampler input is a mapping with ``type`` (B,K,16), ``u`` (B,K,3; centred chart
coordinates, also accepted as ``u_c``), and ``z`` (B,6). ``skel`` is a sampler
mapping (G, row, pad_mask), one Skeleton shared by the batch, or a sequence of B
Skeletons. Skeleton rows occupy the leading token slots. Optional Skeleton.A0
or mapping ``fixed_A0`` overrides type predictions; a data batch's target A0
does not. Outputs and checks run on CPU, with float64 geometry.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.util.coord import pbc_shortest_vectors
from scipy.optimize import linear_sum_assignment

import orbitol.core  # noqa: F401 -- bind the pinned core before importing src
from orbitol.contracts import Skeleton
from src.crystalite.sampler import clamp_lattice_latent
from src.data.mp20_tokens import tokens_to_structure
from src.models.lattice_repr import gram_to_y1

CHECK_A = dict(cart_tol=1e-6, metric_rtol=1e-10)
CHECK_B = dict(cart_tol=1e-4, metric_rtol=1e-6)
MIN_TRANSLATION = 0.5  # Å; no crystal has a shorter lattice translation
MAX_NIGGLI_IMAGES = 1e6  # images in the final Niggli mapping search; real crystals need ~100


@dataclass
class Conventional:
    """Expanded members in token/member order, or a failure reason.

    ``prim_mask`` is row_prim_mask restricted to the expanded real members.
    ``orbit_index`` distinguishes tokens, including repeated non-fixed rows.
    """

    species: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    frac: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    lattice: np.ndarray = field(default_factory=lambda: np.empty((3, 3), dtype=np.float64))
    prim_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    orbit_index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    failure: str | None = None


class CrystaliteRows(dict):
    """Exactly the four Crystalite tensor keys, plus failure metadata.

    ``failures`` is an attribute, not an extra evaluator tensor/dict key.
    Failed rows have zero A0/F1/Y1 and an entirely true pad_mask.
    """

    def __init__(self, n_samples: int, n_max: int):
        super().__init__(
            A0=torch.zeros((n_samples, n_max), dtype=torch.int64),
            F1=torch.zeros((n_samples, n_max, 3), dtype=torch.float32),
            Y1=torch.zeros((n_samples, 6), dtype=torch.float32),
            pad_mask=torch.ones((n_samples, n_max), dtype=torch.bool),
        )
        self.failures: dict[int, str] = {}


@dataclass
class DecodedBatch:
    """Decoded rows and metadata; ``structures=None`` means not built."""

    structures: list[Structure | None] | None
    rows: CrystaliteRows
    failures: dict[int, str]
    contains_a: np.ndarray
    requested_G: np.ndarray


def _numpy(value, dtype=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _array(tables, name):
    """Accept TableSet, its array mapping, or a float64 tensor namespace."""
    arrays = getattr(tables, "arrays", tables)
    return _numpy(arrays[name] if isinstance(arrays, Mapping) else getattr(arrays, name))


def decode_types(type_feats, pad_mask, encoding, allowed_mask):
    """Use the encoding's own snap rule, including its allowed-mask policy."""
    return encoding.decode_logits_to_A0(
        type_logits=type_feats, pad_mask=pad_mask, allowed_mask=allowed_mask,
    )


def _validate_tokens(G, rows, A0, u_c, tables):
    if not 1 <= G < len(_array(tables, "group_ops_count")):
        return "invalid_group"
    if _array(tables, "group_ops_count")[G] == 0:
        return "invalid_group"
    if rows.ndim != 1 or A0.shape != rows.shape or u_c.shape != (len(rows), 3):
        return "invalid_shape"
    if not len(rows):
        return "empty_structure"
    if not all(np.isfinite(x).all() for x in (rows, A0, u_c)):
        return "non_finite"
    if (np.any(rows != np.floor(rows)) or np.any(rows <= 0)
            or np.any(rows >= len(_array(tables, "row_group")))):
        return "invalid_rows"
    rows = rows.astype(np.int64)
    if np.any(_array(tables, "row_group")[rows] != G):
        return "invalid_rows"
    fixed = rows[_array(tables, "row_dof")[rows] == 0]
    if len(np.unique(fixed)) != len(fixed):
        return "reused_fixed_row"
    if np.any((A0 < 1) | (A0 > 118) | (A0 != np.floor(A0))):
        return "invalid_species"
    return None


def _expand_members(G, rows, A0, u_c, lattice, tables):
    """Expand already validated tokens; shared by scalar and batch decoding."""
    lattice = _numpy(lattice, np.float64)
    if not np.isfinite(lattice).all():
        return Conventional(failure="non_finite")
    if lattice.shape != (3, 3):
        return Conventional(failure="invalid_lattice")
    with np.errstate(over="ignore", invalid="ignore"):
        determinant = np.linalg.det(lattice)
        gram = lattice @ lattice.T
    if not np.isfinite(determinant) or not np.isfinite(gram).all():
        return Conventional(failure="non_finite")
    if determinant <= 0:
        return Conventional(failure="invalid_lattice")
    mask = _array(tables, "row_member_mask")[rows]
    W, v = _array(tables, "row_W")[rows], _array(tables, "row_v")[rows]
    frac = (np.einsum("kmij,kj->kmi", W, u_c + 0.5) + v)[mask] % 1.0
    if not np.isfinite(frac).all():
        return Conventional(failure="non_finite")
    orbit = np.broadcast_to(np.arange(len(rows))[:, None], mask.shape)[mask]
    species = np.broadcast_to(A0[:, None], mask.shape)[mask].astype(np.int64)
    prim = _array(tables, "row_prim_mask")[rows][mask]
    if len(frac) > 1:
        # Cartesian minimum images, including highly skew cells. Comparing
        # wrapped fractional differences alone is not a closest-vector search.
        _, d2 = pbc_shortest_vectors(Lattice(lattice), frac, frac, return_d2=True)
        coincident = d2 < 1e-6
        np.fill_diagonal(coincident, False)
        same_orbit = orbit[:, None] == orbit[None, :]
        if np.any(~same_orbit & coincident):
            return Conventional(failure="coinciding_orbits")
        if np.any(same_orbit & coincident):
            return Conventional(failure="collapsed_orbit")
    return Conventional(species, frac, lattice, prim, orbit)


def expand_conventional(G, rows, A0, u_c, z, tables) -> Conventional:
    """Expand centred chart tokens; bad sample values return ``failure``.

    z uses the six-slot lattice chart. The sampler owns its end clamp; this
    function preserves the supplied chart and uses orbitol.lattice.to_matrix.
    Missing dependencies and incompatible call shapes are integration errors.
    """
    rows, A0 = _numpy(rows), _numpy(A0)
    u_c, z = _numpy(u_c, np.float64), _numpy(z, np.float64)
    reason = _validate_tokens(G, rows, A0, u_c, tables)
    if reason:
        return Conventional(failure=reason)
    if z.shape != (6,):
        return Conventional(failure="invalid_shape")
    if not np.isfinite(z).all():
        return Conventional(failure="non_finite")
    from orbitol.lattice import to_matrix

    family = torch.tensor([int(_array(tables, "group_family")[G])], dtype=torch.int64)
    lattice = to_matrix(torch.as_tensor(z[None], dtype=torch.float64), family)[0]
    return _expand_members(G, rows.astype(np.int64), A0, u_c, lattice, tables)


def emit_primitive(conv: Conventional, G, tables) -> Structure | None:
    """Emit the frozen primitive basis, selecting exact centring classes."""
    if conv.failure:
        return None
    if int(_array(tables, "group_ncent")[G]) == 1:
        return Structure(conv.lattice, conv.species, conv.frac)
    P = _array(tables, "group_Pc")[G].astype(np.float64)
    keep = conv.prim_mask
    return Structure(
        P.T @ conv.lattice, conv.species[keep],
        (conv.frac[keep] @ np.linalg.inv(P).T) % 1.0,
    )


def contains_ops(structure, group, tables, *, cart_tol, metric_rtol, transform=None) -> bool:
    """Test lattice isometries AND species-preserving periodic bijections.

    Operations are conjugated by the conventional-to-emitted basis columns P.
    Matching uses a bipartite assignment when nearest candidates are ambiguous;
    merely finding a neighbour for each transformed atom is insufficient.
    """
    if structure is None or not len(structure) or not structure.is_ordered:
        return False
    if not (np.isfinite(cart_tol) and np.isfinite(metric_rtol)) or min(cart_tol, metric_rtol) < 0:
        return False
    counts = _array(tables, "group_ops_count")
    if not 1 <= group < len(counts) or counts[group] <= 0:
        return False
    nops = int(counts[group])
    R = _array(tables, "group_ops_R")[group, :nops].astype(np.float64)
    t = _array(tables, "group_ops_t")[group, :nops].astype(np.float64)
    if transform is not None:
        P = _numpy(transform, np.float64)
        if P.shape != (3, 3) or not np.isfinite(P).all():
            return False
        try:
            inv = np.linalg.inv(P)
        except np.linalg.LinAlgError:
            return False
        R, t = inv @ R @ P, t @ inv.T
    lattice, frac = structure.lattice, structure.frac_coords
    L = lattice.matrix
    if not np.isfinite(L).all() or not np.isfinite(frac).all() or abs(np.linalg.det(L)) == 0:
        return False
    K = L @ L.T
    errors = np.linalg.norm(R.transpose(0, 2, 1) @ K @ R - K, axis=(1, 2))
    if np.any(errors > metric_rtol * np.linalg.norm(K)):
        return False
    transformed = (np.einsum("oij,nj->oni", R, frac) + t[:, None]) % 1.0
    _, d2 = pbc_shortest_vectors(lattice, transformed.reshape(-1, 3), frac, return_d2=True)
    n = len(structure)
    species = structure.species
    same_species = np.array([[a == b for b in species] for a in species])
    matches = (d2.reshape(nops, n, n) <= cart_tol**2) & same_species
    if not matches.any(axis=2).all() or not matches.any(axis=1).all():
        return False
    candidates = matches.argmax(axis=2)
    unambiguous = (np.sort(candidates, axis=1) == np.arange(n)).all(axis=1)
    for adjacency in matches[~unambiguous]:
        i, j = linear_sum_assignment(~adjacency)
        if not adjacency[i, j].all():
            return False
    return True


def _niggli_safe(lattice):
    """Whether pymatgen's Niggli reduction terminates on this cell.

    With a nearly zero translation it divides by a vanishing metric entry and returns
    NaN lengths; it then ends with a Cython lattice-point search over a box sized by the
    input basis, which is unbounded for NaN radii and huge for very skewed cells. Cells
    from two-step sampling can make it segfault or hang the sampler.
    A Python exception in this guard means unsafe; reduction then tries LLL.
    """
    try:
        reduced = lattice.get_lll_reduced_lattice()
        if min(reduced.abc) < MIN_TRANSLATION:
            return False
        extent = 1.5 * max(reduced.abc) * np.linalg.norm(np.linalg.inv(lattice.matrix), axis=0)
        return np.prod(2 * np.ceil(extent) + 1) < MAX_NIGGLI_IMAGES
    except Exception:
        return False


def reduced_cell(structure):
    """The same crystal in its Niggli cell, MP-20's cell convention.

    The pinned evaluator (array_dict_to_crystal, crystal.py:352-370) replaces any
    sample with a lattice angle outside (50, 130) degrees by an invalid dummy. The
    standard primitive bases of centred groups often have such angles; Niggli cells
    keep every angle within [60, 120]. A lattice Niggli reduction rejects falls back
    to LLL, then to the input.

    Cells pymatgen's Niggli reduction cannot handle go straight to LLL (numpy); see
    ``_niggli_safe``. Every real crystal takes the Niggli path.
    """
    for algorithm in ("niggli", "LLL") if _niggli_safe(structure.lattice) else ("LLL",):
        try:
            return structure.get_reduced_structure(reduction_algo=algorithm)
        except Exception:
            continue
    return structure


def to_crystalite_rows(structures, n_max) -> CrystaliteRows:
    """Convert emitted cells to the exact four Crystalite tensor fields.

    Coordinates are already uncentred, so no further +1/2 is applied. Each crystal
    is written in its Niggli cell (``reduced_cell``), and Y1 is that cell's exact
    metric under the core's y1 clamp. Failure reasons are exposed
    as ``rows.failures``; None means ``failed_decode`` and a failed Conventional
    preserves its specific reason. Python exceptions during a structure's
    validation or reduction become zero rows with ``decode_error``.
    """
    if not isinstance(n_max, int) or n_max < 1:
        raise ValueError("n_max must be a positive integer")
    structures = list(structures)
    result = CrystaliteRows(len(structures), n_max)
    valid, matrices = [], []
    for i, structure in enumerate(structures):
        try:
            reason = None
            if structure is None:
                reason = "failed_decode"
            elif isinstance(structure, Conventional):
                reason = structure.failure or "not_emitted"
            elif len(structure) > n_max:
                reason = "too_many_atoms"
            elif not len(structure):
                reason = "empty_structure"
            elif not structure.is_ordered:
                reason = "invalid_species"
            elif not (np.isfinite(structure.frac_coords).all()
                      and np.isfinite(structure.lattice.matrix).all()):
                reason = "non_finite"
            elif np.linalg.det(structure.lattice.matrix) == 0:
                reason = "invalid_lattice"
            if reason:
                result.failures[i] = reason
                continue
            structures[i] = reduced_cell(structure)
            matrix = structures[i].lattice.matrix
        except Exception:
            result.failures[i] = "decode_error"
            continue
        valid.append(i)
        matrices.append(matrix)
    if not valid:
        return result
    L = torch.as_tensor(np.stack(matrices), dtype=torch.float64)
    # Only the core's final y1 clamp applies here. Its ltri latent clamp (off-diagonals
    # within 10 Angstrom) bounds Crystalite's own lattice output. The standard primitive
    # basis of a centred group is often skewed, and clamping it would hand the evaluator
    # a different crystal; Orbitol's lattice is already bounded by its chart end clamp.
    y1 = clamp_lattice_latent(gram_to_y1(L @ L.transpose(-1, -2)), "y1").float()
    for j, i in enumerate(valid):
        structure = structures[i]
        if not torch.isfinite(y1[j]).all():
            result.failures[i] = "non_finite"
            continue
        n = len(structure)
        result["A0"][i, :n] = torch.as_tensor(structure.atomic_numbers, dtype=torch.int64)
        # float32 can round a value just below 1 to 1; wrap once after casting.
        result["F1"][i, :n] = torch.as_tensor(structure.frac_coords % 1.0, dtype=torch.float32) % 1.0
        result["Y1"][i] = y1[j]
        result["pad_mask"][i, :n] = False
    return result


def reduce_rows(rows) -> CrystaliteRows:
    """Rewrite saved rows in Niggli cells; every crystal stays the same.

    For pools committed before ``to_crystalite_rows`` reduced cells. Rows without
    atoms stay failures, and a row that cannot be rebuilt is kept unchanged.
    Reducing rows that are already Niggli cells changes them only by rounding.
    """
    values = {key: torch.as_tensor(rows[key]).detach().cpu() for key in ("A0", "F1", "Y1", "pad_mask")}
    structures, kept = [], {}
    for i in range(len(values["A0"])):
        item = {key: value[i] for key, value in values.items()}
        if bool(item["pad_mask"].all()):
            structures.append(None)
            continue
        try:
            structures.append(tokens_to_structure(item))
        except Exception:
            structures.append(None)
            kept[i] = item
    result = to_crystalite_rows(structures, int(values["A0"].shape[1]))
    for i, item in kept.items():
        for key, value in item.items():
            result[key][i] = value.to(result[key].dtype)
        result.failures.pop(i, None)
    return result


def _batch_skeletons(skel, B, K):
    """Normalise public Skeletons and the sampler's padded tensor mapping."""
    if isinstance(skel, Mapping):
        pad = torch.as_tensor(skel["pad_mask"], dtype=torch.bool).detach().cpu()
        rows, groups = _numpy(skel["row"]), _numpy(skel["G"])
        if pad.shape != (B, K) or rows.shape != (B, K) or groups.shape != (B,):
            raise ValueError("Sampler shapes and skeleton batch size disagree")
        positions = [np.flatnonzero(~p) for p in pad.numpy()]
        fixed = None if skel.get("fixed_A0") is None else _numpy(skel["fixed_A0"])
        if fixed is not None and fixed.shape != (B, K):
            raise ValueError("fixed_A0 must have shape (B,K)")
        skeletons = [Skeleton(g, tuple(rows[i, pos]), None if fixed is None else tuple(fixed[i, pos]))
                     for i, (g, pos) in enumerate(zip(groups, positions, strict=True))]
        return skeletons, pad, positions
    skeletons = [skel] * B if isinstance(skel, Skeleton) else list(skel)
    if len(skeletons) != B:
        raise ValueError("Sampler shapes and skeleton batch size disagree")
    pad = torch.ones((B, K), dtype=torch.bool)
    positions = []
    for i, skeleton in enumerate(skeletons):
        n = len(skeleton.rows)
        if n > K:
            raise ValueError("Skeleton has more rows than sampler tokens")
        pad[i, :n] = False
        positions.append(np.arange(n))
    return skeletons, pad, positions


@torch.no_grad()
def decode_batch(sampler_out, skel, tables, encoding, allowed_mask, *, n_max) -> DecodedBatch:
    """Isolate per-structure geometry exceptions as ``decode_error``.

    Expansion, primitive emission, containment, validation and reduction failures
    affect only that structure. Batch shapes and bad call arguments still raise.
    Existing reduction fallbacks are preserved. Native crashes and hangs are out
    of scope; Python exception handling cannot contain them.
    """
    types = torch.as_tensor(sampler_out["type"]).detach().cpu().float()
    u_c = _numpy(sampler_out["u"] if "u" in sampler_out else sampler_out["u_c"], np.float64)
    z = _numpy(sampler_out["z"], np.float64)
    if types.ndim != 3 or types.shape[-1] != 16:
        raise ValueError("type must have shape (B,K,16)")
    B, K = types.shape[:2]
    if u_c.shape != (B, K, 3) or z.shape != (B, 6):
        raise ValueError("u must have shape (B,K,3) and z must have shape (B,6)")
    finite = _numpy(sampler_out.get("finite", np.ones(B, dtype=bool)))
    skeletons, pad_mask, positions = _batch_skeletons(skel, B, K)
    A0 = decode_types(types, pad_mask, encoding, allowed_mask).numpy()
    finite_types = (torch.isfinite(types).all(dim=-1) | pad_mask).all(dim=-1).numpy()
    requested = np.array([s.G for s in skeletons], dtype=np.int64)
    structures: list[Structure | None] = [None] * B
    failures, contains_a = {}, np.zeros(B, dtype=bool)
    # Transfer tensors once, snap types in one matmul, and vectorise lattice
    # charts within each family. Small Structure objects are built individually.
    token_rows, token_species, token_coords, valid_by_family = {}, {}, {}, {}
    for i, skeleton in enumerate(skeletons):
        rows = np.asarray(skeleton.rows)
        species = np.asarray(skeleton.A0) if skeleton.A0 is not None else A0[i, positions[i]]
        coords = u_c[i, positions[i]]
        reason = _validate_tokens(skeleton.G, rows, species, coords, tables)
        if reason is None and (not finite[i] or not np.isfinite(z[i]).all()
                               or (skeleton.A0 is None and not finite_types[i])):
            reason = "non_finite"
        if reason:
            failures[i] = reason
            continue
        token_rows[i], token_species[i] = rows.astype(np.int64), species
        token_coords[i] = coords
        family = int(_array(tables, "group_family")[skeleton.G])
        valid_by_family.setdefault(family, []).append(i)
    if valid_by_family:
        from orbitol.lattice import to_matrix

    for family, indices in valid_by_family.items():
        families = torch.full((len(indices),), family, dtype=torch.int64)
        matrices = _numpy(to_matrix(torch.as_tensor(z[indices], dtype=torch.float64), families), np.float64)
        for i, matrix in zip(indices, matrices, strict=True):
            try:
                G, rows = int(requested[i]), token_rows[i]
                conv = _expand_members(G, rows, token_species[i], token_coords[i], matrix, tables)
                if conv.failure:
                    failures[i] = conv.failure
                    continue
                structure = emit_primitive(conv, G, tables)
                contains_a[i] = contains_ops(
                    structure, G, tables, transform=_array(tables, "group_Pc")[G], **CHECK_A,
                )
                if not contains_a[i]:
                    failures[i] = "containment_failed"
                    continue
                structures[i] = structure
            except Exception:
                failures[i] = "decode_error"
    rows = to_crystalite_rows(structures, n_max)
    failures = {**rows.failures, **failures}
    rows.failures = failures
    for i in failures:
        structures[i] = None
    return DecodedBatch(structures, rows, failures, contains_a, requested)
