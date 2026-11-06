"""Frozen fractional symmetry tables in pyxtal 1.1.2's default Hall settings.

Conventions are those of orbitol.contracts. build_tables derives every row's chart,
member maps, chart images, H metric, period lattice and Voronoi-relevant vectors, and
every group's normaliser cosets, from pyxtal; verify_tables re-checks them against
pyxtal. No Crystalite imports or data files are needed.

Arrays follow TABLE_ARRAYS. Inactive rows/groups are zero except row_H,
row_Hinv, row_lambda_basis, group_MG and group_Pc, which start as identities.
Row 0 has P1 geometry, masks and wrap data, but zero descriptors and empty letter.
Group 0 has P1 geometry, with hall=0, empty symbol and zero descriptor/range.
Image matrices are zero outside their d x d block, including on zero-dof rows.
Q uses numpy's upper-triangle order: xx, xy, xz, yy, yz, zz.

cosets.npz is a packed, unpadded Cosets dataclass. With C cosets and T row
maps, its arrays are:
  group_start, group_count (231,) int32: slice of cosets for group G;
  group (C,) int16; P (C,3,3), p (C,3) float64: x' = P x + p;
  det (C,) int8; proper, reexpress_lattice (C,) bool;
  map_start (C+1,) int32: row-map offsets, starting at 0 and ending at T;
  source_row, target_row (T,) int16: global ids;
  S (T,3,3), s (T,3) float64: u_target = S u_source + s, padded with zero.
Row maps within each coset follow ascending source row ids. They identify the
transformed orbit, so its representative can differ by an operation of G.
Cosets include identity for built G > 1; G=1 has none. Cosets are closed and
quotiented by G and continuous polar translations; proper representatives are
preferred. Improper-only cosets remain stored, flagged proper=False.

Relevant lattice vectors are the unique shortest pairs +/-v in
nonzero cosets of Lambda / 2 Lambda. An ellipsoid bound makes enumeration complete.
Subtracting improving relevant vectors terminates in the closed Voronoi cell,
so wrap_exact_np is an exact closest-vector slicer (up to floating-point roundoff).

sha256 covers every table AND coset array, sorted by qualified name,
including names, shapes, dtypes and canonical little-endian bytes. Dates and
library versions are metadata, not part of the content hash.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import hashlib
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
from pymatgen.core.operations import SymmOp

from orbitol import contracts as C
from orbitol.descriptors import descriptor_tables, groups_metadata, point_metric, Q_INDICES

TOL = 1e-9
NV1 = np.array(list(itertools.product(range(-1, 2), repeat=3)), dtype=float)
NV4 = np.array(list(itertools.product(range(-4, 5), repeat=3)), dtype=float)
# Prefer the smallest lift, and break ties lexicographically.
NV1 = NV1[np.argsort(np.abs(NV1).sum(axis=1), kind="stable")]
NV4 = NV4[np.argsort(np.abs(NV4).sum(axis=1), kind="stable")]

ORIGIN_SHIFTS = {}
for _groups, _q in (
    ((48, 86, 126, 201, 222, 224), (1/4, 1/4, 1/4)),
    ((50, 59, 125), (1/4, 1/4, 0)),
    ((85, 129, 130), (1/4, 3/4, 0)),
    ((133, 134, 137, 138), (1/4, 3/4, 1/4)),
    ((70, 203, 227), (1/8, 1/8, 1/8)),
    ((68,), (0, 1/4, 1/4)), ((88,), (0, 1/4, 1/8)),
    ((141, 142), (0, 3/4, 1/8)), ((228,), (3/8, 3/8, 3/8)),
):
    ORIGIN_SHIFTS.update(dict.fromkeys(_groups, _q))

# Columns in conventional coordinates, positive determinants 1/c(G).
PRIMITIVE_BASES = {
    "P": np.eye(3),
    "A": np.array([[1, 0, 0], [0, .5, -.5], [0, .5, .5]]),
    "B": np.array([[.5, 0, -.5], [0, 1, 0], [.5, 0, .5]]),
    "C": np.array([[.5, -.5, 0], [.5, .5, 0], [0, 0, 1]]),
    "I": np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]]),
    "F": np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
    "R": np.array([[2, -1, -1], [1, 1, -2], [1, 1, 1]], dtype=float) / 3,
}
CENTRE_VECTORS = {
    "P": [[0, 0, 0]], "A": [[0, 0, 0], [0, .5, .5]],
    "B": [[0, 0, 0], [.5, 0, .5]], "C": [[0, 0, 0], [.5, .5, 0]],
    "I": [[0, 0, 0], [.5, .5, .5]],
    "F": [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
    "R": [[0, 0, 0], [2/3, 1/3, 1/3], [1/3, 2/3, 2/3]],
}


def _mod1(x):
    return x - np.floor(x + 1e-12)


def _wrapped(x):
    return x - np.rint(x)


def _rational(x):
    """All affine chart data are multiples of 1/48; snap them to that grid exactly."""
    rounded = np.rint(np.asarray(x) * 48) / 48
    if not np.allclose(x, rounded, atol=TOL, rtol=0):
        raise ValueError("Affine table value is outside the frozen rational grid")
    return rounded + 0.0  # canonical positive zero


def _ops(operations):
    return (np.array([op.rotation_matrix for op in operations]),
            np.array([op.translation_vector for op in operations]))


def _shape(shape):
    return tuple(C.R_TOTAL if s == "R" else C.G_TOTAL if s == "G"
                 else getattr(C, s) if isinstance(s, str) else s for s in shape)


@dataclass
class Cosets:
    """Packed normaliser arrays; see the module docstring for their layout."""
    group_start: np.ndarray
    group_count: np.ndarray
    group: np.ndarray
    P: np.ndarray
    p: np.ndarray
    det: np.ndarray
    proper: np.ndarray
    reexpress_lattice: np.ndarray
    map_start: np.ndarray
    source_row: np.ndarray
    target_row: np.ndarray
    S: np.ndarray
    s: np.ndarray

    @property
    def arrays(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _hash(arrays, cosets):
    digest = hashlib.sha256()
    all_arrays = {"tables/" + k: v for k, v in arrays.items()}
    all_arrays.update({"cosets/" + k: v for k, v in cosets.arrays.items()})
    for name, arr in sorted(all_arrays.items()):
        arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("<"), copy=False))
        header = json.dumps([name, list(arr.shape), arr.dtype.str], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(arr.tobytes())
    return digest.hexdigest()


@dataclass
class TableSet:
    arrays: dict[str, np.ndarray]
    cosets: Cosets
    build_date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pyxtal_version: str = field(default_factory=lambda: version("pyxtal"))
    spglib_version: str = field(default_factory=lambda: version("spglib"))

    @property
    def sha256(self) -> str:
        # Compute from current data, so accidental mutation cannot leave a stale hash.
        return _hash(self.arrays, self.cosets)

    def row_id(self, group: int, letter: str) -> int:
        if not 1 <= group <= C.N_GROUPS:
            raise ValueError(f"Invalid space group {group}")
        a = self.arrays
        start, count = int(a["group_row_start"][group]), int(a["group_row_count"][group])
        matches = np.flatnonzero(a["row_letter"][start:start + count] == letter)
        if len(matches) != 1:
            raise KeyError(f"Row {group}:{letter} is not present in this table build")
        return start + int(matches[0])

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _validate_layout(self)
        np.savez_compressed(directory / "tables.npz", **self.arrays)
        np.savez_compressed(directory / "cosets.npz", **self.cosets.arrays)
        manifest = {
            "format_version": 1, "sha256": self.sha256,
            "pyxtal": self.pyxtal_version, "spglib": self.spglib_version,
            "build_date": self.build_date,
            "groups": (np.flatnonzero(self.arrays["group_hall"][1:]) + 1).tolist(),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def torch(self, device, dtype=torch.float32):
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating torch dtype")
        tensors = SimpleNamespace(**{
            name: torch.as_tensor(arr, device=device,
                                  dtype=dtype if arr.dtype.kind == "f" else None)
            for name, arr in self.arrays.items() if arr.dtype.kind not in "US"
        })
        # LAPACK on the CPU keeps identity and diagonal rows exact on every device.
        values, vectors = torch.linalg.eigh(torch.as_tensor(self.arrays["row_H"], dtype=torch.float64))
        tensors.row_Hinv_sqrt = ((vectors / values.sqrt()[..., None, :]) @ vectors.transpose(-1, -2)).to(device, dtype)
        return tensors


def load_tables(directory) -> TableSet:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported symmetry table format")
    with np.load(directory / "tables.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    with np.load(directory / "cosets.npz", allow_pickle=False) as data:
        coset_arrays = {name: data[name] for name in data.files}
    if set(coset_arrays) != {f.name for f in fields(Cosets)}:
        raise ValueError("Unexpected coset array names")
    ts = TableSet(arrays, Cosets(**coset_arrays),
                  build_date=manifest["build_date"], pyxtal_version=manifest["pyxtal"],
                  spglib_version=manifest["spglib"])
    _validate_layout(ts)
    if ts.sha256 != manifest.get("sha256"):
        raise ValueError("Symmetry table sha256 mismatch")
    return ts


def _validate_layout(ts):
    if set(ts.arrays) != set(C.TABLE_ARRAYS):
        raise ValueError("Unexpected table array names")
    for name, (shape, dtype, _) in C.TABLE_ARRAYS.items():
        a = ts.arrays[name]
        if a.shape != _shape(shape) or a.dtype != np.dtype(dtype):
            raise ValueError(f"Invalid shape/dtype for {name}: {a.shape}, {a.dtype}")
    cs = ts.cosets
    n, t = len(cs.group), len(cs.source_row)
    specs = {
        "group_start": ((C.G_TOTAL,), "int32"), "group_count": ((C.G_TOTAL,), "int32"),
        "group": ((n,), "int16"), "P": ((n, 3, 3), "float64"), "p": ((n, 3), "float64"),
        "det": ((n,), "int8"), "proper": ((n,), "bool"), "reexpress_lattice": ((n,), "bool"),
        "map_start": ((n + 1,), "int32"), "source_row": ((t,), "int16"),
        "target_row": ((t,), "int16"), "S": ((t, 3, 3), "float64"), "s": ((t, 3), "float64"),
    }
    for name, (shape, dtype) in specs.items():
        arr = getattr(cs, name)
        if arr.shape != shape or arr.dtype != np.dtype(dtype):
            raise ValueError(f"Invalid coset shape/dtype for {name}")
    if cs.map_start[0] != 0 or cs.map_start[-1] != t or np.any(np.diff(cs.map_start) < 0):
        raise ValueError("Invalid coset map offsets")


@dataclass
class _Row:
    id: int
    wp: object
    A: np.ndarray
    b: np.ndarray
    W: np.ndarray
    v: np.ndarray
    E: np.ndarray

    def __post_init__(self):
        self.member_left = np.linalg.pinv(self.W)
        self.member_normal = np.eye(3) - self.W @ self.member_left

    @property
    def d(self):
        return self.A.shape[1]

    @property
    def m(self):
        return len(self.W)


def _make_rows(group, start):
    rows = []
    for i, wp in enumerate(reversed(list(group))):
        free = [k for k in range(3) if k not in wp.get_frozen_axis()]
        E = np.eye(3)[:, free]
        R, t = _ops(wp.ops)
        W = R @ E
        rows.append(_Row(start + i, wp, W[0], t[0], W, t, E))
    return rows


def _chart_images(row):
    d = row.d
    if d == 0:
        return np.zeros((1, 0, 0)), np.zeros((1, 0))
    # E^T A = I in every frozen chart; unlike an SVD this stays exact for integers.
    left = row.E.T
    Q = np.eye(3) - row.A @ left
    S = left @ row.W
    linear = np.max(np.abs(row.A @ S - row.W), axis=(1, 2)) < TOL
    images = []
    for j in np.flatnonzero(linear):
        for shifts in (NV1, NV4):
            rhs = row.v[j] - row.b + shifts
            good = np.flatnonzero(np.max(np.abs(rhs @ Q.T), axis=1) < TOL)
            if len(good):
                images.append((_rational(S[j]), _rational(_mod1(left @ rhs[good[0]]))))
                break
    if not images:
        raise ValueError(f"No chart images for row {row.id}")
    return np.array([x[0] for x in images]), np.array([x[1] for x in images])


def _lll(B, H, delta=.99):
    """Textbook LLL with Lovasz parameter delta on a column basis, inner product x^T H y."""
    B = B.copy()
    d = B.shape[1]

    def gso():
        orth, mu = np.zeros_like(B), np.zeros((d, d))
        for i in range(d):
            v = B[:, i].copy()
            for j in range(i):
                mu[i, j] = B[:, i] @ H @ orth[:, j] / (orth[:, j] @ H @ orth[:, j])
                v -= mu[i, j] * orth[:, j]
            orth[:, i] = v
        return orth, mu

    k = 1
    for _ in range(10000):
        if k >= d:
            return _rational(B)
        orth, mu = gso()
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q:
                B[:, k] -= q * B[:, j]
                orth, mu = gso()
        norms = np.einsum("ik,ij,jk->k", orth, H, orth)
        if norms[k] >= (delta - mu[k, k - 1] ** 2) * norms[k - 1] - 1e-14:
            k += 1
        else:
            B[:, [k - 1, k]] = B[:, [k, k - 1]]
            k = max(k - 1, 1)
    raise RuntimeError("LLL failed to terminate")


def _lattice_basis(d, shifts):
    den = 1
    for v in np.asarray(shifts).ravel():
        den = math.lcm(den, Fraction(float(v)).limit_denominator(48).denominator)
    rows = (np.vstack([np.eye(d), shifts]) * den).round().astype(np.int64).tolist()
    basis = []
    for col in range(d):
        rows = [r for r in rows if any(r)]
        piv = [r for r in rows if r[col]]
        rest = [r for r in rows if not r[col]]
        while len(piv) > 1:
            piv.sort(key=lambda r: abs(r[col]))
            p, new = piv[0], [piv[0]]
            for r in piv[1:]:
                q = r[col] // p[col]
                rr = [a - q * b for a, b in zip(r, p)]
                (new if rr[col] else rest).append(rr)
            piv = new
        if piv:
            basis.append(piv[0])
        rows = rest
    B = np.array(basis, dtype=float).T / den
    if B.shape != (d, d) or abs(np.linalg.det(B)) < 1e-12:
        raise ValueError("Period generators do not span a full-rank lattice")
    return B


def _voronoi_vectors(B, H):
    """Certified parity-coset enumeration, not a radius heuristic.

    A parity vector p gives an upper bound p^T K p for that coset's shortest
    vector. For every candidate c with c^T K c <= U, Cauchy-Schwarz gives
    |c_i| <= sqrt(U (K^-1)_ii). Thus the integer box below contains all ties.
    Only cosets with precisely the two minimisers +/-c yield relevant vectors.
    """
    d = len(B)
    K = B.T @ H @ B
    inv = np.linalg.inv(K)
    vectors = []
    for parity in itertools.product((0, 1), repeat=d):
        if not any(parity):
            continue
        p = np.array(parity)
        upper = p @ K @ p
        bounds = np.floor(np.sqrt(upper * np.diag(inv)) + 1e-9).astype(int)
        ranges = [range(-int(b) + ((int(b) + int(q)) % 2), int(b) + 1, 2)
                  for b, q in zip(bounds, p)]
        coeff = np.array(list(itertools.product(*ranges)), dtype=float)
        norms = np.einsum("ni,ij,nj->n", coeff, K, coeff)
        best = norms.min()
        shortest = coeff[np.abs(norms - best) < 1e-10 * max(1, best)]
        if len(shortest) == 2 and np.array_equal(shortest[0], -shortest[1]):
            vectors.extend(shortest @ B.T)
    result = np.array(vectors).reshape(-1, d)
    if len(result) > C.MAX_VRV or np.linalg.matrix_rank(result) != d:
        raise ValueError("Invalid Voronoi-relevant vector certificate")
    return _rational(result)


@lru_cache(maxsize=None)
def _period_data(d, metric, translations):
    H = np.array(metric).reshape(d, d)
    shifts = np.array(translations).reshape(-1, d)
    B = _lll(_lattice_basis(d, shifts), H)
    vectors = _voronoi_vectors(B, H)
    fast = abs(abs(np.linalg.det(B)) - 1) < TOL and np.allclose(H, np.diag(np.diag(H)), atol=TOL)
    return B, vectors, fast


def wrap_exact_np(delta: np.ndarray, row: int, ts: TableSet) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or delta.shape[1] != 3 or not np.isfinite(delta).all():
        raise ValueError("delta must be a finite (n, 3) array")
    if not 0 <= row < C.R_TOTAL:
        raise ValueError(f"Invalid row id {row}")
    a = ts.arrays
    d = int(a["row_dof"][row])
    out = np.zeros_like(delta)
    if not d or not len(delta):
        return out
    x = delta[:, :d]
    if a["row_wrap_fast"][row]:
        out[:, :d] = _wrapped(x)
        return out
    B, H = a["row_lambda_basis"][row, :d, :d], a["row_H"][row, :d, :d]
    # Babai nearest plane in the H metric (QR of a Euclidean square-root image).
    root = np.linalg.cholesky(H).T
    Q, R = np.linalg.qr(root @ B)
    coeff = np.zeros_like(x)
    y = x @ root.T @ Q
    for i in range(d - 1, -1, -1):
        coeff[:, i] = np.rint((y[:, i] - coeff[:, i + 1:] @ R[i, i + 1:]) / R[i, i])
    residual = x - coeff @ B.T
    v = a["row_vrv"][row, :int(a["row_vrv_count"][row]), :d]
    if len(v) == 0:
        raise ValueError(f"Missing Voronoi vectors for row {row}")
    norms = np.einsum("ni,ij,nj->n", v, H, v)
    # Moving by v improves the squared norm by 2<r,v> - <v,v>.
    while True:
        gains = 2 * residual @ H @ v.T - norms
        best = np.argmax(gains, axis=1)
        improve = gains[np.arange(len(x)), best] > 1e-13 * max(1.0, float(norms.max()))
        if not np.any(improve):
            break
        residual[improve] -= v[best[improve]]
    out[:, :d] = residual
    return out


def _op_codes(R, t):
    """Collision-free integer codes for operations with integer R and t on the 1/48 grid."""
    R = np.asarray(R).reshape(-1, 3, 3)
    t = np.asarray(t).reshape(-1, 3)
    ri = np.rint(R).astype(np.int64)
    tt = _mod1(t) * 48
    if np.any(np.abs(R - ri) > TOL) or np.any(np.abs(ri) > 2) or np.any(np.abs(tt - np.rint(tt)) > 1e-7):
        raise ValueError("Operation is outside the frozen affine grid")
    rc = (ri.reshape(-1, 9) + 2) @ (5 ** np.arange(9, dtype=np.int64))
    ti = np.rint(tt).astype(np.int64) % 48
    return rc * 48**3 + ti @ np.array([48**2, 48, 1], dtype=np.int64)


def _coset_key(P, p, R, t, polar):
    shifts = t @ P.T + p
    shifts[:, polar] = 0
    # A single minimum code identifies the entire right coset T G.
    return int(_op_codes(P @ R, shifts).min())


def _normalisers(group, R, t):
    """Close pyxtal alternatives modulo G and continuous polar shifts.

    Repairs group 19's 94 distinct listed cosets to 96. Groups 44, 45 and 80
    are quotiented by their free z-origin. Every supplied letter list is read
    inversely, then independently checked against the geometric row map.
    """
    if group.number == 1:
        return [], {}
    polar = list(group.get_free_axis())
    alternatives = group.get_alternatives()
    letters = [wp.letter for wp in reversed(list(group))]
    reps, expected = {}, {}
    for rep, transformed in zip(alternatives["Coset Representative"], alternatives["Transformed WP"]):
        op = SymmOp.from_xyz_str(rep)
        P, p = op.rotation_matrix.copy(), _mod1(op.translation_vector.copy())
        p[polar] = 0
        key = _coset_key(P, p, R, t, polar)
        reps.setdefault(key, (P, p))
        inverse_list = transformed.split()
        if sorted(inverse_list) != sorted(letters):
            raise ValueError(f"Invalid pyxtal alternative letters for group {group.number}")
        # Entry i names the SOURCE that T maps onto the i-th TARGET row.
        permutation = dict(zip(inverse_list, letters))
        if key in expected and expected[key] != permutation:
            raise ValueError(f"Conflicting inverse alternatives in group {group.number}")
        expected[key] = permutation
    generators = list(reps.values())
    frontier = list(reps.values())
    while frontier:
        new = []
        for P1, p1 in frontier:
            for P2, p2 in generators:
                P, p = _rational(P1 @ P2), _rational(_mod1(P1 @ p2 + p1))
                p[polar] = 0
                key = _coset_key(P, p, R, t, polar)
                if key not in reps:
                    reps[key] = (P, p)
                    new.append((P, p))
        if len(reps) > 192:
            raise ValueError(f"Unexpected unbounded normaliser closure for group {group.number}")
        frontier = new
    improper = np.flatnonzero(np.linalg.det(R) < 0)
    proper_reps = []
    for key, (P, p) in reps.items():
        if np.linalg.det(P) < 0 and len(improper):
            j = improper[0]
            P, p = P @ R[j], P @ t[j] + p
        p = _rational(_mod1(p))
        p[polar] = 0
        proper_reps.append((key, _rational(P), p))
    return proper_reps, expected


def _generic_gram(family):
    # Distinct free entries expose axis permutations requiring re-expression.
    return {
        "triclinic": [[2, .2, .3], [.2, 3, .4], [.3, .4, 5]],
        "monoclinic": [[2, 0, .3], [0, 3, 0], [.3, 0, 5]],
        "orthorhombic": np.diag([2, 3, 5]),
        "tetragonal": np.diag([2, 2, 5]),
        "hexagonal": [[2, -1, 0], [-1, 2, 0], [0, 0, 5]],
        "cubic": np.eye(3) * 2,
    }[family]


def _row_map(source, targets, P, p):
    Y, y = P @ source.A, P @ source.b + p
    matches = []
    for target in targets:
        if (target.m, target.d) != (source.m, source.d):
            continue
        if target.d == 0:
            if np.any(np.max(np.abs(_wrapped(y - target.v)), axis=1) < TOL):
                matches.append((target, np.zeros((0, 0)), np.zeros(0)))
            continue
        left, normal = target.member_left, target.member_normal
        linear = np.max(np.abs(normal @ Y), axis=(1, 2)) < TOL
        for j in np.flatnonzero(linear):
            found = False
            for shifts in (NV1, NV4):
                rhs = y - target.v[j] + shifts
                good = np.flatnonzero(np.max(np.abs(rhs @ normal[j].T), axis=1) < TOL)
                if len(good):
                    S = _rational(left[j] @ Y)
                    s = _rational(_mod1(left[j] @ rhs[good[0]]))
                    matches.append((target, S, s))
                    found = True
                    break
            if found:
                break
    if len(matches) != 1:
        raise ValueError(f"Normaliser row {source.id} has {len(matches)} geometric targets")
    return matches[0]


def _build_cosets(contexts):
    starts, counts = np.zeros(C.G_TOTAL, np.int32), np.zeros(C.G_TOTAL, np.int32)
    groups, matrices, offsets, dets, reexpress = [], [], [], [], []
    map_starts, sources, targets, linear, shifts = [0], [], [], [], []
    for group, rows, R, t, family in contexts:
        n = group.number
        starts[n] = len(groups)
        reps, expected = _normalisers(group, R, t)
        counts[n] = len(reps)
        for key, P, p in reps:
            permutation = {}
            for source in rows:
                target, S, s = _row_map(source, rows, P, p)
                permutation[source.wp.letter] = target.wp.letter
                sources.append(source.id)
                targets.append(target.id)
                padded_S, padded_s = np.zeros((3, 3)), np.zeros(3)
                padded_S[:target.d, :source.d] = S
                padded_s[:target.d] = s
                linear.append(padded_S)
                shifts.append(padded_s)
            if len(set(permutation.values())) != len(rows):
                raise ValueError(f"Non-bijective geometric normaliser in group {n}")
            if key in expected and permutation != expected[key]:
                raise ValueError(f"Inverse pyxtal list disagrees with geometry in group {n}")
            map_starts.append(len(sources))
            groups.append(n)
            matrices.append(P)
            offsets.append(p)
            dets.append(round(np.linalg.det(P)))
            inv = np.linalg.inv(P)
            gram = np.asarray(_generic_gram(family))
            reexpress.append(not np.allclose(inv.T @ gram @ inv, gram, atol=TOL, rtol=0))
    dets = np.array(dets, dtype=np.int8)
    return Cosets(
        starts, counts, np.array(groups, dtype=np.int16),
        np.array(matrices, dtype=float).reshape(-1, 3, 3),
        np.array(offsets, dtype=float).reshape(-1, 3), dets, dets > 0,
        np.array(reexpress, dtype=bool), np.array(map_starts, dtype=np.int32),
        np.array(sources, dtype=np.int16), np.array(targets, dtype=np.int16),
        np.array(linear, dtype=float).reshape(-1, 3, 3), np.array(shifts, dtype=float).reshape(-1, 3),
    )


def _fill_row(a, row, n, metric, root, cent):
    r, d, m = row.id, row.d, row.m
    a["row_group"][r], a["row_letter"][r] = n, row.wp.letter
    a["row_mult"][r], a["row_mprim"][r], a["row_dof"][r] = m, m // len(cent), d
    a["row_A"][r, :, :d], a["row_b"][r] = row.A, row.b
    a["row_W"][r, :m, :, :d], a["row_v"][r, :m] = row.W, row.v
    a["row_member_mask"][r, :m] = True
    seen = set()
    for j, (W, v) in enumerate(zip(row.W, row.v)):
        ck = min(tuple(np.rint(_mod1(v + c) * 48).astype(int) % 48) for c in cent)
        key = (tuple(W.ravel()), ck)
        a["row_prim_mask"][r, j] = key not in seen
        seen.add(key)
    if len(seen) * len(cent) != m:
        raise ValueError(f"Invalid centring classes for row {r}")
    if d:
        H = row.A.T @ metric @ row.A
        inv = np.linalg.inv(H)
        a["row_H"][r, :d, :d], a["row_Hinv"][r, :d, :d] = H, inv
        a["row_Pi"][r, :d] = inv @ row.A.T @ metric
        a["row_N"][r, :d] = inv @ row.A.T @ root
        a["row_Q"][r] = (row.A @ inv @ row.A.T)[Q_INDICES]
    S, s = _chart_images(row)
    count = len(S)
    if count > C.MAX_IMAGES:
        raise ValueError(f"Row {r} needs {count} images, exceeding MAX_IMAGES={C.MAX_IMAGES}")
    a["row_img_count"][r] = count
    a["row_img_S"][r, :count, :d, :d], a["row_img_s"][r, :count, :d] = S, s
    if d:
        translations = s[np.max(np.abs(S - np.eye(d)), axis=(1, 2)) < TOL]
        B, vrv, fast = _period_data(d, tuple(H.ravel()), tuple(translations.ravel()))
        a["row_lambda_basis"][r, :d, :d] = B
        a["row_vrv"][r, :len(vrv), :d] = vrv
        a["row_vrv_count"][r], a["row_wrap_fast"][r] = len(vrv), fast
    else:
        a["row_wrap_fast"][r] = True  # the unique residual in zero dimensions is zero


def build_tables(groups: Iterable[int] | None = None) -> TableSet:
    for package, pinned in (("pyxtal", "1.1.2"), ("spglib", "2.7.0")):
        if version(package) != pinned:
            raise RuntimeError(f"Frozen tables require {package} {pinned}; found {version(package)}")
    selected = list(range(1, C.G_TOTAL)) if groups is None else list(groups)
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
           or not 1 <= n <= C.N_GROUPS for n in selected):
        raise ValueError("groups must contain integer space-group numbers 1..230")
    selected = sorted(set(int(n) for n in selected))
    metadata = groups_metadata()
    a = {name: np.zeros(_shape(shape), dtype=dtype)
         for name, (shape, dtype, _) in C.TABLE_ARRAYS.items()}
    for name in ("row_H", "row_Hinv", "row_lambda_basis", "group_MG", "group_Pc"):
        a[name][:] = np.eye(3)
    start = 1
    for n, group in enumerate(metadata, 1):
        a["group_row_start"][n], a["group_row_count"][n] = start, len(group)
        start += len(group)
    if start != C.R_TOTAL:
        raise ValueError(f"Frozen Wyckoff count changed: {start - 1}")
    # Row 0 is the P1 chart, with its sentinel identity/descriptor fields retained.
    pad = _make_rows(metadata[0], 0)[0]
    _fill_row(a, pad, 0, np.eye(3), np.eye(3), np.zeros((1, 3)))
    a["row_letter"][0] = ""
    a["group_ncent"][0], a["group_ops_count"][0] = 1, 1
    a["group_ops_R"][0, 0] = np.eye(3, dtype=np.int8)
    a["group_polar"][0] = True
    row_desc, group_desc, _, _, _ = descriptor_tables()
    contexts = []
    for n in selected:
        group = metadata[n - 1]
        start = int(a["group_row_start"][n])
        rows = _make_rows(group, start)
        R, t = _ops(group[0].ops)
        centring = group.symbol[0]
        cent = np.array(CENTRE_VECTORS[centring])
        family = "hexagonal" if group.lattice_type == "trigonal" else group.lattice_type
        metric = point_metric(group)
        vals, vecs = np.linalg.eigh(metric)
        root = (vecs * np.sqrt(vals)) @ vecs.T
        a["group_hall"][n], a["group_symbol"][n] = group.hall_number, group.symbol
        a["group_q"][n] = ORIGIN_SHIFTS.get(n, (0, 0, 0))
        a["group_family"][n], a["group_centring"][n] = C.FAMILIES.index(family), C.CENTRINGS.index(centring)
        a["group_ncent"][n], a["group_cent_vecs"][n, :len(cent)] = len(cent), cent
        a["group_Pc"][n], a["group_MG"][n] = PRIMITIVE_BASES[centring], metric
        a["group_ops_R"][n, :len(R)], a["group_ops_t"][n, :len(t)] = R, t
        a["group_ops_count"][n] = len(R)
        a["group_polar"][n, list(group.get_free_axis())] = True
        a["group_desc"][n] = group_desc[n]
        for row in rows:
            _fill_row(a, row, n, metric, root, cent)
            a["row_desc"][row.id] = row_desc[row.id]
        contexts.append((group, rows, R, t, family))
    ts = TableSet(a, _build_cosets(contexts))
    _validate_layout(ts)
    return ts


def _point_keys(points):
    return {tuple(v) for v in (np.rint(_mod1(points) * 1e8).astype(np.int64) % 100000000)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _close(actual, expected, message, tol=1e-9):
    if not np.allclose(actual, expected, atol=tol, rtol=0):
        raise ValueError(message)


def _verify_wrap_certificate(B, H, vectors):
    """Check relevant vectors by a separate, complete ellipsoid search.

    The largest norm of the 2^d binary coefficient vectors bounds the minimum
    in every parity class. Search one enclosing box, then classify minimisers.
    This intentionally differs from the per-coset construction in the builder.
    """
    d = len(B)
    K = B.T @ H @ B
    binary = np.array(list(itertools.product((0, 1), repeat=d)))
    upper = np.einsum("ni,ij,nj->n", binary, K, binary).max()
    bounds = np.ceil(np.sqrt(upper * np.diag(np.linalg.inv(K)))).astype(int)
    coeff = np.array(list(itertools.product(*(range(-b, b + 1) for b in bounds))))
    norms = np.einsum("ni,ij,nj->n", coeff, K, coeff)
    want = []
    for parity in binary[1:]:
        mask = np.all(coeff % 2 == parity, axis=1)
        best = norms[mask].min()
        winners = coeff[mask & (np.abs(norms - best) < 1e-10)]
        if len(winners) == 2:
            _require(np.array_equal(winners[0], -winners[1]), "Asymmetric parity minimisers")
            want.extend(winners.tolist())
    got = np.linalg.solve(B, vectors.T).T
    _close(got, np.rint(got), "Voronoi vectors are not lattice vectors")
    _require({tuple(v) for v in np.rint(got).astype(int)} == {tuple(v) for v in want},
             "Incomplete or spurious Voronoi-relevant vectors")


def verify_tables(ts: TableSet, *, require_all: bool = True) -> dict:
    """Exhaustive table checks.

    Runs algebra on all built groups/rows, three generic chart samples per row,
    all chart images, all normaliser row maps and coset products, Hall settings,
    period and Voronoi certificates, and descriptor normalization. The CLI
    requires all 230 groups. MP-20 and from_seed pipeline checks
    are outside this table-only verifier (no dataset is read).
    """
    import spglib
    from pyxtal.symmetry import spglib_hall_numbers

    _validate_layout(ts)
    a, cs = ts.arrays, ts.cosets
    selected = np.flatnonzero(a["group_hall"][1:]) + 1
    _require(not require_all or len(selected) == C.N_GROUPS,
             f"Full verification requires all 230 groups; this build contains {len(selected)}")
    rng = np.random.default_rng(20261003)
    descriptors, sg_descriptors, _, _, _ = descriptor_tables()
    counts = {"groups": 0, "rows": 0, "images": 0, "cosets": 0, "wrap_certificates": 0}
    histogram = np.zeros(4, dtype=int)
    expected_start = 1
    for n, group in enumerate(groups_metadata(), 1):
        _require(a["group_row_start"][n] == expected_start and a["group_row_count"][n] == len(group),
                 f"Global row range mismatch in group {n}")
        expected_start += len(group)
    for n in selected:
        n = int(n)
        group = groups_metadata()[n - 1]
        start, count = int(a["group_row_start"][n]), int(a["group_row_count"][n])
        ids = np.arange(start, start + count)
        rows = _make_rows(group, start)
        k = int(a["group_ops_count"][n])
        R, t = a["group_ops_R"][n, :k].astype(float), a["group_ops_t"][n, :k]
        metric = a["group_MG"][n]
        _close(R, _ops(group[0].ops)[0], f"Operations R, group {n}")
        _close(t, _ops(group[0].ops)[1], f"Operations t, group {n}")
        codes = _op_codes(R, t)
        _require(len(set(codes)) == k, f"Duplicate group operations, group {n}")
        products_R = (R[:, None] @ R[None]).reshape(-1, 3, 3)
        products_t = (np.einsum("iab,jb->ija", R, t) + t[:, None]).reshape(-1, 3)
        _require(np.isin(_op_codes(products_R, products_t), codes).all(), f"Group closure {n}")
        _close(metric, point_metric(group), f"Point metric {n}")
        _close(R.transpose(0, 2, 1) @ metric @ R, metric, f"Metric invariance {n}")
        _require(np.linalg.eigvalsh(metric).min() > 0, f"Metric definiteness {n}")
        family = C.FAMILIES[int(a["group_family"][n])]
        _require(family == ("hexagonal" if group.lattice_type == "trigonal" else group.lattice_type),
                 f"Lattice family {n}")
        gram = np.asarray(_generic_gram(family))
        _close(R.transpose(0, 2, 1) @ gram @ R, gram, f"Lattice family invariance {n}")
        # Rank of the invariance constraints on the six-dimensional symmetric space.
        symbasis = []
        for i, j in zip(*Q_INDICES):
            mat = np.zeros((3, 3))
            mat[i, j] = mat[j, i] = 1
            symbasis.append(mat)
        constraints = np.array([(R.transpose(0, 2, 1) @ mat @ R - mat).ravel() for mat in symbasis]).T
        dim = 6 - np.linalg.matrix_rank(constraints, tol=1e-9)
        _require(dim == len(C.FREE_SLOTS[family]), f"Lattice invariant dimension {n}")
        polar = a["group_polar"][n]
        _close(np.unique(R, axis=0).mean(0), np.diag(polar.astype(float)), f"Polar completeness {n}")
        centring = C.CENTRINGS[int(a["group_centring"][n])]
        c = int(a["group_ncent"][n])
        _require(centring == group.symbol[0] and c == len(CENTRE_VECTORS[centring]), f"Centring {n}")
        cent = a["group_cent_vecs"][n, :c]
        _close(cent, CENTRE_VECTORS[centring], f"Centring vectors {n}")
        _require(_point_keys(t[np.all(R == np.eye(3), axis=(1, 2))]) == _point_keys(cent),
                 f"Translation subgroup {n}")
        Pc = a["group_Pc"][n]
        _close(np.linalg.det(Pc), 1 / c, f"Primitive volume {n}")
        _close(np.linalg.solve(Pc, cent.T), np.rint(np.linalg.solve(Pc, cent.T)), f"Primitive centring {n}")
        _close(np.linalg.inv(Pc), np.rint(np.linalg.inv(Pc)), f"Primitive includes Z^3, group {n}")
        _require(a["group_hall"][n] == group.hall_number, f"Frozen Hall number {n}")
        db = spglib.get_symmetry_from_database(group.hall_number)
        _require(set(_op_codes(db["rotations"], db["translations"])) == set(codes), f"Hall database {n}")
        q = a["group_q"][n]
        db = spglib.get_symmetry_from_database(int(spglib_hall_numbers[n - 1]))
        shifted = t + q - R @ q
        _require(set(_op_codes(db["rotations"], db["translations"])) == set(_op_codes(R, shifted)),
                 f"Origin shift to spglib default {n}")
        _close(a["group_desc"][n], sg_descriptors[n], f"Group descriptor {n}", tol=1e-7)
        for row in rows:
            r, d, m = row.id, row.d, row.m
            label = f"group {n}, row {row.wp.letter} ({r})"
            _require(a["row_group"][r] == n and a["row_letter"][r] == row.wp.letter, f"Row id {label}")
            _require((a["row_mult"][r], a["row_mprim"][r], a["row_dof"][r]) == (m, m // c, d),
                     f"Row multiplicity/dof {label}")
            A, b = a["row_A"][r, :, :d], a["row_b"][r]
            W, v = a["row_W"][r, :m, :, :d], a["row_v"][r, :m]
            _close(A, row.A, f"Chart A {label}")
            _close(b, row.b, f"Chart b {label}")
            _close(W, row.W, f"Members W {label}")
            _close(v, row.v, f"Members v {label}")
            _require(np.array_equal(a["row_member_mask"][r], np.arange(C.MAX_MULT) < m), f"Member mask {label}")
            _require(a["row_prim_mask"][r].sum() == m // c, f"Primitive member count {label}")
            _close(a["row_A"][r, :, d:], 0, f"A padding {label}")
            _close(a["row_W"][r, :, :, d:], 0, f"W column padding {label}")
            _close(a["row_W"][r, m:], 0, f"W member padding {label}")
            _close(a["row_v"][r, m:], 0, f"v member padding {label}")
            H, Hinv = a["row_H"][r], a["row_Hinv"][r]
            want_H = np.eye(3)
            want_H[:d, :d] = A.T @ metric @ A
            _close(H, want_H, f"Chart metric {label}")
            _close(H @ Hinv, np.eye(3), f"Chart inverse {label}")
            Pi, N = a["row_Pi"][r], a["row_N"][r]
            _close(Pi[d:], 0, f"Pi padding {label}")
            _close(N[d:], 0, f"N padding {label}")
            _close(Pi[:d], Hinv[:d, :d] @ A.T @ metric, f"Projection {label}")
            _close(N[:d] @ N[:d].T, Hinv[:d, :d], f"Noise covariance {label}")
            _close(a["row_Q"][r], (A @ Hinv[:d, :d] @ A.T)[Q_INDICES], f"Q {label}")
            _close(a["row_desc"][r], descriptors[r], f"Row descriptor {label}", tol=1e-7)
            _close((np.eye(3) - A @ Pi[:d])[:, polar], 0, f"Polar axes free in chart {label}")
            # Exact integer-minor criterion for {u: Au in Z^3} = Z^d.
            if d:
                gcd = 0
                for subset in itertools.combinations(range(3), d):
                    gcd = math.gcd(gcd, abs(round(np.linalg.det(A[list(subset)]))))
                _require(gcd == 1, f"Nontrivial chart periods {label}")
            for _ in range(3):
                u = rng.uniform(.05, .95, d)
                x = A @ u + b
                mem = W @ u + v
                orbit = R @ x + t
                mem_keys = _point_keys(mem)
                _require(len(mem_keys) == m and mem_keys == _point_keys(orbit), f"Members equal orbit {label}")
                _close(_wrapped(row.wp.get_position_from_free_xyzs(u) - x), 0, f"pyxtal chart {label}")
                _close(_wrapped(row.wp.apply_ops(x) - mem), 0, f"pyxtal members {label}")
                _close(row.E.T @ (x - b), u, f"Chart roundtrip {label}")
                stab = np.max(np.abs(_wrapped(orbit - x)), axis=1) < 1e-8
                _require(stab.sum() * m == k, f"Stabiliser order {label}")
                _close(A @ Pi[:d], R[stab].mean(0), f"Reynolds projector {label}")
                primitive = np.linalg.solve(Pc, mem.T).T
                first = []
                seen = set()
                for j, key in enumerate(np.rint(_mod1(primitive) * 1e8).astype(np.int64) % 100000000):
                    key = tuple(key)
                    if key not in seen:
                        first.append(j)
                    seen.add(key)
                _require(np.array_equal(np.flatnonzero(a["row_prim_mask"][r, :m]), first),
                         f"First primitive members {label}")
            ni = int(a["row_img_count"][r])
            S, s = a["row_img_S"][r, :ni, :d, :d], a["row_img_s"][r, :ni, :d]
            if d:
                _close(S, np.rint(S), f"Integral image S {label}")
                _close(np.abs(np.linalg.det(S)), 1, f"Unimodular images {label}")
                def image_key(mat, shift):
                    return tuple(np.rint(mat).astype(int).ravel()), tuple(np.rint(_mod1(shift) * 48).astype(int) % 48)
                keys = {image_key(mat, shift) for mat, shift in zip(S, s)}
                _require(len(keys) == ni, f"Duplicate chart images {label}")
                for mat, shift in zip(S, s):
                    _require(_point_keys(W @ (mat @ u + shift) + v) == mem_keys, f"Image orbit {label}")
                    for mat2, shift2 in zip(S, s):
                        _require(image_key(mat @ mat2, mat @ shift2 + shift) in keys, f"Image closure {label}")
                # Count all group operations preserving the affine chart independently.
                Qnormal = np.eye(3) - A @ np.linalg.pinv(A)
                linear_ok = np.max(np.abs(Qnormal @ R @ A), axis=(1, 2)) < TOL
                offsets = R @ b + t - b
                affine_ok = np.any(np.max(np.abs((offsets[:, None] + NV4) @ Qnormal.T), axis=2) < TOL, axis=1)
                _require(ni * stab.sum() == np.count_nonzero(linear_ok & affine_ok), f"Image completeness {label}")
                B = a["row_lambda_basis"][r, :d, :d]
                translations = s[np.max(np.abs(S - np.eye(d)), axis=(1, 2)) < TOL]
                coeff = np.linalg.solve(B, np.vstack([np.eye(d), translations]).T)
                _close(coeff, np.rint(coeff), f"Period generators in lattice {label}")
                # Translation subgroup already closes mod Z^d, hence this exact index.
                _close(abs(np.linalg.det(B)) * len(translations), 1, f"Period lattice covolume {label}")
                nv = int(a["row_vrv_count"][r])
                vectors = a["row_vrv"][r, :nv, :d]
                _verify_wrap_certificate(B, H[:d, :d], vectors)
                delta = np.zeros((12, 3))
                delta[:, :d] = rng.uniform(-5, 5, (12, d))
                wrapped = wrap_exact_np(delta, r, ts)[:, :d]
                coeff = np.linalg.solve(B, (delta[:, :d] - wrapped).T)
                _close(coeff, np.rint(coeff), f"Wrap lattice displacement {label}")
                slack = 2 * wrapped @ H[:d, :d] @ vectors.T - np.einsum("ni,ij,nj->n", vectors, H[:d, :d], vectors)
                _require(np.max(slack) < 1e-9, f"Wrap outside Voronoi cell {label}")
                fast = len(translations) == 1 and np.allclose(H[:d, :d], np.diag(np.diag(H[:d, :d])), atol=TOL)
                _require(bool(a["row_wrap_fast"][r]) == fast, f"Fast-wrap flag {label}")
                counts["wrap_certificates"] += 1
            else:
                _require(ni == 1 and a["row_vrv_count"][r] == 0, f"Fixed chart images {label}")
            counts["rows"] += 1
            counts["images"] += ni
            histogram[d] += 1
        ci, nc = int(cs.group_start[n]), int(cs.group_count[n])
        polar_axes = np.flatnonzero(polar).tolist()
        keys = set()
        for j in range(ci, ci + nc):
            P, p = cs.P[j], cs.p[j]
            _require(cs.group[j] == n, f"Coset group {n}")
            inv = np.linalg.inv(P)
            Rn = P @ R @ inv
            tn = t @ P.T + p - Rn @ p
            _require(set(_op_codes(Rn, tn)) == set(codes), f"Normaliser conjugacy {n}:{j}")
            det = round(np.linalg.det(P))
            _require(cs.det[j] == det and bool(cs.proper[j]) == (det > 0), f"Coset determinant {n}:{j}")
            _require(det > 0 or np.all(np.linalg.det(R) > 0), f"Proper representative preference {n}:{j}")
            reexpress = not np.allclose(inv.T @ gram @ inv, gram, atol=TOL, rtol=0)
            _require(bool(cs.reexpress_lattice[j]) == reexpress, f"Lattice re-expression {n}:{j}")
            keys.add(_coset_key(P, p, R, t, polar_axes))
            mi, mj = int(cs.map_start[j]), int(cs.map_start[j + 1])
            _require(np.array_equal(cs.source_row[mi:mj], ids) and set(cs.target_row[mi:mj]) == set(ids),
                     f"Coset row permutation {n}:{j}")
            for h in range(mi, mj):
                r, target = int(cs.source_row[h]), int(cs.target_row[h])
                u = rng.uniform(.05, .95, 3)
                u[int(a["row_dof"][r]):] = 0
                uu = cs.S[h] @ u + cs.s[h]
                m = int(a["row_mult"][r])
                x = a["row_W"][r, :m] @ u + a["row_v"][r, :m]
                y = a["row_W"][target, :m] @ uu + a["row_v"][target, :m]
                _require(_point_keys(x @ P.T + p) == _point_keys(y), f"Coset u-map {n}:{j}:{r}")
        _require(len(keys) == nc, f"Duplicate cosets {n}")
        for j in range(ci, ci + nc):
            for h in range(ci, ci + nc):
                key = _coset_key(cs.P[j] @ cs.P[h], cs.P[j] @ cs.p[h] + cs.p[j], R, t, polar_axes)
                _require(key in keys, f"Coset closure {n}")
        if n == 19:
            _require(nc == 96, "P212121 must have 96 completed cosets")
        _require(nc > 0 or n == 1, f"Missing normaliser cosets {n}")
        counts["cosets"] += nc
        counts["groups"] += 1
    if require_all:
        _require(histogram.tolist() == [616, 718, 167, 230], "Wyckoff dof histogram")
        for (symbol, name), pin in C.ROW_ID_PINS.items():
            g = int(np.flatnonzero(a["group_symbol"] == symbol)[0])
            letter = name.lstrip("0123456789")
            _require(ts.row_id(g, letter) == pin, f"Row id pin {symbol} {name}")
        _close(a["row_desc"][1:].mean(0, dtype=float), 0, "Global row descriptor means", tol=1e-6)
        _close(a["group_desc"][1:].mean(0, dtype=float), 0, "Global group descriptor means", tol=1e-6)
    return {"status": "passed", **counts, "rows_by_dof": histogram.tolist(), "sha256": ts.sha256}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build frozen tables (use Slurm for all groups)")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--groups", type=int, nargs="+")
    verify = commands.add_parser("verify", help="Verify all 230 groups (run through Slurm)")
    verify.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        ts = build_tables(args.groups)
        ts.save(args.out)
        print(json.dumps({"directory": str(args.out.resolve()), "sha256": ts.sha256,
                          "groups": int(np.count_nonzero(ts.arrays["group_hall"][1:]))}))
    else:
        print(json.dumps(verify_tables(load_tables(args.tables)), indent=2))


if __name__ == "__main__":
    main()
