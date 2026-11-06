"""The 62 raw space-group features, z-scored over all 230 groups.

No PCA, constant-column removal, or block-variance weighting is applied. The eight
subgroup Laplacian coordinates have their sign fixed by their largest component.
Row descriptors use fractional site operations, never Cartesian site-symmetry APIs.
"""
from functools import lru_cache
from collections import Counter
import math

import numpy as np
from pyxtal.symmetry import Group

from orbitol import contracts as C

N_SG = 230
SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]
LATTICE_DOF = {"triclinic": 6, "monoclinic": 4, "orthorhombic": 3, "tetragonal": 2, "trigonal": 2, "hexagonal": 2, "cubic": 1}
OP_TYPES = ["1", "-1", "2", "m", "3", "-3", "4", "-4", "6", "-6"]
SCREWS = ["2_1", "3_1|3_2", "4_1|4_3", "4_2", "6_1|6_5", "6_2|6_4", "6_3"]
GLIDES = ["axial", "diagonal_n", "diamond_d"]
SITE_ORDERS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]


def crystal_system(n):
    if n <= 2: return "triclinic"
    if n <= 15: return "monoclinic"
    if n <= 74: return "orthorhombic"
    if n <= 142: return "tetragonal"
    if n <= 167: return "trigonal"
    if n <= 194: return "hexagonal"
    return "cubic"


def op_type(R):
    det = int(round(np.linalg.det(R)))
    tr = int(round(np.trace(R)))
    if det == 1:
        return {3: "1", -1: "2", 0: "3", 1: "4", 2: "6"}[tr]
    return {-3: "-1", 1: "m", 0: "-3", -1: "-4", -2: "-6"}[tr]


def order_of(R):
    M = np.eye(3)
    for k in range(1, 7):
        M = M @ R
        if np.allclose(M, np.eye(3), atol=1e-8):
            return k
    raise ValueError("operation order > 6")


def frac_mod(v):
    w = np.mod(v, 1.0)
    w[np.isclose(w, 1.0, atol=1e-6)] = 0.0
    return w


def intrinsic(R, t):
    """Screw/glide part: average of R^j t over the cyclic group of R (origin independent)."""
    k = order_of(R)
    acc = np.zeros(3)
    M = np.eye(3)
    for _ in range(k):
        acc += M @ t
        M = R @ M
    return frac_mod(acc / k), k


HEX_BASIS = np.array([[1.0, -0.5, 0.0], [0.0, math.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]])


def screw_label(w, k, R, basis):
    """Screw type, plus handedness: sign of (rotation sense about v) x (translation along v).
    The sign is the same for R and R^-1 of one axis, so enantiomorphs (3_1 vs 3_2) differ."""
    if np.allclose(w, 0, atol=1e-6) or k == 1:
        return None, 0
    d = w / np.linalg.norm(w)
    nz = np.abs(d[np.abs(d) > 1e-6])
    v = np.round(d / nz.min())  # primitive lattice direction along the axis
    s = float(w @ v / (v @ v))
    p = int(round(s * k)) % k
    if p == 0:
        return None, 0
    hand = 0
    if 2 * p != k:
        Rc = basis @ R @ np.linalg.inv(basis)
        vc = basis @ v
        vc /= np.linalg.norm(vc)
        u = np.cross(vc, [1.0, 0.0, 0.0] if abs(vc[0]) < 0.9 else [0.0, 1.0, 0.0])
        sense = float(np.sign(vc @ np.cross(u, Rc @ u)))  # rotation sense about +v
        s_centered = s if s <= 0.5 else s - 1.0  # translation along +v in (-1/2, 1/2]
        hand = int(sense * np.sign(s_centered))
    label = {(2, 1): "2_1", (3, 1): "3_1|3_2", (3, 2): "3_1|3_2", (4, 1): "4_1|4_3", (4, 3): "4_1|4_3",
             (4, 2): "4_2", (6, 1): "6_1|6_5", (6, 5): "6_1|6_5", (6, 2): "6_2|6_4", (6, 4): "6_2|6_4",
             (6, 3): "6_3"}[(k, p)]
    return label, hand


def glide_label(w):
    if np.allclose(w, 0, atol=1e-6):
        return None
    comps = [c for c in w if c > 1e-6]
    if any(abs(c - 0.25) < 1e-6 or abs(c - 0.75) < 1e-6 for c in comps):
        return "diamond_d"
    return "diagonal_n" if len(comps) >= 2 else "axial"


def describe(n, group=None):
    g = Group(n) if group is None else group
    ops = g[0].ops  # general position, conventional cell, centring included
    nops = len(ops)
    system = crystal_system(n)
    feats = {}
    # lattice
    for s in SYSTEMS:
        feats[f"lattice:{s}"] = float(system == s)
    feats["lattice:free_params"] = LATTICE_DOF[system] / 6.0
    # centring
    c = g.symbol[0]
    cent = {"P": "P", "A": "ABC", "B": "ABC", "C": "ABC", "I": "I", "F": "F", "R": "R"}[c]
    for k in ["P", "ABC", "I", "F", "R"]:
        feats[f"centring:{k}"] = float(cent == k)
    # point group and translations
    types = Counter()
    screws, glides, hand = Counter(), Counter(), 0
    rots = []
    basis = HEX_BASIS if system in ("trigonal", "hexagonal") else np.eye(3)
    for op in ops:
        R = np.array(op.rotation_matrix, dtype=float)
        t = np.array(op.translation_vector, dtype=float)
        ty = op_type(R)
        types[ty] += 1
        rots.append(R)
        if ty in ("2", "3", "4", "6"):
            w, k = intrinsic(R, t)
            lab, h = screw_label(w, k, R, basis)
            if lab:
                screws[lab] += 1
                hand += h
        elif ty == "m":
            w, _ = intrinsic(R, t)
            lab = glide_label(w)
            if lab:
                glides[lab] += 1
    for ty in OP_TYPES:
        feats[f"pointgroup:{ty}"] = types[ty] / nops
    n_centring = sum(1 for R in rots if np.allclose(R, np.eye(3)))
    pg_order = nops // n_centring
    feats["pointgroup:log_order"] = math.log2(pg_order) / math.log2(48)
    for lab in SCREWS:
        feats[f"translate:screw_{lab}"] = screws[lab] / nops
    for lab in GLIDES:
        feats[f"translate:glide_{lab}"] = glides[lab] / nops
    mults = [wp.multiplicity for wp in g]
    symmorphic = min(mults) == n_centring
    feats["translate:symmorphic"] = float(symmorphic)
    feats["translate:handedness"] = hand / nops
    # flags
    feats["flags:centrosymmetric"] = float(types["-1"] > 0)
    feats["flags:sohncke"] = float(all(int(round(np.linalg.det(R))) == 1 for R in rots))
    # common fixed subspace of all rotation parts = polar (free-origin) directions
    A = np.vstack([R - np.eye(3) for R in rots])
    polar_dim = 3 - np.linalg.matrix_rank(A, tol=1e-8)
    feats["flags:polar_dim"] = polar_dim / 3.0
    # Wyckoff spectrum
    dofs = [wp.get_dof() for wp in g]
    feats["wyckoff:n_positions"] = len(mults) / 27.0
    for d in range(4):
        feats[f"wyckoff:dof_{d}"] = sum(1 for x in dofs if x == d) / len(dofs)
    site_orders = [nops // m for m in mults]
    for so in SITE_ORDERS:
        feats[f"wyckoff:site_order_{so}"] = sum(1 for x in site_orders if x == so) / len(site_orders)
    meta = {"number": n, "symbol": g.symbol, "system": system, "centring": cent, "n_ops_conv": nops,
            "point_group_order": pg_order, "symmorphic": symmorphic, "polar_dim": int(polar_dim),
            "max_t_subgroups": sorted(set(int(x) for x in g.get_max_t_subgroup()["subgroup"])),
            "max_k_subgroups": sorted(set(int(x) for x in g.get_max_k_subgroup()["subgroup"]))}
    return feats, meta


def spectral_coords(metas, k=8):
    W = np.zeros((N_SG, N_SG))
    for m in metas:
        i = m["number"] - 1
        for s in m["max_t_subgroups"] + m["max_k_subgroups"]:
            j = s - 1
            if i != j:
                W[i, j] = W[j, i] = 1.0
    deg = W.sum(1)
    Dm = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(N_SG) - Dm @ W @ Dm
    vals, vecs = np.linalg.eigh(L)
    coords = vecs[:, 1:k + 1].copy()
    for col in coords.T:
        if col[np.argmax(np.abs(col))] < 0:
            col *= -1
    return coords, vals[1:k + 1], int((W > 0).sum() // 2)



ROW_OP_TYPES = ("1", "2", "3", "4", "6", "-1", "m", "-3", "-4", "-6")
Q_INDICES = np.triu_indices(3)


@lru_cache(maxsize=1)
def groups_metadata():
    """Cheap pyxtal metadata; no images, lattices or cosets are built."""
    return tuple(Group(n) for n in range(1, C.G_TOTAL))


def point_metric(group):
    rotations = np.array([op.rotation_matrix for op in group[0].ops])
    rotations = np.unique(rotations, axis=0)
    return np.mean(rotations.transpose(0, 2, 1) @ rotations, axis=0)


def raw_row_descriptor(wp, metric):
    free = [i for i in range(3) if i not in wp.get_frozen_axis()]
    A = wp.ops[0].rotation_matrix[:, free]
    d = len(free)
    Q = A @ np.linalg.solve(A.T @ metric @ A, A.T) if d else np.zeros((3, 3))
    # symmetry[0] is the fractional stabiliser (verified against group operations).
    operations = wp.symmetry[0]
    counts = Counter(op_type(op.rotation_matrix) for op in operations)
    return np.array([math.log(wp.multiplicity), d, 1 / len(operations),
                     *(counts[t] for t in ROW_OP_TYPES), *Q[Q_INDICES]], dtype=float)


def _zscore(raw):
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    scale = np.where(std > 1e-12, std, 1.0)
    return ((raw - mean) / scale).astype(np.float32), mean, scale


@lru_cache(maxsize=1)
def descriptor_tables():
    """Global statistics, independent of which groups a table build requests.

    Returns row descriptors (1732,19), group descriptors (231,62), feature names,
    row mean and row scale. Padding is zero. This computes only descriptors and
    metadata, not a full symmetry table. Cached arrays are read-only.
    """
    row_raw, sg_raw, metas = [], [], []
    names = None
    for n, group in enumerate(groups_metadata(), 1):
        feats, meta = describe(n, group)
        names = tuple(feats)
        sg_raw.append(list(feats.values()))
        metas.append(meta)
        metric = point_metric(group)
        row_raw.extend(raw_row_descriptor(wp, metric) for wp in reversed(list(group)))
    spectral, _, _ = spectral_coords(metas)
    names += tuple(f"subgroups:eig_{i + 1}" for i in range(spectral.shape[1]))
    sg_raw = np.column_stack([sg_raw, spectral])
    if sg_raw.shape != (C.N_GROUPS, C.SG_DESC_DIM):
        raise ValueError(f"Unexpected space-group descriptor shape: {sg_raw.shape}")
    if np.shape(row_raw) != (C.N_ROWS, C.ROW_DESC_DIM):
        raise ValueError(f"Unexpected Wyckoff descriptor shape: {np.shape(row_raw)}")
    rows, mean, scale = _zscore(np.array(row_raw))
    groups, _, _ = _zscore(sg_raw)
    rows = np.vstack([np.zeros((1, C.ROW_DESC_DIM), np.float32), rows])
    groups = np.vstack([np.zeros((1, C.SG_DESC_DIM), np.float32), groups])
    for a in (rows, groups, mean, scale):
        a.flags.writeable = False
    return rows, groups, names, mean, scale
