"""Frozen acceptance tests for GEM v1 geometry.

References are independent of the implementation under test:
- members and the primitive cell: orbitol.decode (expand_conventional + emit_primitive);
- distances: pymatgen's LLL basis, then an exhaustive search over n in {-3..3}^3.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice

import orbitol.core  # noqa: F401
from orbitol.contracts import CENTRINGS, ORBIT_CACHE_DIR, TABLES_DIR
from orbitol.data import OrbitDataset, collate_orbits
from orbitol.decode import emit_primitive, expand_conventional
from orbitol.gem_geometry import (
    PrimitiveMembers, cell_scale, expand_members, lll_reduce, pair_distances, primitive_lattice,
    sanitize_cell,
)
from orbitol.tables import load_tables

N_MAX = 20


@pytest.fixture(scope="module")
def tables():
    if not Path(TABLES_DIR, "tables.npz").is_file():
        pytest.skip(f"Frozen tables unavailable: {TABLES_DIR}")
    return load_tables(TABLES_DIR)


@pytest.fixture(scope="module")
def batch(tables):
    path = Path(ORBIT_CACHE_DIR, "train.npz")
    if not path.is_file():
        pytest.skip(f"Orbit cache unavailable: {path}")
    data = OrbitDataset(path, tables, augment=False, aug_seed=0)
    groups, orbits = np.asarray(data.cache["G"]), np.asarray(data.cache["n_orbits"])
    centring = tables.arrays["group_centring"][groups]
    chosen = []
    for c in range(len(CENTRINGS)):
        chosen += list(np.flatnonzero(centring == c)[:12])
    chosen += list(np.flatnonzero(orbits >= 8)[:12])
    chosen = sorted(set(int(i) for i in chosen))
    assert len(chosen) >= 60
    return collate_orbits([data[i] for i in chosen], K=20)


def _decoded(tables, batch, i, u_c):
    k = int((~batch["pad_mask"][i]).sum())
    conv = expand_conventional(int(batch["G"][i]), batch["row"][i, :k].numpy(), batch["A0"][i, :k].numpy(),
                               u_c[i, :k].numpy(), batch["lat"][i].double().numpy(), tables)
    assert conv.failure is None
    return k, conv, emit_primitive(conv, int(batch["G"][i]), tables)


def _wrap(delta):
    return (delta + 0.5) % 1.0 - 0.5


def _reference_distances(x, n, L):
    """pymatgen's LLL basis, then an exhaustive search over 7^3 offsets (independent of the code under test)."""
    lattice = Lattice(L)
    lll = lattice.lll_matrix
    to_lll = np.linalg.inv(lattice.lll_mapping)          # fractional coordinates in the LLL basis: f @ inv(mapping)
    f = x[:n] @ to_lll
    offsets = np.array(np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij")).reshape(3, -1).T
    delta = _wrap(f[None, :, :] - f[:, None, :])          # (n, n, 3)
    cart = (delta[:, :, None, :] + offsets[None, None]) @ lll
    return np.sqrt((cart ** 2).sum(-1)).min(-1)


def test_members_and_cell_match_the_decoder(tables, batch):
    tb = tables.torch("cpu", dtype=torch.float64)
    u_c = batch["U"].double() - 0.5
    members = expand_members(u_c, batch["row"], batch["G"], batch["pad_mask"], tb, n_max=N_MAX)
    assert isinstance(members, PrimitiveMembers)
    L = primitive_lattice(batch["lat"].double(), batch["G"], tb)
    for i in range(len(batch["G"])):
        k, conv, structure = _decoded(tables, batch, i, u_c)
        n = len(structure)
        # Valid members are packed first, in emit_primitive's order.
        assert members.valid[i, :n].all() and not members.valid[i, n:].any()
        assert np.abs(_wrap(members.x[i, :n].numpy() - structure.frac_coords)).max() < 1e-9
        assert ((members.x[i, :n] >= 0) & (members.x[i, :n] < 1)).all()
        expected_owner = conv.orbit_index[conv.prim_mask]
        np.testing.assert_array_equal(members.owner[i, :n].numpy(), expected_owner)
        assert (members.owner[i, n:] == -1).all()
        np.testing.assert_array_equal(members.count[i, :k].numpy(),
                                      tables.arrays["row_mprim"][batch["row"][i, :k].numpy()])
        assert (members.count[i, k:] == 0).all() and (members.rep[i, k:] == -1).all()
        for t in range(k):
            assert int(members.rep[i, t]) == int(np.flatnonzero(expected_owner == t)[0])
        np.testing.assert_allclose(L[i].numpy(), structure.lattice.matrix, rtol=1e-9, atol=1e-9)


def test_float32_path_is_close(tables, batch):
    u_c = batch["U"].double() - 0.5
    m64 = expand_members(u_c, batch["row"], batch["G"], batch["pad_mask"], tables.torch("cpu", torch.float64), n_max=N_MAX)
    m32 = expand_members(u_c.float(), batch["row"], batch["G"], batch["pad_mask"], tables.torch("cpu", torch.float32), n_max=N_MAX)
    assert m32.x.dtype == torch.float32
    assert torch.equal(m32.valid, m64.valid) and torch.equal(m32.owner, m64.owner)
    diff = _wrap((m32.x.double() - m64.x).numpy())[m64.valid.numpy()]
    assert np.abs(diff).max() < 1e-5


def _lll_conditions(B, delta=0.75, eps=1e-6):
    """Rows of B are basis vectors. Check size reduction and the Lovasz condition."""
    B = np.asarray(B, dtype=np.float64)
    Q = np.zeros_like(B)
    mu = np.zeros((3, 3))
    for i in range(3):
        Q[i] = B[i]
        for j in range(i):
            mu[i, j] = B[i] @ Q[j] / (Q[j] @ Q[j])
            Q[i] = Q[i] - mu[i, j] * Q[j]
    size = all(abs(mu[i, j]) <= 0.5 + eps for i in range(3) for j in range(i))
    lovasz = all(delta * (Q[i - 1] @ Q[i - 1]) <= Q[i] @ Q[i] + mu[i, i - 1] ** 2 * (Q[i - 1] @ Q[i - 1]) + eps
                 for i in range(1, 3))
    return size and lovasz


def test_lll_reduce_is_a_unimodular_lll_basis(tables, batch):
    L = primitive_lattice(batch["lat"].double(), batch["G"], tables.torch("cpu", torch.float64))
    rng = np.random.default_rng(0)
    skew = []
    for _ in range(64):  # strongly skewed bases of ordinary cells
        U = np.eye(3)
        for _ in range(6):
            i, j = rng.choice(3, 2, replace=False)
            U[i] += rng.integers(-3, 4) * U[j]
        skew.append(U @ Lattice.from_parameters(*rng.uniform(3, 12, 3), *rng.uniform(70, 110, 3)).matrix)
    for cells in (L, torch.as_tensor(np.stack(skew))):
        L_red, M = lll_reduce(cells)
        assert torch.allclose(L_red, M.to(cells.dtype) @ cells, rtol=1e-9, atol=1e-9)
        assert torch.equal(M.double(), M.double().round())
        assert torch.allclose(torch.linalg.det(M.double()).abs(), torch.ones(len(cells), dtype=torch.float64))
        for basis in L_red.numpy():
            assert _lll_conditions(basis)


def test_distances_are_exact_on_clean_cells(tables, batch):
    for dtype, tolerance in ((torch.float64, 1e-6), (torch.float32, 2e-4)):
        tb = tables.torch("cpu", dtype=dtype)
        u_c = (batch["U"].double() - 0.5).to(dtype)
        members = expand_members(u_c, batch["row"], batch["G"], batch["pad_mask"], tb, n_max=N_MAX)
        L = primitive_lattice(batch["lat"].to(dtype), batch["G"], tb)
        d = pair_distances(members.x, members.valid, L)
        assert d.shape == (len(L), N_MAX, N_MAX) and torch.isfinite(d).all()
        for i in range(len(L)):
            n = int(members.valid[i].sum())
            ref = _reference_distances(members.x[i].double().numpy(), n, L[i].double().numpy())
            assert np.abs(d[i, :n, :n].double().numpy() - ref).max() < tolerance
            assert (d[i, n:] == 0).all() and (d[i, :, n:] == 0).all()
            assert (torch.diagonal(d[i])[:n] == 0).all()


@pytest.mark.parametrize("dtype,threshold", [(torch.float64, 1e-5), (torch.float32, 1e-3)])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.5, 1.0])
def test_noisy_cells_miss_rate(tables, batch, sigma, dtype, threshold):
    # float32 is the model's path; a wrong image differs by far more than its rounding.
    tb = tables.torch("cpu", dtype=dtype)
    generator = torch.Generator().manual_seed(int(sigma * 1000))
    u_c = (batch["U"].double() - 0.5 + sigma * 0.1 * torch.randn(batch["U"].shape, generator=generator, dtype=torch.float64)).to(dtype)
    z = (batch["lat"].double() + sigma * torch.randn(batch["lat"].shape, generator=generator, dtype=torch.float64)).to(dtype)
    members = expand_members(u_c, batch["row"], batch["G"], batch["pad_mask"], tb, n_max=N_MAX)
    L = primitive_lattice(z, batch["G"], tb)
    d = pair_distances(members.x, members.valid, L)
    assert torch.isfinite(d).all()
    misses = total = 0
    for i in range(len(L)):
        n = int(members.valid[i].sum())
        ref = _reference_distances(members.x[i].double().numpy(), n, L[i].double().numpy())
        misses += int((np.abs(d[i, :n, :n].double().numpy() - ref) > threshold).sum())
        total += n * n
    print(f"sigma={sigma} {dtype}: {misses}/{total} pairs differ from the exhaustive search")
    assert misses <= 1e-3 * total


def test_degenerate_cells_stay_finite(tables, batch):
    tb = tables.torch("cpu", dtype=torch.float32)
    u_c = batch["U"].float() - 0.5
    members = expand_members(u_c, batch["row"], batch["G"], batch["pad_mask"], tb, n_max=N_MAX)
    z = batch["lat"].clone()
    z[0::4, 0] = 40.0           # enormous cell
    z[1::4, 0] = -40.0          # vanishing cell
    z[2::4, 1] = 1e4            # extreme shear (triclinic slot)
    z[2::4, 3] = 1e4            # extreme shear (a slot most families keep free)
    z[3::4] = float("nan")      # non-finite chart
    L = primitive_lattice(z, batch["G"], tb)
    d = pair_distances(members.x, members.valid, L)
    assert torch.isfinite(d).all()
    L_red, M = lll_reduce(torch.nan_to_num(L))
    assert torch.isfinite(L_red).all() and torch.isfinite(M).all()


def test_batch_independence_and_overflow(tables, batch):
    tb = tables.torch("cpu", dtype=torch.float64)
    u_c = batch["U"].double() - 0.5
    full = expand_members(u_c, batch["row"], batch["G"], batch["pad_mask"], tb, n_max=N_MAX)
    L = primitive_lattice(batch["lat"].double(), batch["G"], tb)
    d = pair_distances(full.x, full.valid, L)
    for i in (0, 7, len(L) - 1):
        one = expand_members(u_c[i:i + 1], batch["row"][i:i + 1], batch["G"][i:i + 1],
                             batch["pad_mask"][i:i + 1], tb, n_max=N_MAX)
        assert torch.equal(one.owner[0], full.owner[i]) and torch.equal(one.valid[0], full.valid[i])
        assert torch.allclose(one.x[0], full.x[i], rtol=0, atol=1e-12)
        assert torch.allclose(pair_distances(one.x, one.valid, L[i:i + 1])[0], d[i], rtol=0, atol=1e-9)
    largest = int(batch["n_prim"].max())
    with pytest.raises(ValueError):
        expand_members(u_c, batch["row"], batch["G"], batch["pad_mask"], tb, n_max=largest - 1)


def test_cell_scale_is_the_sanitised_volume(tables, batch):
    tb = tables.torch("cpu", dtype=torch.float64)
    L = primitive_lattice(batch["lat"].double(), batch["G"], tb)
    scale = cell_scale(L)
    torch.testing.assert_close(scale, torch.linalg.det(L).abs().pow(1 / 3), rtol=1e-12, atol=0)
    # Basis-independent: a unimodular change of basis leaves it unchanged.
    U = torch.tensor([[1., 2., 0.], [0., 1., -1.], [1., 3., 0.]], dtype=torch.float64)
    assert abs(float(torch.linalg.det(U))) == 1.0
    torch.testing.assert_close(cell_scale(U @ L), scale, rtol=1e-10, atol=0)
    # Degenerate and non-finite cells give a finite, positive scale (no 0/0 in d / scale).
    z = batch["lat"].double().clone()
    z[0::3, 0], z[1::3, 0], z[2::3] = 40.0, -40.0, float("nan")
    bad = cell_scale(primitive_lattice(z, batch["G"], tb))
    assert torch.isfinite(bad).all() and (bad > 0).all()
    assert torch.equal(cell_scale(L), torch.linalg.det(sanitize_cell(L)).abs().pow(1 / 3))
