"""Small CPU checks of GEM geometry, independent of model integration."""
from pathlib import Path
import socket
from time import perf_counter

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice
from pymatgen.util.coord import pbc_shortest_vectors

import orbitol.core  # noqa: F401
from orbitol.contracts import ORBIT_CACHE_DIR, TABLES_DIR
from orbitol.data import OrbitDataset, collate_orbits
from orbitol.decode import emit_primitive, expand_conventional
from orbitol.gem_geometry import cell_scale, expand_members, lll_reduce, pair_distances, primitive_lattice
from orbitol.tables import load_tables


@pytest.fixture(scope="module")
def tables():
    if not Path(TABLES_DIR, "tables.npz").is_file():
        pytest.skip(f"Frozen tables unavailable: {TABLES_DIR}")
    return load_tables(TABLES_DIR)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_members_cells_and_distances(tables, dtype):
    # P1, C-monoclinic, R in hexagonal axes, hexagonal, F and I cubic.
    device = torch.device("cpu")  # Never implicitly select a GPU on the login node.
    G = torch.tensor([1, 5, 166, 194, 225, 229], device=device)
    selected = []
    for g in G.tolist():
        a = tables.arrays
        candidates = np.flatnonzero((a["row_group"] == g) & (a["row_dof"] > 0) & (a["row_mprim"] <= 8))
        selected.append(int(candidates[-1]))
    row = torch.tensor([[r, 0, r] for r in selected], device=device)
    pad = row == 0
    u = torch.tensor([[[-0.17, 0.08, 0.21], [0, 0, 0], [0.11, -0.24, -0.09]]], dtype=dtype, device=device).repeat(len(G), 1, 1)
    z = torch.tensor([[1.7, 0.4, 1.5, -4.0, 0.3, 1.1]], dtype=dtype, device=device).repeat(len(G), 1)
    tb = tables.torch(device, dtype)
    members = expand_members(u, row, G, pad, tb, n_max=20)
    cells = primitive_lattice(z, G, tb)
    distances = pair_distances(members.x, members.valid, cells)
    assert cells.dtype == members.x.dtype == distances.dtype == dtype
    assert members.x.device == cells.device == distances.device == device
    assert members.owner.dtype == members.count.dtype == members.rep.dtype == torch.int64
    assert members.valid.dtype == torch.bool
    for i, g in enumerate(G.tolist()):
        conv = expand_conventional(g, row[i, [0, 2]], np.array([6, 8]), u[i, [0, 2]], z[i], tables)
        assert conv.failure is None
        structure = emit_primitive(conv, g, tables)
        n = len(structure)
        wrapped_error = (members.x[i, :n].numpy() - structure.frac_coords + 0.5) % 1.0 - 0.5
        np.testing.assert_allclose(wrapped_error, 0, atol=2e-6 if dtype == torch.float32 else 1e-12)
        np.testing.assert_allclose(cells[i].numpy(), structure.lattice.matrix, rtol=2e-6, atol=1e-12)
        expected_owner = conv.orbit_index[conv.prim_mask] * 2  # middle token is padded
        np.testing.assert_array_equal(members.owner[i, :n].numpy(), expected_owner)
        assert members.rep[i].tolist() == [0, -1, n // 2]
        assert members.count[i].tolist() == [n // 2, 0, n // 2]
        assert members.valid[i, :n].all() and not members.valid[i, n:].any()
        assert (members.owner[i, n:] == -1).all()
        assert ((members.x[i] >= 0) & (members.x[i] < 1)).all()
        _, squared = pbc_shortest_vectors(Lattice(cells[i].numpy()), members.x[i, :n].numpy(), members.x[i, :n].numpy(), return_d2=True)
        np.testing.assert_allclose(distances[i, :n, :n], np.sqrt(squared), atol=2e-5)
        assert not distances[i, n:].any() and not distances[i, :, n:].any()
    with pytest.raises(ValueError, match="exceeds"):
        expand_members(u, row, G, pad, tb, n_max=1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lll_identity_shear_and_fallback(dtype):
    identity = torch.eye(3, dtype=dtype)
    shear = torch.tensor([[1, 0, 0], [8, 1, 0], [-5, 3, 1]], dtype=dtype)
    singular = identity.clone()
    singular[2] = singular[1]
    cells = torch.stack([identity, shear, singular, identity * float("nan"), identity * 1e30])
    reduced, M = lll_reduce(cells)
    assert torch.isfinite(reduced).all() and torch.isfinite(M).all()
    assert torch.equal(reduced[0], identity) and torch.equal(M[0], identity)
    assert torch.equal(M[2], identity) and torch.equal(reduced[2], singular)
    assert torch.equal(M[3], identity) and torch.equal(reduced[3], identity)
    # Any orthonormal basis of this sheared cubic lattice has the same intent;
    # signs and row order are choices of the reduction algorithm.
    torch.testing.assert_close(reduced[1] @ reduced[1].T, identity, rtol=0, atol=0)
    torch.testing.assert_close(reduced[[0, 1, 2, 4]], M[[0, 1, 2, 4]] @ cells[[0, 1, 2, 4]])
    torch.testing.assert_close(torch.linalg.det(M).abs(), torch.ones(len(cells), dtype=dtype))
    generator = torch.Generator().manual_seed(18)
    rows = torch.randint(-8, 9, (128, 2, 3), generator=generator).to(dtype)
    dependent = torch.cat([rows, rows[:, :1] + rows[:, 1:2]], 1)
    reduced, M = lll_reduce(dependent)
    torch.testing.assert_close(M, identity.expand_as(M), rtol=0, atol=0)
    torch.testing.assert_close(reduced, dependent, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reduction_sweeps_are_deterministic_and_batch_independent(dtype):
    # Include cells needing different sweep counts. An independent QR checks
    # size reduction and Lovasz on the returned basis, not its internal state.
    generator = torch.Generator().manual_seed(77)
    transforms = torch.eye(3, dtype=torch.float64).repeat(128, 1, 1)
    for i, j in [(0, 1), (1, 2), (2, 0), (0, 2)]:
        transforms[:, i] += torch.randint(-3, 4, (128, 1), generator=generator) * transforms[:, j]
    base = torch.randn(128, 3, 3, generator=generator, dtype=torch.float64) + 3 * torch.eye(3)
    cells = (transforms @ base).to(dtype)
    reduced, M = lll_reduce(cells)
    repeated, repeated_M = lll_reduce(cells)
    assert torch.equal(repeated, reduced) and torch.equal(repeated_M, M)
    torch.testing.assert_close(reduced, M @ cells, rtol=0, atol=0)
    assert torch.equal(M, M.round())
    torch.testing.assert_close(torch.linalg.det(M.double()).abs(), torch.ones(len(M), dtype=torch.float64))
    for i in (0, 1, 7, 127):
        single, single_M = lll_reduce(cells[i:i + 1])
        assert torch.equal(single[0], reduced[i]) and torch.equal(single_M[0], M[i])
    R = np.linalg.qr(reduced.double().numpy().transpose(0, 2, 1))[1]
    norms = np.diagonal(R, axis1=1, axis2=2) ** 2
    mu10, mu20, mu21 = R[:, 0, 1] / R[:, 0, 0], R[:, 0, 2] / R[:, 0, 0], R[:, 1, 2] / R[:, 1, 1]
    assert np.max(np.abs([mu10, mu20, mu21])) <= 0.5 + 1e-6
    assert np.all(norms[:, 1] + mu10 ** 2 * norms[:, 0] + 1e-6 >= 0.75 * norms[:, 0])
    assert np.all(norms[:, 2] + mu21 ** 2 * norms[:, 1] + 1e-6 >= 0.75 * norms[:, 1])


def test_distance_periodicity_basis_invariance_and_gradients():
    x = torch.tensor([[[0.1, 0.2, 0.3], [0.73, 0.81, 0.59], [0.1, 0.2, 0.3]]], dtype=torch.float64, requires_grad=True)
    cell = torch.tensor([[[3.0, 0, 0], [0.2, 4.0, 0], [0.3, -0.1, 5.0]]], dtype=x.dtype, requires_grad=True)
    valid = torch.ones(x.shape[:2], dtype=torch.bool)
    d = pair_distances(x, valid, cell)
    M = x.new_tensor([[[1, 0, 0], [4, 1, 0], [-3, 2, 1]]])
    transformed = pair_distances(x @ torch.linalg.inv(M), valid, M @ cell)
    torch.testing.assert_close(d, transformed)
    torch.testing.assert_close(d, pair_distances(x + x.new_tensor([2, -3, 5]), valid, cell))
    torch.testing.assert_close(d, d.transpose(-1, -2))
    d.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(cell.grad).all()
    assert x.grad.abs().sum() > 0 and cell.grad.abs().sum() > 0


def test_sanitization_and_empty_padding(tables):
    tb = tables.torch("cpu")
    u = torch.zeros(2, 3, 3)
    row = torch.zeros(2, 3, dtype=torch.int64)
    G = torch.ones(2, dtype=torch.int64)
    members = expand_members(u, row, G, row == 0, tb, n_max=20)
    assert not members.valid.any() and not members.count.any()
    assert (members.owner == -1).all() and (members.rep == -1).all()
    assert not pair_distances(members.x, members.valid, torch.full((2, 3, 3), float("nan"))).any()
    empty = expand_members(u[:, :0], row[:, :0], G, (row == 0)[:, :0], tb, n_max=0)
    assert pair_distances(empty.x, empty.valid, torch.eye(3).expand(2, -1, -1)).shape == (2, 0, 0)
    cells = torch.stack([torch.zeros(3, 3), torch.full((3, 3), float("inf")), torch.eye(3) * 1e-30, torch.eye(3) * 1e30])
    x = torch.tensor([[[0, 0, 0], [0.25, 0, 0]]]).expand(4, -1, -1)
    d = pair_distances(x, torch.ones(4, 2, dtype=torch.bool), cells)
    assert torch.isfinite(d).all()
    torch.testing.assert_close(d[:, 0, 1], torch.tensor([0.25, 0.25, np.exp(-5) / 4, np.exp(5) / 4], dtype=x.dtype))


def test_cpu_batch_timing(tables):
    """Small forward-only timing; includes no table/cache I/O."""
    path = Path(ORBIT_CACHE_DIR, "train.npz")
    if not path.is_file():
        pytest.skip(f"Orbit cache unavailable: {path}")
    data = OrbitDataset(path, tables, augment=False, aug_seed=0)
    batch = collate_orbits([data[i] for i in range(128)], K=20)
    tb = tables.torch("cpu", torch.float32)
    u = batch["U"] - 0.5
    generator = torch.Generator().manual_seed(1)
    noise = torch.randn(batch["lat"].shape, generator=generator)
    cells = [primitive_lattice(batch["lat"] + sigma * noise, batch["G"], tb) for sigma in (0, 1, 3)]
    z = batch["lat"] + torch.randn(batch["lat"].shape, generator=torch.Generator().manual_seed(0))

    def geometry():
        members = expand_members(u, batch["row"], batch["G"], batch["pad_mask"], tb, n_max=20)
        lattice = primitive_lattice(z, batch["G"], tb)
        return pair_distances(members.x, members.valid, lattice), cell_scale(lattice)

    workloads = [(f"lll_reduce sigma={sigma}", lambda L=L: lll_reduce(L)) for sigma, L in zip((0, 1, 3), cells)]
    workloads.append(("full geometry sigma=1", geometry))
    with torch.no_grad():
        for name, function in workloads:
            elapsed = []
            for iteration in range(12):  # two warmups, ten timed calls
                start = perf_counter()
                result = function()
                duration = perf_counter() - start
                if iteration >= 2:
                    elapsed.append(duration * 1000)
            assert all(torch.isfinite(value).all() for value in result)
            print(f"{name}: B=128 K=20 N=20 CPU float32 node={socket.gethostname()} threads={torch.get_num_threads()}, "
                  f"median={np.median(elapsed):.3f} ms, min={min(elapsed):.3f} ms, max={max(elapsed):.3f} ms (10 calls)")
    assert result[0].shape == (128, 20, 20)
