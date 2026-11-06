"""Frozen dispatch-cost budget for GEM v1 geometry (the eager LLL was launch-bound).

Eager GPU cost is dominated by the number of dispatched aten ops on tiny B x 3 x 3 tensors, so the budget is
an op count measured on CPU (deterministic), at the training batch size. At commit f5c1fbe, lll_reduce with early
exit used 4,103 aten ops and the whole geometry about 4,700.
"""
from pathlib import Path

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import orbitol.core  # noqa: F401
from orbitol.contracts import ORBIT_CACHE_DIR, TABLES_DIR
from orbitol.data import OrbitDataset, collate_orbits
from orbitol.gem_geometry import cell_scale, expand_members, lll_reduce, pair_distances, primitive_lattice
from orbitol.tables import load_tables

LLL_BUDGET = 800          # aten ops for lll_reduce on 128 noisy cells
GEOMETRY_BUDGET = 1500    # aten ops for expand_members + pair_distances + cell_scale


def _aten_ops(prof):
    return sum(event.count for event in prof.key_averages() if event.key.startswith("aten::"))


@pytest.fixture(scope="module")
def setup():
    if not Path(TABLES_DIR, "tables.npz").is_file() or not Path(ORBIT_CACHE_DIR, "train.npz").is_file():
        pytest.skip("tables or orbit cache unavailable")
    tables = load_tables(TABLES_DIR)
    data = OrbitDataset(Path(ORBIT_CACHE_DIR, "train.npz"), tables, augment=False, aug_seed=0)
    batch = collate_orbits([data[i] for i in range(128)], K=20)
    tb = tables.torch("cpu", dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    z = batch["lat"] + 1.0 * torch.randn(batch["lat"].shape, generator=generator)    # sigma = 1, as in training
    return batch, tb, z


@pytest.mark.parametrize("sigma_scale", [0.0, 1.0, 3.0])
def test_lll_dispatch_budget(setup, sigma_scale):
    batch, tb, _ = setup
    generator = torch.Generator().manual_seed(1)
    z = batch["lat"] + sigma_scale * torch.randn(batch["lat"].shape, generator=generator)
    L = primitive_lattice(z, batch["G"], tb)
    lll_reduce(L)                                           # warm-up
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
        lll_reduce(L)
    count = _aten_ops(prof)
    print(f"sigma={sigma_scale}: lll_reduce dispatched {count} aten ops (budget {LLL_BUDGET})")
    assert count <= LLL_BUDGET


def test_geometry_dispatch_budget(setup):
    batch, tb, z = setup
    u_c = batch["U"] - 0.5
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
        members = expand_members(u_c, batch["row"], batch["G"], batch["pad_mask"], tb, n_max=20)
        L = primitive_lattice(z, batch["G"], tb)
        d = pair_distances(members.x, members.valid, L)
        scale = cell_scale(L)
    count = _aten_ops(prof)
    print(f"geometry dispatched {count} aten ops (budget {GEOMETRY_BUDGET})")
    assert torch.isfinite(d).all() and torch.isfinite(scale).all()
    assert count <= GEOMETRY_BUDGET
