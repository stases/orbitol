"""Padding gates for noise draws, loader streams, target groups and resume."""
from dataclasses import replace
from functools import partial
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from orbitol import contracts as C, edm, train
from orbitol.data import ResumableSampler, collate_orbits
from orbitol.sample import rng_state, restore_rng
from test_train import fixture_data  # Shared data fixture only; verification below is independent.

ROOT = Path(__file__).resolve().parents[1]


def config(**kwargs):
    # A small GEM model; these CPU gates compare padding against the eager, full-width reference.
    return C.TrainConfig(objective="posterior_periodic", loss_weights=(.5, 50., 5.), gem_mode="distance",
                         d_model=32, n_heads=4, n_layers=2, compile_trunk="none", pad_multiple=0, **kwargs)


def edge_items(tables):
    """Valid chart rows with primitive atom counts within GEM's capacity."""
    a = tables.arrays

    def item(group, letters, tag):
        rows = np.asarray([tables.row_id(group, letter) for letter in letters], dtype=np.int64)
        dof = np.arange(3) < a["row_dof"][rows, None]
        u = ((np.arange(len(rows) * 3).reshape(-1, 3) * .137 + .091) % 1).astype(np.float32)
        u[~dof] = 0
        return dict(mp_id=tag, G=group, family=int(a["group_family"][group]), row=rows,
                    A0=np.full(len(rows), 8, dtype=np.int64), U=u, dof_mask=dof,
                    pad_mask=np.zeros(len(rows), dtype=bool), lat=np.zeros(6, dtype=np.float32),
                    n_prim=int(a["row_prim_mask"][rows].sum()))

    return [item(1, ["a"] * 4, "P1"), item(136, ["a", "f"], "rutile"),
            item(143, ["d"], "hexagonal"), item(166, ["h"], "rhombohedral")]


def assert_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


@pytest.fixture(scope="module")
def tables(frozen_tables):
    return frozen_tables


def test_legacy_noise_contract_and_capacity_guard():
    batch = dict(U=torch.zeros(3, 4, 3), lat=torch.zeros(3, 6))
    gen = torch.Generator().manual_seed(99)
    expected = [torch.randn(shape, generator=gen) for shape in ((3, 4, 16), (3, 4, 3), (3, 6))]
    actual = edm.orbit_noise(batch, torch.Generator().manual_seed(99))
    for ref, value in zip(expected, actual.values()):
        assert torch.equal(ref, value)
    with pytest.raises(ValueError, match="capacity"):
        edm.orbit_noise(batch, K=3)


def test_loader_and_sampler_streams_are_identical(tables):
    data = edge_items(tables) * 4
    before = rng_state()
    records = []
    for multiple in (0, 4, 8):
        sampler = ResumableSampler(len(data), seed=513, batch_size=3)
        generator = torch.Generator().manual_seed(347)
        loader = DataLoader(data, batch_sampler=sampler, generator=generator, num_workers=0,
                            collate_fn=partial(collate_orbits, K=20, pad_multiple=multiple))
        batches = []
        for batch in loader:
            real = ~batch["pad_mask"]
            batches.append({k: v[real] if k in ("row", "U", "A0", "dof_mask", "pad_mask") else v
                            for k, v in batch.items()})
            sampler.mark_consumed()
        records.append((batches, sampler.state_dict(), generator.get_state()))
    assert_equal(records[0], records[1])
    assert_equal(records[0], records[2])
    assert_equal(before, rng_state())


def test_periodic_groups_cross_chunk_boundaries(tables):
    tb = tables.torch("cpu")
    target = edm.PeriodicTarget(tb.row_H, tb.row_dof)
    items = edge_items(tables) * 17
    reference, membership = None, None
    for multiple in (0, 4, 8):
        batch = collate_orbits(items, 20, multiple)
        width = batch["row"].shape[1]
        groups = target.groups(batch["row"])
        real = ~batch["pad_mask"]
        # Compare logical structure/token indices, since flattened offsets change with width.
        logical = {j: {(int(i // width), int(i % width)) for i in indices
                       if real.flatten()[i]} for j, indices in groups}
        sigma = torch.tensor([.02, .4, .6, 10.] * 17)
        noise = edm.orbit_noise(batch, torch.Generator().manual_seed(812), K=20)
        state = edm.corrupt(batch, torch.zeros(len(items), width, 16), sigma, noise, tb,
                            objective="posterior_periodic")
        value = target(state["u_noisy"], state["u_clean"], sigma, groups)[real]
        adaptive = edm.periodic_target(state["u_noisy"], state["u_clean"], tb.row_H[batch["row"]],
                                       tb.row_dof[batch["row"]], sigma)[real]
        torch.testing.assert_close(value, adaptive, rtol=1e-6, atol=1e-6)
        if reference is None:
            reference, membership = value, logical
        else:
            assert logical == membership
            torch.testing.assert_close(value, reference, rtol=1e-6, atol=1e-6)


def test_resume_changed_padding_preserves_next_corruption(fixture_data):
    extra, _ = fixture_data
    cfg = config(precision="fp32", batch_size=8, num_workers=0, sample_count=0,
                 max_steps=4, warmup_steps=1, val_every=0, ckpt_every=0, log_every=50, extra=dict(extra))
    with tempfile.TemporaryDirectory(prefix=".verify-resume-", dir=ROOT) as out:
        run = train.Trainer(cfg, run_name="run", out_root=out, device="cpu", no_wandb=True)
        run.run(until_step=2)
        run.close()
        path = Path(out) / "run/checkpoints/latest.pt"
        saved = torch.load(path, weights_only=False)
        # Also exercise default-filling a pre-pad_multiple checkpoint.
        saved["cfg"].pop("pad_multiple")
        train.atomic_save(saved, path)
        records = []
        for multiple in (0, 4, 8):
            resumed = train.Trainer(replace(cfg, pad_multiple=multiple), run_name="run", out_root=out,
                                    device="cpu", no_wandb=True)
            assert_equal(saved["rng"], rng_state())
            assert_equal(saved["loader_rng"], resumed.loader_generator.get_state())
            assert_equal(saved["sampler"], resumed.sampler.state_dict())
            for key, value in (("model", resumed.model.state_dict()), ("ema", resumed.ema.state_dict()),
                               ("optimizer", resumed.optimizer.state_dict()),
                               ("scheduler", resumed.scheduler.state_dict())):
                assert_equal(saved[key], value)
            batch, groups = resumed._next_batch()
            losses, sigma, state, _ = resumed._loss(batch, groups=groups)
            real = ~batch["pad_mask"]
            records.append(dict(ids=batch["mp_id"], sigma=sigma,
                                noise={k: v[real] if v.ndim == 3 else v for k, v in state.items()},
                                rng=rng_state(), loader_rng=resumed.loader_generator.get_state(),
                                sampler=resumed.sampler.state_dict()))
            resumed.close()
        assert_equal(records[0], records[1])
        assert_equal(records[0], records[2])
        restore_rng(saved["rng"])
