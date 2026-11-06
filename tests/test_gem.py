"""GEM edge cases, validation isolation, and small CPU cost measurements."""
from dataclasses import asdict
from functools import partial
import itertools
import statistics
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from orbitol import edm
from orbitol.contracts import TrainConfig
from orbitol.data import OrbitDataset, collate_orbits
from orbitol.gem import OrbitGEM, aggregate, rbf
from orbitol.gem_geometry import PrimitiveMembers
from orbitol.model import build_model
from orbitol.sample import isolated_rng, rng_state, sample_from_checkpoint
from orbitol.tables import load_tables
from orbitol.train import Trainer, atomic_save
from test_train import equal, fixture_data  # noqa: F401


def members():
    # One missing token between two real tokens, plus an entirely empty crystal.
    return PrimitiveMembers(
        x=torch.zeros(2, 4, 3), owner=torch.tensor([[0, 0, 2, -1], [-1, -1, -1, -1]]),
        valid=torch.tensor([[True, True, True, False], [False] * 4]),
        rep=torch.tensor([[0, -1, 2], [-1, -1, -1]]), count=torch.tensor([[2, 0, 1], [0, 0, 0]]),
    )


def test_rbf_centres_and_width():
    actual = rbf(torch.tensor([0., 1., 2.], dtype=torch.float64), 3, 2.)
    expected = torch.tensor([[0., 1., 4.], [1., 0., 1.], [4., 1., 0.]], dtype=torch.float64).neg().exp()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_aggregate_hand_cases_and_gradients():
    owner = torch.tensor([[0, 0, 2, -1], [-1, -1, -1, -1]])
    valid = owner >= 0
    logits = torch.tensor([1000., 1000., -1000., 9000.], dtype=torch.float64)
    values = logits.expand(2, 2, 3, 4).clone().requires_grad_()
    actual = aggregate(values, owner, valid, 3)
    expected = torch.zeros(2, 2, 3, 3, dtype=torch.float64)
    expected[0, :, :, 0] = 1000 + np.log(2)
    expected[0, :, :, 2] = -1000
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    actual.sum().backward()
    gradient = torch.zeros_like(values)
    gradient[0, :, :, :2] = .5
    gradient[0, :, :, 2] = 1.
    torch.testing.assert_close(values.grad, gradient, rtol=0, atol=1e-12)
    assert aggregate(values[..., :0], owner[:, :0], valid[:, :0], 3).eq(0).all()


@pytest.mark.parametrize("mode", ["multiplicity", "distance"])
def test_padding_masks_queries_keys_and_empty_crystals(mode):
    m = members()
    gem = OrbitGEM(2, mode)
    if mode == "distance":
        torch.nn.init.normal_(gem.mlp[-1].weight)
        torch.nn.init.normal_(gem.mlp[-1].bias)
    bias = gem(m, torch.zeros(2, 4, 4), torch.ones(2), seq_len=4)
    assert torch.isfinite(bias).all() and bias.dtype == torch.float32
    assert bias[1].eq(0).all()
    assert bias[0, :, 1, :].eq(0).all() and bias[0, :, :, 1].eq(0).all()
    assert bias[:, :, :, 3].eq(0).all()
    torch.testing.assert_close(bias[0, :, 3, :3], torch.tensor([np.log(2), 0., 0.]).float().expand(2, -1))
    if mode == "distance":
        bias.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gem.parameters())
    else:
        assert not list(gem.parameters())


def test_zero_initialisation_matches_multiplicity_exactly():
    m = members()
    distance = OrbitGEM(4, "distance")
    multiplicity = OrbitGEM(4, "multiplicity")
    actual = distance(m, torch.rand(2, 4, 4), torch.ones(2), seq_len=4)
    assert torch.equal(actual, multiplicity(m, seq_len=4))


@pytest.mark.parametrize("features", [("distance_abs",), ("distance_abs", "distance_norm")])
def test_feature_selection_and_order(features):
    m = members()
    gem = OrbitGEM(2, "distance", features=features, n_rbf=3, hidden=4)
    seen = []
    handle = gem.mlp[0].register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    d = torch.rand(2, 4, 4)
    scale = torch.tensor([2., 3.])
    gem(m, d, scale, seq_len=4)
    handle.remove()
    expected = torch.cat([rbf(d, 3, 8.) if f == "distance_abs"
                          else rbf(d / scale[:, None, None], 3, 2.) for f in features], -1)
    assert torch.equal(seen[0], expected)


def test_validation_and_config_errors():
    for mode, features in (("edge", ("distance_norm",)), ("distance", ("direction",)),
                           ("multiplicity", ("fourier",))):
        with pytest.raises(ValueError, match="direction-dependent features are not allowed"):
            OrbitGEM(2, mode, features=features)
    with pytest.raises(TypeError, match="use_gem"):
        TrainConfig(use_gem=False)
    with pytest.raises(ValueError, match="gem_mode"):
        TrainConfig(gem_mode="edge")
    old = asdict(TrainConfig())
    del old["gem_mode"]
    assert TrainConfig(**old).gem_mode == "none"


@pytest.fixture(scope="module")
def model_inputs(fixture_data):
    extra, _ = fixture_data
    tables = load_tables(extra["tables_dir"])
    data = OrbitDataset(extra["cache_dir"] + "/val.npz", tables, augment=False, aug_seed=0)
    batch = collate_orbits([data[i % len(data)] for i in range(32)], K=20)
    return tables.torch("cpu"), batch


def tiny_cfg(**kwargs):
    return TrainConfig(d_model=32, n_heads=4, n_layers=2, batch_size=8, num_workers=0,
                       precision="fp32", sample_count=0, sample_num_steps=3,
                       compile_trunk="none", pad_multiple=0, sample_token_budget=0, **kwargs)


def forward_inputs(batch):
    B, K = batch["row"].shape
    return dict(type_in=torch.zeros(B, K, 16), x_in=batch["U"], lat_in=batch["lat"],
                pad_mask=batch["pad_mask"], c_noise=torch.zeros(B), row=batch["row"], G=batch["G"],
                u=batch["U"] - .5, z=batch["lat"], family=batch["family"])


@pytest.mark.parametrize("mode", ["multiplicity", "distance"])
def test_cpu_bf16_forward_and_nonpersistent_tables(model_inputs, mode):
    tb, batch = model_inputs
    model = build_model(tiny_cfg(gem_mode=mode), tb.row_desc, tb.group_desc, tables=tb).eval()
    inputs = forward_inputs(batch)
    seen = []
    handle = model.gem.register_forward_hook(lambda module, args, output: seen.append(output))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(**inputs)
    handle.remove()
    assert len(seen) == 1 and seen[0].dtype == torch.float32
    assert all(torch.isfinite(v).all() for v in out.values())
    assert not any(k.startswith("gem_tables.") for k in model.state_dict())
    assert set(model.gem_tables._buffers) == set(model.gem_tables._non_persistent_buffers_set)
    for missing in ("u", "z"):
        with pytest.raises(ValueError, match="noisy chart state u"):
            model(**{k: v for k, v in inputs.items() if k != missing})


def test_denoise_passes_the_chart_and_unscaled_lattice(model_inputs):
    tb, batch = model_inputs
    sigma = torch.full((32,), .5)
    state = edm.corrupt(batch, torch.zeros(32, 20, 16), sigma,
                        edm.orbit_noise(batch, torch.Generator().manual_seed(1)), tb, objective="P")
    seen = []

    class Capture:
        gem_mode = "distance"  # only GEM models receive the chart state and lattice

        def __call__(self, ty, x, lat, pad, cnoise, *, row, G, u, z, family):
            seen.append((u, z, family))
            return dict(type_logits=torch.zeros_like(ty), coord_vel=torch.zeros_like(x),
                        lattice_vel=torch.zeros_like(lat))

    def plain(ty, x, lat, pad, cnoise, *, row, G):  # a call without GEM keywords
        return dict(type_logits=torch.zeros_like(ty), coord_vel=torch.zeros_like(x),
                    lattice_vel=torch.zeros_like(lat))

    edm.denoise_orbit(plain, state, sigma, batch, tb, objective="P", sigma_data=(.3,) * 3)
    edm.denoise_orbit(Capture(), state, sigma, batch, tb, objective="P", sigma_data=(.3,) * 3)
    assert len(seen) == 1
    for actual, key in zip(seen[0][:2], ("u_noisy", "z_noisy")):
        assert actual.dtype == torch.float32 and torch.equal(actual, state[key])
    assert torch.equal(seen[0][2], batch["family"])


def test_distance_learns_after_adaln_warmup(model_inputs):
    tb, batch = model_inputs
    torch.manual_seed(9)
    model = build_model(tiny_cfg(gem_mode="distance"), tb.row_desc, tb.group_desc, tables=tb).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sigma = torch.full((32,), .5)
    state = edm.corrupt(batch, torch.zeros(32, 20, 16), sigma,
                        edm.orbit_noise(batch, torch.Generator().manual_seed(1)), tb, objective="P")
    norms = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        den = edm.denoise_orbit(model, state, sigma, batch, tb, objective="P", sigma_data=(.3,) * 3)
        losses = edm.compute_orbit_loss(den, state, sigma, batch, tb, objective="P",
                                        sigma_data=(.3,) * 3, loss_weights=(1., 50., 5.))
        losses["total"].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        norms.append(float(model.gem.mlp[-1].weight.grad.abs().sum()))
        optimizer.step()
    assert norms[0] == 0.  # Inherited AdaLN-zero blocks initially bypass attention.
    assert norms[1] > 0. and model.gem.mlp[-1].weight.detach().abs().sum() > 0
    print(f"GEM last-weight gradient L1 before/after one optimizer step: {norms}")


def test_sigma_grid_keys_weights_and_legacy_values(fixture_data, monkeypatch):
    extra, root = fixture_data
    cfg = tiny_cfg(extra=dict(extra))
    run = Trainer(cfg, run_name="sigma-grid", out_root=root / "runs", device="cpu", no_wandb=True)
    try:
        before = rng_state()
        actual = run.validate()
        equal(before, rng_state())
        assert run.model.training and actual == run.validate()
        grid = (0.02, 0.1, 0.5, 2, 10)
        new_keys = {f"val/sigma_{s}/loss_{name}" for s in grid for name in ("type", "coord", "lattice")}
        assert new_keys <= actual.keys()
        assert all(np.isfinite(value) for value in actual.values())
        # Reproduce the pre-GEM validation loop, including its random sigma draws.
        seed = int(np.random.SeedSequence([cfg.extra["val_seed"], run.step]).generate_state(1)[0])
        loader = DataLoader(run.val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
                            collate_fn=partial(collate_orbits, K=cfg.K),
                            generator=torch.Generator().manual_seed(seed))
        generator = torch.Generator().manual_seed(seed)
        totals, count = dict(total=0., A=0., F=0., Y=0.), 0
        with isolated_rng(), torch.no_grad():
            run.model.eval()
            for batch in itertools.islice(loader, cfg.extra["val_batches"]):
                losses, _, _, _ = run._loss(run._to_device(batch), generator)
                for key in totals:
                    totals[key] += float(losses[key])
                count += 1
        run.model.train()
        for name, key in (("total", "total"), ("type", "A"), ("coord", "F"), ("lattice", "Y")):
            assert actual[f"val/loss_{name}"] == totals[key] / count
        # Known loss components pin the weights, sigma values and key formatting.
        def known_loss(batch, generator=None, *, sigma=None):
            value = 1. if sigma is None else float(sigma[0])
            assert sigma is None or (sigma == sigma[0]).all()
            return dict(total=6 * value, A=value, F=2 * value, Y=3 * value), None, None, None
        monkeypatch.setattr(run, "_loss", known_loss)
        # The stub has no corrupted state, so the coordinate diagnostics (tested in test_train.py) are off here.
        monkeypatch.setattr(run, "_coord_diagnostics", lambda *args: (0., {}))
        weighted = run.validate()
        for s in grid:
            for name, factor, weight in zip(("type", "coord", "lattice"), (1, 2, 3), cfg.loss_weights):
                assert weighted[f"val/sigma_{s}/loss_{name}"] == pytest.approx(s * factor * weight)
    finally:
        run.close()


@pytest.mark.parametrize("mode", ["none", "multiplicity", "distance"])
def test_checkpoint_sampling_and_old_config_resume(fixture_data, mode):
    extra, root = fixture_data
    cfg = tiny_cfg(gem_mode=mode, extra=dict(extra))
    run = Trainer(cfg, run_name=f"gem-snapshot-{mode}", out_root=root / "runs", device="cpu", no_wandb=True)
    try:
        run.save(snapshot=True)
        if mode == "none":
            # Older full checkpoints lack the newer default field.
            path = run.checkpoint_dir / "latest.pt"
            old = torch.load(path, weights_only=False)
            del old["cfg"]["gem_mode"]
            atomic_save(old, path)
            resumed = Trainer(cfg, run_name=f"gem-snapshot-{mode}", out_root=root / "runs",
                              device="cpu", no_wandb=True)
            equal(run.model.state_dict(), resumed.model.state_dict())
            resumed.close()
        out = sample_from_checkpoint(run.checkpoint_dir / "step_0000000.pt", n=2, seed=31,
                                     device="cpu", chunk=2)
        assert all(torch.isfinite(v).all() for v in out["raw"].values())
    finally:
        run.close()


def test_aggregate_memory():
    # Cumulative positive allocations are an upper bound on live extra memory.
    # Include backward so empty-key handling and its autograd storage are counted.
    values = torch.randn(128, 16, 20, 20, requires_grad=True)
    owner = (torch.arange(20) // 2).expand(128, -1)
    valid = torch.ones_like(owner, dtype=torch.bool)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
        aggregate(values, owner, valid, 20).sum().backward()
    allocated = sum(max(0, e.self_cpu_memory_usage) for e in prof.events())
    print(f"aggregate B=128 H=16 K=N=20: cumulative positive CPU allocations {allocated / 2**20:.2f} MiB (fwd+bwd)")
    assert allocated < 200_000_000
    assert torch.isfinite(values.grad).all()


def test_cpu_forward_microbenchmark(model_inputs):
    tb, batch = model_inputs
    inputs = forward_inputs(batch)
    results = {}
    for mode in ("none", "multiplicity", "distance"):
        torch.manual_seed(13)
        model = build_model(tiny_cfg(gem_mode=mode), tb.row_desc, tb.group_desc, tables=tb).eval()
        elapsed = []
        with torch.no_grad():
            for step in range(8):
                start = time.perf_counter()
                out = model(**inputs)
                if step >= 3:
                    elapsed.append(time.perf_counter() - start)
        assert all(torch.isfinite(v).all() for v in out.values())
        results[mode] = statistics.median(elapsed) * 1000
    print("CPU forward B=32 K=N_max=20 d_model=32 H=4 layers=2, one thread, fp32: "
          + ", ".join(f"{mode}={elapsed:.3f} ms" for mode, elapsed in results.items()))
