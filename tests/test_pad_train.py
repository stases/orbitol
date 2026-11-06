"""Batch-local padding, fixed-noise equivalence and resume gates."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from orbitol import contracts as C, edm, train
from orbitol.data import collate_orbits
from orbitol.model import build_model
from orbitol.sample import load_encoding, rng_state
from test_model_perf import perturb
from test_train import cfg, equal, fixture_data, trainer  # noqa: F401


def snapshot(model, batch, noise, sigma, tb, cfg, encoding, *, dtype=None, prepared=None):
    """Compare real tokens per structure, scalar losses and all parameter gradients."""
    model.zero_grad(set_to_none=True)
    clean = encoding.encode_from_A0(batch["A0"], batch["pad_mask"])
    state = edm.corrupt(batch, clean, sigma, noise, tb, objective=cfg.objective)
    den = edm.denoise_orbit(model, state, sigma, batch, tb, objective=cfg.objective,
                           sigma_data=cfg.sigma_data, autocast_dtype=dtype)
    target = (prepared(state["u_noisy"], state["u_clean"], sigma,
                       prepared.groups(batch["row"].cpu())) if prepared is not None else None)
    losses = edm.compute_orbit_loss(den, state, sigma, batch, tb, objective=cfg.objective,
                                   sigma_data=cfg.sigma_data, loss_weights=cfg.loss_weights,
                                   coord_target=target)
    losses["total"].backward()
    result = {"loss/" + k: v.detach().cpu().clone() for k, v in losses.items()}
    for prefix, values in (("out/den/", {k: v for k, v in den.items() if k != "raw"}),
                           ("out/raw/", den["raw"])):
        for key, value in values.items():
            for i, real in enumerate(~batch["pad_mask"]):
                selected = value[i, real] if value.ndim == 3 else value[i]
                result[f"{prefix}{key}/{i}"] = selected.detach().float().cpu().clone()
    result.update({"grad/" + name: p.grad.detach().cpu().clone()
                   for name, p in model.named_parameters() if p.grad is not None})
    return result


def item(length, *, group=1, rows=None, tb=None):
    rows = np.ones(length, dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    dof = np.full(length, 3) if tb is None else tb.row_dof[rows].numpy()
    return dict(mp_id=f"{group}-{length}", G=group,
                family=0 if tb is None else int(tb.group_family[group]), row=rows,
                A0=np.full(length, 8, dtype=np.int64),
                U=(np.arange(length * 3).reshape(length, 3) / 31 % 1).astype(np.float32),
                dof_mask=np.arange(3) < dof[:, None], pad_mask=np.zeros(length, dtype=bool),
                lat=np.zeros(6, dtype=np.float32), n_prim=length)


@pytest.mark.parametrize("maximum,multiple,capacity,expected", [
    (1, 0, 20, 20), (1, 4, 20, 4), (4, 4, 20, 4), (5, 4, 20, 8),
    (9, 4, 20, 12), (9, 8, 20, 16), (17, 8, 20, 20), (20, 8, 20, 20),
    (9, 8, 10, 10), (5, 1, 20, 5), (5, 32, 20, 20),
])
def test_collation_rounding_order_masks_and_capacity(maximum, multiple, capacity, expected):
    items = [item(maximum), item(1), item(max(1, maximum - 1))]
    full = collate_orbits(items, capacity)
    actual = collate_orbits(items, capacity, pad_multiple=multiple)
    assert actual["row"].shape == (3, expected)
    for name, value in full.items():
        if name in ("row", "A0", "U", "dof_mask", "pad_mask"):
            value = value[:, :expected]
        equal(actual[name], value)
    assert torch.count_nonzero(actual["row"][actual["pad_mask"]]) == 0
    assert not actual["dof_mask"][actual["pad_mask"]].any()
    with pytest.raises(ValueError, match="do not fit"):
        collate_orbits([item(capacity + 1)], capacity, multiple)
    with pytest.raises(ValueError, match="do not fit"):
        collate_orbits([item(0)], capacity, multiple)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_padding(value):
    with pytest.raises(ValueError, match="pad_multiple"):
        C.TrainConfig(pad_multiple=value)
    with pytest.raises(ValueError, match="pad_multiple"):
        collate_orbits([item(2)], 20, value)


@pytest.mark.parametrize("gem,coord,readout", [
    ("none", "explicit", "projection"), ("distance", "explicit", "projection"),
    ("distance", "internal", "direct"),
])
@pytest.mark.parametrize("prepared_target", [False, True])
@pytest.mark.parametrize("dtype", [None, torch.bfloat16], ids=["fp32", "bf16"])
def test_fixed_noise_outputs_losses_and_gradients(gem, coord, readout, prepared_target, dtype, frozen_tables):
    tables = frozen_tables
    tb = tables.torch("cpu")
    items = [item(5, tb=tb), item(2, group=2, rows=[2, 3], tb=tb),
             item(3, group=47, rows=[226, 252, 226], tb=tb),
             item(1, group=194, rows=[tables.row_id(194, "h")], tb=tb),
             item(1, group=166, rows=[tables.row_id(166, "h")], tb=tb)]
    full, short = (collate_orbits(items, 20, p) for p in (0, 4))
    torch.manual_seed(31)
    cfg = C.TrainConfig(d_model=32, n_heads=4, n_layers=2, gem_mode=gem,
                        coord_input=coord, coord_readout=readout, objective="posterior_periodic",
                        compile_trunk="none")
    model = build_model(cfg, tb.row_desc, tb.group_desc, tables=tb).train()
    perturb(model)  # Exercise trained attention and GEM gradients, beyond zero gates.
    encoding = load_encoding()
    noise = edm.orbit_noise(full, torch.Generator().manual_seed(753))
    sigma = torch.tensor([.02, .3, 2., .1, .6])
    prepared = edm.PeriodicTarget(tb.row_H, tb.row_dof) if prepared_target else None
    before = rng_state()
    fp32 = snapshot(model, full, noise, sigma, tb, cfg, encoding, prepared=prepared)
    expected = (fp32 if dtype is None else snapshot(
        model, full, noise, sigma, tb, cfg, encoding, prepared=prepared, dtype=dtype))
    K = short["row"].shape[1]
    cropped = {k: v[:, :K].contiguous() if v.ndim == 3 else v for k, v in noise.items()}
    actual = snapshot(model, short, cropped, sigma, tb, cfg, encoding, prepared=prepared, dtype=dtype)
    equal(before, rng_state())
    assert expected.keys() == actual.keys()
    for name in expected:
        if dtype is None:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6, msg=name)
        else:
            # bf16 kernels can round differently across sequence lengths and CPU
            # instruction sets. Bound each tensor by its padded bf16/fp32 floor.
            floor = (expected[name] - fp32[name]).abs().max()
            delta = (actual[name] - expected[name]).abs().max()
            roundoff = 1e-6 + 1e-5 * expected[name].abs().max()
            assert delta <= floor + roundoff, name
    assert expected["grad/trunk.blocks.0.attn.in_proj_weight"].abs().max() > 0
    if gem != "none":
        assert model.gem.mlp[-1].weight.grad.abs().max() > 0


def test_both_loaders_and_diagnostics_use_padding(cfg, fixture_data):
    run = trainer(replace(cfg, pad_multiple=4), fixture_data, "padded-loaders")
    try:
        batch, groups = run._next_batch()
        assert batch["row"].shape == (cfg.batch_size, 4)
        target = run.periodic_target
        expected = target.groups(batch["row"])
        equal(groups, expected)
        seen = []
        handle = run.model.register_forward_pre_hook(lambda model, args: seen.append(args[3].shape[1]))
        before = rng_state()
        metrics = run.validate()
        equal(before, rng_state())
        handle.remove()
        assert seen and set(seen) == {4}
        assert all(np.isfinite(v) for v in metrics.values())
    finally:
        run.close()


@pytest.mark.parametrize("before,after,legacy", [(0, 4, True), (4, 0, False), (4, 8, False)])
def test_resume_padding_restores_state_and_stays_strict(cfg, fixture_data, before, after, legacy):
    name = f"padding-resume-{before}-{after}"
    original = replace(cfg, pad_multiple=before)
    run = trainer(original, fixture_data, name)
    run.run(until_step=2)
    path = run.checkpoint_dir / "latest.pt"
    run.close()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if legacy:
        saved["cfg"].pop("pad_multiple")
        train.atomic_save(saved, path)
        assert C.TrainConfig(**C.normalize_historical_config(saved["cfg"])).pad_multiple == 0
    resumed = trainer(replace(original, pad_multiple=after), fixture_data, name)
    try:
        assert resumed.step == 2 and resumed.cfg.pad_multiple == after
        for key, value in (("model", resumed.model.state_dict()), ("ema", resumed.ema.state_dict()),
                           ("optimizer", resumed.optimizer.state_dict()), ("sampler", resumed.sampler.state_dict()),
                           ("scheduler", resumed.scheduler.state_dict()), ("rng", rng_state()),
                           ("loader_rng", resumed.loader_generator.get_state())):
            equal(value, saved[key])
        resumed.run()
    finally:
        resumed.close()
    for change in (dict(K=19), dict(lr=cfg.lr * 2), dict(batch_size=4), dict(objective="wrapped")):
        with pytest.raises(ValueError, match="Resume mismatch: cfg"):
            trainer(replace(original, pad_multiple=after, **change), fixture_data, name)


def test_compile_cache_budget_never_lowers_existing_limit(monkeypatch, frozen_tables):
    tb = frozen_tables.torch("cpu")
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: calls.append(kw) or fn)
    cfg = C.TrainConfig(d_model=32, n_heads=4, n_layers=1, compile_trunk="default")
    for limit in (8, 96):
        with torch._dynamo.config.patch(cache_size_limit=limit):
            build_model(cfg, tb.row_desc, tb.group_desc, tables=tb)
            assert torch._dynamo.config.cache_size_limit == max(64, limit)
    assert all(c["dynamic"] is False and c["options"]["emulate_precision_casts"] for c in calls)


def test_compile_reuses_more_than_eight_train_and_validation_shapes(monkeypatch, frozen_tables):
    from orbitol.model import _run_trunk
    compiled_graphs = []
    compile_fn = torch.compile

    def backend(graph, inputs):
        compiled_graphs.append(graph)
        return graph.forward

    def compile_cpu(fn, *, dynamic, options):
        return compile_fn(fn, dynamic=dynamic, backend=backend, fullgraph=True)

    monkeypatch.setattr(torch, "compile", compile_cpu)
    tb = frozen_tables.torch("cpu")
    cfg = C.TrainConfig(d_model=32, n_heads=4, n_layers=1, compile_trunk="default")
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(cache_size_limit=8):
            model = build_model(cfg, tb.row_desc, tb.group_desc, tables=tb)
            for repeat in range(2):
                for training in (True, False):
                    model.train(training)
                    with torch.set_grad_enabled(training):
                        for K in (4, 8, 12, 16, 20):
                            x, t = torch.randn(2, K + 1, 32), torch.randn(2, 32)
                            pad = torch.zeros(2, K + 1, dtype=torch.bool)
                            expected = _run_trunk(model.trunk, x, t, pad)
                            actual = model._run_trunk(model.trunk, x, t, pad)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert len(compiled_graphs) == 10  # The second pass must reuse every shape.
    finally:
        torch._dynamo.reset()


def test_padding_cli_and_legacy_config():
    old = asdict(C.TrainConfig())
    old.pop("pad_multiple")
    assert C.TrainConfig(**C.normalize_historical_config(old)).pad_multiple == 0
    assert C.TrainConfig(**old).pad_multiple == 4
    # Omitted CLI options must not override a saved configuration on resume.
    assert "pad_multiple" not in vars(train.parser().parse_args(["--run-name", "padding"]))
    assert train.parser().parse_args(["--run-name", "padding", "--pad-multiple", "8"]).pad_multiple == 8
