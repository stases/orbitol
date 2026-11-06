"""Numerical gates for training-loop batching."""
import pytest
import torch

from orbitol import contracts as C, edm, lattice, train
from test_train import cfg, equal, fixture_data  # noqa: F401
from test_objective_acceptance import TB


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_prepared_target_matches_adaptive_windows(dtype):
    prepared = edm.PeriodicTarget(TB.row_H, TB.row_dof)
    rows = torch.arange(len(TB.row_dof)).repeat(7, 1)
    generator = torch.Generator().manual_seed(72)
    u = torch.rand(*rows.shape, 3, generator=generator, dtype=dtype) - .5
    uc = torch.rand(*rows.shape, 3, generator=generator, dtype=dtype) - .5
    sigma = torch.tensor([1e-6, .002, .15, .4, .7, 2., 80.], dtype=dtype)
    expected = edm.periodic_target(u, uc, TB.row_H[rows], TB.row_dof[rows], sigma)
    actual = prepared(u, uc, sigma, prepared.groups(rows))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-12)
    # A lift changes the mean by the same integer, including at the primal/dual switch.
    shift = (torch.randint(-3, 4, u.shape, generator=generator)
             * (torch.arange(3) < TB.row_dof[rows][..., None])).to(dtype)
    shifted = prepared(u + shift, uc, sigma, prepared.groups(rows))
    torch.testing.assert_close(shifted, actual + shift, rtol=1e-5, atol=1e-6 if dtype == torch.float32 else 1e-12)


def test_prepared_target_bounds_switches_and_chunking():
    prepared = edm.PeriodicTarget(TB.row_H, TB.row_dof)
    generator = torch.Generator().manual_seed(12)
    # 257 tokens of each metric crosses the original 256-token chunk boundary.
    rows = torch.arange(len(TB.row_dof)).repeat(257, 1)
    u = torch.rand(*rows.shape, 3, generator=generator, dtype=torch.float64) - .5
    uc = torch.rand(*rows.shape, 3, generator=generator, dtype=torch.float64) - .5
    sigma = torch.logspace(-6, 2, len(rows), dtype=torch.float64)
    actual = prepared(u, uc, sigma, prepared.groups(rows))
    expected = edm.periodic_target(u, uc, TB.row_H[rows], TB.row_dof[rows], sigma)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    for j, (d, switch, sigma_hi, logtol, h, hi, images, k, norm) in prepared.metrics.items():
        row = torch.nonzero(prepared.row_group == j)[0, 0]
        rows = row.expand(6, 1)
        sigma = torch.tensor([switch * (1 - 1e-10), switch, switch * (1 + 1e-10),
                              sigma_hi * (1 - 1e-10), sigma_hi, sigma_hi * (1 + 1e-10)], dtype=torch.float64)
        # Half-cell corners exercise the r0 bound and equidistant images.
        u, uc = torch.full((6, 1, 3), .5, dtype=torch.float64), torch.zeros(6, 1, 3, dtype=torch.float64)
        expected = edm.periodic_target(u, uc, TB.row_H[rows], TB.row_dof[rows], sigma)
        actual = prepared(u, uc, sigma, prepared.groups(rows))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)


def test_foreach_ema_matches_old_updates_and_state():
    torch.manual_seed(56)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 3))
    model[1].bias.requires_grad_(False)
    old, new = train.CoreEMA(model, .9999), train.EMA(model, .9999)
    for _ in range(5):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * .01)
        old.update(model)
        new.update(model)
        equal(old.state_dict(), new.state_dict())
    new.load_state_dict(old.state_dict())
    backup = old.apply(model)
    old.restore(model, backup)
    equal(backup, new.apply(model))
    new.restore(model, backup)


@pytest.mark.parametrize("objective", [name for name, (target, _) in C.OBJECTIVES.items()
                                       if target == "posterior"] + ["P"])
@pytest.mark.parametrize("sigma_min", [0., -1.])
def test_posterior_rejects_nonpositive_sigma_at_construction(objective, sigma_min):
    with pytest.raises(ValueError, match="Posterior objectives require sigma_min > 0"):
        train.Trainer(C.TrainConfig(objective=objective, sigma_min=sigma_min),
                      run_name="invalid-sigma", device="cpu", no_wandb=True)


def test_wrapped_objective_still_accepts_zero_sigma_floor():
    train.validate_config(C.TrainConfig(objective="wrapped", sigma_min=0.))


def test_target_groups_stay_outside_batches(cfg, fixture_data):
    run = train.Trainer(cfg, run_name="loop-groups", out_root=fixture_data[1] / "runs",
                        device="cpu", no_wandb=True)
    try:
        batch, groups = run._next_batch()
        assert set(batch) == set(C.ORBIT_BATCH_KEYS)
        assert set(run._to_device(batch)) == set(C.ORBIT_BATCH_KEYS)
        assert groups is not None
        # Same random draws, whether groups are supplied or adaptive grouping is used.
        fast = run._loss(batch, torch.Generator().manual_seed(63), groups=groups)
        adaptive = run._loss(batch, torch.Generator().manual_seed(63))
        equal(fast, adaptive)
        run.model.eval()
        sigma = torch.full((len(batch["G"]),), .4)
        expected = run._loss(batch, torch.Generator().manual_seed(12), sigma=sigma)
        actual = run._loss(batch, torch.Generator().manual_seed(12), sigma=sigma, groups=groups)
        equal(actual, expected)
        with pytest.raises(ValueError, match="positive"):
            run._loss(batch, sigma=torch.zeros_like(sigma), groups=groups)
    finally:
        run.close()


def test_lattice_masks_are_cached_and_return_independent_values():
    family = torch.tensor([5, 0, 4, 2, 1, 3, 5])
    expected = torch.tensor([[i in C.FREE_SLOTS[f] for i in range(6)] for f in C.FAMILIES])[family]
    before = lattice._free_masks(family.device)
    actual = lattice.free_mask(family)
    assert torch.equal(actual, expected)
    actual.fill_(False)
    lattice.loss_weights(family)
    assert lattice._free_masks(family.device) is before
    assert torch.equal(lattice.free_mask(family), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_lattice_clamp_cache_matches_old_values(dtype):
    family = torch.arange(len(C.FAMILIES))
    z = torch.linspace(-15, 15, len(family) * 6, dtype=dtype).reshape(-1, 6)
    bounds = z.new_tensor([5., 10., 5., 10., 10., 5.])
    expected = torch.where(lattice.free_mask(family), torch.maximum(torch.minimum(z, bounds), -bounds), 0.)
    assert torch.equal(lattice.end_clamp(z, family), expected)
    assert lattice._clamp_bounds(z.device, z.dtype) is lattice._clamp_bounds(z.device, z.dtype)


def test_nonfinite_gradient_aborts_before_optimizer(cfg, fixture_data):
    run = train.Trainer(cfg, run_name="loop-nonfinite", out_root=fixture_data[1] / "runs",
                        device="cpu", no_wandb=True)
    before = {name: value.detach().clone() for name, value in run.model.named_parameters()}
    parameter = next(run.model.parameters())
    hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            run.run(until_step=1)
        assert run.step == 0
        assert not run.optimizer.state
        equal(before, dict(run.model.named_parameters()))
        equal(before, run.ema.state_dict())
    finally:
        hook.remove()
        run.close()


def test_clamp_cache_created_in_inference_supports_autograd():
    lattice._clamp_bounds.cache_clear()
    family = torch.zeros(1, dtype=torch.long)
    with torch.inference_mode():
        lattice.end_clamp(torch.ones(1, 6), family)
    z = torch.ones(1, 6, requires_grad=True)
    lattice.end_clamp(z, family).sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))
