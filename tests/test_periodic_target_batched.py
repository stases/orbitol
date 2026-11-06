"""CPU gates for the batched prepared target; the reference is the loop implementation."""
from dataclasses import replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from orbitol import edm
from orbitol.sample import rng_state
from test_train import cfg, equal, fixture_data, trainer  # noqa: F401


@pytest.fixture(scope="module")
def targets(frozen_tables):
    tb = frozen_tables.torch("cpu")
    loop = edm.PeriodicTarget(tb.row_H, tb.row_dof)
    batched = edm.PeriodicTarget(tb.row_H, tb.row_dof, impl="batched")
    return tb, loop, batched


def representatives(target):
    return torch.stack([torch.nonzero(target.row_group == j)[0, 0]
                        for j in range(int(target.row_group.max()) + 1)])


def coordinates(rows, seed, dtype):
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand(*rows.shape, 3, generator=generator, dtype=dtype) - .5
    uc = torch.rand(*rows.shape, 3, generator=generator, dtype=dtype) - .5
    return u, uc


def close(actual, expected):
    # Float64 padded reductions and per-token dual tails differ at roundoff.
    # Float32 additionally allows one final-cast rounding near a bin boundary.
    tol = dict(atol=1e-12, rtol=1e-10) if expected.dtype == torch.float64 else dict(atol=1e-7, rtol=1e-6)
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seed", [17, 123, 813])
def test_all_table_rows_and_random_batches(targets, dtype, seed):
    tb, loop, batched = targets
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randperm(len(tb.row_dof), generator=generator)[:, None]
    u, uc = coordinates(rows, seed, dtype)
    sigma = torch.exp(torch.randn(len(rows), generator=generator, dtype=dtype) * 2 - 1)
    before = rng_state()
    expected = loop(u, uc, sigma, loop.groups(rows))
    actual = batched(u, uc, sigma, batched.groups(rows))
    close(actual, expected)
    assert not actual[~(torch.arange(3) < tb.row_dof[rows][..., None])].any()
    equal(before, rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chunk_boundaries_and_row_order(targets, dtype):
    _, loop, batched = targets
    # Every metric crosses 256 twice; sigma varies within the old chunks.
    rows = representatives(loop).repeat(513, 1)
    u, uc = coordinates(rows, 318, dtype)
    sigma = torch.logspace(-6, 2, len(rows), dtype=dtype)
    actual = batched(u, uc, sigma, batched.groups(rows))
    close(actual, loop(u, uc, sigma, loop.groups(rows)))
    generator = torch.Generator().manual_seed(18)
    permutation = torch.randperm(rows.numel(), generator=generator)
    permuted_rows = rows.flatten()[permutation, None]
    permuted_sigma = sigma[:, None].expand_as(rows).flatten()[permutation]
    shuffled = batched(u.reshape(-1, 1, 3)[permutation], uc.reshape(-1, 1, 3)[permutation],
                       permuted_sigma, batched.groups(permuted_rows))
    # CPU vector math can round differently when a value moves to another lane.
    close(shuffled[:, 0], actual.reshape(-1, 3)[permutation])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_switches_ties_and_high_sigma(targets, dtype):
    _, loop, batched = targets
    for j, (_, switch, sigma_hi, *_) in loop.metrics.items():
        row = torch.nonzero(loop.row_group == j)[0, 0]
        sigma = torch.tensor([1e-6, .002, switch * (1 - 1e-6), switch,
                              switch * (1 + 1e-6), sigma_hi * (1 - 1e-6), sigma_hi,
                              sigma_hi * (1 + 1e-6), 80.], dtype=dtype)
        rows = row.expand(len(sigma), 4)
        u, uc = coordinates(rows, j, dtype)
        u[:, :2] = torch.tensor([[-.5, .5, -.5], [.5, .5, .5]], dtype=dtype)
        uc[:, :2] = 0
        close(batched(u, uc, sigma, batched.groups(rows)), loop(u, uc, sigma, loop.groups(rows)))


def test_padding_noncontiguous_inputs_and_lifts(targets):
    tb, loop, batched = targets
    rows = torch.cat([representatives(loop), torch.zeros(3, dtype=torch.long)]).repeat(5, 2)[:, ::2]
    u, uc = (t.transpose(0, 1) for t in coordinates(rows.T, 79, torch.float64))
    sigma = torch.tensor([.002, .2, .4, .7, 80.], dtype=torch.float64)
    actual = batched(u, uc, sigma, batched.groups(rows))
    close(actual, loop(u, uc, sigma, loop.groups(rows)))
    shift = torch.tensor([2., -3., 1.], dtype=torch.float64) * (torch.arange(3) < tb.row_dof[rows][..., None])
    close(batched(u + shift, uc, sigma, batched.groups(rows)), actual + shift)
    # Padding has a P1 chart, including when unused input values are nonzero.
    pad = rows == 0
    close(actual[pad], loop(u, uc, sigma, loop.groups(rows))[pad])


@pytest.mark.parametrize("shape", [(3, 5), (0, 5), (3, 0)])
def test_only_zero_dof_and_empty_batches(shape):
    H, dof = torch.eye(3)[None], torch.zeros(1, dtype=torch.long)
    rows = torch.zeros(shape, dtype=torch.long)
    u, uc = coordinates(rows, 13, torch.float64)
    sigma = torch.ones(shape[0], dtype=torch.float64)
    target = edm.PeriodicTarget(H, dof, impl="batched")
    assert torch.equal(target(u, uc, sigma, target.groups(rows)), torch.zeros_like(u))


def test_batched_gradients_and_autocast(targets):
    _, loop, batched = targets
    rows = representatives(loop).repeat(4, 1)
    u, uc = coordinates(rows, 16, torch.float64)
    args = [t.requires_grad_() for t in (u, uc, torch.tensor([.1, .3, .7, 2.], dtype=torch.float64))]
    expected = loop(*args, loop.groups(rows))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = batched(*args, batched.groups(rows))
    close(actual, expected)
    for new, old in zip(torch.autograd.grad(actual.square().sum(), args),
                        torch.autograd.grad(expected.square().sum(), args)):
        close(new, old)


def test_device_path_has_no_scalar_reads_or_dynamic_shapes(targets):
    _, loop, batched = targets

    class Calls(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            assert not any(op in name for op in ("_local_scalar_dense", "nonzero", "unique")), name
            self.calls.append(name)
            return func(*args, **(kwargs or {}))

    calls = []
    for n in (2, 513):
        rows = representatives(loop).repeat(n, 1)
        u, uc = coordinates(rows, 71, torch.float32)
        sigma, groups = torch.full((n,), .3), batched.groups(rows)
        with Calls() as mode:
            batched(u, uc, sigma, groups)
        calls.append(mode.calls)
    assert calls[0] == calls[1]  # No launch-count growth with groups or chunks.


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_parity(targets, dtype):
    """Optional GPU gate; CPU-only runs deselect this test."""
    tb, cpu_loop, _ = targets
    loop = edm.PeriodicTarget(tb.row_H.cuda(), tb.row_dof.cuda())
    batched = edm.PeriodicTarget(tb.row_H.cuda(), tb.row_dof.cuda(), impl="batched")
    rows = representatives(cpu_loop).repeat(513, 1)
    u, uc = (t.cuda() for t in coordinates(rows, 318, dtype))
    sigma = torch.logspace(-6, 2, len(rows), dtype=dtype, device="cuda")
    actual = batched(u, uc, sigma, batched.groups(rows))
    close(actual, loop(u, uc, sigma, loop.groups(rows)))
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(83))
    shuffled = batched(u[permutation], uc[permutation], sigma[permutation], batched.groups(rows[permutation]))
    close(shuffled, actual[permutation])


def test_trainer_uses_batched_groups_with_matching_loss_and_gradients(cfg, fixture_data):
    config = replace(cfg, objective="posterior_periodic", periodic_target_impl="batched")
    run = trainer(config, fixture_data, "periodic-loss")
    try:
        batch, groups = run._next_batch()
        assert torch.is_tensor(groups)
        prepared = run.periodic_target
        parameters = [p for p in run.model.parameters() if p.requires_grad]
        before = rng_state()
        actual = run._loss(batch, torch.Generator().manual_seed(819), groups=groups)
        actual_grad = torch.autograd.grad(actual[0]["total"], parameters, allow_unused=True)
        run.periodic_target = edm.PeriodicTarget(run.tb.row_H, run.tb.row_dof)
        expected = run._loss(batch, torch.Generator().manual_seed(819),
                             groups=run.periodic_target.groups(batch["row"]))
        expected_grad = torch.autograd.grad(expected[0]["total"], parameters, allow_unused=True)
        equal(actual[1:], expected[1:])  # Sigma, corruption and denoiser are untouched.
        for key in actual[0]:
            close(actual[0][key], expected[0][key])
        for new, old in zip(actual_grad, expected_grad):
            if old is None:
                assert new is None
            else:
                close(new, old)
        run.periodic_target = prepared
        equal(before, rng_state())
    finally:
        run.close()


@pytest.mark.parametrize("source,destination", [("loop", "batched"), ("batched", "loop")])
def test_checkpoint_can_resume_across_target_modes(cfg, fixture_data, source, destination):
    """No training: save/load a tiny fixture checkpoint in both directions."""
    config = replace(cfg, objective="posterior_periodic", periodic_target_impl=source)
    name = f"periodic-resume-{source}"
    original = trainer(config, fixture_data, name)
    original.save()
    saved = torch.load(original.checkpoint_dir / "latest.pt", weights_only=False)
    original.close()
    resumed = trainer(replace(config, periodic_target_impl=destination), fixture_data, name)
    try:
        assert resumed.periodic_target.impl == destination
        for key, value in (("model", resumed.model.state_dict()), ("ema", resumed.ema.state_dict()),
                           ("optimizer", resumed.optimizer.state_dict()), ("scheduler", resumed.scheduler.state_dict()),
                           ("sampler", resumed.sampler.state_dict()), ("rng", rng_state()),
                           ("loader_rng", resumed.loader_generator.get_state())):
            equal(value, saved[key])
    finally:
        resumed.close()
