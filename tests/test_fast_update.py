"""Bitwise gates of fast_update against the default update path; CUDA gates run only on a GPU.

No tolerance is needed: on CPU both paths run the same AdamW step and reduction order.
The batched CUDA kernels retain explicit rounding; GPU bit parity is a separate
acceptance gate, not something the CPU run can establish.
"""
import copy
from dataclasses import replace
from functools import partial
import io

import pytest
import torch

from orbitol import optim, train
from test_train import cfg, equal, fixture_data, load_latest  # noqa: F401


def assert_bits(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                           b.detach().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_bits(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_bits(x, y)
    else:
        assert a == b


def make_update(*, fast_update, dtype, device="cpu", grouped=False):
    generator = torch.Generator().manual_seed(723)
    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.randn(shape, generator=generator, dtype=dtype, device="cpu").to(device))
        for shape in ((5, 7), (0,), (1025,), (), (17, 3))])
    model.append(torch.nn.Parameter(torch.tensor([-0., 0., 1.], dtype=dtype, device=device),
                                    requires_grad=False))
    params = ([{"params": list(model)[:2], "lr": 1e-4},
               {"params": list(model)[2:], "lr": 3e-5}] if grouped else model.parameters())
    optimizer = optim.ExactAdamW(params, fast_update=fast_update)
    ema = train.EMA(model, .9999, fast_update=fast_update)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, partial(train.lr_multiplier, warmup_steps=7, max_steps=120))
    return model, optimizer, scheduler, ema


def advance(update, iteration):
    model, optimizer, scheduler, ema = update
    generator = torch.Generator().manual_seed(981 + iteration)
    optimizer.zero_grad(set_to_none=True)
    for j, p in enumerate(model):
        if not p.requires_grad or iteration % 19 == 0 or (iteration + j) % 7 == 0:
            continue
        p.grad = (torch.randn(p.shape, generator=generator, dtype=p.dtype, device="cpu")
                  * 10. ** (iteration % 7 - 3)).to(p.device)
        if iteration % 11 == 0:
            p.grad.zero_()
    # Same norm and multiply reduction order in both paths, including empty gradients.
    torch.nn.utils.clip_grad_norm_(model.parameters(), [0., .7, 1e6][iteration % 3],
                                   error_if_nonfinite=not getattr(optimizer, "fast_update", False))
    optimizer.step()
    scheduler.step()
    ema.update(model)


def assert_update_equal(a, b):
    assert_bits(a[0].state_dict(), b[0].state_dict())
    assert_bits(a[1].state_dict(), b[1].state_dict())
    assert_bits(a[2].state_dict(), b[2].state_dict())
    assert_bits(a[3].state_dict(), b[3].state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("grouped", [False, True])
def test_many_updates_match_default_path(dtype, grouped):
    reference = make_update(fast_update=False, dtype=dtype, grouped=grouped)
    actual = make_update(fast_update=True, dtype=dtype, grouped=grouped)
    rng = torch.random.get_rng_state().clone()
    for i in range(125):
        advance(reference, i)
        advance(actual, i)
        assert_update_equal(reference, actual)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("source_fast", [False, True])
def test_optimizer_ema_checkpoint_switch_both_directions(source_fast):
    source = make_update(fast_update=source_fast, dtype=torch.float32)
    for i in range(23):
        advance(source, i)
    stream = io.BytesIO()
    torch.save([component.state_dict() for component in source], stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=False)
    untouched = copy.deepcopy(saved)
    destination = make_update(fast_update=not source_fast, dtype=torch.float32)
    for component, state in zip(destination, saved):
        component.load_state_dict(state)
    # Loading preserves the input. PyTorch subsequently aliases CPU moment
    # tensors, so this assertion belongs before advancing the loaded optimizer.
    assert_bits(saved, untouched)
    assert destination[1].fast_update is not source_fast
    assert destination[3].fast_update is not source_fast
    for i in range(23, 70):
        advance(source, i)
        advance(destination, i)
        assert_update_equal(source, destination)


@pytest.mark.parametrize("weight_decay", [0., .01])
def test_trainer_matches_default_path(cfg, fixture_data, weight_decay):
    cfg = replace(cfg, weight_decay=weight_decay)
    root = fixture_data[1] / "runs"
    checkpoints, histories = [], []
    for fast_update in (False, True):
        name = f"update-{fast_update}-{weight_decay}"
        run = train.Trainer(replace(cfg, fast_update=fast_update), run_name=name, out_root=root,
                            device="cpu", no_wandb=True)
        losses = []
        try:
            assert run.ema.fast_update == fast_update
            if weight_decay == 0:
                assert run.optimizer.fast_update == fast_update
            else:
                assert type(run.optimizer) is torch.optim.AdamW
            run.run(on_step=lambda step, values: losses.append(
                {key: value.detach().clone() for key, value in values.items()}))
            histories.append(losses)
            checkpoints.append(load_latest(fixture_data, name))
        finally:
            run.close()
    assert_bits(histories[0], histories[1])
    for key in ("model", "ema", "optimizer", "scheduler", "sampler", "rng", "loader_rng", "scaler"):
        equal(checkpoints[0][key], checkpoints[1][key])


def test_scaler_skip_backoff_growth_and_schedule_match(cfg, fixture_data):
    root = fixture_data[1] / "runs"
    outcomes = []
    for label, fast in (("default", False), ("fast", True)):
        config = replace(cfg, fast_update=fast, max_steps=6, ckpt_every=0)
        run = train.Trainer(config, run_name=f"update-scaler-{label}", out_root=root,
                            device="cpu", no_wandb=True)
        run.scaler = torch.amp.GradScaler("cpu", init_scale=128., growth_interval=2)
        parameter = next(run.model.parameters())
        hook = parameter.register_hook(lambda grad: torch.full_like(
            grad, float("inf") if run.step == 1 else float("nan")) if run.step in (1, 3) else grad)
        scales = []
        try:
            run.run(on_step=lambda step, values: scales.append(run.scaler.get_scale()))
            outcomes.append(load_latest(fixture_data, f"update-scaler-{label}"))
            assert scales == [128., 64., 64., 32., 32., 64.]
            assert run.scheduler.last_epoch == 6
            assert all(state["step"] == 4 for state in run.optimizer.state.values())
        finally:
            hook.remove()
            run.close()
    for key in ("model", "ema", "optimizer", "scheduler", "scaler", "rng"):
        equal(outcomes[0][key], outcomes[1][key])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1e30])
def test_nonfinite_norm_fails_before_update_even_between_logs(cfg, fixture_data, bad):
    config = replace(cfg, fast_update=True, log_every=100, ckpt_every=0)
    run = train.Trainer(config, run_name=f"update-bad-{bad}", out_root=fixture_data[1] / "runs",
                        device="cpu", no_wandb=True)
    before = copy.deepcopy(run.model.state_dict())
    ema = run.ema.state_dict()
    hook = next(run.model.parameters()).register_hook(lambda grad: torch.full_like(grad, bad))
    try:
        with pytest.raises(RuntimeError, match="non-finite gradient norm"):
            run.run(until_step=1)
        assert run.step == 0 and not run.optimizer.state
        assert_bits(before, run.model.state_dict())
        assert_bits(ema, run.ema.state_dict())
        assert not (run.checkpoint_dir / "latest.pt").exists()
    finally:
        hook.remove()
        run.close()


@pytest.mark.parametrize("decay", [.01, .5, .9999])
def test_ema_empty_frozen_and_noncontiguous_cpu(decay):
    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.arange(35.).reshape(5, 7).T),
        torch.nn.Parameter(torch.empty(0)),
        torch.nn.Parameter(torch.ones(1), requires_grad=False)])
    reference = train.EMA(model, decay)
    actual = train.EMA(model, decay, fast_update=True)
    for i in range(40):
        with torch.no_grad():
            model[0].add_((-1.) ** i * .031)
        reference.update(model)
        actual.update(model)
    assert_bits(reference.state_dict(), actual.state_dict())
    empty = torch.nn.Identity()
    train.EMA(empty, decay, fast_update=True).update(empty)


@pytest.mark.gpu
@pytest.mark.parametrize("grouped", [False, True])
def test_cuda_batched_arithmetic_and_resume(grouped):
    reference = make_update(fast_update=False, dtype=torch.float32, device="cuda", grouped=grouped)
    actual = make_update(fast_update=True, dtype=torch.float32, device="cuda", grouped=grouped)
    for i in range(125):
        advance(reference, i)
        advance(actual, i)
        assert_update_equal(reference, actual)
        if i in (21, 57):
            # Replace moment allocations without replacing the optimizer: cached
            # metadata must not retain checkpoint-era pointers or old gradients.
            state = copy.deepcopy(actual[1].state_dict())
            actual[1].load_state_dict(state)
            actual[3].load_state_dict(reference[3].state_dict())
