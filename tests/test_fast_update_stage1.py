"""Bitwise gates of the fast update against the default update path; CUDA gates run only on a GPU."""
import copy
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
import io
import os
import weakref

import pytest
import torch

from orbitol import optim, train
from test_fast_update import assert_bits, assert_update_equal
from test_train import cfg, equal, fixture_data, load_latest  # noqa: F401


def make_update(*, fast=True, device="cpu", max_steps=150, warmup=17):
    generator = torch.Generator().manual_seed(811)
    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.randn(shape, generator=generator).to(device))
        for shape in ((7, 3), (1025,), (), (0,), (5,))])
    model.append(torch.nn.Parameter(torch.tensor([-0., 0., 1.], device=device), requires_grad=False))
    optimizer = optim.ExactAdamW([
        {"params": list(model)[:2], "lr": 1e-4}, {"params": list(model)[2:], "lr": 3e-5}],
        fast_update=fast)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, partial(train.lr_multiplier, warmup_steps=warmup, max_steps=max_steps))
    if fast:
        optimizer.configure_schedule(scheduler, max_steps)
        optimizer._test_clip = optim.CachedClip(model.parameters())
    ema = train.EMA(model, .9999, fast_update=fast)
    return model, optimizer, scheduler, ema


def backward(update, iteration, *, missing=False, set_to_none=True):
    model, optimizer, _, _ = update
    optimizer.zero_grad(set_to_none=set_to_none)
    if set_to_none:
        assert all(p.grad is None for p in model)
    generator = torch.Generator().manual_seed(271 + iteration)
    terms = []
    for j, p in enumerate(model):
        if not p.requires_grad or (missing and (iteration + j) % 11 == 0):
            continue
        g = (torch.randn(p.shape, generator=generator) * 10. ** (iteration % 5 - 2)).to(p.device)
        if iteration % 13 == 0:
            g.zero_()
        # Multiple contributions at a leaf exercise backward's accumulation.
        terms.append((p * g).sum() + (p * (g * -.25)).sum())
    sum(terms).backward()


def advance(update, iteration, *, missing=False, skip=False, set_to_none=True):
    model, optimizer, scheduler, ema = update
    backward(update, iteration, missing=missing, set_to_none=set_to_none)
    max_norm = (0., .7, 1e6)[iteration % 3]
    if hasattr(optimizer, "_test_clip"):
        norm = optimizer._test_clip(max_norm)
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)
    if not skip:
        optimizer.step()
    scheduler.step()
    if getattr(optimizer, "_schedule", None) is not None:
        optimizer.finish_schedule_step()
    ema.update(model)
    return norm


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("set_to_none", [False, True])
def test_125_backward_updates_match_default_path(missing, set_to_none):
    expected, actual = make_update(fast=False), make_update()
    pointers = None
    rng = torch.random.get_rng_state().clone()
    for i in range(125):
        assert_bits(advance(expected, i, missing=missing, set_to_none=set_to_none),
                    advance(actual, i, missing=missing, set_to_none=set_to_none))
        assert_update_equal(expected, actual)
        assert actual[1]._next_step == actual[2].last_epoch == i + 1
        assert int(actual[1]._device_step) == i + 1
        if not missing and not set_to_none:
            current = [p.grad.data_ptr() for p in actual[0] if p.requires_grad]
            assert pointers is None or current == pointers
            pointers = current
    assert_bits(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("source_fast,target_fast", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_mid_warmup_checkpoint_and_toggle_roundtrip(source_fast, target_fast, device):
    reference, source = make_update(fast=False, device=device), make_update(fast=source_fast, device=device)
    for i in range(9):
        advance(reference, i, missing=True)
        advance(source, i, missing=True)
    stream = io.BytesIO()
    torch.save([part.state_dict() for part in source], stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    untouched = copy.deepcopy(saved)
    destination = make_update(fast=target_fast, device=device)
    for part, state in zip(destination, saved):
        part.load_state_dict(state)
    assert_bits(saved, untouched)
    assert destination[1].fast_update is target_fast
    for i in range(9, 130):
        # Also exercise a scaler skip before the first optimizer.step after load.
        assert_bits(advance(reference, i, missing=True, skip=i == 9),
                    advance(destination, i, missing=True, skip=i == 9))
        assert_update_equal(reference, destination)


def test_tables_are_python_rounded_and_rebuilt_for_new_resume_schedule():
    expected, actual = make_update(fast=False), make_update()
    for i in range(23):
        advance(expected, i)
        advance(actual, i)
    saved = copy.deepcopy([part.state_dict() for part in actual])
    resumed = make_update(max_steps=180, warmup=41)
    for part, state in zip(resumed, saved):
        part.load_state_dict(state)
    # Load order is optimizer then scheduler; rebuilding must be deferred until
    # both have loaded. The first update still uses the saved optimizer LR.
    resumed[1]._ensure_schedule()
    for i, group in enumerate(resumed[1].param_groups):
        assert resumed[1]._rate_tables[i, 0].shape == (180, 2)
        for k in range(23, 180):
            lr = group["lr"] if k == 23 else resumed[2].base_lrs[i] * resumed[2].lr_lambdas[i](k)
            for offset in (0, 4):
                n = float(k + 1 - offset)  # The default CUDA path reads a float32 CPU step's .item().
                old_rates = torch.tensor([-(lr / (1 - .9 ** n)), (1 - .999 ** n) ** .5])
                assert_bits(resumed[1]._rate_table(i, offset)[k], old_rates)
    # Give the reference the same new schedule, retaining its saved current LR.
    expected[2].lr_lambdas = resumed[2].lr_lambdas
    for i in range(23, 150):
        assert_bits(advance(expected, i), advance(resumed, i))
        assert_update_equal(expected, resumed)


def test_zero_grad_none_skips_unused_but_false_updates_zero_moments():
    expected, actual = make_update(fast=False), make_update()
    for i in range(2):
        advance(expected, i)
        advance(actual, i)
    before = copy.deepcopy(actual[1].state_dict())
    for update in (expected, actual):
        update[1].zero_grad(set_to_none=True)
        update[1].step()
        update[2].step()
        if getattr(update[1], "_schedule", None) is not None:
            update[1].finish_schedule_step()
    assert_update_equal(expected, actual)
    assert_bits(before["state"], actual[1].state_dict()["state"])
    assert all(p.grad is None for p in actual[0])
    # Reusing the buffers with False intentionally updates previously used leaves.
    for i in range(3, 6):
        assert_bits(advance(expected, i, missing=True, set_to_none=False),
                    advance(actual, i, missing=True, set_to_none=False))
        assert_update_equal(expected, actual)


@pytest.mark.parametrize("between", ["clear_false", "load"])
def test_none_clear_survives_second_clear_or_load(between):
    expected, actual = make_update(fast=False), make_update()
    advance(expected, 0)
    advance(actual, 0)
    for update in (expected, actual):
        optimizer = update[1]
        optimizer.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in update[0])
        if between == "clear_false":
            optimizer.zero_grad(set_to_none=False)
        else:
            optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        optimizer.step()
    assert_update_equal(expected, actual)


def test_scaler_style_skips_keep_schedule_and_bias_counters_distinct():
    expected, actual = make_update(fast=False), make_update()
    for i in range(125):
        skip = i in (5, 19, 20, 89)
        assert_bits(advance(expected, i, skip=skip), advance(actual, i, skip=skip))
        assert_update_equal(expected, actual)
    assert actual[2].last_epoch == 125
    assert int(actual[1]._device_step) == 125
    assert all(int(s["step"]) == 121 for s in actual[1].state.values())


def test_cached_ema_lists_survive_load_and_parameter_storage_replacement():
    expected, actual = make_update(fast=False), make_update()
    advance(expected, 0)
    advance(actual, 0)
    pairs = actual[3]._pairs
    actual[3].update(actual[0])
    expected[3].update(expected[0])
    assert actual[3]._pairs is pairs
    for update in (expected, actual):
        with torch.no_grad():
            update[0][0].data = update[0][0].detach().clone().add_(.25)
        update[3].update(update[0])
    assert_update_equal(expected, actual)
    actual[3].load_state_dict(expected[3].state_dict())
    actual[3].update(actual[0])
    expected[3].update(expected[0])
    assert actual[3]._pairs is not pairs
    assert_update_equal(expected, actual)


@pytest.mark.parametrize("dtypes", [(torch.float32,), (torch.float64,), (torch.float32, torch.float64)])
def test_cached_clip_arithmetic_empty_replacement_and_nonfinite(dtypes):
    expected = [torch.nn.Parameter(torch.ones(n, dtype=dtype)) for dtype in dtypes for n in (0, 1, 17)]
    actual = copy.deepcopy(expected)
    clip = optim.CachedClip(actual)
    assert_bits(clip(1.), torch.nn.utils.clip_grad_norm_(expected, 1.))
    for i in range(9):
        for p, q in zip(expected, actual):
            p.grad = torch.sin(torch.arange(p.numel(), dtype=p.dtype) + i)
            q.grad = p.grad.clone()
        assert_bits(clip(.1), torch.nn.utils.clip_grad_norm_(expected, .1))
        for p, q in zip(expected, actual):
            assert_bits(p.grad, q.grad)
        assert not clip._grads
    for p in actual:
        p.grad.fill_(float("inf"))
    assert not torch.isfinite(clip(1.))


def test_schedule_guard_and_explicit_reconfiguration():
    update = make_update(max_steps=2, warmup=1)
    advance(update, 0)
    advance(update, 1)
    with pytest.raises(ValueError, match="exhausted"):
        update[1].step()
    update[1].configure_schedule(update[2], 4)
    update[2].step()
    with pytest.raises(AssertionError, match="disagree"):
        update[1].step()


def test_pointer_staging_reuses_allocations_and_waits_before_overwriting(monkeypatch):
    uploads, launches, pinned, events = [], [], [], []
    empty = torch.empty

    def allocate(*args, **kwargs):
        pin = kwargs.pop("pin_memory", False)
        tensor = empty(*args, **kwargs)
        if pin:
            pinned.append(tensor)
        return tensor

    class Event:
        def __init__(self):
            self.index = len(events)
            self.pending = False
            self.waits = 0
            events.append(self)

        def query(self):
            return not self.pending

        def synchronize(self):
            # Simulate a delayed copy: staging must still contain its old data.
            assert_bits(pinned[self.index], self.contents)
            self.pending = False
            self.waits += 1

        def record(self, stream):
            assert stream == "stream"
            self.pending = True
            self.contents = pinned[self.index].clone()

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((grid, args, kwargs))

    def upload(values, *, dtype, device):
        uploads.append(values)
        return torch.tensor(values, dtype=dtype, device=device)

    monkeypatch.setattr(optim, "_multi_tensor_kernel", Kernel)
    monkeypatch.setattr(optim, "_upload", upload)
    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "stream")
    rows = [tuple(torch.ones(n) for _ in range(4)) for n in (1, 1025)]
    rates, step = torch.ones(150, 2), torch.tensor(0, dtype=torch.int64)
    batch = optim._MultiTensorUpdate()
    batch(rows, rates=rates, step=step)
    assert len(uploads) == 1 and len(pinned) == 2  # Geometry, then two persistent staging buffers.
    metadata, blocks = batch._meta, batch._blocks
    for _ in range(4):
        batch(rows, rates=rates, step=step)
    assert len(uploads) == 1 and batch._meta is metadata and batch._blocks is blocks
    assert batch._slot == 1  # Unchanged pointers, including EMA, need no copies.
    # Keep shape but replace moment storage, as load_state_dict does.
    rows[1][2].data = rows[1][2].clone()
    batch(rows, rates=rates, step=step)
    assert len(pinned) == 2 and batch._meta is metadata and batch._blocks is blocks
    assert batch._slot == 0 and not any(event.waits for event in events)
    # New gradient storage invalidates only pointers.
    old_grads = [weakref.ref(row[1]) for row in rows]
    rows = [(row[0], row[1].clone(), row[2], row[3]) for row in rows]
    batch(rows, rates=rates, step=step)
    assert all(ref() is None for ref in old_grads)
    assert len(uploads) == 1 and len(pinned) == 2
    assert batch._meta is metadata and batch._blocks is blocks
    assert events[0].waits == 1 and events[1].waits == 0
    assert_bits(metadata, torch.tensor([[*(t.data_ptr() for t in row), row[0].numel()] for row in rows]))
    assert len(launches) == 7


@pytest.mark.parametrize("source_fast", [False, True])
def test_trainer_mid_warmup_resume_across_toggles(cfg, fixture_data, source_fast):
    config = replace(cfg, max_steps=6, warmup_steps=4, ckpt_every=0, log_every=100)
    root = fixture_data[1] / "runs"
    name = f"stage1-split-{source_fast}"
    reference_name = f"stage1-reference-{source_fast}"
    reference = train.Trainer(replace(config, fast_update=False), run_name=reference_name,
                              out_root=root, device="cpu", no_wandb=True)
    try:
        reference.run()
    finally:
        reference.close()
    source = train.Trainer(replace(config, fast_update=source_fast), run_name=name,
                           out_root=root, device="cpu", no_wandb=True)
    try:
        source.run(until_step=3)
    finally:
        source.close()
    destination = train.Trainer(replace(config, fast_update=not source_fast), run_name=name,
                                out_root=root, device="cpu", no_wandb=True)
    try:
        destination.run()
    finally:
        destination.close()
    expected, actual = load_latest(fixture_data, reference_name), load_latest(fixture_data, name)
    for key in ("model", "ema", "optimizer", "scheduler", "scaler", "sampler", "rng", "loader_rng"):
        # RNG includes NumPy state, whose arrays are compared by the trainer helper.
        equal(expected[key], actual[key])


@pytest.mark.gpu
@pytest.mark.parametrize("missing", [False, True])
def test_cuda_125_updates_resume_and_reallocated_storage(missing):
    expected, actual = make_update(fast=False, device="cuda"), make_update(device="cuda")
    for i in range(125):
        skip = i in (23, 61)
        assert_bits(advance(expected, i, missing=missing, skip=skip),
                    advance(actual, i, missing=missing, skip=skip))
        assert_update_equal(expected, actual)
        if i == 53:
            actual[1].load_state_dict(copy.deepcopy(actual[1].state_dict()))
            actual[3].load_state_dict(expected[3].state_dict())
            # External gradient replacement must rebuild pointers safely.
            for p in actual[0]:
                if p.grad is not None:
                    p.grad = p.grad.clone()


@pytest.mark.gpu
def test_cuda_steady_update_reuses_staging_without_rate_upload_or_scalar_read(monkeypatch):
    expected, actual = make_update(fast=False, device="cuda"), make_update(device="cuda")
    advance(expected, 0)
    advance(actual, 0)
    batch = next(iter(actual[1]._batched_updates.values()))
    metadata, blocks = batch._meta, batch._blocks
    staging = tuple(buffer.data_ptr() for buffer in batch._staging)
    empty = torch.empty

    def forbidden(*args, **kwargs):
        raise AssertionError("steady update must not allocate pinned storage, upload rates or read scalars")

    def allocate(*args, **kwargs):
        if kwargs.get("pin_memory"):
            forbidden()
        return empty(*args, **kwargs)

    for i in range(1, 10):
        advance(expected, i)
        with monkeypatch.context() as context:
            context.setattr(optim, "_upload", forbidden)
            context.setattr(torch, "empty", allocate)
            context.setattr(torch.Tensor, "item", forbidden)
            context.setattr(torch.Tensor, "__int__", forbidden)
            advance(actual, i)
        assert batch._meta is metadata and batch._blocks is blocks
        assert tuple(buffer.data_ptr() for buffer in batch._staging) == staging
        assert_update_equal(expected, actual)


@pytest.mark.gpu
def test_cuda_reduce_overhead_trainer_24_steps_matches_fast_off(cfg, fixture_data):
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "run with CUBLAS_WORKSPACE_CONFIG=:4096:8"
    torch.use_deterministic_algorithms(True)
    config = replace(cfg, max_steps=24, warmup_steps=7, ckpt_every=0, log_every=100,
                     precision="bf16", compile_trunk="reduce-overhead")
    histories, checkpoints = [], []
    for fast in (False, True):
        # Independent graph trees, identical initialization and training RNG.
        torch._dynamo.reset()
        name = f"stage1-cudagraph-trees-{fast}"
        run = train.Trainer(replace(config, fast_update=fast), run_name=name,
                            out_root=fixture_data[1] / "runs", device="cuda", no_wandb=True)
        history = []
        try:
            run.run(on_step=lambda step, losses: history.append(
                {name: p.detach().cpu().clone() for name, p in run.model.named_parameters()}))
            assert run.step == 24 and run.scheduler.last_epoch == 24
            histories.append(history)
            checkpoints.append(load_latest(fixture_data, name))
        finally:
            run.close()
            del run
    assert len(histories[0]) == len(histories[1]) == 24
    assert_bits(histories[0], histories[1])
    for key in ("model", "ema", "optimizer", "scheduler", "scaler"):
        assert_bits(checkpoints[0][key], checkpoints[1][key])
    for key in ("rng", "sampler", "loader_rng"):
        equal(checkpoints[0][key], checkpoints[1][key])
    torch._dynamo.reset()
