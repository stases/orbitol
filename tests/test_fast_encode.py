"""Exact encoder/group gates against the pinned core."""
from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import orbitol.core  # noqa: F401 -- bind the pinned core before src imports
from orbitol import edm
from orbitol.encode import FastEncoder, validate_a0_cache
from orbitol.sample import load_encoding, rng_state
from orbitol.train import Trainer
from test_train import cfg, equal, fixture_data, trainer  # noqa: F401


@pytest.fixture(scope="module")
def encoding():
    return load_encoding()


@pytest.fixture(scope="module")
def targets(frozen_tables):
    tb = frozen_tables.torch("cpu")
    return tb, edm.PeriodicTarget(tb.row_H, tb.row_dof, impl="batched")


def encoding_inputs(max_z, seed):
    generator = torch.Generator().manual_seed(seed)
    for a0 in (torch.arange(1, max_z + 1)[None], torch.full((3, 5), max_z),
               torch.randint(1, max_z + 1, (5, 13), generator=generator)):
        yield a0, torch.zeros_like(a0, dtype=torch.bool)
        pad = torch.rand(a0.shape, generator=generator) < .4
        # The core ignores arbitrary (even out-of-range) numbers on padding.
        yield torch.where(pad, torch.full_like(a0, -999), a0), pad
    yield torch.tensor([[-999, 0, max_z + 1]]), torch.ones((1, 3), dtype=torch.bool)
    yield torch.empty((2, 0), dtype=torch.long), torch.empty((2, 0), dtype=torch.bool)


def bitwise(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("input_dtype", [torch.int64, torch.int32, torch.float32])
def test_encoder_matches_core_for_every_element_and_edges(encoding, dtype, input_dtype):
    reference = copy(encoding)
    reference.z_to_enc = encoding.z_to_enc.to(dtype).clone()
    # The core masks the padding row even if it contains nonzero values.
    reference.z_to_enc[0].fill_(-1)
    original = reference.z_to_enc.clone()
    fast = FastEncoder(reference, "cpu")
    for seed in (7, 101, 639):
        for a0, pad in encoding_inputs(encoding.max_z, seed):
            a0, pad = a0.to(input_dtype), pad.to(torch.uint8)
            bitwise(fast(a0, pad), reference.encode_from_A0(a0, pad))
    bitwise(reference.z_to_enc, original)
    assert fast.z_to_enc.data_ptr() != reference.z_to_enc.data_ptr()


@pytest.mark.parametrize("bad", [-1, 0, 119])
def test_cache_range_check_ignores_padding_but_rejects_real_tokens(bad):
    cache = {"A0": np.array([[1, 118, -999], [118, 119, 0], [-1, 0, 999]]),
             "n_orbits": np.array([2, 1, 0])}
    validate_a0_cache(cache, 118)
    cache["A0"][0, 0] = bad
    with pytest.raises(ValueError, match=r"atomic numbers.*\[1, 118\]"):
        validate_a0_cache(cache, 118)


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("bad", [-1, 0, "above_max"])
def test_trainer_rejects_bad_cache_at_load(cfg, fixture_data, encoding, tmp_path, split, bad):
    for name in ("train", "val"):
        with np.load(Path(cfg.extra["cache_dir"]) / f"{name}.npz", allow_pickle=False) as source:
            cache = dict(source)
        if name == split:
            cache["A0"][0, 0] = encoding.max_z + 1 if bad == "above_max" else bad
        np.savez_compressed(tmp_path / f"{name}.npz", **cache)
    config = replace(cfg, fast_encode=True, extra={**cfg.extra, "cache_dir": str(tmp_path)})
    # Validation must fail before manifests, a loader, or a model are needed.
    with pytest.raises(ValueError, match="Cached atomic numbers outside descriptor range"):
        trainer(config, fixture_data, f"bad-cache-{split}-{bad}")


class Calls(TorchDispatchMode):
    def __init__(self):
        self.names = []
        super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.names.append(func._schema.name)
        return func(*args, **(kwargs or {}))


def test_encode_and_device_groups_have_no_reads_dynamic_shapes_or_copies(encoding, targets):
    _, target = targets
    fast = FastEncoder(encoding, "cpu")
    a0 = torch.tensor([[1, encoding.max_z, -999]])
    pad = torch.tensor([[False, False, True]])
    rows = torch.tensor([[0, 1, 2]])
    with Calls() as calls:
        fast(a0, pad)
        target.groups(rows)
    assert not set(calls.names) & {"aten::_local_scalar_dense", "aten::nonzero", "aten::any",
                                   "aten::_to_copy", "aten::copy_"}
    # An eager fullgraph trace also rejects accidental Python data dependencies.
    compiled = torch.compile(fast, backend="eager", fullgraph=True)
    bitwise(compiled(a0, pad), fast(a0, pad))


@pytest.mark.parametrize("pad_multiple", [0, 4])
@pytest.mark.parametrize("objective,impl", [("M", "loop"), ("P", "loop"), ("P", "batched")])
def test_trainer_loss_and_gradients_match_with_fast_encode_on_and_off(
        cfg, fixture_data, pad_multiple, objective, impl):
    config = replace(cfg, objective=objective, periodic_target_impl=impl, pad_multiple=pad_multiple)
    name = f"encode-loss-{objective}-{impl}-{pad_multiple}"
    reference = trainer(config, fixture_data, name + "-off")
    actual = trainer(replace(config, fast_encode=True), fixture_data, name + "-on")
    try:
        assert reference.fast_encoder is None and actual.fast_encoder is not None
        for seed in (23, 147, 386):
            old_batch, old_groups = reference._next_batch()
            batch, groups = actual._next_batch()
            equal((batch, groups), (old_batch, old_groups))
            before = rng_state()
            expected = reference._loss(old_batch, torch.Generator().manual_seed(seed), groups=old_groups)
            result = actual._loss(batch, torch.Generator().manual_seed(seed), groups=groups)
            equal(result, expected)
            expected_grads = torch.autograd.grad(expected[0]["total"], tuple(reference.model.parameters()),
                                                 allow_unused=True)
            actual_grads = torch.autograd.grad(result[0]["total"], tuple(actual.model.parameters()),
                                               allow_unused=True)
            equal(actual_grads, expected_grads)
            equal(before, rng_state())
            reference.sampler.mark_consumed()
            actual.sampler.mark_consumed()
    finally:
        reference.close()
        actual.close()


@pytest.mark.parametrize("source,destination", [(False, True), (True, False)])
def test_checkpoint_resumes_across_encode_settings(cfg, fixture_data, source, destination):
    config = replace(cfg, periodic_target_impl="batched", fast_encode=source)
    name = f"encode-resume-{source}"
    original = trainer(config, fixture_data, name)
    try:
        original.save()
        saved = torch.load(original.checkpoint_dir / "latest.pt", weights_only=False)
        batch, groups = original._next_batch()
        expected = original._loss(batch, torch.Generator().manual_seed(191), groups=groups)
    finally:
        original.close()
    resumed = trainer(replace(config, fast_encode=destination), fixture_data, name)
    try:
        for key, value in (("model", resumed.model.state_dict()), ("ema", resumed.ema.state_dict()),
                           ("optimizer", resumed.optimizer.state_dict()), ("scheduler", resumed.scheduler.state_dict()),
                           ("sampler", resumed.sampler.state_dict()), ("rng", rng_state()),
                           ("loader_rng", resumed.loader_generator.get_state()), ("scaler", resumed.scaler.state_dict())):
            equal(value, saved[key])
        batch, groups = resumed._next_batch()
        equal(resumed._loss(batch, torch.Generator().manual_seed(191), groups=groups), expected)
    finally:
        resumed.close()


@pytest.mark.gpu
def test_cuda_encoder_groups_capture_and_parity(encoding, targets):
    """GPU gate: changed A0/rows on replay must remain exact."""
    tb, _ = targets
    fast = FastEncoder(encoding, "cuda")
    target = edm.PeriodicTarget(tb.row_H.cuda(), tb.row_dof.cuda(), impl="batched")
    a0 = torch.ones((4, 20), dtype=torch.long, device="cuda")
    pad = torch.zeros_like(a0, dtype=torch.bool)
    rows = torch.zeros_like(a0)
    u = torch.zeros((4, 20, 3), device="cuda")
    uc = torch.zeros_like(u)
    sigma = torch.tensor([.02, .4, 2., 80.], device="cuda")

    def forward():
        groups = target.groups(rows)
        return fast(a0, pad), groups, target(u, uc, sigma, groups)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    for seed in (17, 101, 368):
        generator = torch.Generator().manual_seed(seed)
        host_a0 = torch.randint(1, encoding.max_z + 1, a0.shape, generator=generator)
        host_pad = torch.rand(pad.shape, generator=generator) < .3
        host_a0[host_pad] = -999
        host_rows = torch.randint(len(tb.row_dof), rows.shape, generator=generator)
        a0.copy_(host_a0)
        pad.copy_(host_pad)
        rows.copy_(host_rows)
        for value in (u, uc):
            value.copy_(torch.rand(value.shape, generator=generator) - .5)
        with Calls() as calls:
            fast(a0, pad)
            target.groups(rows)
        assert not set(calls.names) & {"aten::_local_scalar_dense", "aten::nonzero", "aten::_to_copy"}
        eager = forward()
        graph.replay()
        bitwise(captured[0], encoding.encode_from_A0(a0, pad))
        for actual, wanted in zip(captured, eager):
            bitwise(actual, wanted)


@pytest.mark.gpu
def test_cuda_trainer_loss_and_gradients(cfg, fixture_data):
    """GPU gate for real Trainer device rows and encoder wiring."""
    config = replace(cfg, fast_encode=True, periodic_target_impl="batched")
    run = Trainer(config, run_name="cuda-encode-loss", out_root=fixture_data[1] / "runs",
                  device="cuda", no_wandb=True)
    try:
        for seed in (23, 147, 386):
            batch, groups = run._next_batch()
            assert groups.device == batch["row"].device == run.fast_encoder.z_to_enc.device
            result = run._loss(batch, torch.Generator(device="cuda").manual_seed(seed), groups=groups)
            result_grad = torch.autograd.grad(result[0]["total"], tuple(run.model.parameters()), allow_unused=True)
            run.cfg = replace(config, fast_encode=False)
            expected = run._loss(batch, torch.Generator(device="cuda").manual_seed(seed), groups=groups)
            expected_grad = torch.autograd.grad(expected[0]["total"], tuple(run.model.parameters()), allow_unused=True)
            equal(result, expected)
            equal(result_grad, expected_grad)
            run.cfg = config
            run.sampler.mark_consumed()
    finally:
        run.close()
