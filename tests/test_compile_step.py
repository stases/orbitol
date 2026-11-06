"""Whole training-forward capture, RNG, lifetime and resume gates."""
from dataclasses import asdict, replace
import json
import os
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.utils import counters

from orbitol import contracts as C, train
from orbitol.edm import PeriodicTarget
from orbitol.encode import FastEncoder
from orbitol.model import OrbitolModel, _run_trunk, build_model
from test_gem_compile import MODES, activate, inputs, tables
from test_train import cfg, equal, fixture_data, load_latest  # noqa: F401


# Fused fp32 reductions, transcendental functions and SDPA can round differently.
# The absolute floor covers near-zero/cancelled gradients, where relative error
# is uninformative; each tensor is checked, not just the summed loss.
ATOL, RTOL = 1e-5, 1e-4


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    counters.clear()
    yield
    torch.compiler.reset()


@pytest.fixture
def capture(monkeypatch):
    """Trace the actual fullgraph boundary without slow CPU code generation."""
    compile_fn = torch.compile
    graphs = []

    def compile_forward(fn, **kwargs):
        assert fn.__name__ == "_train_forward", "nested compilation"
        assert kwargs == dict(fullgraph=True, dynamic=False, mode="reduce-overhead")

        def backend(graph, example_inputs):
            graphs.append(graph)
            return graph.forward

        return compile_fn(fn, backend=backend, fullgraph=True, dynamic=False)

    monkeypatch.setattr(torch, "compile", compile_forward)
    return graphs


def tiny(mode="distance", *, compiled=False, device="cpu"):
    """Real Trainer loss/model code with small, self-contained tensor tables."""
    tb = tables()
    eye = torch.eye(3).expand(C.R_TOTAL, 3, 3).clone()
    active = torch.arange(3) < tb.row_dof[:, None]
    for name in ("row_A", "row_N", "row_Pi", "row_Hinv_sqrt"):
        setattr(tb, name, eye * active[..., None])
    tb.row_H = eye
    tb.row_b = torch.zeros(C.R_TOTAL, 3)
    tb = SimpleNamespace(**{key: value.to(device) for key, value in vars(tb).items()})
    run = train.Trainer.__new__(train.Trainer)
    run.cfg = C.TrainConfig(d_model=16, n_heads=2, n_layers=1, coord_n_freqs=4,
                            gem_mode=mode, compile_trunk="reduce-overhead" if compiled else "none",
                            gem_compile=compiled, compile_step=compiled, fast_encode=True,
                            periodic_target_impl="batched", objective="posterior_periodic",
                            batch_size=2, precision="fp32", pad_multiple=4, K=20)
    run.device, run.tb = torch.device(device), tb
    gen = torch.Generator().manual_seed(71)
    run.encoding = SimpleNamespace(max_z=118, z_to_enc=torch.randn(119, 16, generator=gen))
    run.fast_encoder = FastEncoder(run.encoding, run.device)
    run.periodic_target = PeriodicTarget(tb.row_H, tb.row_dof, impl="batched")
    torch.manual_seed(83)
    run.model = build_model(run.cfg, tb.row_desc, tb.group_desc, tb)
    activate(run.model)
    run.model.to(device)
    if compiled:
        run._configure_train_forward()
        assert run.model._run_inputs is OrbitolModel._forward_inputs
        assert run.model._run_heads is OrbitolModel._forward_heads
        assert run.model._run_trunk is _run_trunk
    return run


def batch(run, seed, width=4):
    data = inputs(seed, K=width)
    row = data["row"]
    return {key: value.to(run.device) if torch.is_tensor(value) else value for key, value in dict(
        row=row, G=data["G"], family=data["G"] % 6, A0=(row + seed) % 118 + 1,
        U=data["u"] + .5, lat=data["z"], pad_mask=data["pad_mask"],
        dof_mask=torch.arange(3) < (row % 4)[..., None],
        mp_id=[f"changing-metadata-{seed}-{i}" for i in range(len(row))],
        n_prim=torch.zeros(len(row), dtype=torch.long)).items()}


def evaluate(run, data, generator):
    run.model.zero_grad(set_to_none=True)
    groups = None if run.cfg.compile_step else run.periodic_target.groups(data["row"])
    result = run._loss(data, generator, groups=groups)
    result[0]["total"].backward()
    # Take independent copies before another replay, including CUDA gradients.
    return ({key: value.detach().clone() for key, value in result[0].items()},
            {name: None if p.grad is None else p.grad.detach().clone()
             for name, p in run.model.named_parameters()})


@pytest.mark.parametrize("changes,match", [
    ({"fast_encode": False}, "fast_encode"),
    ({"periodic_target_impl": "loop"}, "periodic_target_impl"),
    ({"gem_mode": "unsupported"}, "compilable GEM"),
    ({"objective": "M"}, "posterior objective"),
    ({"pad_multiple": 1}, "at most five"),
    ({"K": 0}, "positive integer K"),
])
def test_config_rejects_uncapturable_settings(changes, match):
    options = dict(compile_step=True, fast_encode=True, periodic_target_impl="batched")
    with pytest.raises(ValueError, match=match):
        C.TrainConfig(**(options | changes))


def test_config_default_cli_and_fixed_width():
    assert C.TrainConfig().compile_step is False
    assert train.parser().parse_args(["--run-name", "test", "--compile-step", "true"]).compile_step
    for K, pad in ((20, 4), (18, 4), (40, 0), (1, 4)):
        config = C.TrainConfig(compile_step=True, fast_encode=True, periodic_target_impl="batched",
                               K=K, pad_multiple=pad)
        assert asdict(config)["compile_step"]


@pytest.mark.parametrize("mode", MODES)
def test_fullgraph_loss_gradient_and_rng_parity(capture, mode):
    eager, compiled = tiny(mode), tiny(mode, compiled=True)
    expected_rng, actual_rng = [torch.Generator().manual_seed(59) for _ in range(2)]
    for seed in range(3):
        data = batch(eager, seed)
        expected = evaluate(eager, data, expected_rng)
        actual = evaluate(compiled, data, actual_rng)
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        equal(actual_rng.get_state(), expected_rng.get_state())
    assert len(capture) == 1
    assert counters["stats"]["unique_graphs"] == 1
    assert not counters["graph_break"]


def test_five_graphs_over_fifty_random_batches(capture):
    run = tiny(compiled=True)
    generator = torch.Generator().manual_seed(997)
    order = torch.Generator().manual_seed(931)
    for repeat in range(10):
        for index in torch.randperm(5, generator=order).tolist():
            width = (4, 8, 12, 16, 20)[index]
            evaluate(run, batch(run, repeat * 5 + index, width), generator)
        assert len(capture) == 5
    assert counters["stats"]["unique_graphs"] == 5
    assert not counters["graph_break"]
    for graph in capture:
        calls = " ".join(str(node.target) for node in graph.graph.nodes if node.op == "call_function")
        assert "rand" not in calls and "_assert_async" not in calls


def test_aot_autograd_loss_and_gradients(monkeypatch):
    """Exercise AOTAutograd functionalization/backward on CPU as well."""
    compile_fn = torch.compile

    def compile_forward(fn, **kwargs):
        return compile_fn(fn, backend="aot_eager", fullgraph=True, dynamic=False)

    monkeypatch.setattr(torch, "compile", compile_forward)
    eager, compiled = tiny(), tiny(compiled=True)
    for seed in (2, 8):
        data = batch(eager, seed)
        expected = evaluate(eager, data, torch.Generator().manual_seed(seed))
        actual = evaluate(compiled, data, torch.Generator().manual_seed(seed))
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    assert counters["stats"]["unique_graphs"] == 1


def test_unexpected_batch_shapes_do_not_compile(capture):
    run = tiny(compiled=True)
    with pytest.raises(ValueError, match="batch_size and padded width"):
        run._loss(batch(run, 0, width=5))
    data = batch(run, 0)
    data = {key: value[:1] for key, value in data.items()}
    with pytest.raises(ValueError, match="batch_size and padded width"):
        run._loss(data)
    assert not capture


def test_capacity_assert_stays_outside_graph(capture, monkeypatch):
    run = tiny("multiplicity", compiled=True)
    data = batch(run, 0, width=20)
    data["row"].fill_(2)  # three primitive members each -> 60 > n_max=20
    data["pad_mask"].zero_()
    original = torch._assert_async
    checked = []

    def outside(condition, message):
        assert not torch.compiler.is_compiling()
        checked.append(message)
        return original(condition, message)

    monkeypatch.setattr(torch, "_assert_async", outside)
    with pytest.raises(RuntimeError, match="primitive member count exceeds n_max"):
        run._loss(data)
    assert checked == ["primitive member count exceeds n_max"]


def test_no_grad_loss_uses_eager_path(capture):
    eager, run = tiny(), tiny(compiled=True)
    data = batch(run, 1)
    with torch.no_grad():
        result = run._loss(data, torch.Generator().manual_seed(9))
        expected = eager._loss(data, torch.Generator().manual_seed(9))
        equal(result, expected)
        assert torch.isfinite(result[0]["total"])
        assert "raw" in result[3]
    assert not capture


def real_config(cfg, **kwargs):
    return replace(cfg, fast_encode=True, fast_update=True, periodic_target_impl="batched",
                   gem_mode="distance", objective="posterior_periodic", pad_multiple=4,
                   ckpt_every=0, log_every=2, **kwargs)


def real_trainer(config, fixture_data, name, *, device="cpu"):
    return train.Trainer(config, run_name=name, out_root=fixture_data[1] / "runs", device=device, no_wandb=True)


def test_real_trainer_rng_logging_and_retained_callback_losses(capture, cfg, fixture_data):
    marks = []
    original = torch.compiler.cudagraph_mark_step_begin
    # Mark tracking is eager; no bookkeeping enters the traced function.
    from unittest.mock import patch
    with patch.object(torch.compiler, "cudagraph_mark_step_begin", side_effect=lambda: (marks.append(1), original())):
        for enabled in (False, True):
            run = real_trainer(real_config(cfg, compile_step=enabled, val_every=2),
                               fixture_data, f"step-rng-{enabled}")
            history, copies = [], []

            def retain(step, losses):
                history.append(losses)  # deliberately retain the objects passed to the callback
                copies.append({key: value.detach().clone() for key, value in losses.items()})

            try:
                run.run(on_step=retain)
                equal(history, copies)
                if enabled:
                    assert all(not value.requires_grad for losses in history for value in losses.values())
            finally:
                run.close()
    assert len(marks) == 2 * cfg.max_steps
    off, on = [load_latest(fixture_data, f"step-rng-{enabled}") for enabled in (False, True)]
    for key in ("rng", "sampler", "loader_rng", "scheduler", "scaler"):
        equal(on[key], off[key])
    for key in ("model", "optimizer", "ema"):
        torch.testing.assert_close(on[key], off[key], atol=ATOL, rtol=RTOL)
    assert len(capture) == 1


@pytest.mark.parametrize("source", [False, True])
def test_resume_allows_compile_step_in_both_directions(capture, cfg, fixture_data, source):
    config = real_config(cfg, compile_step=source)
    name = f"step-resume-{source}"
    run = real_trainer(config, fixture_data, name)
    try:
        run.run(until_step=2)
    finally:
        run.close()
    saved = load_latest(fixture_data, name)
    run = real_trainer(replace(config, compile_step=not source), fixture_data, name)
    try:
        equal(run.model.state_dict(), saved["model"])
        equal(run.optimizer.state_dict(), saved["optimizer"])
        run.run()
        assert run.step == cfg.max_steps
    finally:
        run.close()


def test_nonfinite_gradient_assert_remains_eager(capture, cfg, fixture_data, monkeypatch):
    run = real_trainer(real_config(cfg, compile_step=True), fixture_data, "step-nonfinite")
    monkeypatch.setattr(run, "_clip", lambda limit: torch.tensor(float("nan")))
    try:
        with pytest.raises(RuntimeError, match="non-finite gradient norm"):
            run.run(until_step=1)
        assert run.step == run.scheduler.last_epoch == 0
        assert not run.optimizer.state
    finally:
        run.close()


@pytest.mark.slow
def test_inductor_cpu_loss_and_gradients():
    """Optional cold-codegen gate; CPU trace coverage above is fast."""
    eager, compiled = tiny("multiplicity"), tiny("multiplicity", compiled=True)
    data = batch(eager, 2)
    expected = evaluate(eager, data, torch.Generator().manual_seed(7))
    actual = evaluate(compiled, data, torch.Generator().manual_seed(7))
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["distance"])
def test_cuda_fp32_losses_gradients_rng_and_widths(mode):
    """Real Inductor/AOTAutograd, five widths, 50 steps."""
    # conftest enables deterministic algorithms for every test; torch 2.8 Inductor then bans scatter in
    # CUDA graphs and skips capture. Production runs nondeterministically, so test the captured path.
    torch.use_deterministic_algorithms(False)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        eager, compiled = tiny(mode, device="cuda"), tiny(mode, compiled=True, device="cuda")
        expected_rng, actual_rng = [torch.Generator(device="cuda").manual_seed(59) for _ in range(2)]
        order = torch.Generator().manual_seed(931)
        for repeat in range(10):
            for index in torch.randperm(5, generator=order).tolist():
                data = batch(eager, repeat * 5 + index, (4, 8, 12, 16, 20)[index])
                expected = evaluate(eager, data, expected_rng)
                torch.compiler.cudagraph_mark_step_begin()
                actual = evaluate(compiled, data, actual_rng)
                torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
                equal(actual_rng.get_state(), expected_rng.get_state())
                del actual, expected
            assert counters["stats"]["unique_graphs"] == 5
        assert not counters["inductor"]["cudagraph_skips"]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_cuda_real_trainer_twenty_steps(cfg, fixture_data, precision):
    """Callback tensors survive logging, validation and replay."""
    # conftest enables deterministic algorithms for every test; torch 2.8 Inductor then bans scatter in
    # CUDA graphs and skips capture. Production runs nondeterministically, so test the captured path.
    torch.use_deterministic_algorithms(False)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    config = real_config(cfg, max_steps=20, warmup_steps=7, precision=precision,
                         compile_step=True, gem_compile=True, compile_trunk="reduce-overhead", val_every=10)
    snapshots = []
    for enabled in (False, True):
        torch.compiler.reset()
        counters.clear()
        # The reference's model is eager, so only the whole-forward toggle differs in arithmetic.
        options = replace(config, compile_step=enabled, gem_compile=enabled,
                          compile_trunk="reduce-overhead" if enabled else "none")
        run = real_trainer(options, fixture_data, f"step-cuda-{precision}-{enabled}", device="cuda")
        history, copies = [], []
        logged = []
        original_metrics = run.training_metrics

        def retain(step, losses):
            history.append(losses)
            copies.append({key: value.detach().clone() for key, value in losses.items()})

        def metrics(batch, losses, sigma, state, den):
            logged.append((losses, state, den))
            return original_metrics(batch, losses, sigma, state, den)

        if enabled:
            run.training_metrics = metrics
        try:
            run.run(on_step=retain)
            assert run.step == run.scheduler.last_epoch == 20
            equal(history, copies)  # read all previous steps after the final replay
            for losses, state, den in logged:
                assert all(torch.isfinite(tensor).all() for obj in (losses, state, den) for tensor in obj.values())
            records = [json.loads(line) for line in (run.run_dir / "metrics.jsonl").read_text().splitlines()]
            assert sum("train/loss_total" in record for record in records) == 10
            assert sum("val/loss_total" in record for record in records) == 2
            snapshots.append(load_latest(fixture_data, f"step-cuda-{precision}-{enabled}"))
            if enabled:
                assert counters["stats"]["unique_graphs"] == 1
                assert not counters["inductor"]["cudagraph_skips"]
        finally:
            run.close()
    for key in ("rng", "sampler", "loader_rng", "scheduler", "scaler"):
        equal(snapshots[1][key], snapshots[0][key])
