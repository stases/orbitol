"""Opt-in GEM compilation: Inductor parity and graph counts."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from orbitol import contracts as C
from orbitol.model import OrbitolModel, build_model
from orbitol.packed import packed_forward


MODES = ("none", "multiplicity", "distance")

# Fused fp32 transcendental functions and reductions need not round like eager.
ATOL, RTOL = 3e-6, 3e-5


def tables():
    generator = torch.Generator().manual_seed(104)
    R, S = C.R_TOTAL, 4
    tb = SimpleNamespace(
        row_desc=torch.randn(R, C.ROW_DESC_DIM, generator=generator),
        group_desc=torch.randn(C.G_TOTAL, C.SG_DESC_DIM, generator=generator),
        row_dof=torch.arange(R) % 4,
        row_W=torch.eye(3).expand(R, S, 3, 3).clone(),
        row_v=torch.rand(R, S, 3, generator=generator),
        row_member_mask=torch.arange(S)[None] < (torch.arange(R) % S + 1)[:, None],
        row_prim_mask=torch.ones(R, S, dtype=torch.bool),
        group_Pc=torch.eye(3).expand(C.G_TOTAL, 3, 3).clone(),
        group_family=torch.arange(C.G_TOTAL) % 6,
    )
    tb.row_member_mask[0] = False
    tb.row_prim_mask[3, -1] = False
    tb.group_Pc[2] = torch.tensor([[.5, -.5, 0], [.5, .5, 0], [0, 0, 1]])
    return tb


def config(mode, **kwargs):
    return C.TrainConfig(d_model=16, n_heads=2, n_layers=2, coord_n_freqs=4,
                         gem_mode=mode, compile_trunk="none", **kwargs)


def inputs(seed, K=4, B=2):
    gen = torch.Generator().manual_seed(seed)
    row = torch.randint(1, 5, (B, K), generator=gen)
    pad = torch.rand(B, K, generator=gen) < .35
    # Keep at most 20 primitive members, without requiring external tables.
    for b in range(B):
        count = 0
        for k in range(K):
            n = (1, 2, 3, 3, 1)[row[b, k]]
            if not pad[b, k] and count + n <= 20:
                count += n
            else:
                pad[b, k] = True
    if seed % 3 == 0:
        pad[0] = True  # lattice-only structure
    elif seed % 3 == 1:
        pad[0] = True
        pad[0, -1] = False  # a single orbit, not necessarily a single member
    row[pad] = 0
    u = torch.rand(B, K, 3, generator=gen) - .5
    if seed % 4 == 0:
        u.zero_()  # coincident members and minimum-image ties
    z = torch.randn(B, 6, generator=gen) * .5
    return dict(type_in=torch.randn(B, K, 16, generator=gen), x_in=u + .5,
                lat_in=z.clone(), pad_mask=pad, c_noise=torch.randn(B, generator=gen),
                row=row, G=(torch.arange(B) + seed) % 6, u=u, z=z)


def activate(model):
    with torch.no_grad():
        gen = torch.Generator().manual_seed(93)
        for p in model.parameters():
            p.add_(.03 * torch.randn(p.shape, generator=gen))


def evaluate(model, batch):
    model.zero_grad(set_to_none=True)
    rng = torch.random.get_rng_state().clone()
    biases = []
    run_trunk = model._run_trunk

    def capture(trunk, x, t_emb, pad_seq, bias):
        biases.extend(() if bias is None else (bias,))
        return run_trunk(trunk, x, t_emb, pad_seq, bias)

    model._run_trunk = capture
    try:
        out = model(**batch)
    finally:
        model._run_trunk = run_trunk
    # Compare GEM itself too: small AdaLN gates can hide bias/gradient errors.
    loss = sum(v.square().mean() for v in out.values())
    loss = loss + sum(.1 * bias.square().mean() for bias in biases)
    loss.backward()
    assert torch.equal(rng, torch.random.get_rng_state())
    return out, {n: p.grad for n, p in model.named_parameters()}, tuple(biases)


def assert_close(actual, expected, atol=ATOL, rtol=RTOL):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.slow  # Cold Inductor code generation can take tens of seconds.
def test_inductor_outputs_gradients_and_checkpoint(mode):
    torch.compiler.reset()
    tb, cfg = tables(), config(mode, coord_input="internal")
    eager = build_model(cfg, tb.row_desc, tb.group_desc, tb)
    fast = build_model(replace(cfg, gem_compile=True), tb.row_desc, tb.group_desc, tb)
    activate(eager)
    fast.load_state_dict(eager.state_dict(), strict=True)
    try:
        gem_has_gradient = False
        for seed in range(6):
            batch = inputs(seed)
            if seed == 3:
                batch["pad_mask"].fill_(True)
                batch["row"].zero_()
                batch["z"].fill_(float("nan"))
            expected = evaluate(eager, batch)
            actual = evaluate(fast, batch)
            assert_close(actual, expected)
            if mode.startswith("distance"):
                gem_has_gradient |= any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in fast.gem.parameters())
        if mode.startswith("distance"):
            assert gem_has_gradient
        # A copied model must receive its own parameters, not a captured self.
        copied = copy.deepcopy(fast)
        with torch.no_grad():
            copied.heads.type_head.bias.add_(1)
        assert_close(copied(**batch)["type_logits"], fast(**batch)["type_logits"] + 1)
        eager.load_state_dict(copied.state_dict(), strict=True)
        assert_close(eager(**batch), copied(**batch))
        # Packed sampling must never enter the newly compiled regions.
        def forbidden(*args, **kwargs):
            raise AssertionError("packed sampling entered a compiled model region")
        copied._run_inputs = copied._run_heads = forbidden
        with torch.no_grad():
            assert_close(packed_forward(copied, **batch, backend="dense"),
                         packed_forward(eager, **batch, backend="dense"), 0, 0)
            batch["pad_mask"].fill_(True)
            batch["row"].zero_()
            assert_close(packed_forward(copied, **batch, backend="dense"),
                         packed_forward(eager, **batch, backend="dense"), 0, 0)
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("mode", MODES)
def test_graph_bound_across_random_batches(monkeypatch, mode):
    """At most five input/readout graphs each for K=4,8,12,16,20.

    Fix the batch size and execution context, as in a training epoch. The
    static buckets cover the five widths; member counts never create graphs.
    Fullgraph capture also rejects scalar extraction and Python tensor branches.
    """
    torch.compiler.reset()
    graphs = {"_forward_inputs": [], "_forward_heads": []}
    compile_fn = torch.compile

    def counted(fn, **kwargs):
        assert kwargs["options"]["triton.cudagraphs"] is False
        def backend(graph, example_inputs):
            graphs[fn.__name__].append(graph)
            return graph.forward
        return compile_fn(fn, backend=backend, fullgraph=kwargs["fullgraph"], dynamic=kwargs["dynamic"])

    monkeypatch.setattr(torch, "compile", counted)
    tb = tables()
    model = build_model(config(mode, gem_compile=True), tb.row_desc, tb.group_desc, tb)
    activate(model)
    try:
        widths = torch.tensor([4, 8, 12, 16, 20])
        gen = torch.Generator().manual_seed(98)
        for round_ in range(6):
            for index in torch.randperm(5, generator=gen):
                batch = inputs(10 * round_ + int(index), int(widths[index]))
                actual = model(**batch)
                # Eager operation order is exact with the capture backend.
                eager = copy.copy(model)
                eager._run_inputs = OrbitolModel._forward_inputs
                eager._run_heads = OrbitolModel._forward_heads
                assert_close(actual, eager(**batch), 0, 0)
        assert all(len(captures) == 5 for captures in graphs.values())
        print({name: len(captures) for name, captures in graphs.items()})
    finally:
        torch.compiler.reset()


def test_compiled_capacity_assert_is_outside_inductor(monkeypatch):
    """Inductor drops graph-local _assert_async; keep its eager call."""
    torch.compiler.reset()
    compile_fn = torch.compile

    def capture(fn, **kwargs):
        return compile_fn(fn, backend="eager", fullgraph=True, dynamic=kwargs["dynamic"])

    monkeypatch.setattr(torch, "compile", capture)
    tb = tables()
    model = build_model(config("multiplicity", gem_compile=True), tb.row_desc, tb.group_desc, tb)
    batch = inputs(2, K=20)
    batch["pad_mask"].zero_()
    batch["row"].fill_(2)  # 60 members, above the fixed capacity 20
    try:
        with pytest.raises(RuntimeError, match="primitive member count exceeds n_max"):
            model(**batch)
        with torch.no_grad(), pytest.raises(ValueError, match="primitive member count exceeds n_max"):
            model(**batch)
    finally:
        torch.compiler.reset()


@pytest.mark.slow
@pytest.mark.parametrize("K", [4, 8, 12, 16, 20])
def test_inductor_width_bucket(monkeypatch, K):
    """One Inductor forward/backward graph per region in each bucket.

    Split widths so cold compilation can be checked in sub-minute CPU groups.
    The capture test above checks all five buckets together in one process.
    """
    torch.compiler.reset()
    graphs = {"_forward_inputs": 0, "_forward_heads": 0}
    compile_fn = torch.compile

    def counted(fn, **kwargs):
        def backend(graph, example_inputs, **options):
            graphs[fn.__name__] += 1
            return torch._inductor.compile(graph, example_inputs, **options)
        return compile_fn(fn, backend=backend, **kwargs)

    monkeypatch.setattr(torch, "compile", counted)
    tb, cfg = tables(), config("distance")
    eager = build_model(cfg, tb.row_desc, tb.group_desc, tb)
    fast = build_model(replace(cfg, gem_compile=True), tb.row_desc, tb.group_desc, tb)
    activate(eager)
    fast.load_state_dict(eager.state_dict(), strict=True)
    try:
        for seed in range(12):
            batch = inputs(seed, K=K)
            assert_close(evaluate(fast, batch), evaluate(eager, batch))
        assert graphs == {"_forward_inputs": 1, "_forward_heads": 1}
        print(graphs)
    finally:
        torch.compiler.reset()


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["distance"])
@pytest.mark.parametrize("bf16", [False, True])
def test_cuda_trunk_replay(mode, bf16):
    """GPU gate: independent CUDA-graph trunk and repeated backward."""
    torch.compiler.reset()
    torch.use_deterministic_algorithms(False)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    tb, cfg = tables(), config(mode)
    eager = build_model(cfg, tb.row_desc, tb.group_desc, tb)
    fast = build_model(replace(cfg, gem_compile=True, compile_trunk="reduce-overhead"),
                       tb.row_desc, tb.group_desc, tb)
    activate(eager)
    fast.load_state_dict(eager.state_dict(), strict=True)
    eager, fast = eager.cuda(), fast.cuda()
    try:
        for step, K in enumerate((4, 8, 4, 8, 4, 8)):
            batch = {name: value.cuda() for name, value in inputs(step, K=K).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                expected = evaluate(eager, batch)
                torch.compiler.cudagraph_mark_step_begin()
                actual = evaluate(fast, batch)
            assert_close(actual, expected, atol=1e-2 if bf16 else 1e-5,
                         rtol=2e-2 if bf16 else 1e-4)
            assert all(bias.dtype == torch.float32 for bias in actual[2])
            assert_close(actual[2], expected[2], atol=1e-5, rtol=1e-4)
            # No previous step's outputs may be read after the next marker.
            del actual, expected
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.compiler.reset()
