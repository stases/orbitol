"""Fixed compiled LLL: eager parity, cap fallback and capture gates."""
import pytest
import torch

from orbitol import gem_geometry as geometry
from orbitol.model import build_model
from test_gem_compile import activate, config, evaluate, inputs, tables


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def cells(seed, dtype):
    """Mixed random, skewed, near-singular, tied and invalid cells."""
    gen = torch.Generator().manual_seed(seed)
    random = torch.randn(64, 3, 3, generator=gen, dtype=dtype) + 3 * torch.eye(3, dtype=dtype)
    transform = torch.eye(3, dtype=dtype).repeat(64, 1, 1)
    for i, j in ((0, 1), (1, 2), (2, 0), (0, 2)):
        transform[:, i] += torch.randint(-3, 4, (64, 1), generator=gen) * transform[:, j]
    edges = torch.tensor([
        [[1, 0, 0], [.5, 1, 0], [-.5, .5, 1]],
        [[1, 0, 0], [1, 1e-5, 0], [1, 1e-5, 1e-5]],
        [[1, 0, 0], [1, 1e-8, 0], [0, 0, 1]],
        [[1, 0, 0], [8, 1, 0], [-5, 3, 1]],
        [[1e-20, 0, 0], [0, 1, 0], [0, 0, 1e20]],
        [[1, 0, 0], [0, 1, 0], [0, 1, 0]],
        [[1, 0, 0], [2**20, 1, 0], [0, 0, 1]],
    ], dtype=dtype)
    eye = torch.eye(3, dtype=dtype)
    return torch.cat([random, transform @ random, edges,
                      torch.stack([eye, eye * 1e30, eye * 1e-30, eye * float("nan"),
                                   eye * float("inf"), eye * 0])])


def hard_cell(dtype):
    """Consecutive Fibonacci entries require nine reduction sweeps."""
    return torch.tensor([[[610, 0, 0], [987, .001, 0], [0, 0, 1]]], dtype=dtype)


def assert_equal(actual, expected):
    for a, e in zip(actual, expected, strict=True):
        assert torch.equal(a, e)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_matches_eager_and_compiled(dtype):
    fast = torch.compile(geometry.lll_reduce, backend="eager", fullgraph=True, dynamic=False)
    counter = torch.zeros((), dtype=torch.int64)
    for seed in (7, 18, 77):
        lattice = cells(seed, dtype)
        expected = geometry.lll_reduce(lattice)
        assert_equal(fast(lattice, exceedance_counter=counter), expected)
        assert counter.item() == 0
        # Gradients stay on the original cell; decisions remain detached.
        finite = lattice[:128].clone().requires_grad_()
        reduced, _ = fast(finite, exceedance_counter=counter)
        actual_grad, = torch.autograd.grad(reduced.square().sum(), finite)
        reference, _ = geometry.lll_reduce(finite)
        expected_grad, = torch.autograd.grad(reference.square().sum(), finite)
        assert torch.equal(actual_grad, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exceedance_count_and_exact_fallback(dtype):
    hard = hard_cell(dtype)
    eye = torch.eye(3, dtype=dtype)[None]
    lattice = torch.cat([hard, eye, hard * 2, eye * float("nan"), eye * 0])
    converged = geometry.lll_reduce(lattice)  # eager: up to 64 sweeps
    assert not torch.equal(converged[1][0], eye[0])
    fast = torch.compile(geometry.lll_reduce, backend="eager", fullgraph=True, dynamic=False)
    counter = torch.zeros((), dtype=torch.int64)
    actual = fast(lattice, exceedance_counter=counter)
    assert counter.dtype == torch.int64 and counter.device == lattice.device
    assert counter.shape == () and counter.item() == 2
    assert torch.equal(actual[0][0], hard[0])
    assert torch.equal(actual[1][0], eye[0])
    fast(eye.expand_as(lattice), exceedance_counter=counter)
    assert counter.item() == 2  # cumulative: a converged batch adds nothing


def test_fullgraph_has_eight_sweeps_and_no_host_predicate(monkeypatch):
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    lattice = cells(7, torch.float32)
    counter = torch.zeros((), dtype=torch.int64)
    fast = torch.compile(geometry.lll_reduce, backend=backend, fullgraph=True, dynamic=False)
    fast(lattice, exceedance_counter=counter)
    assert len(graphs) == 1
    nodes = list(graphs[0].graph.nodes)
    # Two matrix products per sweep, initial sorting, overflow check and output.
    assert geometry.LLL_SWEEPS == 8
    assert sum(n.target is torch.bmm for n in nodes) == 2 * geometry.LLL_SWEEPS + 3
    forbidden = ("while_loop", "_local_scalar_dense", "_assert", "synchronize")
    assert not any(any(name in str(n.target) for name in forbidden) for n in nodes)
    assert not any(n.op == "call_method" and n.target in ("item", "tolist", "cpu", "__bool__")
                   for n in nodes)

    def fail(*args, **kwargs):
        raise AssertionError("compiled LLL read a tensor on the host")

    # Execute the captured graph directly so Dynamo's own bookkeeping is excluded.
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", fail)
        patch.setattr(torch.Tensor, "__bool__", fail)
        patch.setattr(torch._C, "_cuda_synchronize", fail)
        graphs[0](lattice, counter)


@pytest.mark.parametrize("mode", ["distance"])
def test_model_counter_parity_and_checkpoint(monkeypatch, mode):
    compile_fn = torch.compile

    def capture(fn, **kwargs):
        return compile_fn(fn, backend="eager", fullgraph=True, dynamic=False)

    monkeypatch.setattr(torch, "compile", capture)
    tb = tables()
    fast = build_model(config(mode, gem_compile=True), tb.row_desc, tb.group_desc, tb)
    eager = build_model(config(mode), tb.row_desc, tb.group_desc, tb)
    activate(fast)
    eager.load_state_dict(fast.state_dict(), strict=True)
    assert "lll_exceedances" not in fast.state_dict()
    counter = fast.lll_exceedances
    for seed in range(3):
        counter.zero_()  # cumulative counter: real batches add no exceedances
        batch = inputs(seed)
        actual = evaluate(fast, batch)
        torch.testing.assert_close(actual, evaluate(eager, batch), rtol=0, atol=0)
        assert fast.lll_exceedances is counter and counter.item() == 0
    fast.load_state_dict(eager.state_dict(), strict=True)
    counter.fill_(7)
    with torch.no_grad():
        fast(**batch)
    assert counter.item() == 7  # sampling does not update training diagnostics


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_graph_replay(dtype):
    """GPU gate: CUDA graph capture/replay matches direct compiled calls."""
    torch.use_deterministic_algorithms(False)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    options = {"triton.cudagraphs": False, "emulate_precision_casts": True}
    fast = torch.compile(geometry.lll_reduce, fullgraph=True, dynamic=False, options=options)
    static = cells(7, dtype).cuda()
    counter = torch.zeros((), dtype=torch.int64, device="cuda")
    try:
        for seed in (7, 18, 77):
            fast(cells(seed, dtype).cuda(), exceedance_counter=counter)
            assert counter.item() == 0
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fast(static, exceedance_counter=counter)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fast(static, exceedance_counter=counter)
        for seed in (18, 77):
            lattice = cells(seed, dtype).cuda()
            expected = fast(lattice, exceedance_counter=counter)
            static.copy_(lattice)
            graph.replay()
            torch.cuda.synchronize()
            assert_equal(captured, expected)
            assert counter.item() == 0
        # Counter storage and fallback must also update during graph replay.
        static.copy_(hard_cell(dtype).cuda().expand_as(static))
        graph.replay()
        torch.cuda.synchronize()
        assert counter.item() == len(static)
        assert torch.equal(captured[0], static)
        assert torch.equal(captured[1], torch.eye(3, dtype=dtype, device="cuda").expand_as(static))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
