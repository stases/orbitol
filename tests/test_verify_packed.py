"""Independent adversarial checks of the packed public interface.

CPU checks exercise the dense packed oracle; the gpu-marked test runs the actual
Flex kernels.
"""
from types import SimpleNamespace

import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol import contracts as C
from orbitol.model import build_model
from orbitol.packed import document_mask, pack_layout, packed_forward


def make_case(kind="B", lengths=(1, 0, 7, 20), width=20, seed=927, d_model=32):
    """Different fixtures, seeds and loss from the implementation's tests."""
    torch.manual_seed(seed)
    r, g = C.R_TOTAL, C.G_TOTAL
    dof = torch.arange(r) % 4
    axes = torch.arange(3)[None] < dof[:, None]
    chart = torch.diag_embed(axes.float())
    tb = SimpleNamespace(
        row_desc=torch.randn(r, C.ROW_DESC_DIM),
        group_desc=torch.randn(g, C.SG_DESC_DIM), row_dof=dof,
        row_W=chart[:, None], row_v=torch.rand(r, 1, 3),
        row_member_mask=torch.ones(r, 1, dtype=torch.bool),
        row_prim_mask=torch.ones(r, 1, dtype=torch.bool),
        group_Pc=torch.eye(3).repeat(g, 1, 1),
        group_family=torch.zeros(g, dtype=torch.long),
        row_A=chart, row_b=torch.rand(r, 3), row_N=chart,
        row_Pi=chart, row_Hinv_sqrt=chart,
    )
    cfg = C.TrainConfig(d_model=d_model, n_heads=4, n_layers=2,
                        compile_trunk="none",
                        gem_mode="none" if kind == "B" else "distance",
                        coord_input="internal" if kind == "C" else "explicit",
                        coord_readout="direct" if kind == "C" else "projection")
    model = build_model(cfg, tb.row_desc, tb.group_desc, tables=tb)
    with torch.no_grad():
        # Activate attention, GEM and conditioning without overwhelming the model.
        for name, p in model.named_parameters():
            if "adaLN_mod.1" in name or "gem.mlp.2" in name or "embed" in name:
                p.add_(torch.randn_like(p) * .07)
    batch = len(lengths)
    pad = torch.ones(batch, width, dtype=torch.bool)
    for b, length in enumerate(lengths):
        pad[b, torch.randperm(width)[:length]] = False
    rows = (torch.arange(batch * width).reshape(batch, width) % 12 + 1).masked_fill(pad, 0)
    inp = dict(type_in=torch.randn(batch, width, 16), x_in=torch.randn(batch, width, 3),
               lat_in=torch.randn(batch, 6), c_noise=torch.linspace(-2, 2, batch),
               pad_mask=pad, row=rows, G=torch.arange(batch) % (g - 1) + 1,
               u=torch.randn(batch, width, 3), z=torch.randn(batch, 6) * .1)
    return model, inp, tb


def snapshot(model, inputs, packed, *, bf16=False, slack=0, backend="dense"):
    rng = torch.random.get_rng_state().clone()
    device = inputs["x_in"].device
    cuda_rng = torch.cuda.get_rng_state().clone() if device.type == "cuda" else None
    budget = int((~inputs["pad_mask"]).sum()) + len(inputs["pad_mask"]) + slack
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        out = (packed_forward(model, **inputs, layout=pack_layout(inputs["pad_mask"], budget),
                              backend=backend) if packed else model(**inputs))
    assert torch.equal(rng, torch.random.get_rng_state()), "forward consumed RNG"
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state()), "forward consumed CUDA RNG"
    return {"out/" + k: v.detach().float() for k, v in out.items()}


def assert_scaled_close(actual, expected, tolerance):
    assert actual.keys() == expected.keys()
    for key, ref in expected.items():
        got = actual[key]
        if ref is None:
            assert got is None, key
        else:
            assert got is not None, key
            assert torch.isfinite(got).all(), key
            scale = max(float(ref.abs().max()), 1e-6)
            assert float((got - ref).abs().max()) <= 2e-6 + tolerance * scale, key


@pytest.mark.parametrize("kind", ["B", "A", "C"])
@pytest.mark.parametrize("bf16", [False, True])
def test_independent_outputs(kind, bf16):
    model, inputs, _ = make_case(kind)
    model.eval()
    ref = snapshot(model, inputs, False, bf16=bf16)
    packed = snapshot(model, inputs, True, bf16=bf16, slack=13)
    assert_scaled_close(packed, ref, .065 if bf16 else 4e-5)


@pytest.mark.parametrize("kind", ["B", "A", "C"])
@pytest.mark.parametrize("lengths,width", [((1,), 1), ((20, 20), 20), ((0,), 20),
                                         ((1, 1, 1), 20), ((2, 9, 0, 1, 17), 20)])
def test_extreme_lengths_holes_and_fixed_rows(kind, lengths, width):
    model, inputs, _ = make_case(kind, lengths, width)
    model.eval()
    assert_scaled_close(snapshot(model, inputs, True), snapshot(model, inputs, False), 3e-5)


def test_many_length_mixes_and_document_isolation():
    for seed in range(9):
        lengths = torch.randint(0, 21, (9,), generator=torch.Generator().manual_seed(seed)).tolist()
        model, inp, _ = make_case("B", lengths, seed=seed)
        model.eval()
        assert_scaled_close(snapshot(model, inp, True), snapshot(model, inp, False), 3e-5)
        before = snapshot(model, inp, True)
        changed = {k: v.clone() for k, v in inp.items()}
        changed["type_in"][0].add_(100)
        after = snapshot(model, changed, True)
        for name in before:
            torch.testing.assert_close(after[name][1:], before[name][1:], atol=0, rtol=0)


def test_1024_document_layout_and_partial_tile():
    pad = torch.ones(1024, 20, dtype=torch.bool)
    for b in range(len(pad)):
        pad[b, :b % 21] = False
    layout = pack_layout(pad)
    assert torch.equal(layout.column[layout.lattice_positions], torch.full((1024,), 20))
    assert torch.equal(layout.document[layout.lattice_positions], torch.arange(1024))
    # Check tile boundaries without allocating the dense quadratic CPU oracle.
    tiny = pack_layout(pad[:11], 129)
    mask = document_mask(tiny, compiled=False)
    q = torch.arange(256)[:, None]
    k = torch.arange(256)[None, :]
    expected = (q < 129) & (k < 129) & (tiny.document[q.clamp(max=128)] == tiny.document[k.clamp(max=128)])
    assert torch.equal(mask.mask_mod(0, 0, q, k), expected)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["B", "A", "C"])
@pytest.mark.parametrize("bf16", [False, True])
def test_independent_cuda_flex_outputs_across_layouts(kind, bf16):
    torch.compiler.reset()
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, original, _ = make_case(kind, d_model=64)
    model.cuda().eval()
    # Reuse the same compiled Flex entry point across changing capacities and
    # document membership; resetting between layouts would hide cache defects.
    for index, slack in enumerate((0, 99, 3, 99)):
        inp = {k: v.cuda() for k, v in original.items()}
        if index % 2:
            inp["pad_mask"][3, -2:] = True
            inp["row"][3, -2:] = 0
        reference = snapshot(model, inp, False, bf16=bf16)
        result = snapshot(model, inp, True, bf16=bf16, slack=slack, backend="flex")
        assert_scaled_close(result, reference, .065 if bf16 else 6e-5)


@pytest.mark.parametrize("kind", ["B", "A", "C"])
@pytest.mark.parametrize("token_budget", [16, 64])
def test_sampler_hook_preserves_initial_and_churn_draws(kind, token_budget):
    from dataclasses import asdict
    from orbitol.sampler import orbit_edm_sampler
    from orbitol.packed import PackedDenoiser

    model, inp, tb = make_case(kind, (1, 5, 3), seed=431)
    model.eval()
    skel = {k: inp[k] for k in ("row", "G", "pad_mask")}
    skel["family"] = torch.zeros(3, dtype=torch.long)
    skel["dof_mask"] = torch.arange(3) < tb.row_dof[inp["row"]][..., None]
    # One churned Heun step (two network calls): longer trajectories amplify rounding past any fixed tolerance,
    # and per-call network parity is tested separately.
    settings = dict(asdict(C.TrainConfig()), num_steps=2, sigma_max=2., S_churn=2.)
    g1, g2 = (torch.Generator().manual_seed(921) for _ in range(2))
    ref = orbit_edm_sampler(model, skel, tb, objective="posterior_periodic", settings=settings, generator=g1)
    adapter = PackedDenoiser(model, skel, token_budget=token_budget)
    got = orbit_edm_sampler(model, skel, tb, objective="posterior_periodic", settings=settings,
                            generator=g2, denoise_fn=adapter.denoise)
    assert torch.equal(g1.get_state(), g2.get_state())
    for key in ("u", "z", "type"):
        torch.testing.assert_close(got[key], ref[key], atol=1e-4, rtol=1e-4)
    assert torch.equal(got["finite"], ref["finite"])


@pytest.mark.parametrize("kind", ["B", "A", "C"])
def test_fixed_trunk_shape_across_structure_counts(kind):
    from orbitol.packed import packed_trunk
    from torch._dynamo.testing import CompileCounter

    torch.compiler.reset()
    counter = CompileCounter()
    run = torch.compile(packed_trunk, backend=counter, dynamic=False, fullgraph=True)
    model, original, _ = make_case(kind, (1, 5, 3, 12))
    model.eval()
    with torch.no_grad():
        for n in (4, 1, 3, 2, 4):
            inp = {k: v[:n].clone() for k, v in original.items()}
            layout = pack_layout(inp["pad_mask"], 64)
            assert layout.structure_capacity == 32
            got = packed_forward(model, **inp, layout=layout, backend="dense", run_trunk=run)
            ref = model(**inp)
            for key in got:
                torch.testing.assert_close(got[key], ref[key], atol=3e-6, rtol=3e-5)
    assert counter.frame_count == 1
    torch.compiler.reset()


def _pool_fixture(kind, lengths):
    from src.utils.ema import EMA

    model, _, tb = make_case(kind, lengths)
    skeletons = [C.Skeleton(i + 1, tuple(range(1, length + 1)))
                 for i, length in enumerate(lengths)]
    bank = SimpleNamespace(sample=lambda n, generator: skeletons[:n])
    ema = EMA(model, .99)
    for value in ema.shadow.values():
        value.add_(.001)
    cfg = C.TrainConfig(sample_num_steps=5, sigma_max=2., S_churn=2.,
                        objective="posterior_periodic", sample_token_budget=0)
    return model, ema, cfg, tb, bank, skeletons


def _decode_pool(output, selected, tables, encoding, allowed_mask, *, n_max):
    import numpy as np

    # Isolate the pool's indexing/assembly from crystallographic decoding (which
    # is covered by the existing decode tests and the real GPU benchmark).
    groups = np.array([s.G for s in selected], dtype=np.int64)
    return SimpleNamespace(rows={"A0": torch.from_numpy(groups)[:, None]},
                           contains_a=groups % 2 == 1, requested_G=groups,
                           failures={i: "fixture_failure" for i, s in enumerate(selected) if s.G % 2 == 0})


@pytest.mark.parametrize("kind", ["B", "A", "C"])
def test_sample_pool_same_chunk_noise_and_outputs(monkeypatch, kind):
    from dataclasses import replace
    import orbitol.sample as sampling

    model, ema, cfg, tb, bank, skeletons = _pool_fixture(kind, (1, 5, 3))
    monkeypatch.setattr(sampling, "decode_batch", _decode_pool)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    kwargs = dict(n=len(skeletons), seed=921, device="cpu", chunk=3)
    cfg = replace(cfg, sample_num_steps=2)  # one Heun step, as in the sampler-hook test above
    ref = sampling.sample_pool(model, ema, cfg, None, tb, None, None, bank, **kwargs)
    got = sampling.sample_pool(model, ema, replace(cfg, sample_token_budget=16),
                               None, tb, None, None, bank, **kwargs)
    for key in ("u", "z", "type"):
        torch.testing.assert_close(got["raw"][key], ref["raw"][key], atol=1e-4, rtol=1e-4)
    assert torch.equal(got["raw"]["finite"], ref["raw"]["finite"])
    assert got["skeletons"] == ref["skeletons"]
    assert got["failures"] == ref["failures"] == {1: "fixture_failure"}
    assert torch.equal(got["requested_G"], ref["requested_G"])
    assert got["raw"]["type"].shape == (3, cfg.K, 16)
    assert model.training
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])


def test_sample_pool_greedy_order_failures_and_budget(monkeypatch):
    from dataclasses import asdict, replace
    import orbitol.sample as sampling
    from orbitol.packed import token_chunks

    model, ema, cfg, tb, bank, skeletons = _pool_fixture("B", (1, 3, 2, 1, 4))
    monkeypatch.setattr(sampling, "decode_batch", _decode_pool)
    assert token_chunks(skeletons, 6) == [(0, 2), (2, 4), (4, 5)]
    chunks = []
    original = sampling.orbit_edm_sampler

    def record(model, batch, *args, **kwargs):
        chunks.append(batch["G"].tolist())
        return original(model, batch, *args, **kwargs)

    got = sampling.sample_pool(model, ema, replace(cfg, sample_token_budget=6), None, tb,
                               None, None, bank, n=5, seed=921, device="cpu", chunk=4, sampler=record)
    assert chunks == [[1, 2], [3, 4], [5]]
    assert got["requested_G"].tolist() == [1, 2, 3, 4, 5]
    assert got["rows"]["A0"].flatten().tolist() == [1, 2, 3, 4, 5]
    assert got["skeletons"] == [asdict(s) for s in skeletons]
    assert got["failures"] == {1: "fixture_failure", 3: "fixture_failure"}
    assert got["failure_counts"] == {"fixture_failure": 2}
    assert got["raw"]["type"].shape == (5, cfg.K, 16)
    chunks.clear()
    with pytest.raises(ValueError, match="cannot hold structure"):
        sampling.sample_pool(model, ema, replace(cfg, sample_token_budget=4), None, tb,
                             None, None, bank, n=5, seed=921, device="cpu")
    assert chunks == []  # Overflow is rejected before sampling any chunk.
