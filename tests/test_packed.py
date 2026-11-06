"""CPU dense-reference parity and optional compiled CUDA Flex parity."""
from types import SimpleNamespace

import pytest
import torch

from orbitol import contracts as C
from orbitol.model import build_model
from orbitol.packed import (document_mask, pack_layout, packed_forward,
                            packed_trunk)


def case(gem_mode="none", coord_input="explicit", device="cpu", d_model=32):
    torch.manual_seed(173)
    R = C.R_TOTAL
    tb = SimpleNamespace(
        row_desc=torch.randn(R, C.ROW_DESC_DIM),
        group_desc=torch.randn(C.G_TOTAL, C.SG_DESC_DIM),
        row_dof=torch.full((R,), 3),
        row_W=torch.eye(3).expand(R, 2, 3, 3).clone(),
        row_v=torch.zeros(R, 2, 3),
        row_member_mask=torch.zeros(R, 2, dtype=torch.bool),
        row_prim_mask=torch.ones(R, 2, dtype=torch.bool),
        group_Pc=torch.eye(3).expand(C.G_TOTAL, 3, 3).clone(),
        group_family=torch.zeros(C.G_TOTAL, dtype=torch.long),
    )
    tb.row_member_mask[:, 0] = True
    tb.row_member_mask[2, 1] = True
    tb.row_v[2, 1] = .5
    tb.row_dof[3] = 0
    tb.row_dof[4] = 1
    cfg = C.TrainConfig(d_model=d_model, n_heads=4, n_layers=2,
                        gem_mode=gem_mode, coord_input=coord_input, compile_trunk="none")
    model = build_model(cfg, tb.row_desc, tb.group_desc, tables=tb).to(device)
    # Fresh zero adaLN gates would make an attention test vacuous.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.03 * torch.randn_like(parameter))
    pad = torch.ones(4, 20, dtype=torch.bool, device=device)
    pad[1, 2] = False
    pad[2, [0, 3, 7, 9]] = False
    pad[3] = False
    row = torch.ones_like(pad, dtype=torch.long)
    row[2, [0, 3, 7, 9]] = torch.tensor([2, 3, 4, 1], device=device)
    row[pad] = 0
    inputs = dict(type_in=torch.randn(4, 20, 16, device=device),
                  x_in=torch.rand(4, 20, 3, device=device),
                  lat_in=torch.randn(4, 6, device=device) * .1,
                  pad_mask=pad, c_noise=torch.randn(4, device=device),
                  row=row, G=torch.tensor([1, 2, 3, 1], device=device),
                  u=torch.rand(4, 20, 3, device=device) - .5,
                  z=torch.randn(4, 6, device=device) * .1)
    return model, inputs


@torch.no_grad()
def snapshot(model, inputs, forward):
    rng = torch.random.get_rng_state().clone()
    out = forward(inputs)
    assert torch.equal(rng, torch.random.get_rng_state())
    return ({k: v.detach() for k, v in out.items()},)


def assert_equivalent(actual, reference, atol=3e-6, rtol=3e-5):
    for got, expected in zip(actual, reference):
        assert got.keys() == expected.keys()
        for name in got:
            if expected[name] is None:
                assert got[name] is None, name
            else:
                assert got[name] is not None, name
                torch.testing.assert_close(got[name], expected[name], atol=atol, rtol=rtol,
                                           msg=lambda detail: f"{name}: {detail}")


@pytest.mark.parametrize("gem_mode", ["none", "multiplicity", "distance"])
@pytest.mark.parametrize("coord_input", ["explicit", "internal"])
def test_cpu_packed_outputs_and_filler_invariance(gem_mode, coord_input):
    model, inputs = case(gem_mode, coord_input)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    reference = snapshot(model, inputs, lambda inp: model(**inp))
    for budget in (None, 32, 64):
        layout = pack_layout(inputs["pad_mask"], budget)
        actual = snapshot(model, inputs, lambda inp: packed_forward(
            model, **inp, layout=layout, backend="dense"))
        assert_equivalent(actual, reference)
    assert state.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name]), name
    model.load_state_dict(state, strict=True)


def test_layout_budget_and_document_mask():
    _, inputs = case()
    layout = pack_layout(inputs["pad_mask"], 32)
    assert layout.real_tokens == 29
    assert layout.document.tolist() == [0] + [1] * 2 + [2] * 5 + [3] * 21 + [4, 5, 6]
    assert layout.column[layout.lattice_positions].tolist() == [20] * 4
    mask = document_mask(layout, compiled=False)
    q, k = torch.arange(32)[:, None], torch.arange(32)[None, :]
    assert torch.equal(mask.mask_mod(0, 0, q, k), layout.document[:, None] == layout.document[None, :])
    assert not mask.mask_mod(0, 0, torch.tensor(32), torch.tensor(0))
    with pytest.raises(ValueError, match="cannot hold"):
        pack_layout(inputs["pad_mask"], 28)


def test_reject_dropout():
    model, inputs = case()
    model.trunk.blocks[0].attn.dropout = .1
    with pytest.raises(ValueError, match="zero dropout"):
        packed_forward(model, **inputs, backend="dense")


def test_trunk_uses_p0_bias_in_flex_scores(monkeypatch):
    import orbitol.packed as packed

    model, inputs = case("distance")
    layout = pack_layout(inputs["pad_mask"], 32)
    bias = model._gem_bias(inputs["u"], inputs["z"], inputs["row"], inputs["G"],
                           inputs["pad_mask"], 21).detach()
    token = torch.arange(layout.real_tokens)
    expected = bias[layout.document[token], 0, layout.column[token], layout.column[token]]
    assert not torch.equal(expected, expected.bfloat16().float())
    observed = []

    def inspect_scores(q, k, v, *, score_mod, block_mask, kernel_options):
        observed.append(score_mod(torch.zeros(len(token)), 0, 0, token, token))
        return v  # Only the score modifier is under test; no attention oracle.

    monkeypatch.setattr(packed, "_compiled_flex", inspect_scores)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        packed_forward(model, **inputs, layout=layout, block_mask=object())
    assert len(observed) == len(model.trunk.blocks)
    for scores in observed:
        torch.testing.assert_close(scores, expected.bfloat16().float(), atol=0, rtol=0)


@pytest.mark.parametrize("width,expected", [(32, 8192), (512, 8192), (513, 6144), (1024, 6144)])
def test_auto_sample_token_budget(width, expected):
    cfg = C.TrainConfig(d_model=width, sample_token_budget=-1)
    assert cfg.sample_token_budget == expected
    assert C.TrainConfig(d_model=width).sample_token_budget == expected
    assert C.TrainConfig(d_model=width, sample_token_budget=0).sample_token_budget == 0
    assert C.TrainConfig(d_model=width, sample_token_budget=17).sample_token_budget == 17


@pytest.mark.parametrize("budget", [-2, -100, 1.5, True])
def test_invalid_sample_token_budget(budget):
    with pytest.raises(ValueError, match="sample_token_budget"):
        C.TrainConfig(sample_token_budget=budget)


@pytest.mark.gpu
@pytest.mark.parametrize("gem_mode", ["none", "distance"])
@pytest.mark.parametrize("budget", [None, 128])
def test_cuda_flex_outputs(gem_mode, budget):
    # GPU-only: the CPU tests intentionally use the independent dense SDPA path.
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    # PyTorch 2.8's CUDA Flex kernel requires at least 16 channels per head.
    model, inputs = case(gem_mode, device="cuda", d_model=64)
    layout = pack_layout(inputs["pad_mask"], budget)
    mask = document_mask(layout)
    ref = snapshot(model, inputs, lambda inp: model(**inp))
    actual = snapshot(model, inputs, lambda inp: packed_forward(
        model, **inp, layout=layout, block_mask=mask))
    assert_equivalent(actual, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.parametrize("gem_mode", ["none", "distance"])
@pytest.mark.parametrize("bf16", [False, True])
def test_cuda_compiled_changing_batches(gem_mode, bf16):
    # Regression: automatic dynamic mask compilation failed after
    # changing capacity. Exercise one callable across lengths and new skeletons.
    torch.compiler.reset()
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, original = case(gem_mode, device="cuda", d_model=64)
    run = torch.compile(packed_trunk, dynamic=False, fullgraph=True,
                        options={"emulate_precision_casts": True})
    for index, budget in enumerate((None, 128, 64, 128)):
        inputs = {name: value.clone() for name, value in original.items()}
        if index == 1:
            inputs["pad_mask"][3, -2:] = True
            inputs["row"][3, -2:] = 0
        if index == 3:
            inputs = {name: value.flip(0) for name, value in inputs.items()}
        layout = pack_layout(inputs["pad_mask"], budget)
        mask = document_mask(layout)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            ref = snapshot(model, inputs, lambda inp: model(**inp))
            actual = snapshot(model, inputs, lambda inp: packed_forward(
                model, **inp, layout=layout, block_mask=mask, run_trunk=run))
        assert_equivalent(actual, ref, atol=1e-2 if bf16 else 1e-5, rtol=2e-2 if bf16 else 1e-4)
