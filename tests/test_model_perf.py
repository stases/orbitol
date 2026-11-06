"""Network optimizations preserve the original outputs and gradient paths."""
import copy
from dataclasses import asdict, replace
import random

import numpy as np
import pytest
import torch

from orbitol import contracts as C, edm
from orbitol.model import OrbitolModel, build_model
from src.crystalite.crystalite import mod1


class ReferenceModel(OrbitolModel):
    """Original forward, independently retaining all per-block masking."""

    def forward(self, type_in, x_in, lat_in, pad_mask, c_noise, *, row, G,
                u=None, z=None, family=None):
        frac_mod = mod1(x_in)
        h_type = self.type_proj(type_in) + (
            self._internal_coord_embed(u, frac_mod, row) if self.coord_input == "internal"
            else self.coord_embed(frac_mod)
        ) + self.segment_embed.weight[0]
        h_type = h_type + self.row_embed(row)
        h_lat = self.lattice_embed(lat_in) + self.segment_embed.weight[1]
        x = torch.cat([h_type, h_lat[:, None, :]], dim=1)
        pad_seq = torch.cat([pad_mask.bool(), torch.zeros(
            (pad_mask.shape[0], 1), device=pad_mask.device, dtype=torch.bool)], dim=1)
        t_emb = self.time(c_noise, c_noise) + self.sg_embed(G)
        if self.gem_mode == "none":
            sigma_for_gem = torch.exp(4.0 * c_noise.to(dtype=torch.float32))
            h = self.trunk(x, t_emb, pad_mask=pad_seq, coords=frac_mod,
                           lattice=lat_in, t_sigma=sigma_for_gem)
        else:
            bias = self._gem_bias(u, z, row, G, pad_mask, x.shape[1])
            for block in self.trunk.blocks:
                x = x.masked_fill(pad_seq[..., None], 0.0)
                x = block(x, t_emb, pad_mask=pad_seq, attn_head_bias=bias.to(dtype=x.dtype))
            h = self.trunk.norm_out(x).masked_fill(pad_seq[..., None], 0.0)
        return self.heads(h, frac_coords=frac_mod, pad_mask=pad_mask.bool())


def model_kwargs(cfg, tb):
    return dict(row_desc=tb.row_desc, group_desc=tb.group_desc,
                row_embed=cfg.row_embed, sg_embed=cfg.sg_embed,
                coord_input=cfg.coord_input, coord_readout=cfg.coord_readout,
                row_dof=tb.row_dof, gem_mode=cfg.gem_mode, gem_tables=tb,
                gem_n_max=cfg.n_prim_max, d_model=cfg.d_model, n_heads=cfg.n_heads,
                n_layers=cfg.n_layers, mlp_ratio=cfg.mlp_ratio, type_dim=16, vz=94,
                coord_embed_mode=cfg.coord_embed_mode, n_freqs=cfg.coord_n_freqs,
                lattice_embed_mode=cfg.lattice_embed_mode, lattice_repr="ltri",
                coord_head_mode="direct", use_edge_bias=False, use_distance_bias=False)


def perturb(model):
    # Nonzero gates and heads exercise gradients which fresh zero gates hide.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))


@pytest.fixture(scope="module")
def tb(frozen_tables):
    return frozen_tables.torch("cpu")


def inputs(tb):
    row = torch.tensor([[1, 1, 0, 0], [2, 3, 0, 0], [226, 252, 226, 0]])
    u = torch.randn(3, 4, 3) * 0.1
    return dict(type_in=torch.randn(3, 4, 16),
                x_in=edm.map3(tb.row_A[row], u + 0.5) + tb.row_b[row],
                lat_in=torch.randn(3, 6) * 0.1, pad_mask=row == 0,
                c_noise=torch.tensor([-0.3, 0., 0.5]), row=row,
                G=torch.tensor([1, 2, 47]), u=u, z=torch.randn(3, 6) * 0.1)


@pytest.mark.parametrize("gem_mode", ["none", "distance"])
@pytest.mark.parametrize("coord_input", ["explicit", "internal"])
@pytest.mark.parametrize("training", [False, True])
def test_outputs_loss_gradients_and_rng_match_reference(tb, gem_mode, coord_input, training):
    torch.manual_seed(31)
    cfg = C.TrainConfig(d_model=32, n_heads=4, n_layers=3, gem_mode=gem_mode,
                        coord_input=coord_input)
    kw = model_kwargs(cfg, tb)
    # Exercise RNG preservation too; inference uses the same weights with dropout disabled.
    kw.update(dropout=0.1, attn_dropout=0.1)
    reference = ReferenceModel(**kw).train(training)
    perturb(reference)
    model = OrbitolModel(**kw).train(training)
    model.load_state_dict(reference.state_dict(), strict=True)
    assert model.state_dict().keys() == reference.state_dict().keys()
    batch = inputs(tb)
    ref_inputs = {k: v.clone().requires_grad_(v.is_floating_point()) for k, v in batch.items()}
    new_inputs = {k: v.clone().requires_grad_(v.is_floating_point()) for k, v in batch.items()}
    torch.manual_seed(7)
    expected = reference(**ref_inputs)
    ref_loss = sum(v.square().mean() for v in expected.values())
    ref_loss.backward()
    ref_rng = torch.random.get_rng_state()
    torch.manual_seed(7)
    actual = model(**new_inputs)
    loss = sum(v.square().mean() for v in actual.values())
    loss.backward()
    assert torch.equal(torch.random.get_rng_state(), ref_rng)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    torch.testing.assert_close(loss, ref_loss, atol=0, rtol=0)
    for name, parameter in reference.named_parameters():
        grad = dict(model.named_parameters())[name].grad
        assert (grad is None) == (parameter.grad is None), name
        if grad is not None:
            torch.testing.assert_close(grad, parameter.grad, atol=0, rtol=0, msg=name)
    for name, value in ref_inputs.items():
        if value.grad is not None:
            torch.testing.assert_close(new_inputs[name].grad, value.grad, atol=0, rtol=0, msg=name)
    # No-grad inference takes MultiheadAttention's native path without a GEM bias.
    if not training:
        with torch.no_grad():
            expected, actual = reference(**batch), model(**batch)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.parametrize("gem_mode", ["none", "distance"])
def test_checkpoint_and_deepcopy_keep_parameter_names_and_ownership(tb, gem_mode):
    cfg = C.TrainConfig(d_model=32, n_heads=4, n_layers=2, gem_mode=gem_mode)
    model = OrbitolModel(**model_kwargs(cfg, tb))
    reference = ReferenceModel(**model_kwargs(cfg, tb))
    model.load_state_dict(reference.state_dict(), strict=True)
    ema = copy.deepcopy(model)
    for name, parameter in model.named_parameters():
        copied = dict(ema.named_parameters())[name]
        assert copied.data_ptr() != parameter.data_ptr()
        assert torch.equal(copied, parameter)
    batch = inputs(tb)
    with torch.no_grad():
        original = ema(**batch)
        for parameter in model.parameters():
            parameter.add_(0.1)
        copied = ema(**batch)
    for key in original:
        assert torch.equal(copied[key], original[key])


@pytest.mark.parametrize("gem_mode", ["none", "distance"])
def test_resume_original_adamw_state(tb, gem_mode):
    torch.manual_seed(42)
    cfg = C.TrainConfig(d_model=32, n_heads=4, n_layers=2, gem_mode=gem_mode)
    reference = ReferenceModel(**model_kwargs(cfg, tb))
    perturb(reference)
    old_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4)
    batch = inputs(tb)

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(v.square().mean() for v in model(**batch).values())
        loss.backward()
        optimizer.step()

    step(reference, old_optimizer)
    checkpoint = copy.deepcopy(dict(model=reference.state_dict(), optimizer=old_optimizer.state_dict()))
    model = OrbitolModel(**model_kwargs(cfg, tb))
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer"])
    step(reference, old_optimizer)
    step(model, optimizer)
    for name, value in reference.state_dict().items():
        assert torch.equal(model.state_dict()[name], value), name
    for old, resumed in zip(old_optimizer.state.values(), optimizer.state.values()):
        for key in old:
            assert torch.equal(old[key], resumed[key]), key


@pytest.mark.parametrize("gem_mode", ["none", "distance"])
def test_compile_option_outputs_gradients_and_copy(tb, gem_mode, monkeypatch):
    # AOTAutograd exercises graph capture and backward on CPU without slow codegen.
    # Actual Inductor kernels and their precision errors are checked on the bench GPU.
    compile_fn = torch.compile

    def compile_for_cpu(fn, *, dynamic, options):
        assert options["emulate_precision_casts"] is True
        return compile_fn(fn, backend="aot_eager", fullgraph=True, dynamic=dynamic)

    monkeypatch.setattr(torch, "compile", compile_for_cpu)
    cfg = C.TrainConfig(d_model=32, n_heads=4, n_layers=2, gem_mode=gem_mode)
    old_config = asdict(cfg)
    old_config.pop("compile_trunk")
    assert C.TrainConfig(**C.normalize_historical_config(old_config)).compile_trunk == "none"
    assert C.TrainConfig(**old_config).compile_trunk == "reduce-overhead"
    with pytest.raises(ValueError, match="compile_trunk"):
        C.TrainConfig(compile_trunk="fast")
    torch.manual_seed(11)
    reference = ReferenceModel(**model_kwargs(cfg, tb))
    rng = torch.random.get_rng_state()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch.manual_seed(11)
    model = build_model(replace(cfg, compile_trunk="default"), tb.row_desc, tb.group_desc, tables=tb)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert python_rng == random.getstate()
    assert all(np.array_equal(a, b) for a, b in zip(numpy_rng, np.random.get_state()))
    perturb(reference)
    model.load_state_dict(reference.state_dict(), strict=True)
    batch = inputs(tb)
    expected, actual = reference(**batch), model(**batch)
    ref_loss = sum(v.square().mean() for v in expected.values())
    loss = sum(v.square().mean() for v in actual.values())
    ref_loss.backward()
    loss.backward()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    for name, parameter in reference.named_parameters():
        grad = dict(model.named_parameters())[name].grad
        assert (grad is None) == (parameter.grad is None), name
        if grad is not None:
            torch.testing.assert_close(grad, parameter.grad, rtol=1e-5, atol=1e-6, msg=name)
    copied = copy.deepcopy(model)
    copied.load_state_dict(reference.state_dict(), strict=True)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)
        output = copied(**batch)
        expected = reference(**batch)
    for name in expected:
        torch.testing.assert_close(output[name], expected[name], rtol=1e-5, atol=1e-6)
