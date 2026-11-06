"""Default model size, conditioning, equivariance and CPU gradient checks."""
import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol import contracts as C
from orbitol.embeddings import RowEmbedding, SGConditioning
from orbitol.model import OrbitolModel, build_model
from src.crystalite.crystalite import CrystaliteModel
from src.data.mp20_tokens import VZ


ROW_MODES = ("categorical", "physical", "hybrid")
SG_MODES = ("none", "categorical", "physical", "hybrid")
DEFAULT_KW = dict(
    d_model=512,
    n_heads=16,
    n_layers=14,
    mlp_ratio=4.0,
    type_dim=16,
    vz=VZ,
    n_freqs=32,
    coord_embed_mode="fourier",
    lattice_embed_mode="mlp",
    lattice_repr="ltri",
    coord_head_mode="direct",
    use_edge_bias=False,
    use_distance_bias=False,
)
SMALL_KW = {**DEFAULT_KW, "d_model": 32, "n_heads": 4, "n_layers": 2}


def _descriptors():
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(C.R_TOTAL, C.ROW_DESC_DIM, generator=generator),
        torch.randn(C.G_TOTAL, C.SG_DESC_DIM, generator=generator),
    )


def _inputs():
    generator = torch.Generator().manual_seed(11)
    B, K = 2, 20
    pad = torch.zeros(B, K, dtype=torch.bool)
    pad[0, -5:] = True
    pad[1, -2:] = True
    row = torch.randint(1, C.R_TOTAL, (B, K), generator=generator)
    row[pad] = 0
    return dict(
        type_in=torch.randn(B, K, 16, generator=generator),
        x_in=torch.rand(B, K, 3, generator=generator),
        lat_in=torch.randn(B, 6, generator=generator),
        pad_mask=pad,
        c_noise=torch.randn(B, generator=generator),
        row=row,
        G=torch.tensor([136, 225]),
    )


@pytest.fixture(scope="module")
def crystalite_base():
    # Only parameter shapes are needed here; the full-size CPU test below uses real weights.
    with torch.device("meta"):
        return CrystaliteModel(**DEFAULT_KW)


@pytest.mark.parametrize("row_mode", ROW_MODES)
@pytest.mark.parametrize("sg_mode", SG_MODES)
def test_build_model_parameter_count(crystalite_base, row_mode, sg_mode):
    cfg = C.TrainConfig(row_embed=row_mode, sg_embed=sg_mode)
    with torch.device("meta"):
        model = build_model(
            cfg,
            torch.empty(C.R_TOTAL, C.ROW_DESC_DIM),
            torch.empty(C.G_TOTAL, C.SG_DESC_DIM),
        )
    d = 512
    row_extra = (
        (C.R_TOTAL * d if row_mode in {"categorical", "hybrid"} else 0)
        + (d * (C.ROW_DESC_DIM + d + 2) if row_mode in {"physical", "hybrid"} else 0)
    )
    sg_extra = (
        (C.G_TOTAL * d if sg_mode in {"categorical", "hybrid"} else 0)
        + (d * (C.SG_DESC_DIM + d + 2) if sg_mode in {"physical", "hybrid"} else 0)
    )
    base_count = sum(p.numel() for p in crystalite_base.parameters())
    model_count = sum(p.numel() for p in model.parameters())
    assert model_count - base_count == row_extra + sg_extra
    assert sum(p.numel() for p in model.row_embed.parameters()) == row_extra
    assert sum(p.numel() for p in model.sg_embed.parameters()) == sg_extra
    base_state, model_state = crystalite_base.state_dict(), model.state_dict()
    assert base_state.keys() <= model_state.keys()
    for name in base_state:
        assert model_state[name].shape == base_state[name].shape
    assert model_state.keys() - base_state.keys() == {
        name for name, _ in model.named_parameters() if name.startswith(("row_embed.", "sg_embed."))
    }
    assert model.type_dim == 16
    assert model.coord_embed.mode == "fourier"
    assert model.coord_embed.n_freqs == 32
    assert model.lattice_embed.mode == "mlp"
    assert model.trunk.lattice_repr == "ltri"
    assert model.heads.coord_head_mode == "direct"
    assert model.trunk.gem is None and model.trunk.gems is None
    print(f"Crystalite: {base_count:,}; added: {model_count - base_count:,}")


@pytest.mark.parametrize("row_mode", ROW_MODES)
@pytest.mark.parametrize("sg_mode", SG_MODES)
def test_embeddings_zero_initialisation_and_buffers(row_mode, sg_mode):
    row_desc, group_desc = _descriptors()
    row_emb = RowEmbedding(row_mode, row_desc.requires_grad_(), 32)
    sg_cond = SGConditioning(sg_mode, group_desc.requires_grad_(), 32)
    for module, ids, buffer_name in (
        (row_emb, torch.arange(C.R_TOTAL).reshape(1, -1), "row_desc"),
        (sg_cond, torch.arange(C.G_TOTAL), "group_desc"),
    ):
        output = module(ids)
        assert output.shape == (*ids.shape, 32)
        assert torch.count_nonzero(output) == 0
        assert buffer_name in dict(module.named_buffers())
        assert buffer_name not in module.state_dict()
        assert not getattr(module, buffer_name).requires_grad
        module.double()
        assert getattr(module, buffer_name).dtype == torch.float64
        assert module(ids).dtype == torch.float64


@pytest.mark.parametrize("row_mode", ROW_MODES)
@pytest.mark.parametrize("sg_mode", SG_MODES)
def test_token_permutation_equivariance(row_mode, sg_mode):
    torch.manual_seed(17)
    row_desc, group_desc = _descriptors()
    model = OrbitolModel(
        row_desc=row_desc, group_desc=group_desc, row_embed=row_mode, sg_embed=sg_mode, **SMALL_KW
    ).eval()
    # Activate embeddings and AdaLN gates: fresh gates would bypass token mixing.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    inputs = _inputs()
    permutation = torch.randperm(20)
    permuted = {
        name: value[:, permutation] if name in {"type_in", "x_in", "pad_mask", "row"} else value
        for name, value in inputs.items()
    }
    for grad in (True, False):
        with torch.set_grad_enabled(grad):
            reference = model(**inputs)
            output = model(**permuted)
        for name in ("type_logits", "coord_vel"):
            torch.testing.assert_close(output[name], reference[name][:, permutation], atol=1e-5, rtol=0)
        torch.testing.assert_close(output["lattice_vel"], reference["lattice_vel"], atol=1e-5, rtol=0)


def test_default_size_cpu_forward_backward():
    torch.manual_seed(19)
    model = build_model(C.TrainConfig(), *_descriptors())
    output = model(**_inputs())
    assert {name: tuple(value.shape) for name, value in output.items()} == {
        "type_logits": (2, 20, 16),
        "coord_vel": (2, 20, 3),
        "lattice_vel": (2, 6),
    }
    loss = sum(value.square().mean() for value in output.values())
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.device.type == "cpu"
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert torch.count_nonzero(model.row_embed.residual.weight.grad) > 0
    assert torch.count_nonzero(model.row_embed.mlp[-1].weight.grad) > 0


def test_nonzero_backbone_parity_and_gradients():
    torch.manual_seed(23)
    base = CrystaliteModel(**SMALL_KW).eval()
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    row_desc, group_desc = _descriptors()
    model = OrbitolModel(row_desc=row_desc, group_desc=group_desc, **SMALL_KW).eval()
    loaded = model.load_state_dict(base.state_dict(), strict=False)
    assert loaded.unexpected_keys == []
    assert set(loaded.missing_keys) == {
        name for name, _ in model.named_parameters() if name.startswith(("row_embed.", "sg_embed."))
    }
    inputs = _inputs()
    base_inputs = [inputs[name].clone() for name in ("type_in", "x_in", "lat_in", "pad_mask", "c_noise")]
    for value in base_inputs:
        if value.is_floating_point():
            value.requires_grad_()
    orbit_inputs = [value.detach().clone().requires_grad_(value.requires_grad) for value in base_inputs]
    for grad in (True, False):
        with torch.set_grad_enabled(grad):
            reference = base(*base_inputs)
            output = model(*orbit_inputs, row=inputs["row"], G=inputs["G"])
        for name in reference:
            torch.testing.assert_close(output[name], reference[name], atol=0, rtol=0)
        if grad:
            sum(value.square().sum() for value in reference.values()).backward()
            sum(value.square().sum() for value in output.values()).backward()
    for base_input, orbit_input in zip(base_inputs, orbit_inputs):
        if base_input.requires_grad:
            torch.testing.assert_close(orbit_input.grad, base_input.grad, atol=0, rtol=0)
    model_parameters = dict(model.named_parameters())
    for name, parameter in base.named_parameters():
        torch.testing.assert_close(model_parameters[name].grad, parameter.grad, atol=0, rtol=0)
    assert torch.count_nonzero(model.sg_embed.residual.weight.grad) > 0
    assert torch.count_nonzero(model.sg_embed.mlp[-1].weight.grad) > 0


@pytest.mark.parametrize("flag", ["use_edge_bias", "use_distance_bias"])
def test_reject_gem(flag):
    row_desc, group_desc = _descriptors()
    with pytest.raises(ValueError, match="edge and distance biases are not supported"):
        OrbitolModel(row_desc=row_desc, group_desc=group_desc, **{**SMALL_KW, flag: True})
