"""Frozen acceptance tests for orbitol.model."""
import pytest
import torch

from orbitol import contracts as C
from orbitol.model import OrbitolModel
from src.crystalite.crystalite import CrystaliteModel

KW = dict(d_model=32, n_heads=4, n_layers=2, vz=94, type_dim=16, n_freqs=32, coord_embed_mode="fourier",
          lattice_embed_mode="mlp", lattice_repr="ltri", use_edge_bias=False, use_distance_bias=False,
          coord_head_mode="direct")


def _pair(row_mode, sg_mode):
    torch.manual_seed(0)
    base = CrystaliteModel(**KW)
    g = torch.Generator().manual_seed(5)
    row_desc = torch.randn(C.R_TOTAL, C.ROW_DESC_DIM, generator=g)
    group_desc = torch.randn(C.G_TOTAL, C.SG_DESC_DIM, generator=g)
    torch.manual_seed(1)
    m = OrbitolModel(row_desc=row_desc, group_desc=group_desc, row_embed=row_mode, sg_embed=sg_mode, **KW)
    res = m.load_state_dict(base.state_dict(), strict=False)
    assert list(res.unexpected_keys) == []
    return base.eval(), m.eval()


def _inputs(B=3, K=7):
    g = torch.Generator().manual_seed(2)
    type_in = torch.randn(B, K, 16, generator=g)
    x_in = torch.rand(B, K, 3, generator=g)
    lat_in = torch.randn(B, 6, generator=g)
    c_noise = 0.25 * torch.randn(B, generator=g)
    pad = torch.zeros(B, K, dtype=torch.bool)
    pad[0, 5:] = True
    pad[2, 3:] = True
    return type_in, x_in, lat_in, pad, c_noise


@pytest.mark.parametrize("row_mode", ["categorical", "physical", "hybrid"])
@pytest.mark.parametrize("sg_mode", ["none", "categorical", "physical", "hybrid"])
def test_p1_equivalence_at_init(row_mode, sg_mode):
    base, m = _pair(row_mode, sg_mode)
    type_in, x_in, lat_in, pad, c_noise = _inputs()
    row = torch.where(pad, torch.zeros_like(pad, dtype=torch.long), torch.ones_like(pad, dtype=torch.long))
    G = torch.ones(type_in.shape[0], dtype=torch.long)
    for grad in (True, False):
        with torch.set_grad_enabled(grad):
            ref = base(type_in, x_in, lat_in, pad, c_noise)
            out = m(type_in, x_in, lat_in, pad, c_noise, row=row, G=G)
        for k in ("type_logits", "coord_vel", "lattice_vel"):
            torch.testing.assert_close(out[k], ref[k], atol=0, rtol=0)


@pytest.mark.parametrize("row_mode", ["categorical", "physical", "hybrid"])
@pytest.mark.parametrize("sg_mode", ["categorical", "physical", "hybrid"])
def test_embeddings_are_wired(row_mode, sg_mode):
    _, m = _pair(row_mode, sg_mode)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    type_in, x_in, lat_in, pad, c_noise = _inputs()
    row = torch.full(pad.shape, 300, dtype=torch.long)
    row[pad] = 0
    G = torch.tensor([136, 136, 221])
    with torch.no_grad():
        out = m(type_in, x_in, lat_in, pad, c_noise, row=row, G=G)
        row2 = row.clone()
        row2[1, 0] = 301
        out_row = m(type_in, x_in, lat_in, pad, c_noise, row=row2, G=G)
        G2 = G.clone()
        G2[2] = 225
        out_g = m(type_in, x_in, lat_in, pad, c_noise, row=row, G=G2)
    assert not torch.allclose(out_row["coord_vel"][1], out["coord_vel"][1])
    torch.testing.assert_close(out_row["coord_vel"][[0, 2]], out["coord_vel"][[0, 2]], atol=0, rtol=0)
    assert not torch.allclose(out_g["coord_vel"][2], out["coord_vel"][2])
    torch.testing.assert_close(out_g["coord_vel"][:2], out["coord_vel"][:2], atol=0, rtol=0)


def test_padding_content_is_ignored():
    _, m = _pair("hybrid", "hybrid")
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    type_in, x_in, lat_in, pad, c_noise = _inputs()
    row = torch.where(pad, torch.zeros_like(pad, dtype=torch.long), torch.full_like(pad, 300, dtype=torch.long))
    G = torch.tensor([136, 136, 221])
    with torch.no_grad():
        out = m(type_in, x_in, lat_in, pad, c_noise, row=row, G=G)
        t2, x2, r2 = type_in.clone(), x_in.clone(), row.clone()
        t2[pad] = 7.0
        x2[pad] = 0.37
        r2[pad] = 1234
        out2 = m(t2, x2, lat_in, pad, c_noise, row=r2, G=G)
    real = ~pad
    for k in ("type_logits", "coord_vel"):
        torch.testing.assert_close(out2[k][real], out[k][real], atol=0, rtol=0)
    torch.testing.assert_close(out2["lattice_vel"], out["lattice_vel"], atol=0, rtol=0)
