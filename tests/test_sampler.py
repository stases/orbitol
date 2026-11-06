"""P1 trajectory parity and constrained orbit state checks."""
from types import SimpleNamespace

import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol.sampler import orbit_edm_sampler
from src.crystalite.crystalite import CrystaliteModel
from src.crystalite.sampler import edm_sampler
from test_edm import batch, tables, Stub


def settings(aa=0, steps=8):
    return SimpleNamespace(num_steps=steps, sigma_min=0.002, sigma_max=80., rho=7., S_churn=60.,
                           S_min=0., S_max=999., S_noise=1.003, sigma_data=(0.3,) * 3,
                           aa_rho_coords=aa, aa_rho_lattice=aa, aa_rho_types=0.)


class Contract(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, ty, x, lat, pad, cnoise, *, row, G):
        return self.base(ty, x, lat, pad, cnoise)


@pytest.mark.parametrize("aa", [0., 10.])
def test_p1_bitwise_sampler_parity(aa):
    torch.manual_seed(23)
    base = CrystaliteModel(d_model=32, n_heads=4, n_layers=2, vz=94, type_dim=16, n_freqs=32,
                           coord_embed_mode="fourier", lattice_embed_mode="mlp", lattice_repr="ltri",
                           use_edge_bias=False, use_distance_bias=False).eval()
    b, tb, s = batch(), tables(), settings(aa)
    b["family"][:] = 0
    b["row"] = (~b["pad_mask"]).long()
    b["dof_mask"] = (~b["pad_mask"])[..., None].expand(-1, -1, 3)
    ref = edm_sampler(base, b["pad_mask"], 16, s.num_steps, s.sigma_min, s.sigma_max, s.rho,
                       s.S_churn, s.S_min, s.S_max, s.S_noise, *s.sigma_data,
                       generator=torch.Generator().manual_seed(5), aa_rho_coords=aa,
                       aa_rho_lattice=aa, lattice_repr="ltri")
    got = orbit_edm_sampler(Contract(base), b, tb, objective="M", settings=s,
                            generator=torch.Generator().manual_seed(5))
    for k, rk in (("type", "type"), ("u", "frac"), ("z", "lat")):
        torch.testing.assert_close(got[k], ref[rk], atol=0, rtol=0)
    assert got["finite"].all()


@pytest.mark.parametrize("objective", ["M", "P"])
def test_sampler_bounds_frozen_and_determinism(objective):
    b, tb = batch(), tables()
    outputs = [orbit_edm_sampler(Stub(), b, tb, objective=objective, settings=settings(),
                                 generator=torch.Generator().manual_seed(8)) for _ in range(2)]
    for key in outputs[0]:
        assert torch.equal(outputs[0][key], outputs[1][key])
    assert outputs[0]["u"].abs().max() <= 0.5
    assert not outputs[0]["u"][~b["dof_mask"]].any()
    assert not outputs[0]["type"][b["pad_mask"]].any()


def test_analytic_hook_step_count_and_zero_dof():
    b, tb = batch(), tables()
    seen = []
    def oracle(state, sigma, skel, tables, *, objective, sigma_data):
        seen.append(sigma.clone())
        assert not state["u_noisy"][~b["dof_mask"]].any()
        return {"D_type": state["type_noisy"], "D_u": state["u_noisy"], "D_z": state["z_noisy"]}
    out = orbit_edm_sampler(None, b, tb, objective="P", settings=settings(), denoise_fn=oracle,
                            generator=torch.Generator().manual_seed(7))
    assert len(seen) == 15
    assert not out["u"][~b["dof_mask"]].any()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("key,index,finite", [
    ("D_type", (0, 0, 0), False),
    ("D_type", (0, 2, 0), False),  # A real zero-dof token still has type features.
    ("D_u", (0, 1, 0), False),
    ("D_z", (0, 5), False),
    ("D_type", (0, 3, 0), True),  # Padding and inactive slots are ignored.
    ("D_u", (0, 1, 1), True),
    ("D_u", (0, 2, 0), True),
    ("D_u", (0, 3, 0), True),
    ("D_z", (0, 1), True),
])
def test_sampler_finite_before_sanitisation(bad_value, key, index, finite):
    b, tb = batch(), tables()
    b["family"][0] = 4

    def sample(value):
        def oracle(state, sigma, skel, tables, *, objective, sigma_data):
            den = {"D_type": torch.zeros_like(state["type_noisy"]),
                   "D_u": torch.zeros_like(state["u_noisy"]),
                   "D_z": torch.zeros_like(state["z_noisy"])}
            den[key][index] = value
            return den
        generator = torch.Generator().manual_seed(7)
        out = orbit_edm_sampler(None, b, tb, objective="P", settings=settings(steps=2),
                                denoise_fn=oracle, generator=generator)
        return out, generator.get_state()

    clean, clean_rng = sample(0.0)
    out, rng = sample(bad_value)
    assert out["finite"].dtype == torch.bool
    assert out["finite"].shape == (2,)
    assert out["finite"].tolist() == [finite, True]
    assert torch.equal(rng, clean_rng)
    for name in ("type", "u", "z"):
        assert torch.isfinite(out[name]).all()
        torch.testing.assert_close(out[name][1], clean[name][1], atol=0, rtol=0)
        if finite:
            torch.testing.assert_close(out[name], clean[name], atol=0, rtol=0)
