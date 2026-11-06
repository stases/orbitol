"""Small hand-built tables keep these tests independent of table generation."""
import math
from types import SimpleNamespace

import pytest
import torch

from orbitol import edm, lattice


def tables():
    eye = torch.eye(3).repeat(4, 1, 1)
    A = eye.clone()
    A[2, :, 1:] = 0
    A[3] = 0
    vr = torch.zeros(4, 14, 3)
    vr[2, 0, 0], vr[2, 1, 0] = 0.5, -0.5
    basis = eye.clone()
    basis[2, 0, 0] = 0.5
    return SimpleNamespace(row_dof=torch.tensor([3, 3, 1, 0]), row_A=A, row_b=torch.zeros(4, 3),
                           row_H=eye.clone(), row_Hinv=eye.clone(), row_Pi=A.clone(), row_N=A.clone(),
                           row_wrap_fast=torch.tensor([True, True, False, True]), row_lambda_basis=basis,
                           row_vrv=vr, row_vrv_count=torch.tensor([0, 0, 2, 0]),
                           group_family=torch.zeros(231, dtype=torch.long))


def batch():
    row = torch.tensor([[1, 2, 3, 0], [1, 1, 0, 0]])
    pad = row == 0
    dm = (torch.arange(3) < tables().row_dof[row, None]) & (~pad)[..., None]
    return dict(row=row, pad_mask=pad, dof_mask=dm, family=torch.tensor([0, 4]), G=torch.ones(2, dtype=torch.long),
                U=torch.rand(2, 4, 3) * dm, lat=torch.zeros(2, 6))


class Stub(torch.nn.Module):
    def forward(self, ty, x, lat, pad, cnoise, *, row, G):
        return dict(type_logits=ty * 0.1, coord_vel=x.sin(), lattice_vel=lat * 0.2)


def test_noise_order_and_global_stream():
    b = batch()
    torch.manual_seed(17)
    want = [torch.randn_like(torch.empty(2, 4, 16)), torch.randn_like(b["U"]), torch.randn_like(b["lat"])]
    torch.manual_seed(17)
    actual = edm.orbit_noise(b)
    for x, y in zip(actual.values(), want):
        assert torch.equal(x, y)
    gen = torch.Generator().manual_seed(17)
    for x, y in zip(edm.orbit_noise(b, gen).values(), want):
        assert torch.equal(x, y)


@pytest.mark.parametrize("objective", ["M", "P"])
def test_masks_shapes_and_zero_dof(objective):
    b, tb = batch(), tables()
    sigma = torch.tensor([0.2, 2.0])
    state = edm.corrupt(b, torch.zeros(2, 4, 16), sigma, edm.orbit_noise(b), tb, objective=objective)
    den = edm.denoise_orbit(Stub(), state, sigma, b, tb, objective=objective, sigma_data=(0.3,) * 3)
    assert den["D_type"].shape == (2, 4, 16)
    assert den["D_u"].shape == (2, 4, 3)
    assert den["D_z"].shape == (2, 6)
    assert not state["u_noisy"][~b["dof_mask"]].any()
    assert not den["D_u"][0, 2].any()
    assert not den["D_z"][~lattice.free_mask(b["family"])].any()
    if objective == "P":
        assert state["u_noisy"].abs().max() <= 0.5


def test_periodic_loss_has_correct_unwrapped_target():
    b, tb = batch(), tables()
    sigma = torch.tensor([0.15, 0.7])
    s = edm.corrupt(b, torch.zeros(2, 4, 16), sigma, edm.orbit_noise(b), tb, objective="P")
    target = edm.periodic_target(s["u_noisy"], s["u_clean"], tb.row_H[b["row"]], tb.row_dof[b["row"]], sigma)
    perfect = dict(D_type=s["type_clean"], D_u=target, D_z=s["z_clean"])
    shifted = dict(perfect, D_u=target + b["dof_mask"].float())
    args = dict(objective="P", sigma_data=(0.3,) * 3, loss_weights=(1, 50, 5))
    good = edm.compute_orbit_loss(perfect, s, sigma, b, tb, **args)
    bad = edm.compute_orbit_loss(shifted, s, sigma, b, tb, **args)
    assert good["total"] == 0
    assert torch.isfinite(bad["total"]) and bad["F"] > good["F"]
    # A whole period remains an error under P.
    assert bad["F"] > 1


def test_lattice_readouts_weights_clamps_and_parameters():
    family = torch.arange(6)
    z = torch.tensor([[0.9, 0.3, 0.4, -0.2, 0.1, 1.2]]).repeat(6, 1)
    p = lattice.params(z, family)
    L = lattice.to_matrix(z, family)
    assert L.dtype == p.dtype == torch.float64
    assert p[3, 0] == p[3, 1] and p[5, 0] == p[5, 1] == p[5, 2]
    assert p[4, 0] == p[4, 1] and p[4, 5] == 120
    assert torch.all(p[2:, 3:5] == 90)
    torch.testing.assert_close(L.square().sum(-1).sqrt(), p[:, :3], atol=1e-14, rtol=1e-14)
    F = torch.arange(6.0).expand(6, 6)
    out = lattice.readout(F, family)
    assert out[3, 0] == out[4, 0] == 1 and out[5, 0] == 7 / 3
    w = lattice.loss_weights(family)
    assert torch.equal(w.sum(-1), torch.tensor([6., 4., 3., 3., 3., 3.]))
    clamped = lattice.end_clamp(z * 100, family)
    assert not clamped[~lattice.free_mask(family)].any()
    assert clamped[0, 0] == 5 and clamped[0, 3] == -10
    # Inactive hexagonal branches must not poison gradients at noisy p0 >> 1.
    big = (z * 100).requires_grad_()
    lattice.ell(big, family).sum().backward()
    assert torch.isfinite(big.grad).all()


def test_target_at_switch_and_very_small_sigma():
    h = torch.tensor([[[4 / 3, -2 / 3, 0], [-2 / 3, 4 / 3, 0], [0, 0, 1.0]]], dtype=torch.float64)
    u = torch.tensor([[0.49999, -0.49, 0.0]], dtype=torch.float64)
    uc = torch.tensor([[-0.01, 0.02, 0.0]], dtype=torch.float64)
    sw = 0.4 * math.sqrt(2)
    a = edm.periodic_target(u, uc, h, torch.tensor([2]), torch.tensor([sw * (1 - 1e-10)], dtype=torch.float64))
    b = edm.periodic_target(u, uc, h, torch.tensor([2]), torch.tensor([sw * (1 + 1e-10)], dtype=torch.float64))
    torch.testing.assert_close(a, b, atol=1e-9, rtol=0)
    c = edm.periodic_target(u, uc, h, torch.tensor([2]), torch.tensor([1e-6], dtype=torch.float64))
    assert torch.isfinite(c).all()


def test_all_zero_dof_coordinate_loss_is_zero():
    b, tb = batch(), tables()
    b["row"][:] = 3
    b["dof_mask"][:] = False
    sigma = torch.ones(2)
    s = edm.corrupt(b, torch.zeros(2, 4, 16), sigma, edm.orbit_noise(b), tb, objective="P")
    den = edm.denoise_orbit(Stub(), s, sigma, b, tb, objective="P", sigma_data=(0.3,) * 3)
    loss = edm.compute_orbit_loss(den, s, sigma, b, tb, objective="P", sigma_data=(0.3,) * 3, loss_weights=(1, 50, 5))
    assert loss["F"] == 0 and torch.isfinite(loss["total"])
