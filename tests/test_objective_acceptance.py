"""Frozen acceptance tests for orbitol.edm and orbitol.lattice."""
import itertools
import math
import types

import numpy as np
import pytest
import torch

from orbitol import contracts as C
from orbitol import edm, lattice
from src.crystalite.crystalite import CrystaliteModel
from src.crystalite.edm_utils import compute_edm_loss, denoise_edm, sample_sigma

MG_HEX = np.array([[4 / 3, -2 / 3, 0], [-2 / 3, 4 / 3, 0], [0, 0, 1.0]])


def _sqrtm(M):
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(w)) @ V.T


def _vrvs(B, H):
    d = B.shape[1]
    out = []
    for c in itertools.product(range(-2, 3), repeat=d):
        if not any(c):
            continue
        v = B @ np.array(c, float)
        ok = True
        for w in itertools.product(range(-3, 4), repeat=d):
            ww = B @ np.array(w, float)
            if np.allclose(ww, 0) or np.allclose(ww, v):
                continue
            e = v - 2 * ww
            if e @ H @ e <= v @ H @ v + 1e-12:
                ok = False
                break
        if ok:
            out.append(v)
    return out


def _row(A_cols, b, MG, lam=None):
    d = A_cols.shape[1]
    A = np.zeros((3, 3))
    A[:, :d] = A_cols
    H = np.eye(3)
    Pi, N, basis, vr = np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((C.MAX_VRV, 3))
    nv = 0
    if d:
        Ad = A[:, :d]
        Hd = Ad.T @ MG @ Ad
        H[:d, :d] = Hd
        Pi[:d] = np.linalg.solve(Hd, Ad.T @ MG)
        N[:d] = np.linalg.solve(Hd, Ad.T @ _sqrtm(MG))
        if lam is not None:
            basis[:d, :d] = lam
        vs = _vrvs(basis[:d, :d], Hd)
        nv = len(vs)
        for i, v in enumerate(vs):
            vr[i, :d] = v
    fast = lam is None and np.allclose(H, np.diag(np.diag(H)))
    return dict(dof=d, A=A, b=np.asarray(b, float), H=H, Hinv=np.linalg.inv(H), Pi=Pi, N=N, basis=basis, vrv=vr,
                nvrv=nv, fast=fast)


ROWS = [
    _row(np.eye(3), [0, 0, 0], np.eye(3)),                                       # 0 pad (P1 chart)
    _row(np.eye(3), [0, 0, 0], np.eye(3)),                                       # 1 P1 1a
    _row(np.array([[1.0], [1.0], [0.0]]), [0, 0, 0], np.eye(3)),                 # 2 rutile 4f (x, x, 0)
    _row(np.array([[1.0], [2.0], [0.0]]), [0, 0, 0.25], MG_HEX),                 # 3 P6_3/mmc 6h (x, 2x, 1/4)
    _row(np.array([[0.0], [0.0], [1.0]]), [0, 0, 0], np.eye(3), lam=np.array([[0.5]])),  # 4 Pcc2 2a, Lambda = Z/2
    _row(np.eye(3), [0, 0, 0], MG_HEX),                                          # 5 hexagonal general position
    _row(np.zeros((3, 0)), [0, 0, 0.5], np.eye(3)),                              # 6 a d = 0 row
]


def _tables(dtype):
    t = lambda k, dt=dtype: torch.tensor(np.stack([r[k] for r in ROWS]), dtype=dt)  # noqa: E731
    fam = torch.zeros(C.G_TOTAL, dtype=torch.long)  # every group triclinic in this fixture
    return types.SimpleNamespace(
        row_dof=t("dof", torch.long), row_A=t("A"), row_b=t("b"), row_H=t("H"), row_Hinv=t("Hinv"), row_Pi=t("Pi"),
        row_N=t("N"), row_wrap_fast=t("fast", torch.bool), row_lambda_basis=t("basis"), row_vrv=t("vrv"),
        row_vrv_count=t("nvrv", torch.long), group_family=fam)


TB = _tables(torch.float32)
TB64 = _tables(torch.float64)


class Contract(torch.nn.Module):
    """CrystaliteModel behind Orbitol's model-call contract (GEM off, so row and G are unused)."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, type_in, x_in, lat_in, pad_mask, c_noise, *, row, G):
        return self.base(type_in, x_in, lat_in, pad_mask, c_noise)


def _p1_batch(B=4, K=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pad = torch.zeros(B, K, dtype=torch.bool)
    pad[1, 4:] = True
    pad[3, 2:] = True
    row = torch.where(pad, torch.zeros_like(pad, dtype=torch.long), torch.ones_like(pad, dtype=torch.long))
    U = torch.rand(B, K, 3, generator=g)
    U[pad] = 0
    batch = {"G": torch.ones(B, dtype=torch.long), "family": torch.zeros(B, dtype=torch.long), "row": row, "U": U,
             "dof_mask": (~pad)[..., None].expand(B, K, 3).clone(), "pad_mask": pad,
             "lat": 0.3 * torch.randn(B, 6, generator=g)}
    type_clean = torch.randn(B, K, 16, generator=g)
    return batch, type_clean


def test_objective_M_equals_crystalite_at_p1():
    torch.manual_seed(0)
    base = CrystaliteModel(d_model=32, n_heads=4, n_layers=2, vz=94, type_dim=16, n_freqs=32,
                           coord_embed_mode="fourier", lattice_embed_mode="mlp", lattice_repr="ltri",
                           use_edge_bias=False, use_distance_bias=False).eval()
    model = Contract(base)
    batch, type_clean = _p1_batch()
    B, K = batch["row"].shape
    g = torch.Generator().manual_seed(3)
    sigma = sample_sigma(B, torch.device("cpu"), -1.2, 1.2, 0.002, 80.0, generator=g)
    noise = {"type": torch.randn(B, K, 16, generator=g), "coord": torch.randn(B, K, 3, generator=g),
             "lat": torch.randn(B, 6, generator=g)}
    pad = batch["pad_mask"]
    frac_c = batch["U"] - 0.5
    frac_noisy = torch.where((~pad)[..., None], frac_c + sigma[:, None, None] * noise["coord"], torch.zeros_like(frac_c))
    type_noisy = type_clean + sigma[:, None, None] * noise["type"]
    lat_noisy = batch["lat"] + sigma[:, None] * noise["lat"]
    with torch.no_grad():
        ref = denoise_edm(base, type_noisy, frac_noisy, lat_noisy, pad, sigma, 0.3, 0.3, 0.3, 0.002, 80.0)
        ref_loss = compute_edm_loss(ref, {"type": type_clean, "frac_c": frac_c, "lat": batch["lat"]}, frac_noisy, sigma,
                                    pad, 0.3, 0.3, 0.3, {"A": 1.0, "F": 50.0, "Y": 5.0}, coord_loss_mode="frac_mse",
                                    lattice_repr="ltri")
        state = edm.corrupt(batch, type_clean, sigma, noise, TB, objective="M")
        torch.testing.assert_close(state["u_noisy"], frac_noisy, atol=0, rtol=0)
        den = edm.denoise_orbit(model, state, sigma, batch, TB, objective="M", sigma_data=(0.3, 0.3, 0.3))
        loss = edm.compute_orbit_loss(den, state, sigma, batch, TB, objective="M", sigma_data=(0.3, 0.3, 0.3),
                                      loss_weights=(1.0, 50.0, 5.0))
    torch.testing.assert_close(den["D_type"], ref["type"], atol=0, rtol=0)
    torch.testing.assert_close(den["D_u"], ref["frac"], atol=0, rtol=0)
    torch.testing.assert_close(den["D_z"], ref["lat"], atol=0, rtol=0)
    for k_o, k_r in (("total", "loss_total"), ("A", "loss_a"), ("F", "loss_f"), ("Y", "loss_y")):
        torch.testing.assert_close(loss[k_o], ref_loss[k_r], atol=0, rtol=0)


def _brute_target(u, uc, H, sigma, radius):
    d = len(u)
    num, den = np.zeros(d), 0.0
    terms = []
    for n in itertools.product(range(-radius, radius + 1), repeat=d):
        e = u - uc - np.array(n, float)
        terms.append((-0.5 * (e @ H @ e) / sigma**2, e))
    m = max(t[0] for t in terms)
    for lw, e in terms:
        w = math.exp(lw - m)
        num += w * e
        den += w
    return u - num / den


@pytest.mark.parametrize("sigma", [0.005, 0.05, 0.2, 0.5, 1.0, 2.0, 80.0])
@pytest.mark.parametrize("r", [2, 3, 5, 1])
def test_periodic_target_is_exact(sigma, r):
    rng = np.random.default_rng(4)
    d = ROWS[r]["dof"]
    H = ROWS[r]["H"][:d, :d]
    n = 6
    u = np.zeros((n, 3))
    uc = np.zeros((n, 3))
    u[:, :d] = rng.uniform(-0.5, 0.5, size=(n, d))
    uc[:, :d] = rng.uniform(-0.5, 0.5, size=(n, d))
    got = edm.periodic_target(torch.tensor(u), torch.tensor(uc), torch.tensor(np.stack([ROWS[r]["H"]] * n)),
                              torch.full((n,), d), torch.full((n,), sigma, dtype=torch.float64)).numpy()
    assert got.dtype == np.float64
    for i in range(n):
        if sigma >= 5:
            np.testing.assert_allclose(got[i, :d], u[i, :d], atol=1e-9)
            continue
        radius = int(math.ceil(8 * sigma + 2)) if d < 3 else int(math.ceil(6 * sigma + 2))
        want = _brute_target(u[i, :d], uc[i, :d], H, sigma, radius)
        np.testing.assert_allclose(got[i, :d], want, atol=1e-9)
        assert not np.any(got[i, d:])


class Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.lin = torch.nn.Linear(6 + 16, 3 + 16)

    def forward(self, type_in, x_in, lat_in, pad_mask, c_noise, *, row, G):
        h = self.lin(torch.cat([torch.sin(2 * math.pi * x_in), torch.cos(2 * math.pi * x_in), type_in], -1))
        return {"type_logits": h[..., 3:], "coord_vel": h[..., :3], "lattice_vel": 0.5 * lat_in}


def test_objective_P_is_lift_equivariant_and_M_is_not():
    K = 6
    rows = torch.tensor([[1, 2, 3, 4, 5, 6]])
    dof = TB.row_dof[rows]
    dmask = torch.arange(3)[None, None, :] < dof[..., None]
    g = torch.Generator().manual_seed(11)
    batch = {"G": torch.ones(1, dtype=torch.long), "family": torch.zeros(1, dtype=torch.long), "row": rows,
             "U": torch.rand(1, K, 3, generator=g) * dmask, "dof_mask": dmask, "pad_mask": torch.zeros(1, K, dtype=torch.bool),
             "lat": 0.3 * torch.randn(1, 6, generator=g)}
    type_clean = torch.randn(1, K, 16, generator=g)
    sigma = torch.tensor([0.3])
    noise = {"type": torch.randn(1, K, 16, generator=g), "coord": torch.randn(1, K, 3, generator=g),
             "lat": torch.randn(1, 6, generator=g)}
    shift = (torch.randint(-3, 4, (1, K, 3), generator=g) * dmask).float()
    model = Stub()
    for objective in ("P", "M"):
        state = edm.corrupt(batch, type_clean, sigma, noise, TB, objective=objective)
        shifted = dict(state)
        shifted["u_noisy"] = state["u_noisy"] + shift
        with torch.no_grad():
            d0 = edm.denoise_orbit(model, state, sigma, batch, TB, objective=objective, sigma_data=(0.3, 0.3, 0.3))
            d1 = edm.denoise_orbit(model, shifted, sigma, batch, TB, objective=objective, sigma_data=(0.3, 0.3, 0.3))
        diff = d1["D_u"] - d0["D_u"]
        if objective == "P":
            torch.testing.assert_close(diff, shift, atol=1e-5, rtol=0)
        else:
            assert not torch.allclose(diff, shift, atol=1e-3)
        assert not torch.any(d0["D_u"][~dmask])  # frozen coordinates stay exactly zero


def test_exact_wrap_matches_brute_force():
    rng = np.random.default_rng(5)
    for r in (2, 3, 4, 5):
        d = ROWS[r]["dof"]
        basis, H = ROWS[r]["basis"][:d, :d], ROWS[r]["H"][:d, :d]
        n = 24
        delta = np.zeros((n, 3))
        delta[:, :d] = rng.uniform(-3, 3, size=(n, d))
        got = edm.wrap_exact(torch.tensor(delta), torch.full((n,), r), TB64).numpy()
        rad = 6 if d < 3 else 5
        for i in range(n):
            best = min((delta[i, :d] - basis @ np.array(c, float) for c in itertools.product(range(-rad, rad + 1), repeat=d)),
                       key=lambda e: e @ H @ e)
            e = got[i, :d]
            assert abs(e @ H @ e - best @ H @ best) < 1e-9, (r, i)
            coeff = np.linalg.solve(basis, delta[i, :d] - e)
            np.testing.assert_allclose(coeff, np.round(coeff), atol=1e-9)


def test_lattice_family_charts():
    z = torch.tensor([[0.9, 0.3, 0.4, -0.2, 0.1, 1.2]] * 6, dtype=torch.float32)
    fam = torch.arange(6)
    ell = lattice.ell(z, fam)
    tri, mono, ortho, tet, hexa, cub = (ell[i] for i in range(6))
    torch.testing.assert_close(tri, z[0], atol=0, rtol=0)
    assert mono[1] == 0 and mono[4] == 0
    assert ortho[1] == ortho[3] == ortho[4] == 0
    assert tet[2] == tet[0] and tet[1] == tet[3] == tet[4] == 0
    assert cub[2] == cub[0] and cub[5] == cub[0]
    p0 = min(max(0.9, -5 - math.log(math.sqrt(3) / 2)), math.log(20))
    torch.testing.assert_close(hexa[0], torch.tensor(p0))
    torch.testing.assert_close(hexa[2], torch.tensor(p0 + math.log(math.sqrt(3) / 2)))
    torch.testing.assert_close(hexa[1], torch.tensor(-0.5 * math.exp(p0)))
    big = lattice.ell(torch.tensor([[5.0, 0, 0, 0, 0, 1.0]]), torch.tensor([4]))[0]
    torch.testing.assert_close(big[0], torch.tensor(math.log(20.0)))  # p0 -> p0' (clamped)
    for f, slots in enumerate(C.FREE_SLOTS.values()):
        mask = lattice.free_mask(torch.tensor([f]))[0]
        assert tuple(torch.nonzero(mask).flatten().tolist()) == slots
