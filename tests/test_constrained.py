"""Denoiser replacement, trajectory noise, and pool-level element constraints."""
from dataclasses import replace

import pytest
import torch

from orbitol import lattice, sample
from orbitol.constrained import (Constraint, ConstrainedSampler, FixedElements,
                                 constrain_denoiser, constrained_edm_sampler, element_constraint)
from orbitol.decode import decode_types
from orbitol.edm import wrap_exact
from orbitol.packed import PackedDenoiser
from orbitol.sampler import orbit_edm_sampler
from test_edm import Stub, batch, tables
from test_sampler import settings


def oracle(state, sigma, skel, tb, *, objective, sigma_data):
    return {"D_" + block: 0.5 * state[block + "_noisy"] for block in ("type", "u", "z")}


def mixed_constraint(b):
    ty = torch.zeros_like(b["pad_mask"])
    ty[:, 0] = True
    ty[0, 2] = True  # Zero chart degrees of freedom still permits a fixed element.
    u = torch.zeros_like(b["dof_mask"])
    u[0, 0, 0] = u[0, 1, 0] = True  # Periods 1 and 1/2, respectively.
    z = torch.zeros_like(b["lat"], dtype=torch.bool)
    z[:, 0] = True
    return Constraint(ty, torch.full((*ty.shape, 16), .37),
                      u, torch.full_like(b["U"], .49), z, torch.full_like(b["lat"], .8))


@pytest.mark.parametrize("objective", ["M", "P"])
@pytest.mark.parametrize("empty", [True, False])
@pytest.mark.parametrize("mapping", [True, False])
def test_noop_bitwise_with_churn(objective, empty, mapping):
    b, tb, s = batch(), tables(), settings()
    s.skip_type_scaling = True
    s.autocast_dtype = torch.bfloat16
    if mapping:
        s = vars(s)
    c = Constraint() if empty else Constraint(
        torch.zeros_like(b["pad_mask"]), torch.ones(2, 4, 16, dtype=torch.float64),
        torch.zeros_like(b["dof_mask"]), torch.ones_like(b["U"], dtype=torch.float64),
        torch.zeros_like(b["lat"], dtype=torch.bool), torch.ones_like(b["lat"], dtype=torch.float64))
    g1, g2 = (torch.Generator().manual_seed(91) for _ in range(2))
    ref = orbit_edm_sampler(Stub(), b, tb, objective=objective, settings=s, generator=g1)
    out = constrained_edm_sampler(Stub(), b, tb, constraint=c, objective=objective, settings=s, generator=g2)
    assert ref.keys() == out.keys()
    for key in ref:
        assert torch.equal(ref[key], out[key]), key
    assert torch.equal(g1.get_state(), g2.get_state())


@pytest.mark.parametrize("objective", ["M", "P", "posterior", "posterior_periodic"])
def test_exact_fixed_and_bitwise_free_entries(objective):
    b, tb = batch(), tables()
    c = mixed_constraint(b)
    g1, g2 = (torch.Generator().manual_seed(7) for _ in range(2))
    ref = orbit_edm_sampler(None, b, tb, objective=objective, settings=settings(), generator=g1, denoise_fn=oracle)
    out = constrained_edm_sampler(None, b, tb, constraint=c, objective=objective,
                                   settings=settings(), generator=g2, denoise_fn=oracle)
    for block in ("type", "z"):
        mask, value = getattr(c, block + "_mask"), getattr(c, block + "_value")
        torch.testing.assert_close(out[block][mask], value[mask], atol=1e-6, rtol=0)
    residual = wrap_exact(out["u"] - c.u_value, b["row"], tb)
    torch.testing.assert_close(residual[c.u_mask], torch.zeros_like(residual[c.u_mask]), atol=1e-6, rtol=0)
    for block in ("type", "u", "z"):
        free = ~getattr(c, block + "_mask")
        assert torch.equal(out[block][free], ref[block][free]), block
    assert torch.equal(out["finite"], ref["finite"])
    assert torch.equal(g1.get_state(), g2.get_state())


def test_value_precision_preserves_sampler_dtypes_and_draws():
    b, tb = batch(), tables()
    c = mixed_constraint(b)
    wide = replace(c, type_value=c.type_value.double(), u_value=c.u_value.double(), z_value=c.z_value.double())
    g1, g2 = (torch.Generator().manual_seed(73) for _ in range(2))
    ref = constrained_edm_sampler(None, b, tb, constraint=c, objective="P", settings=settings(),
                                   generator=g1, denoise_fn=oracle)
    out = constrained_edm_sampler(None, b, tb, constraint=wide, objective="P", settings=settings(),
                                   generator=g2, denoise_fn=oracle)
    for key in ref:
        assert out[key].dtype == ref[key].dtype
        assert torch.equal(out[key], ref[key])
    assert torch.equal(g1.get_state(), g2.get_state())


def test_noisy_known_values_at_every_denoiser_call():
    b, tb = batch(), tables()
    b = {key: value.repeat(256, *([1] * (value.ndim - 1))) for key, value in b.items()}
    c = Constraint(type_mask=~b["pad_mask"], type_value=torch.full((512, 4, 16), .37),
                   z_mask=lattice.free_mask(b["family"]), z_value=torch.full_like(b["lat"], .8))
    seen = []

    def record(state, sigma, skel, tb, **kwargs):
        seen.append((sigma.clone(), state["type_noisy"].clone(), state["z_noisy"].clone()))
        return oracle(state, sigma, skel, tb, **kwargs)

    s = settings()  # S_churn=60, including both predictor and corrector inputs.
    constrained_edm_sampler(None, b, tb, constraint=c, objective="P", settings=s,
                             generator=torch.Generator().manual_seed(31), denoise_fn=record)
    assert len(seen) == 2 * s.num_steps - 1
    for sigma, ty, z in seen:
        ty_eps = ((ty - c.type_value) / sigma[:, None, None])[c.type_mask]
        z_eps = ((z - c.z_value) / sigma[:, None])[c.z_mask]
        assert .95 <= ty_eps.std().item() <= 1.05
        assert .95 <= z_eps.std().item() <= 1.05


@pytest.mark.parametrize("objective", ["M", "P"])
@torch.no_grad()
def test_cpu_packed_constrained_parity(objective):
    from test_packed import case

    model, _ = case()
    model.eval()
    b, tb, s = batch(), tables(), settings(steps=2)
    s.sigma_max, s.S_churn = 2., 2.
    c = mixed_constraint(b)
    g1, g2 = (torch.Generator().manual_seed(13) for _ in range(2))
    ref = constrained_edm_sampler(model, b, tb, constraint=c, objective=objective, settings=s, generator=g1)
    adapter = PackedDenoiser(model, b, token_budget=16)
    assert adapter.backend == "dense"
    out = constrained_edm_sampler(model, b, tb, constraint=c, objective=objective,
                                   settings=s, generator=g2, denoise_fn=adapter.denoise)
    # Packing changes float32 summation order; one element in 128 moved by 1.05e-5 on some CPUs.
    for key in ref:
        torch.testing.assert_close(out[key], ref[key], atol=2e-5, rtol=3e-5)
    assert torch.equal(g1.get_state(), g2.get_state())


@pytest.fixture(scope="module")
def encoding():
    return sample.load_encoding()


@pytest.mark.parametrize("budget", [0, 6])
def test_fixed_elements_pool_decode_and_noop(monkeypatch, encoding, budget):
    from test_train import equal
    from test_verify_packed import _decode_pool, _pool_fixture

    model, ema, cfg, tb, bank, skeletons = _pool_fixture("B", (1, 3, 2, 1, 4))
    cfg = replace(cfg, sample_token_budget=budget)
    # As in the existing pool tests, isolate assembly from crystallographic decoding.
    # Element decoding below uses the real frozen encoder and decode_types.
    monkeypatch.setattr(sample, "decode_batch", _decode_pool)
    fixed = torch.zeros(5, cfg.K, dtype=torch.long)
    fixed[:, 0] = torch.tensor([7, 8, 14, 22, 26])
    fixed[1, 2], fixed[4, 3] = 6, 3
    allowed = torch.ones(encoding.vz, dtype=torch.bool)
    args = (model, ema, cfg, None, tb, encoding, allowed, bank)
    kwargs = dict(n=5, seed=919, device="cpu", chunk=2)
    ref = sample.sample_pool(*args, **kwargs)
    noop = sample.sample_pool(*args, **kwargs, sampler=FixedElements(torch.zeros_like(fixed), encoding, allowed))
    equal(noop, ref)
    out = sample.sample_pool(*args, **kwargs, sampler=FixedElements(fixed, encoding, allowed))
    skel = sample.skeleton_batch(skeletons, tb, K=cfg.K, device="cpu")
    assert "index" not in skel
    decoded = decode_types(out["raw"]["type"], skel["pad_mask"], encoding, allowed)
    assert torch.equal(decoded[fixed > 0], fixed[fixed > 0])
    assert out["skeletons"] == ref["skeletons"]
    assert all(s["A0"] is None for s in out["skeletons"])


def test_general_pool_builder_and_noncontiguous_indices(encoding):
    b, tb = batch(), tables()
    c = mixed_constraint(b)
    calls = []

    def build(skel):
        calls.append(skel)
        return c

    out = ConstrainedSampler(build)(None, b, tb, objective="P", settings=settings(), denoise_fn=oracle)
    assert calls == [b]
    torch.testing.assert_close(out["z"][c.z_mask], c.z_value[c.z_mask], atol=1e-6, rtol=0)
    fixed = torch.tensor([[3, 0, 0, 0], [7, 0, 0, 0], [8, 0, 0, 0]])
    b["index"] = torch.tensor([2, 0])
    selected = FixedElements(fixed, encoding, None).build(b)
    decoded = decode_types(selected.type_value, ~selected.type_mask, encoding, None)
    assert torch.equal(decoded, fixed[[2, 0]])


def test_override_preserves_inputs_and_base_predictions():
    b, tb = batch(), tables()
    state = dict(type_noisy=torch.ones(2, 4, 16), u_noisy=torch.full_like(b["U"], -.49),
                 z_noisy=torch.ones_like(b["lat"]))
    den = oracle(state, None, b, tb, objective="P", sigma_data=(.3,) * 3)
    saved = {k: v.clone() for k, v in den.items()}
    hook = constrain_denoiser(lambda *args, **kwargs: den, mixed_constraint(b))
    out = hook(state, torch.ones(2), b, tb, objective="P", sigma_data=(.3,) * 3)
    for key in saved:
        assert torch.equal(den[key], saved[key])
    assert torch.equal(state["u_noisy"], torch.full_like(b["U"], -.49))
    assert out["D_u"][0, 0, 0].item() == pytest.approx(-.51)


@pytest.mark.parametrize("block", ["type", "u", "z"])
@pytest.mark.parametrize("problem", ["mask_missing", "value_missing", "shape", "value_shape", "dtype", "inactive"])
def test_constraint_validation(block, problem):
    b, tb = batch(), tables()
    shape = {"type": (2, 4, 16), "u": (2, 4, 3), "z": (2, 6)}[block]
    mask = torch.zeros(shape[:-1] if block == "type" else shape, dtype=torch.bool)
    value = torch.zeros(shape)
    if problem == "mask_missing":
        mask = None
    elif problem == "value_missing":
        value = None
    elif problem == "shape":
        mask = mask[:1]
    elif problem == "value_shape":
        value = value[:1]
    elif problem == "dtype":
        mask = mask.float()
    else:
        index = {"type": (0, 3), "u": (0, 1, 1), "z": (1, 1)}[block]
        mask[index] = True
    with pytest.raises(ValueError):
        c = Constraint(**{block + "_mask": mask, block + "_value": value})
        constrained_edm_sampler(None, b, tb, constraint=c, objective="P", settings=settings(), denoise_fn=oracle)


def test_coordinate_padding_rejected_even_if_dof_mask_is_set():
    b, tb = batch(), tables()
    b["dof_mask"][0, 3] = True
    mask = torch.zeros_like(b["dof_mask"])
    mask[0, 3, 0] = True
    c = Constraint(u_mask=mask, u_value=torch.zeros_like(b["U"]))
    with pytest.raises(ValueError, match="padding"):
        constrained_edm_sampler(None, b, tb, constraint=c, objective="P", settings=settings(), denoise_fn=oracle)


@pytest.mark.parametrize("block,target", [("type", "types"), ("u", "coords"), ("z", "lattice")])
def test_anti_annealing_rejects_only_constrained_blocks(block, target):
    b, tb, s = batch(), tables(), vars(settings())
    c = mixed_constraint(b)
    c = Constraint(**{block + suffix: getattr(c, block + suffix) for suffix in ("_mask", "_value")})
    s["aa_rho_" + target] = 1.
    with pytest.raises(ValueError, match="aa_rho_" + target):
        constrained_edm_sampler(None, b, tb, constraint=c, objective="P", settings=s, denoise_fn=oracle)
    constrained_edm_sampler(None, b, tb, constraint=Constraint(), objective="P", settings=s, denoise_fn=oracle)


@pytest.mark.parametrize("problem", ["padding", "negative", "range", "disallowed", "float", "shape", "allowed_shape"])
def test_element_validation(encoding, problem):
    b = batch()
    fixed = torch.zeros_like(b["row"])
    fixed[0, 0] = 7
    allowed = torch.ones(encoding.vz, dtype=torch.bool)
    if problem == "padding":
        fixed[0, 3] = 7
    elif problem == "negative":
        fixed[0, 0] = -1
    elif problem == "range":
        fixed[0, 0] = encoding.max_z + 1
    elif problem == "disallowed":
        allowed[6] = False
    elif problem == "float":
        fixed = fixed.float()
    elif problem == "shape":
        fixed = fixed[:1]
    else:
        allowed = allowed[:-1]
    with pytest.raises(ValueError):
        element_constraint(fixed, b["pad_mask"], encoding, allowed)
