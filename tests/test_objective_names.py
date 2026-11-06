"""Coordinate objective names, the c_skip fix, and backward compatibility with the historical names M and P."""
from dataclasses import replace

import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol import contracts as C
from orbitol import edm
from orbitol.train import parser

from test_objective_acceptance import TB, Stub
from test_train import cfg, fixture_data, trainer, load_latest  # noqa: F401  -- shared tiny-run fixtures

SD = (0.3, 0.3, 0.3)


def _toy_batch():
    K = 6
    rows = torch.tensor([[1, 2, 3, 4, 5, 6]])
    dmask = torch.arange(3)[None, None, :] < TB.row_dof[rows][..., None]
    g = torch.Generator().manual_seed(11)
    batch = {"G": torch.ones(1, dtype=torch.long), "family": torch.zeros(1, dtype=torch.long), "row": rows,
             "U": torch.rand(1, K, 3, generator=g) * dmask, "dof_mask": dmask,
             "pad_mask": torch.zeros(1, K, dtype=torch.bool), "lat": 0.3 * torch.randn(1, 6, generator=g)}
    type_clean = torch.randn(1, K, 16, generator=g)
    noise = {"type": torch.randn(1, K, 16, generator=g), "coord": torch.randn(1, K, 3, generator=g),
             "lat": torch.randn(1, 6, generator=g)}
    shift = (torch.randint(-3, 4, (1, K, 3), generator=g) * dmask).float()
    return batch, type_clean, noise, shift, dmask


def _run(objective, batch, type_clean, noise, sigma, model, *, state=None):
    state = state or edm.corrupt(batch, type_clean, sigma, noise, TB, objective=objective)
    with torch.no_grad():
        den = edm.denoise_orbit(model, state, sigma, batch, TB, objective=objective, sigma_data=SD)
        loss = edm.compute_orbit_loss(den, state, sigma, batch, TB, objective=objective, sigma_data=SD,
                                      loss_weights=(1.0, 50.0, 5.0))
    return state, den, loss


def test_names_aliases_and_default():
    assert C.TrainConfig().objective == "posterior"
    assert C.TrainConfig(objective="M").objective == "wrapped"
    assert C.TrainConfig(objective="P").objective == "posterior_unit"
    for name in ("wrapped", "posterior", "posterior_unit", "posterior_periodic"):
        assert C.TrainConfig(objective=name).objective == name
    assert edm._objective("M") == ("wrapped", "edm") and edm._objective("P") == ("posterior", "unit")
    assert edm._objective("posterior") == ("posterior", "edm")
    assert edm._objective("posterior_periodic") == ("posterior", "periodic")
    with pytest.raises(ValueError):
        C.TrainConfig(objective="Q")
    assert parser().parse_args(["--run-name", "x", "--objective", "posterior_periodic"]).objective == "posterior_periodic"


@pytest.mark.parametrize("alias, name", [("M", "wrapped"), ("P", "posterior_unit")])
def test_historical_names_are_bitwise_unchanged(alias, name):
    batch, type_clean, noise, _, _ = _toy_batch()
    sigma = torch.tensor([0.3])
    model = Stub()
    s_alias, d_alias, l_alias = _run(alias, batch, type_clean, noise, sigma, model)
    s_name, d_name, l_name = _run(name, batch, type_clean, noise, sigma, model)
    for key in ("u_noisy", "u_clean", "type_noisy", "z_noisy"):
        assert torch.equal(s_alias[key], s_name[key])
    for key in ("D_type", "D_u", "D_z"):
        assert torch.equal(d_alias[key], d_name[key])
    for key in ("total", "A", "F", "Y"):
        assert torch.equal(l_alias[key], l_name[key])


@pytest.mark.parametrize("sigma_value", [0.02, 0.3, 2.0])
def test_posterior_uses_edm_skip_and_unit_keeps_input(sigma_value):
    batch, type_clean, noise, _, dmask = _toy_batch()
    sigma = torch.tensor([sigma_value])
    model = Stub()
    state, d_unit, _ = _run("posterior_unit", batch, type_clean, noise, sigma, model)
    _, d_edm, _ = _run("posterior", batch, type_clean, noise, sigma, model, state=state)
    c_skip, _, _ = edm._coeffs(sigma[:, None, None], SD[1])
    correction = d_unit["D_u"] - state["u_noisy"]          # c_out * F, shared by both
    expected = torch.where(dmask, c_skip * state["u_noisy"] + correction, 0.0)
    torch.testing.assert_close(d_edm["D_u"], expected, atol=1e-6, rtol=0)
    # Both posterior variants train on the same wrapped state and the same periodic target.
    s_edm = edm.corrupt(batch, type_clean, sigma, noise, TB, objective="posterior")
    assert torch.equal(s_edm["u_noisy"], state["u_noisy"])


def test_periodic_skip_is_lift_equivariant_and_edm_skip_is_not():
    batch, type_clean, noise, shift, dmask = _toy_batch()
    sigma = torch.tensor([0.3])
    model = Stub()
    for objective in ("posterior_periodic", "posterior_unit", "posterior"):
        state = edm.corrupt(batch, type_clean, sigma, noise, TB, objective=objective)
        shifted = dict(state, u_noisy=state["u_noisy"] + shift)
        with torch.no_grad():
            d0 = edm.denoise_orbit(model, state, sigma, batch, TB, objective=objective, sigma_data=SD)
            d1 = edm.denoise_orbit(model, shifted, sigma, batch, TB, objective=objective, sigma_data=SD)
        diff = d1["D_u"] - d0["D_u"]
        if objective == "posterior":
            assert not torch.allclose(diff, shift, atol=1e-3)
        else:
            torch.testing.assert_close(diff, shift, atol=1e-5, rtol=0)
        assert not torch.any(d0["D_u"][~dmask])


@pytest.mark.parametrize("objective", ["posterior", "posterior_periodic"])
def test_new_objectives_train_and_sample(cfg, fixture_data, objective):
    run = trainer(replace(cfg, objective=objective, max_steps=2, sample_count=8, sample_frequency=2),
                  fixture_data, f"objective-{objective}")
    run.run()
    latest = load_latest(fixture_data, f"objective-{objective}")
    assert latest["cfg"]["objective"] == objective and latest["step"] == 2


def test_resume_accepts_historical_names_and_refuses_a_changed_objective(cfg, fixture_data):
    run = trainer(replace(cfg, objective="P", max_steps=2), fixture_data, "historical-P")
    run.run()
    path = run.checkpoint_dir / "latest.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["cfg"]["objective"] = "P"                  # as written by runs before the rename
    torch.save(checkpoint, path)
    resumed = trainer(replace(cfg, objective="P", max_steps=2), fixture_data, "historical-P", resume="auto")
    assert resumed.step == 2 and resumed.cfg.objective == "posterior_unit"
    with pytest.raises(ValueError, match="Resume mismatch: cfg"):
        trainer(replace(cfg, objective="posterior", max_steps=2), fixture_data, "historical-P", resume="auto")


def _sampler_calls(objective, oracle):
    from orbitol.sampler import orbit_edm_sampler
    from test_edm import batch as edm_batch, tables as edm_tables
    from test_sampler import settings as sampler_settings
    orbit_edm_sampler(None, edm_batch(), edm_tables(), objective=objective, settings=sampler_settings(),
                      denoise_fn=oracle, generator=torch.Generator().manual_seed(7))


def test_edm_posterior_denoiser_only_sees_canonical_chart_states():
    for objective, canonical in (("posterior", True), ("posterior_unit", False), ("posterior_periodic", False)):
        seen = []

        def oracle(state, sigma, skel, tables, *, objective, sigma_data):
            seen.append(float(state["u_noisy"].abs().max()))
            return {"D_type": state["type_noisy"], "D_u": state["u_noisy"], "D_z": state["z_noisy"]}

        _sampler_calls(objective, oracle)
        if canonical:
            assert max(seen) <= 0.5
        else:
            assert max(seen) > 0.5          # lift-equivariant variants keep the historical sampler states


def test_uniform_target_oracle_has_zero_drift_under_edm_posterior():
    """The EDM-skip optimum for a uniform target is D = c_skip u + (1 - c_skip) wrap(u); its drift must vanish."""
    drifts = []

    def oracle(state, sigma, skel, tables, *, objective, sigma_data):
        u = state["u_noisy"]
        c_skip, _, _ = edm._coeffs(sigma[:, None, None], sigma_data[1])
        d_u = c_skip * u + (1 - c_skip) * (u - torch.round(u))
        drifts.append(float(((u - d_u) / sigma[:, None, None]).abs().max()))
        return {"D_type": state["type_noisy"], "D_u": d_u, "D_z": state["z_noisy"]}

    _sampler_calls("posterior", oracle)
    assert max(drifts) < 1e-6
