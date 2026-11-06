"""Frozen acceptance tests for Orbitol-I: internal chart input and direct chart readout."""
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol import edm
from orbitol.contracts import ORBIT_CACHE_DIR, TABLES_DIR, TrainConfig
from orbitol.data import OrbitDataset, collate_orbits
from orbitol.model import build_model
from orbitol.tables import load_tables

H = 4
CELLS = [(i, r) for i in ("explicit", "internal") for r in ("projection", "direct")]


@pytest.fixture(scope="module")
def tables():
    if not Path(TABLES_DIR, "tables.npz").is_file():
        pytest.skip(f"Frozen tables unavailable: {TABLES_DIR}")
    return load_tables(TABLES_DIR)


@pytest.fixture(scope="module")
def dataset(tables):
    path = Path(ORBIT_CACHE_DIR, "train.npz")
    if not path.is_file():
        pytest.skip(f"Orbit cache unavailable: {path}")
    return OrbitDataset(path, tables, augment=False, aug_seed=0)


def _select(dataset, tables, keep, count):
    G = np.asarray(dataset.cache["G"])
    rows, n = np.asarray(dataset.cache["row"]), np.asarray(dataset.cache["n_orbits"])
    dof = tables.arrays["row_dof"]
    chosen = [i for i in range(len(G)) if keep(int(G[i]), dof[rows[i, :n[i]]])][:count]
    assert len(chosen) == count
    return collate_orbits([dataset[i] for i in chosen], K=20)


@pytest.fixture(scope="module")
def p1_batch(dataset, tables):
    return _select(dataset, tables, lambda g, d: g == 1, 6)


@pytest.fixture(scope="module")
def fixed_batch(dataset, tables):
    """Structures whose orbits all sit on d = 0 rows (no free coordinate)."""
    return _select(dataset, tables, lambda g, d: g > 1 and np.all(d == 0), 6)


@pytest.fixture(scope="module")
def mixed_batch(dataset, tables):
    """Non-P1 structures with at least one d >= 1 row, across centrings and crystal systems."""
    G = np.asarray(dataset.cache["G"])
    centring = tables.arrays["group_centring"][G]
    rows, n = np.asarray(dataset.cache["row"]), np.asarray(dataset.cache["n_orbits"])
    dof = tables.arrays["row_dof"]
    free = np.array([np.any(dof[rows[i, :n[i]]] > 0) for i in range(len(G))])
    chosen = {int(i) for c in range(7) for i in np.flatnonzero((centring == c) & (G > 1) & free)[:4]}
    H = tables.arrays["row_H"]
    nondiag = np.abs(H - np.eye(3) * np.diagonal(H, axis1=1, axis2=2)[:, None, :]).max(axis=(1, 2)) > 0
    has_nondiag = np.array([np.any(nondiag[rows[i, :n[i]]]) for i in range(len(G))])
    chosen |= {int(i) for i in np.flatnonzero(has_nondiag)[:4]}
    return collate_orbits([dataset[i] for i in sorted(chosen)], K=20)


def tiny_cfg(**kwargs):
    return TrainConfig(d_model=32, n_layers=2, n_heads=H, batch_size=4, num_workers=0, precision="fp32", **kwargs)


def tiny_model(tables, *, coord_input="explicit", coord_readout="projection", gem_mode="none", seed=0):
    torch.manual_seed(seed)
    tb = tables.torch("cpu")
    cfg = tiny_cfg(coord_input=coord_input, coord_readout=coord_readout, gem_mode=gem_mode)
    return build_model(cfg, tb.row_desc, tb.group_desc, tables=tb).eval()


def randomise(module, seed=0):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(0.5 * torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype))


def cells(tables, gem_mode="none"):
    """The four 2x2 cells with identical weights; strict loading also proves identical parameter sets."""
    reference = tiny_model(tables, gem_mode=gem_mode)
    randomise(reference)
    models = {}
    for coord_input, coord_readout in CELLS:
        model = tiny_model(tables, coord_input=coord_input, coord_readout=coord_readout, gem_mode=gem_mode, seed=1)
        model.load_state_dict(reference.state_dict(), strict=True)
        models[coord_input, coord_readout] = model
    return models


def noisy_state(tables, batch, objective="posterior_periodic", seed=5):
    tb = tables.torch("cpu")
    generator = torch.Generator().manual_seed(seed)
    sigma = torch.linspace(0.05, 2.0, len(batch["G"]))
    noise = edm.orbit_noise(batch, generator)
    state = edm.corrupt(batch, torch.zeros(len(batch["G"]), batch["row"].shape[1], 16), sigma, noise, tb,
                        objective=objective)
    return state, sigma, tb


def denoise(model, state, sigma, batch, tb, objective="posterior_periodic"):
    with torch.no_grad():
        return edm.denoise_orbit(model, state, sigma, batch, tb, objective=objective, sigma_data=(0.3,) * 3)


def assert_equal(a, b):
    for key in ("D_type", "D_u", "D_z"):
        assert torch.equal(a[key], b[key]), key
    for key in a["raw"]:
        assert torch.equal(a["raw"][key], b["raw"][key]), key


def active_mask(tables, batch):
    dof = torch.as_tensor(tables.arrays["row_dof"])[batch["row"]]
    return (torch.arange(3) < dof[..., None]) & ~batch["pad_mask"][..., None]


# ---------------------------------------------------------------- configuration

def test_defaults_are_orbitol_x_and_old_configs_load():
    cfg = TrainConfig()
    assert (cfg.coord_input, cfg.coord_readout) == ("explicit", "projection")
    old = {k: v for k, v in asdict(cfg).items() if k not in ("coord_input", "coord_readout")}
    assert (TrainConfig(**old).coord_input, TrainConfig(**old).coord_readout) == ("explicit", "projection")


@pytest.mark.parametrize("bad", [dict(coord_input="ambient"), dict(coord_readout="chart"),
                                 dict(coord_input="internal", coord_embed_mode="rff")])
def test_invalid_combinations_are_rejected(bad):
    with pytest.raises(ValueError):
        TrainConfig(**bad)


# ---------------------------------------------------------------- the P1 identity and d = 0 rows

@pytest.mark.parametrize("gem_mode", ["none", "distance"])
def test_all_four_cells_agree_bitwise_at_p1(tables, p1_batch, gem_mode):
    state, sigma, tb = noisy_state(tables, p1_batch)
    outputs = {cell: denoise(model, state, sigma, p1_batch, tb) for cell, model in cells(tables, gem_mode).items()}
    reference = outputs["explicit", "projection"]
    for cell, output in outputs.items():
        assert_equal(output, reference)


@pytest.mark.parametrize("gem_mode", ["none", "distance"])
def test_internal_input_matches_explicit_on_fixed_rows(tables, fixed_batch, gem_mode):
    """With no free coordinate, Orbitol-I must still see the fixed Wyckoff position that Orbitol-X sees."""
    models = cells(tables, gem_mode)
    state, sigma, tb = noisy_state(tables, fixed_batch)
    assert_equal(denoise(models["internal", "projection"], state, sigma, fixed_batch, tb),
                 denoise(models["explicit", "projection"], state, sigma, fixed_batch, tb))


def test_internal_input_is_a_different_representation_off_p1(tables, mixed_batch):
    models = cells(tables)
    state, sigma, tb = noisy_state(tables, mixed_batch)
    explicit = denoise(models["explicit", "projection"], state, sigma, mixed_batch, tb)
    internal = denoise(models["internal", "projection"], state, sigma, mixed_batch, tb)
    assert not torch.equal(explicit["raw"]["coord_vel"], internal["raw"]["coord_vel"])


# ---------------------------------------------------------------- periodicity and inactive axes

def test_internal_input_is_integer_lift_invariant(tables, mixed_batch):
    model = cells(tables)["internal", "projection"]
    state, sigma, tb = noisy_state(tables, mixed_batch)
    active = active_mask(tables, mixed_batch)
    generator = torch.Generator().manual_seed(3)
    # Dyadic coordinates keep u + n + 1/2 exact in float32, so invariance must hold bitwise.
    u = torch.randint(-512, 512, active.shape, generator=generator).float() / 1024
    shift = torch.randint(-2, 3, active.shape, generator=generator).float()
    base = {**state, "u_noisy": torch.where(active, u, 0.0)}
    lifted = {**state, "u_noisy": torch.where(active, u + shift, 0.0)}
    a, b = denoise(model, base, sigma, mixed_batch, tb), denoise(model, lifted, sigma, mixed_batch, tb)
    for key in a["raw"]:
        assert torch.equal(a["raw"][key], b["raw"][key]), key


def test_internal_input_ignores_inactive_axes(tables, mixed_batch):
    model = cells(tables)["internal", "projection"]
    state, sigma, tb = noisy_state(tables, mixed_batch)
    inactive = ~active_mask(tables, mixed_batch) & ~mixed_batch["pad_mask"][..., None]
    assert inactive.any()
    junk = torch.rand(state["u_noisy"].shape, generator=torch.Generator().manual_seed(4)) - 0.5
    perturbed = {**state, "u_noisy": torch.where(inactive, junk, state["u_noisy"])}
    a, b = denoise(model, state, sigma, mixed_batch, tb), denoise(model, perturbed, sigma, mixed_batch, tb)
    for key in a["raw"]:
        assert torch.equal(a["raw"][key], b["raw"][key]), key


def test_internal_features_use_the_parent_layout(tables, mixed_batch):
    model = cells(tables)["internal", "projection"]
    state, sigma, tb = noisy_state(tables, mixed_batch)
    captured = []
    hook = model.coord_embed.mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    denoise(model, state, sigma, mixed_batch, tb)
    hook.remove()
    q = state["u_noisy"] + 0.5
    q = q - torch.floor(q)
    args = 2 * math.pi * q[..., None] * model.coord_embed.freqs
    dof = torch.as_tensor(tables.arrays["row_dof"])[mixed_batch["row"]]
    on = (torch.arange(3) < dof[..., None])[..., None]
    expected = (torch.cat([torch.sin(args), torch.cos(args)], -1) * on).reshape(*q.shape[:2], -1)
    chart = ~mixed_batch["pad_mask"] & (dof >= 1)
    assert any(c.shape == expected.shape and torch.allclose(c[chart], expected[chart], atol=1e-6) for c in captured)


# ---------------------------------------------------------------- readouts

def _expected_chart_update(tables, batch, f, readout):
    """Projection Pi_w F, or the metric-normalised direct readout H_w^{-1/2} f, in float64."""
    rows, dof = batch["row"].numpy(), tables.arrays["row_dof"]
    out = np.zeros(f.shape, dtype=np.float64)
    for b, k in zip(*np.nonzero(~batch["pad_mask"].numpy())):
        r, d = rows[b, k], int(dof[rows[b, k]])
        if d == 0:
            continue
        vector = f[b, k].double().numpy()
        if readout == "projection":
            out[b, k, :d] = tables.arrays["row_Pi"][r][:d] @ vector
        else:
            values, vectors = np.linalg.eigh(tables.arrays["row_H"][r][:d, :d].astype(np.float64))
            out[b, k, :d] = (vectors / np.sqrt(values)) @ vectors.T @ vector[:d]
    return torch.from_numpy(out)


@pytest.mark.parametrize("coord_input", ["explicit", "internal"])
@pytest.mark.parametrize("readout", ["projection", "direct"])
def test_readout_maps_the_raw_head_as_specified(tables, mixed_batch, coord_input, readout):
    model = cells(tables)[coord_input, readout]
    state, sigma, tb = noisy_state(tables, mixed_batch, objective="posterior")
    out = denoise(model, state, sigma, mixed_batch, tb, objective="posterior")
    sd = 0.3
    s = sigma.double()[:, None, None]
    c_skip, c_out = sd**2 / (s**2 + sd**2), s * sd / torch.sqrt(s**2 + sd**2)
    chart = _expected_chart_update(tables, mixed_batch, out["raw"]["coord_vel"], readout)
    expected = torch.where(active_mask(tables, mixed_batch), c_skip * state["u_noisy"].double() + c_out * chart, 0.0)
    # Padding is row 0 (the P1 chart, d = 3); its prediction is deliberately left unmasked, as in Crystalite.
    real = ~mixed_batch["pad_mask"]
    torch.testing.assert_close(out["D_u"].double()[real], expected[real], rtol=1e-5, atol=1e-5)


def test_direct_and_projection_readouts_differ_off_p1(tables, mixed_batch):
    models = cells(tables)
    state, sigma, tb = noisy_state(tables, mixed_batch)
    proj = denoise(models["internal", "projection"], state, sigma, mixed_batch, tb)
    direct = denoise(models["internal", "direct"], state, sigma, mixed_batch, tb)
    assert torch.equal(proj["raw"]["coord_vel"], direct["raw"]["coord_vel"])
    assert not torch.equal(proj["D_u"], direct["D_u"])


# ---------------------------------------------------------------- end to end

@pytest.mark.parametrize("gem_mode", ["none", "distance"])
def test_trainer_runs_internal_direct(gem_mode, request):
    extra, root = request.getfixturevalue("fixture_data")
    from orbitol.train import Trainer
    cfg = TrainConfig(d_model=32, n_layers=2, n_heads=H, batch_size=8, num_workers=0, max_steps=3, warmup_steps=1,
                      precision="fp32", objective="posterior_periodic", sample_count=8, sample_num_steps=5,
                      sample_frequency=3, val_every=0, log_every=1, gem_mode=gem_mode,
                      coord_input="internal", coord_readout="direct", extra=dict(extra))
    trainer = Trainer(cfg, run_name=f"orbitol-i-{gem_mode}", out_root=root / "runs", device="cpu", no_wandb=True)
    trainer.run()
    assert trainer.step == 3
    snapshot = torch.load(root / "runs" / f"orbitol-i-{gem_mode}" / "checkpoints" / "latest.pt",
                          map_location="cpu", weights_only=False)
    assert (snapshot["cfg"]["coord_input"], snapshot["cfg"]["coord_readout"]) == ("internal", "direct")


from test_train import fixture_data  # noqa: E402,F401  -- tiny cache and tables fixture shared with test_train
