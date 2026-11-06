"""Frozen acceptance tests for GEM v1 bias and model integration."""
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol import edm
from orbitol.contracts import ORBIT_CACHE_DIR, TrainConfig
from orbitol.data import OrbitDataset, collate_orbits
from orbitol.gem import GEM_FEATURES, OrbitGEM, aggregate, rbf
from orbitol.gem_geometry import PrimitiveMembers, cell_scale, expand_members, pair_distances, primitive_lattice
from orbitol.model import build_model
from src.crystalite.crystalite import mod1

H, N_MAX = 4, 20


@pytest.fixture(scope="module")
def tables(frozen_tables):
    return frozen_tables


@pytest.fixture(scope="module")
def dataset(tables):
    path = Path(ORBIT_CACHE_DIR, "train.npz")
    if not path.is_file():
        pytest.skip(f"Orbit cache unavailable: {path}; set ORBITOL_MP20_ORBIT_CACHE_DIR or ORBITOL_DATA_HOME")
    return OrbitDataset(path, tables, augment=False, aug_seed=0)


@pytest.fixture(scope="module")
def batch(tables, dataset):
    groups = np.asarray(dataset.cache["G"])
    centring = tables.arrays["group_centring"][groups]
    chosen = {int(i) for c in range(7) for i in np.flatnonzero(centring == c)[:6]}
    chosen |= {int(i) for i in np.flatnonzero(groups == 1)[:4]}
    return collate_orbits([dataset[i] for i in sorted(chosen)], K=20)


def geometry(tables, batch, dtype=torch.float64):
    tb = tables.torch("cpu", dtype=dtype)
    members = expand_members((batch["U"].double() - 0.5).to(dtype), batch["row"], batch["G"],
                             batch["pad_mask"], tb, n_max=N_MAX)
    L = primitive_lattice(batch["lat"].to(dtype), batch["G"], tb)
    return members, pair_distances(members.x, members.valid, L), cell_scale(L)


def randomise(module, seed=0):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(0.5 * torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype))
    return module


def features(d, scale):
    return torch.cat([rbf(d / scale, 32, 2.0), rbf(d, 32, 8.0)], dim=-1)


# ---------------------------------------------------------------- the bias itself

def test_aggregation_is_tied_atom_attention():
    # Orbit attention with B = logsumexp over members equals atom attention in
    # which every member carries its orbit's key and value.
    generator = torch.Generator().manual_seed(1)
    B, K, N, D = 3, 5, 9, 7
    owner = torch.tensor([[0, 0, 1, 1, 1, 1, 2, 3, -1],
                          [0, 1, 2, 3, 4, -1, -1, -1, -1],
                          [0, 0, 0, 0, 0, 0, 1, 1, 2]])
    valid = owner >= 0
    b_member = torch.randn(B, H, K, N, generator=generator, dtype=torch.float64)
    scores = torch.randn(B, H, K, K, generator=generator, dtype=torch.float64)
    values = torch.randn(B, K, D, generator=generator, dtype=torch.float64)
    bias = aggregate(b_member, owner, valid, K)
    assert bias.shape == (B, H, K, K) and torch.isfinite(bias).all()
    for i in range(B):
        keys = sorted(set(owner[i][valid[i]].tolist()))
        empty = [l for l in range(K) if l not in keys]
        assert (bias[i][:, :, empty] == 0).all()                     # no members: exactly 0, never -inf/NaN
        orbit = torch.softmax(scores[i][:, :, keys] + bias[i][:, :, keys], dim=-1) @ values[i, keys]
        atoms = valid[i].nonzero().squeeze(-1)
        tied = scores[i][:, :, owner[i, atoms]] + b_member[i][:, :, atoms]
        atom = torch.softmax(tied, dim=-1) @ values[i, owner[i, atoms]]
        torch.testing.assert_close(orbit, atom, rtol=1e-12, atol=1e-12)


@pytest.mark.data
def test_multiplicity_mode_values(tables, batch):
    members, _, _ = geometry(tables, batch)
    K = batch["row"].shape[1]
    bias = OrbitGEM(H, "multiplicity")(members, seq_len=K + 1)
    assert bias.shape == (len(batch["G"]), H, K + 1, K + 1) and bias.dtype == torch.float32
    for i in range(len(batch["G"])):
        k = int((~batch["pad_mask"][i]).sum())
        log_m = members.count[i, :k].double().log()
        torch.testing.assert_close(bias[i, :, :k, :k].double(), log_m.expand(H, k, k), rtol=0, atol=1e-6)
        torch.testing.assert_close(bias[i, :, K, :k].double(), log_m.expand(H, k), rtol=0, atol=1e-6)
        assert (bias[i, :, :, K] == 0).all()
        assert (bias[i, :, k:K, :] == 0).all() and (bias[i, :, :, k:K] == 0).all()


@pytest.mark.data
def test_distance_mode_starts_as_multiplicity(tables, batch):
    members, d, scale = geometry(tables, batch, torch.float32)
    K = batch["row"].shape[1]
    start = OrbitGEM(H, "distance")(members, d, scale, seq_len=K + 1)
    reference = OrbitGEM(H, "multiplicity")(members, seq_len=K + 1)
    torch.testing.assert_close(start, reference, rtol=0, atol=1e-6)


@pytest.mark.data
def test_values_pin_the_own_orbit_rule(tables, batch):
    # B[k,l] = logsumexp over l's members b of MLP(phi(d(rep_k, b))); ONLY (rep_k, rep_k) is
    # zeroed. The representative's orbit-mates keep their true distances.
    members, d, scale = geometry(tables, batch)
    gem = randomise(OrbitGEM(H, "distance").double(), seed=2)
    K = batch["row"].shape[1]
    bias = gem(members, d, scale, seq_len=K + 1)
    multi = 0
    for i in range(len(batch["G"])):
        n, k = int(members.valid[i].sum()), int((members.rep[i] >= 0).sum())
        b = gem.mlp(features(d[i, :n, :n], scale[i])).permute(2, 0, 1).clone()     # (H, n, n)
        for t in range(k):
            rep = int(members.rep[i, t])
            b[:, rep, rep] = 0.0
            for l in range(k):
                mask = members.owner[i, :n] == l
                expected = torch.logsumexp(b[:, rep, mask], dim=-1)
                torch.testing.assert_close(bias[i, :, t, l].double(), expected, rtol=1e-6, atol=1e-6)
        multi += int((members.count[i, :k] > 1).any())
    assert multi >= 10


@pytest.mark.data
def test_self_pair_is_zero_by_index_only(tables, dataset):
    # Two coinciding P1 tokens: the cross pair sees d = 0 and gets MLP(phi(0)); only (rep, rep) is 0.
    one = next(dataset[i] for i in range(len(dataset)) if dataset[i]["G"] == 1 and len(dataset[i]["row"]) < 20)
    twin = {key: (np.concatenate([value[:1], value[:1], value[1:]]) if isinstance(value, np.ndarray) and value.ndim >= 1
                  and key in {"row", "A0", "U", "dof_mask", "pad_mask"} else value) for key, value in one.items()}
    batch = collate_orbits([twin], K=20)
    members, d, scale = geometry(tables, batch)
    assert float(d[0, 0, 1]) == 0.0
    gem = randomise(OrbitGEM(H, "distance").double(), seed=4)
    bias = gem(members, d, scale, seq_len=21)
    cross = gem.mlp(features(torch.zeros(1, 1, dtype=torch.float64), scale[0]))[0, 0]       # (H,)
    torch.testing.assert_close(bias[0, :, 0, 0].double(), torch.zeros(H, dtype=torch.float64), rtol=0, atol=1e-7)
    torch.testing.assert_close(bias[0, :, 0, 1].double(), cross, rtol=1e-6, atol=1e-6)


@pytest.mark.data
def test_padding_hygiene_with_a_trained_looking_bias(tables, batch):
    members, d, scale = geometry(tables, batch)
    gem = randomise(OrbitGEM(H, "distance").double(), seed=5)
    K = batch["row"].shape[1]
    bias = gem(members, d, scale, seq_len=K + 1)
    assert bias.dtype == torch.float32 and torch.isfinite(bias).all()
    for i in range(len(batch["G"])):
        k = int((~batch["pad_mask"][i]).sum())
        torch.testing.assert_close(bias[i, :, K, :k].double(), members.count[i, :k].double().log().expand(H, k),
                                   rtol=0, atol=1e-6)
        assert (bias[i, :, :, K] == 0).all()
        assert (bias[i, :, k:K, :] == 0).all() and (bias[i, :, :, k:K] == 0).all()


@pytest.mark.data
def test_representative_independence_on_real_crystals(tables, batch):
    members, d, scale = geometry(tables, batch)
    gem = randomise(OrbitGEM(H, "distance").double(), seed=3)
    K = batch["row"].shape[1]
    first = gem(members, d, scale, seq_len=K + 1)
    last_rep = members.rep.clone()
    for i in range(len(batch["G"])):
        for t in range(int((members.rep[i] >= 0).sum())):
            last_rep[i, t] = int((members.owner[i] == t).nonzero().max())
    last = gem(replace(members, rep=last_rep), d, scale, seq_len=K + 1)
    torch.testing.assert_close(first, last, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bad", [("fourier",), ("distance_norm", "displacement"), ("direction",), ("y1",)])
def test_direction_features_are_rejected(bad):
    assert set(GEM_FEATURES) == {"distance_norm", "distance_abs"}
    with pytest.raises(ValueError):
        OrbitGEM(H, "distance", features=bad)
    with pytest.raises(ValueError):
        OrbitGEM(H, "edge")


# ---------------------------------------------------------------- the model

def tiny_cfg(**kwargs):
    return TrainConfig(d_model=32, n_layers=2, n_heads=H, batch_size=4, num_workers=0,
                       precision="fp32", compile_trunk="none", **kwargs)


def tiny_model(tables, seed=0, **kwargs):
    torch.manual_seed(seed)
    tb = tables.torch("cpu")
    return build_model(tiny_cfg(**kwargs), tb.row_desc, tb.group_desc, tables=tb).eval()


def noisy_state(tables, batch, objective="P", seed=5):
    tb = tables.torch("cpu")
    generator = torch.Generator().manual_seed(seed)
    sigma = torch.linspace(0.05, 2.0, len(batch["G"]))
    noise = edm.orbit_noise(batch, generator)
    state = edm.corrupt(batch, torch.zeros(len(batch["G"]), batch["row"].shape[1], 16), sigma, noise, tb,
                        objective=objective)
    return state, sigma, tb


def reference_forward(model, type_in, x_in, lat_in, pad_mask, c_noise, row, G):
    """orbitol/model.py's forward without GEM, re-implemented from its submodules."""
    frac_mod = mod1(x_in)
    h_type = model.type_proj(type_in) + model.coord_embed(frac_mod) + model.segment_embed.weight[0]
    h_type = h_type + model.row_embed(row)
    h_lat = model.lattice_embed(lat_in) + model.segment_embed.weight[1]
    x = torch.cat([h_type, h_lat[:, None, :]], dim=1)
    pad_seq = torch.cat([pad_mask.bool(), torch.zeros((pad_mask.shape[0], 1), dtype=torch.bool)], dim=1)
    t_emb = model.time(c_noise, c_noise) + model.sg_embed(G)
    h = model.trunk(x, t_emb, pad_mask=pad_seq, coords=frac_mod, lattice=lat_in,
                    t_sigma=torch.exp(4.0 * c_noise.to(dtype=torch.float32)))
    return model.heads(h, frac_coords=frac_mod, pad_mask=pad_mask.bool())


@pytest.mark.data
def test_none_mode_matches_reference_forward(tables, batch):
    assert TrainConfig().gem_mode == "none"
    model = tiny_model(tables)
    generator = torch.Generator().manual_seed(9)
    B, K = batch["row"].shape
    type_in = torch.randn(B, K, 16, generator=generator)
    x_in, lat_in = torch.rand(B, K, 3, generator=generator), torch.randn(B, 6, generator=generator)
    c_noise = torch.randn(B, generator=generator)
    with torch.no_grad():
        new = model(type_in, x_in, lat_in, batch["pad_mask"], c_noise, row=batch["row"], G=batch["G"])
        ref = reference_forward(model, type_in, x_in, lat_in, batch["pad_mask"], c_noise, batch["row"], batch["G"])
        extra = model(type_in, x_in, lat_in, batch["pad_mask"], c_noise, row=batch["row"], G=batch["G"],
                      u=torch.zeros(B, K, 3), z=torch.zeros(B, 6), family=batch["family"])
    for key in ref:
        assert torch.equal(new[key], ref[key]) and torch.equal(extra[key], ref[key]), key


def test_gem_modes_require_tables(tables):
    tb = tables.torch("cpu")
    for mode in ("multiplicity", "distance"):
        with pytest.raises(ValueError):
            build_model(tiny_cfg(gem_mode=mode), tb.row_desc, tb.group_desc, tables=None)


@pytest.mark.data
def test_block_loop_with_zero_bias_matches_the_core_trunk(tables, batch, monkeypatch):
    plain = tiny_model(tables)
    gem = tiny_model(tables, gem_mode="multiplicity")
    gem.load_state_dict(plain.state_dict(), strict=True)
    monkeypatch.setattr(OrbitGEM, "forward", lambda self, members, *args, seq_len, **kwargs:
                        torch.zeros(members.x.shape[0], H, seq_len, seq_len))
    state, sigma, tb = noisy_state(tables, batch)
    with torch.no_grad():
        a = edm.denoise_orbit(plain, state, sigma, batch, tb, objective="P", sigma_data=(0.3,) * 3)
        b = edm.denoise_orbit(gem, state, sigma, batch, tb, objective="P", sigma_data=(0.3,) * 3)
    # A zero float mask may select another attention kernel, so allow rounding only.
    for key in ("D_type", "D_u", "D_z"):
        torch.testing.assert_close(a[key], b[key], rtol=1e-4, atol=1e-5)


@pytest.mark.data
def test_padding_invariance(tables, batch):
    model = tiny_model(tables, gem_mode="distance")
    randomise(model.gem, seed=6)
    small = ((~batch["pad_mask"]).sum(-1) < 20).nonzero().squeeze(-1)
    batch = {key: (value[small] if torch.is_tensor(value) else [value[j] for j in small.tolist()])
             for key, value in batch.items()}
    state, sigma, tb = noisy_state(tables, batch)
    kmax = int((~batch["pad_mask"]).sum(-1).max())
    assert kmax < 20
    keyed = {"row", "A0", "U", "dof_mask", "pad_mask"}
    short = {key: (value[:, :kmax] if key in keyed else value) for key, value in batch.items()}
    short_state = {key: (value[:, :kmax] if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] == 20 else value)
                   for key, value in state.items()}
    with torch.no_grad():
        full = edm.denoise_orbit(model, state, sigma, batch, tb, objective="P", sigma_data=(0.3,) * 3)
        cut = edm.denoise_orbit(model, short_state, sigma, short, tb, objective="P", sigma_data=(0.3,) * 3)
    real = ~short["pad_mask"]
    for key in ("D_type", "D_u"):
        torch.testing.assert_close(full[key][:, :kmax][real], cut[key][real], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(full["D_z"], cut["D_z"], rtol=1e-5, atol=1e-5)


@pytest.mark.data
def test_gradients_are_finite_in_distance_mode(tables, batch):
    model = tiny_model(tables, gem_mode="distance").train()
    # Crystalite's blocks start with AdaLN-zero gates, so attention (and with it the bias) has no
    # effect at initialisation. Open the gates to test that gradients reach the bias MLP.
    for index, block in enumerate(model.trunk.blocks):
        randomise(block.adaLN_mod[-1], seed=20 + index)
    state, sigma, tb = noisy_state(tables, batch)
    out = edm.denoise_orbit(model, state, sigma, batch, tb, objective="P", sigma_data=(0.3,) * 3)
    loss = edm.compute_orbit_loss(out, state, sigma, batch, tb, objective="P", sigma_data=(0.3,) * 3,
                                  loss_weights=(1.0, 50.0, 5.0))["total"]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    last = model.gem.mlp[-1].weight.grad
    assert last is not None and torch.isfinite(last).all() and last.abs().sum() > 0


@pytest.mark.data
def test_bf16_autocast_keeps_the_bias_float32(tables, batch):
    model = tiny_model(tables, gem_mode="distance")
    randomise(model.gem, seed=7)
    seen = []
    model.gem.register_forward_hook(lambda module, inputs, output: seen.append(output.dtype))
    state, sigma, tb = noisy_state(tables, batch)
    with torch.no_grad():
        out = edm.denoise_orbit(model, state, sigma, batch, tb, objective="P", sigma_data=(0.3,) * 3,
                                autocast_dtype=torch.bfloat16)
    assert seen == [torch.float32]
    assert all(torch.isfinite(out[key]).all() for key in ("D_type", "D_u", "D_z"))


def test_distance_checkpoint_round_trip(tables):
    model = tiny_model(tables, gem_mode="distance")
    randomise(model.gem, seed=8)
    fresh = tiny_model(tables, seed=1, gem_mode="distance")
    missing, unexpected = fresh.load_state_dict(model.state_dict(), strict=True)
    assert not missing and not unexpected
    assert all(torch.equal(a, b) for a, b in zip(model.gem.parameters(), fresh.gem.parameters()))


@pytest.mark.parametrize("mode", ["multiplicity", "distance"])
def test_trainer_runs_with_gem(mode, request):
    extra, root = request.getfixturevalue("fixture_data")
    from orbitol.train import Trainer
    cfg = TrainConfig(d_model=32, n_layers=2, n_heads=H, batch_size=8, num_workers=0, max_steps=3, warmup_steps=1,
                      precision="fp32", objective="P", sample_count=8, sample_num_steps=5, sample_frequency=3,
                      compile_trunk="none", sample_token_budget=0, eval_mode="sync", nequip_eval=False,
                      val_every=0, log_every=1, gem_mode=mode, extra=dict(extra))
    trainer = Trainer(cfg, run_name=f"gem-{mode}", out_root=root / "runs", device="cpu", no_wandb=True)
    trainer.run()
    assert trainer.step == 3
    import json, math
    rows = [json.loads(line) for line in (root / "runs" / f"gem-{mode}" / "metrics.jsonl").read_text().splitlines() if line]
    losses = [row[key] for row in rows for key in ("train/loss_total", "train/loss_type", "train/loss_coord", "train/loss_lattice") if key in row]
    assert losses and all(value is not None and math.isfinite(value) for value in losses)


# ---------------------------------------------------------------- full size, GPU only

@pytest.mark.gpu
@pytest.mark.data
def test_full_size_cost_on_gpu(tables, dataset):
    # conftest enables deterministic algorithms for CPU tests; on CUDA that rejects cuBLAS and scatter
    # kernels and would distort the timing. Training runs without it; conftest restores it afterwards.
    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda")
    batch = collate_orbits([dataset[i] for i in range(128)], K=20)
    batch = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}
    tb = tables.torch(device)
    results = {}
    for mode in ("none", "multiplicity", "distance"):
        torch.manual_seed(0)
        model = build_model(TrainConfig(gem_mode=mode), tb.row_desc, tb.group_desc, tables=tb).to(device)
        generator = torch.Generator(device=device).manual_seed(0)
        sigma = torch.exp(-1.2 + 1.2 * torch.randn(128, device=device, generator=generator))
        noise = edm.orbit_noise(batch, generator)
        state = edm.corrupt(batch, torch.zeros(128, 20, 16, device=device), sigma, noise, tb, objective="P")
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(8):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = edm.denoise_orbit(model, state, sigma, batch, tb, objective="P", sigma_data=(0.3,) * 3,
                                    autocast_dtype=torch.bfloat16)
            loss = edm.compute_orbit_loss(out, state, sigma, batch, tb, objective="P", sigma_data=(0.3,) * 3,
                                          loss_weights=(1.0, 50.0, 5.0))["total"]
            assert torch.isfinite(loss)
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        results[mode] = (1000 * float(np.median(times[3:])), torch.cuda.max_memory_allocated() / 2**30)
        print(f"{mode}: {results[mode][0]:.1f} ms/step (fwd+bwd), peak {results[mode][1]:.2f} GiB")
    assert results["distance"][0] <= 1.5 * results["none"][0]
    assert results["multiplicity"][0] <= 1.5 * results["none"][0]


from test_train import fixture_data  # noqa: E402,F401  -- tiny cache and tables fixture shared with test_train
