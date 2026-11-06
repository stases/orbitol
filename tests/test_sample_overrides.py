"""Sampler overrides: execution choices for GPUs without bf16 (docs/hardware.md) and churn settings."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from orbitol import contracts as C, sample
from orbitol.edm import karras_sigma_steps
from orbitol.sampler import orbit_edm_sampler
from test_edm import batch, tables


class Built(Exception):
    pass


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    cfg = dict(dataset="alex_mp20", precision="bf16", compile_trunk="reduce-overhead", sample_token_budget=8192)
    path = tmp_path / "snapshot.pt"
    torch.save(dict(cfg=cfg, core_manifest=json.loads((sample.ROOT / "orbitol/core_manifest.json").read_text()),
                    table_sha256="table", encoding_sha256=C.TYPE_ENCODING_SHA256, bank_sha256="bank"), path)
    tb = SimpleNamespace(row_desc=None, group_desc=None)
    monkeypatch.setattr(sample, "load_tables", lambda path: SimpleNamespace(sha256="table", torch=lambda device: tb))
    monkeypatch.setattr(sample, "load_encoding", lambda: None)
    monkeypatch.setattr(sample.SkeletonBank, "from_cache",
                        lambda path: SimpleNamespace(table_sha256="table", sha256="bank"))

    def build_model(cfg, *args, **kwargs):
        raise Built(cfg)

    monkeypatch.setattr(sample, "build_model", build_model)
    return path


def built_cfg(path, overrides=None):
    with pytest.raises(Built) as built:
        sample.load_snapshot(path, device="cpu", overrides=overrides)
    return built.value.args[0]


def built_config(path, overrides=None):
    cfg = built_cfg(path, overrides)
    return cfg.precision, cfg.compile_trunk, cfg.sample_token_budget


def test_overrides_reach_the_model_and_sampler_config(snapshot):
    assert built_config(snapshot) == ("bf16", "reduce-overhead", 8192)
    overrides = dict(precision="fp32", compile_trunk="none", sample_token_budget=0)
    assert built_config(snapshot, overrides) == ("fp32", "none", 0)
    assert built_config(snapshot, dict(precision="fp32")) == ("fp32", "reduce-overhead", 8192)


def test_churn_overrides_reach_the_sampler_config(snapshot):
    overrides = dict(sample_num_steps=100, S_churn=0.0, S_min=0.05, S_max=50.0, S_noise=1.0,
                     sigma_min=0.01, sigma_max=3.0, rho=5.0)
    cfg = built_cfg(snapshot, overrides)
    assert {key: getattr(cfg, key) for key in overrides} == overrides
    assert (cfg.precision, cfg.sample_token_budget) == ("bf16", 8192)


def test_only_execution_choices_can_be_overridden(snapshot):
    with pytest.raises(ValueError, match="limited to"):
        sample.load_snapshot(snapshot, device="cpu", overrides=dict(d_model=256))


def test_cli_forwards_only_the_given_flags(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sample, "sample_from_checkpoint",
                        lambda path, **kwargs: calls.append(kwargs) or dict(failure_counts={}))
    base = [str(tmp_path / "snapshot.pt"), "--out", str(tmp_path / "samples.pt")]
    sample.main(base + ["--precision", "fp32", "--compile-trunk", "none", "--sample-token-budget", "0"])
    sample.main(base)
    assert [call["overrides"] for call in calls] == [
        dict(precision="fp32", compile_trunk="none", sample_token_budget=0), {}]
    with pytest.raises(SystemExit):
        sample.main(base + ["--precision", "fp64"])
    sample.main(base + ["--S-min", "0.05", "--S-churn", "0", "--sample-num-steps", "100"])
    assert calls[-1]["overrides"] == dict(S_min=0.05, S_churn=0.0, sample_num_steps=100)
    assert type(calls[-1]["overrides"]["S_min"]) is float and type(calls[-1]["overrides"]["sample_num_steps"]) is int


def test_s_min_changes_only_the_steps_below_it_and_keeps_the_random_stream():
    """Arms that differ in S_min share every state until sigma falls below S_min (the paired design).

    gamma = 0 still draws the churn noise, so every arm, S_churn = 0 included, consumes the same random stream.
    """
    num_steps = 12

    def run(S_min, S_churn=60.0):
        calls = []

        def denoise(state, sigma, skel, tb, *, objective, sigma_data):
            calls.append((float(sigma[0]), *(state[k].clone() for k in ("type_noisy", "u_noisy", "z_noisy"))))
            return dict(D_type=0.9 * state["type_noisy"], D_u=0.5 * state["u_noisy"].sin(),
                        D_z=0.8 * state["z_noisy"])

        settings = dict(num_steps=num_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, S_churn=S_churn,
                        S_min=S_min, S_max=999.0, S_noise=1.003, sigma_data=(0.3,) * 3)
        generator = torch.Generator().manual_seed(3)
        out = orbit_edm_sampler(None, batch(), tables(), objective="P", settings=settings,
                                generator=generator, denoise_fn=denoise)
        return calls, out, torch.randn(4, generator=generator)

    steps = karras_sigma_steps(num_steps=num_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device="cpu")
    base, base_out, base_next = run(0.0)
    assert len(base) == 2 * num_steps - 1
    for S_min in (0.05, 0.3):
        calls, out, following = run(S_min)
        k = int((steps[:-1] >= S_min).sum())  # first step without churn; its first call is calls[2k]
        assert 0 < k < num_steps - 1
        for (sigma, *state), (sigma0, *state0) in zip(calls[:2 * k], base[:2 * k], strict=True):
            assert sigma == sigma0 and all(torch.equal(x, y) for x, y in zip(state, state0))
        assert calls[2 * k][0] == pytest.approx(float(steps[k]))
        assert base[2 * k][0] == pytest.approx((1 + min(60.0 / num_steps, math.sqrt(2) - 1)) * float(steps[k]))
        assert not torch.equal(out["u"], base_out["u"])
        assert torch.equal(following, base_next)
    calls, out, following = run(0.0, S_churn=0.0)
    assert torch.equal(following, base_next) and not torch.equal(out["u"], base_out["u"])
