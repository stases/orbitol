"""CPU gates for opt-in setup, evaluation recovery and diagnostic persistence."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from orbitol import nequip_relax, train
from orbitol.posthoc_eval import thermo_settings
from orbitol.sample import rng_state

from test_train import cfg, equal, fixture_data, load_latest, trainer  # noqa: F401


@pytest.fixture
def nequip_cfg(cfg, fixture_data):
    root = fixture_data[1]
    model, phase = root / "mock.nequip.pth", root / "mock-phase.pkl"
    model.write_bytes(b"mock TorchScript model")
    phase.write_bytes(b"mock phase diagram; evaluator is mocked")
    return replace(cfg, sample_count=4, sample_frequency=2, sun_k=2, nequip_eval=True,
                   extra={**cfg.extra, "nequip_model": str(model), "phase_diagram": str(phase)})


@pytest.fixture
def factories(monkeypatch):
    calls = []

    def build(mode, *args, **kwargs):
        # Backend setup must not change the seeded model or training draws.
        random.random()
        np.random.random()
        torch.rand(1)
        backend = object()
        calls.append((mode, args, kwargs, backend))
        return backend

    for mode in ("batch", "sequential"):
        monkeypatch.setattr(nequip_relax, f"make_{mode}_relaxer",
                            lambda *args, _mode=mode, **kwargs: build(_mode, *args, **kwargs))
    return calls


@pytest.mark.parametrize("mode", ["batch", "sequential"])
def test_offline_settings_and_startup_hash(nequip_cfg, fixture_data, factories, capsys, mode):
    config = replace(nequip_cfg, nequip_relax_mode=mode, nequip_relax_batch_size=3)
    run = trainer(config, fixture_data, f"settings-{mode}")
    expected = thermo_settings(SimpleNamespace(
        nequip_ts_path=Path(config.extra["nequip_model"]),
        ppd_path=Path(config.extra["phase_diagram"]),
        device="cpu", relax_mode=mode, relax_batch_size=3,
        nequip_backend="torchscript", relax_steps=200,
    ))
    assert run.evaluator.thermo == expected
    assert run.evaluator.no_thermo_before_steps == 0
    assert len(factories) == 1
    made_mode, args, kwargs, backend = factories[0]
    assert made_mode == mode and args == (expected.nequip_ts_path, "cpu")
    assert kwargs == (dict(max_force_abort=1e6, fmax=0.005) if mode == "sequential"
                      else dict(max_force_abort=1e6))
    assert isinstance(run.evaluator.relaxer, nequip_relax.RobustRelaxer)
    assert run.evaluator.relaxer.backend is backend
    digest = hashlib.sha256(b"mock TorchScript model").hexdigest()
    hull_digest = hashlib.sha256(b"mock phase diagram; evaluator is mocked").hexdigest()
    out = capsys.readouterr().out
    assert f"nequip_model_sha256={digest}" in out and f"phase_diagram_sha256={hull_digest}" in out
    run.save(snapshot=True)
    for path in (run.checkpoint_dir / "latest.pt", run.checkpoint_dir / "step_0000000.pt"):
        payload = torch.load(path, weights_only=False)
        assert payload["nequip_model_sha256"] == digest and payload["phase_diagram_sha256"] == hull_digest


@pytest.mark.parametrize("missing", ["nequip_model", "phase_diagram"])
def test_missing_inputs_precede_backend_loading(nequip_cfg, fixture_data, factories, missing):
    config = replace(nequip_cfg, extra={**nequip_cfg.extra, missing: str(fixture_data[1] / "missing")})
    with pytest.raises(FileNotFoundError):
        trainer(config, fixture_data, f"missing-{missing}")
    assert factories == []


def test_backend_setup_error_is_fatal(nequip_cfg, fixture_data, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("broken TorchScript build")

    monkeypatch.setattr(nequip_relax, "make_batch_relaxer", fail)
    with pytest.raises(RuntimeError, match="broken TorchScript build"):
        trainer(nequip_cfg, fixture_data, "broken-backend")


@pytest.mark.parametrize("enabled,count", [(False, 4), (True, 0)])
def test_inactive_nequip_needs_no_artifacts(cfg, fixture_data, factories, enabled, count):
    config = replace(cfg, nequip_eval=enabled, sample_count=count,
                     extra={**cfg.extra, "nequip_model": "/missing/model", "phase_diagram": "/missing/phase"})
    run = trainer(config, fixture_data, f"inactive-{enabled}")
    assert factories == [] and "nequip_model_sha256" not in run.snapshot()
    if count:
        assert run.evaluator.thermo is None and run.evaluator.sun_k == 0
    else:
        assert run.evaluator is None


def test_evaluation_error_continues_and_preserves_training(nequip_cfg, fixture_data, factories,
                                                         monkeypatch, capsys):
    reference = trainer(replace(nequip_cfg, nequip_eval=False), fixture_data, "error-reference")
    monkeypatch.setattr(reference.evaluator, "evaluate", lambda *a, **kw: {})
    reference.run()
    expected = load_latest(fixture_data, "error-reference")

    run = trainer(nequip_cfg, fixture_data, "error-recovery")
    attempts, logged = [], []

    def evaluate(*args, step, **kwargs):
        attempts.append(step)
        checkpoint = load_latest(fixture_data, "error-recovery")
        assert checkpoint["step"] == step and checkpoint["last_sample_step"] != step
        random.random()
        np.random.random()
        torch.rand(4)
        if len(attempts) == 1:
            raise RuntimeError("SUN evaluation failed")
        return {"sample_ema/SUN": 0.25, "joint/checked_count": 0}

    monkeypatch.setattr(run.evaluator, "evaluate", evaluate)
    monkeypatch.setattr(run.evaluator, "relaxation_diagnostics", lambda: [])
    run.wandb_run = SimpleNamespace(log=lambda metrics, step: logged.append((step, dict(metrics))))
    run.run()
    assert attempts == [2, 4] and run.step == run.last_sample_step == 4
    assert len(factories) == 1  # reused after the exception
    metrics = dict(logged)
    assert metrics[2]["status/eval_error"] == 1
    assert metrics[2]["perf/nequip_eval_seconds"] >= 0
    assert metrics[4]["eval/SUN"] == 0.25 and "eval/SUN_valid" in metrics[4]
    assert "status/eval_error" not in metrics[4]
    assert "RuntimeError: SUN evaluation failed" in capsys.readouterr().err
    assert (run.run_dir / "samples/step_0000002.pt").is_file()
    assert (run.run_dir / "samples/step_0000004.relaxed.jsonl").read_text() == ""
    actual = load_latest(fixture_data, "error-recovery")
    for key in ("model", "ema", "optimizer", "scheduler", "sampler", "rng", "loader_rng", "scaler"):
        equal(expected[key], actual[key])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_diagnostic_denominator_timing_and_atomic_failure(nequip_cfg, fixture_data, factories,
                                                        monkeypatch, device):
    run = trainer(nequip_cfg, fixture_data, f"diagnostic-{device}")
    # Only exercise the CUDA cleanup branch: sampling and every CUDA call are mocked.
    run.device = torch.device(device)
    monkeypatch.setattr(train, "sample_pool", lambda *a, **kw: dict(
        rows={}, requested_G=[], metadata={}, contains_a=torch.tensor([True]), failure_counts={}))
    monkeypatch.setattr(run.evaluator, "evaluate", lambda *a, **kw: {"joint/checked_count": 2})
    records = [dict(pool_index=0, is_un=True, stable=True, metastable=True, raw_valid=True,
                    relaxation_outcome="success", e_above_hull=0.0,
                    relaxed_min_distance=1.0, relaxed_volume_per_atom=10.0)]
    monkeypatch.setattr(run.evaluator, "relaxation_diagnostics", lambda: records)
    cleared = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))
    times = iter([0., 100., 109., 111., 200., 300., 309., 311.])
    monkeypatch.setattr(train.time, "perf_counter", lambda: next(times))
    before = rng_state()
    metrics = run.evaluate_samples()
    equal(before, rng_state())
    assert metrics["diag/SUN_plausible"] == metrics["diag/MSUN_valid"] == 0.5
    assert metrics["orbit/nequip_seconds"] == 9.0 and metrics["orbit/eval_seconds"] == 111.0
    path = run.run_dir / "samples/step_0000000.relaxed.jsonl"
    saved = path.read_bytes()
    assert [json.loads(line) for line in saved.splitlines()] == records
    # A partial serialization must leave the prior complete records untouched.
    records.append({**records[0], "e_above_hull": float("nan")})
    assert run.evaluate_samples()["orbit/nequip_error"] == 1
    assert path.read_bytes() == saved
    assert not list(path.parent.glob("*.tmp"))
    assert len(cleared) == (2 if device == "cuda" else 0)


@pytest.mark.parametrize("artifact,key", [("nequip_model", "nequip_model_sha256"),
                                          ("phase_diagram", "phase_diagram_sha256")])
def test_resume_rejects_replaced_artifact(nequip_cfg, fixture_data, factories, artifact, key):
    run = trainer(nequip_cfg, fixture_data, f"changed-{artifact}")
    run.save()
    Path(nequip_cfg.extra[artifact]).write_bytes(b"replacement")
    with pytest.raises(ValueError, match=f"Resume mismatch: {key}"):
        trainer(nequip_cfg, fixture_data, f"changed-{artifact}")


@pytest.mark.parametrize("flag,expected", [("true", True), ("false", False)])
def test_cli_nequip_flags(flag, expected):
    args = train.parser().parse_args([
        "--run-name", "cli", "--nequip-eval", flag,
        "--nequip-relax-mode", "sequential", "--nequip-relax-batch-size", "16",
    ])
    assert args.nequip_eval is expected
    assert args.nequip_relax_mode == "sequential" and args.nequip_relax_batch_size == 16
