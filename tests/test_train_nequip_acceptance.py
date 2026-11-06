"""Frozen acceptance tests for in-training NequIP SUN/MSUN (synchronous, Crystalite-style)."""
import json
import math
import pickle
from types import SimpleNamespace

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol import nequip_relax
from orbitol.contracts import TrainConfig
from orbitol.train import Trainer

from test_train import fixture_data  # noqa: F401  -- tiny cache and tables fixture shared with test_train


class MockBackend:
    """Relaxation stand-in: returns the input crystal with a fixed energy per atom."""

    calls = 0

    def relax(self, structure, steps=200, verbose=False):
        MockBackend.calls += 1
        return {"final_structure": structure.copy(),
                "trajectory": SimpleNamespace(energies=[-3.0 * len(structure)], forces=None)}

    def relax_many(self, structures, steps=200, verbose=False):
        return [self.relax(s, steps=steps, verbose=verbose) for s in structures]


@pytest.fixture
def mock_relaxer(monkeypatch, fixture_data):
    MockBackend.calls = 0
    made = []
    # Mocked relaxation still uses real hull scoring, with tiny local inputs.
    extra, root = fixture_data
    model, phase = root / "mock.nequip.pth", root / "mock-phase.pkl"
    model.write_bytes(b"mock TorchScript model")
    with phase.open("wb") as stream:
        pickle.dump(PhaseDiagram([ComputedEntry(el, 0.) for el in ("Li", "Na", "O", "Ti")]), stream)
    monkeypatch.setitem(extra, "nequip_model", str(model))
    monkeypatch.setitem(extra, "phase_diagram", str(phase))

    def factory(*args, **kwargs):
        made.append((args, kwargs))
        return MockBackend()

    monkeypatch.setattr(nequip_relax, "make_batch_relaxer", factory)
    monkeypatch.setattr(nequip_relax, "make_sequential_relaxer", factory)
    return made


def tiny(extra, **kwargs):
    base = dict(d_model=32, n_layers=2, n_heads=4, batch_size=8, num_workers=0, warmup_steps=1, precision="fp32",
                compile_trunk="none", pad_multiple=0, sample_token_budget=0,
                eval_mode="sync", nequip_eval=False, nequip_backend="torchscript", thermo_relax_steps=200, sun_k=512,
                objective="P", sample_count=8, sample_num_steps=5, val_every=0, log_every=1, extra=dict(extra))
    base.update(kwargs)
    return TrainConfig(**base)


def rows(root, name):
    return [json.loads(line) for line in (root / "runs" / name / "metrics.jsonl").read_text().splitlines() if line]


def test_default_config_enables_nequip_and_explicit_opt_out(fixture_data):
    extra, root = fixture_data
    defaults = TrainConfig()
    assert defaults.nequip_eval is True and defaults.eval_mode == "async"
    assert defaults.nequip_backend == "oeq" and defaults.thermo_relax_steps == 500
    assert defaults.sample_count == 10000 and defaults.sun_k == 0
    trainer = Trainer(tiny(extra, max_steps=1), run_name="nequip-off", out_root=root / "runs", device="cpu",
                      no_wandb=True)
    assert trainer.evaluator.thermo is None and trainer.evaluator.sun_k == 0


def test_nequip_eval_logs_sun_msun_and_diagnostics(fixture_data, mock_relaxer):
    extra, root = fixture_data
    cfg = tiny(extra, max_steps=3, sample_frequency=3, nequip_eval=True, sun_k=4)
    trainer = Trainer(cfg, run_name="nequip-on", out_root=root / "runs", device="cpu", no_wandb=True)
    assert mock_relaxer, "the relaxer must be built once at start-up (fail fast), not at the first evaluation"
    assert trainer.evaluator.thermo is not None and trainer.evaluator.sun_k == 4
    trainer.run()
    logged = {key: value for row in rows(root, "nequip-on") for key, value in row.items()}
    for key in ("eval/SUN", "eval/MSUN", "eval_legacy/joint_checked_count", "eval_legacy/joint_SUN", "eval/SUN_plausible",
                "eval/MSUN_valid", "eval_count/suspect_stable", "perf/nequip_eval_seconds"):
        assert key in logged, key
    assert logged["eval_legacy/joint_checked_count"] >= 0 and math.isfinite(logged["perf/nequip_eval_seconds"])
    records = root / "runs" / "nequip-on" / "samples" / "step_0000003.relaxed.jsonl"
    assert records.is_file()
    lines = [json.loads(line) for line in records.read_text().splitlines() if line]
    assert len(lines) == int(logged["eval_legacy/joint_checked_count"])
    if lines:
        assert MockBackend.calls > 0 and {"pool_index", "e_above_hull", "stable", "metastable"} <= lines[0].keys()


def test_nequip_eval_does_not_change_training(fixture_data, mock_relaxer):
    extra, root = fixture_data
    losses = {}
    for flag in (False, True):
        name = f"isolation-{flag}"
        cfg = tiny(extra, max_steps=4, sample_frequency=2, nequip_eval=flag, sun_k=4)
        Trainer(cfg, run_name=name, out_root=root / "runs", device="cpu", no_wandb=True).run()
        losses[flag] = [row["train/loss_total"] for row in rows(root, name) if "train/loss_total" in row]
    assert len(losses[False]) == 4 and losses[False] == losses[True]


def test_resume_accepts_checkpoints_without_the_new_fields(fixture_data):
    extra, root = fixture_data
    cfg = tiny(extra, max_steps=2, sample_frequency=0)
    trainer = Trainer(cfg, run_name="old-cfg", out_root=root / "runs", device="cpu", no_wandb=True)
    trainer.run()
    latest = root / "runs" / "old-cfg" / "checkpoints" / "latest.pt"
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    for key in ("nequip_eval", "nequip_relax_mode", "nequip_relax_batch_size", "gem_mode"):
        checkpoint["cfg"].pop(key, None)
    torch.save(checkpoint, latest)
    resumed = Trainer(tiny(extra, max_steps=2, sample_frequency=0), run_name="old-cfg", out_root=root / "runs",
                      device="cpu", no_wandb=True, resume="auto")
    assert resumed.step == 2


def test_missing_nequip_model_fails_at_startup(fixture_data):
    extra, root = fixture_data
    cfg = tiny(dict(extra, nequip_model="/nonexistent/model.nequip.pth"), max_steps=1, nequip_eval=True)
    with pytest.raises((FileNotFoundError, ValueError)):
        Trainer(cfg, run_name="no-model", out_root=root / "runs", device="cpu", no_wandb=True)


@pytest.mark.gpu
def test_real_nequip_during_training_on_gpu(fixture_data):
    torch.use_deterministic_algorithms(False)
    extra, root = fixture_data
    cfg = tiny(extra, max_steps=5, sample_frequency=5, sample_count=64, nequip_eval=True, sun_k=16)
    trainer = Trainer(cfg, run_name="nequip-gpu", out_root=root / "runs", device="cuda", no_wandb=True)
    trainer.run()
    logged = {key: value for row in rows(root, "nequip-gpu") for key, value in row.items()}
    print({key: logged[key] for key in ("eval_legacy/joint_checked_count", "eval/SUN", "eval/MSUN",
                                        "perf/nequip_eval_seconds") if key in logged})
    assert "eval/MSUN" in logged and math.isfinite(logged["perf/nequip_eval_seconds"])
