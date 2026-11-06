"""EMA storage, publication failures and compatibility with the real sampler."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from orbitol import contracts as C
from orbitol import eval_snapshot
from orbitol.model import build_model
from orbitol.sample import rng_state, sample_from_checkpoint, sample_pool
from orbitol.train import EMA, Trainer
from test_train import cfg, equal, fixture_data, trainer  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"format_version", "model", "ema", "cfg", "step", "table_sha256", "encoding_sha256",
        "core_manifest", "commit", "bank_sha256", "data_hashes", "allowed_mask"}


def snapshot_trainer(model=None):
    run = object.__new__(Trainer)
    run.cfg = C.TrainConfig(compile_trunk="none")
    run.model = model if model is not None else torch.nn.Linear(128, 128)
    run.ema = EMA(run.model, run.cfg.ema_decay)
    with torch.no_grad():
        next(run.model.parameters()).add_(1.)  # Online weights deliberately differ.
    run.step = 17
    run.table_sha256, run.bank_sha256 = "tables", "bank"
    run.data_hashes = {"train": "train", "val": "val"}
    run.core_manifest = json.loads((ROOT / "orbitol/core_manifest.json").read_text())
    run.commit = "test"
    run.allowed_mask = torch.ones(100, dtype=torch.bool)
    run.nequip_model_sha256 = run.phase_diagram_sha256 = None
    return run


def test_ema_only_keys_size_and_live_state_unchanged(tmp_path):
    model = torch.nn.Linear(128, 128)
    model.register_buffer("reference", torch.arange(4))
    model.register_parameter("frozen", torch.nn.Parameter(torch.ones(3), requires_grad=False))
    run = snapshot_trainer(model)
    before = {name: tensor.clone() for name, tensor in run.model.state_dict().items()}
    ordinary = tmp_path / "ordinary.pt"
    torch.save(run.snapshot(), ordinary)
    path = eval_snapshot.write_eval_snapshot(run, tmp_path, run.step)
    payload = eval_snapshot.load_eval_snapshot(path)
    assert path == tmp_path / "eval_snapshots/step_0000017.pt"
    assert set(payload) == KEYS
    assert payload["step"] == 17 and payload["cfg"] == run.snapshot()["cfg"]
    assert payload["model"].keys() == before.keys()
    for name, tensor in payload["ema"].items():
        assert torch.equal(payload["model"][name], run.ema.shadow[name])
        assert payload["model"][name].data_ptr() == tensor.data_ptr()
        assert tensor.device.type == "cpu"
    assert torch.equal(payload["model"]["reference"], before["reference"])
    assert torch.equal(payload["model"]["frozen"], before["frozen"])
    assert path.stat().st_size < .6 * ordinary.stat().st_size
    equal(before, run.model.state_dict())


def test_atomic_publication_and_pending_filter(tmp_path, monkeypatch):
    run = snapshot_trainer()
    target = tmp_path / "eval_snapshots/step_0000017.pt"
    assert eval_snapshot.pending_eval_snapshots(tmp_path) == []
    save = torch.save

    def observe(payload, stream):
        assert not target.exists()
        assert eval_snapshot.pending_eval_snapshots(tmp_path) == []
        save(payload, stream)
        assert not target.exists()
        assert eval_snapshot.pending_eval_snapshots(tmp_path) == []
        assert len(list(target.parent.glob(".step_0000017.pt.*.tmp"))) == 1

    monkeypatch.setattr(torch, "save", observe)
    eval_snapshot.write_eval_snapshot(run, tmp_path, 17)
    assert eval_snapshot.pending_eval_snapshots(tmp_path) == [target]
    assert list(target.parent.iterdir()) == [target]
    monkeypatch.setattr(torch, "save", save)
    second = eval_snapshot.write_eval_snapshot(run, tmp_path, 10000000)
    first = eval_snapshot.write_eval_snapshot(run, tmp_path, 2)
    for name in ("step_0000001.pt.tmp", ".step_0000003.pt.tmp", "step_invalid.pt", "step_4.pt", "latest.pt"):
        (target.parent / name).touch()
    (target.parent / "step_0000008.pt").mkdir()
    assert eval_snapshot.pending_eval_snapshots(tmp_path) == [first, target, second]
    target.unlink()  # The GPU stage signals completion by removing its snapshot.
    assert eval_snapshot.pending_eval_snapshots(tmp_path) == [first, second]


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_atomic_failure_preserves_previous_snapshot(tmp_path, monkeypatch, failure):
    run = snapshot_trainer()
    path = eval_snapshot.write_eval_snapshot(run, tmp_path, 17)
    before = path.read_bytes()

    def fail(*args):
        if failure == "serialize":
            args[1].write(b"partial snapshot")
        raise OSError("disk failure")

    if failure == "serialize":
        monkeypatch.setattr(torch, "save", fail)
    else:
        from orbitol import posthoc_io
        monkeypatch.setattr(posthoc_io.os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        eval_snapshot.write_eval_snapshot(run, tmp_path, 17)
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


def test_load_normalizes_historical_sequential_config(tmp_path):
    run = snapshot_trainer()
    payload = run.snapshot()
    payload["cfg"] = dict(nequip_relax_mode="sequential", sample_count=8, sun_k=2)
    path = tmp_path / "historical.pt"
    torch.save(payload, path)
    cfg = C.TrainConfig(**eval_snapshot.load_eval_snapshot(path)["cfg"])
    assert (cfg.nequip_backend, cfg.eval_mode, cfg.thermo_relax_steps) == ("torchscript", "sync", 200)
    assert (cfg.nequip_relax_mode, cfg.sample_count, cfg.sun_k) == ("sequential", 8, 2)


def test_real_sampler_accepts_ema_snapshot(cfg, fixture_data):
    run = trainer(replace(cfg, eval_mode="async"), fixture_data, "eval-snapshot-sampler")
    with torch.no_grad():
        next(run.model.parameters()).add_(1.)
    before = rng_state()
    expected = sample_pool(run.model, run.ema, run.cfg, run.tables, run.tb, run.encoding,
                          run.allowed_mask, run.bank, n=2, seed=123, device="cpu", chunk=2)
    path = eval_snapshot.write_eval_snapshot(run, run.run_dir, run.step)
    actual = sample_from_checkpoint(path, n=2, seed=123, device="cpu", chunk=2)
    for key in ("rows", "requested_G", "skeletons", "raw", "contains_a", "failures"):
        equal(expected[key], actual[key])
    equal(before, rng_state())
    assert actual["metadata"]["step"] == run.step
    run.close()


def test_d512_snapshot_size(tmp_path):
    cfg = C.TrainConfig(compile_trunk="none")
    model = build_model(cfg, torch.zeros(C.R_TOTAL, C.ROW_DESC_DIM), torch.zeros(C.G_TOTAL, C.SG_DESC_DIM))
    run = snapshot_trainer(model)
    path = eval_snapshot.write_eval_snapshot(run, tmp_path, run.step)
    size = path.stat().st_size
    weight_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    # One set of model storage plus archive/metadata overhead, never two weight copies.
    assert weight_bytes <= size < weight_bytes + 1_000_000
    print(f"d512: {size} bytes ({size / 1e6:.3f} MB); tensor bytes={weight_bytes}; keys={sorted(KEYS)}")
    path.unlink()
