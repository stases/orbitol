"""Finite Slurm submission and bounded async/sync trainer dispatch."""
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from orbitol import contracts as C, eval_jobs, eval_snapshot, train
from orbitol.train import Trainer
from test_train import cfg, fixture_data, trainer  # noqa: F401


@pytest.mark.parametrize("output,job_id", [("4123\n", "4123"), ("9876;cluster\n", "9876")])
def test_submit_command(tmp_path, monkeypatch, output, job_id):
    calls = []

    def submit(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(eval_jobs.subprocess, "run", submit)
    cfg = C.TrainConfig(eval_partition="other", eval_account="otherusers", eval_gres="gpu:rtx6000:1")
    run_dir = tmp_path / "run with spaces"
    assert eval_jobs.submit_eval(run_dir, 20000, cfg) == job_id
    command, kwargs = calls.pop()
    assert "WANDB_MODE" not in kwargs["env"] or kwargs["env"]["WANDB_MODE"] == os.environ.get("WANDB_MODE")
    assert eval_jobs.submit_eval(run_dir, 20000, cfg, wandb=False) == job_id
    assert calls.pop()[1]["env"]["WANDB_MODE"] == "disabled"
    assert command == [
        str(eval_jobs.ROOT / "slurm/submit"), "--parsable", "--partition=other", "--account=otherusers", "--gres=gpu:rtx6000:1",
        str(eval_jobs.ROOT / "slurm/eval_gpu.sbatch"), "--run-dir", str(run_dir), "--step", "20000",
    ]
    assert kwargs["cwd"] == eval_jobs.ROOT
    assert kwargs["env"]["ORBITOL_REPO_ROOT"] == str(eval_jobs.ROOT)
    assert {key: kwargs[key] for key in ("check", "capture_output", "text", "timeout")} == dict(
        check=True, capture_output=True, text=True, timeout=30)
    assert "shell" not in kwargs and not calls


@pytest.mark.parametrize("failure", [
    FileNotFoundError("sbatch missing"),
    subprocess.CalledProcessError(1, ["sbatch"], stderr="account denied"),
    subprocess.TimeoutExpired(["sbatch"], timeout=30),
    "", "Submitted batch job 42", "42\n43", "42;cluster;extra",
])
def test_submission_failure_is_logged(tmp_path, monkeypatch, caplog, failure):
    def submit(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return SimpleNamespace(stdout=failure)

    monkeypatch.setattr(eval_jobs.subprocess, "run", submit)
    with caplog.at_level(logging.ERROR, logger="orbitol.eval_jobs"):
        assert eval_jobs.submit_eval(tmp_path, 20, C.TrainConfig()) == ""
    assert "orbit/eval_submit_failed=1" in caplog.text
    if isinstance(failure, subprocess.CalledProcessError):
        assert "account denied" in caplog.text


@pytest.mark.parametrize("count,maximum", [(2, 2), (3, 2), (1, 1)])
def test_backlog_cap_skips_without_writing_or_submitting(tmp_path, monkeypatch, count, maximum):
    run = object.__new__(Trainer)
    run.cfg = C.TrainConfig(eval_backlog_max=maximum)
    run.run_dir, run.step, run.last_sample_step = tmp_path, 20000, -1
    directory = tmp_path / "eval_snapshots"
    directory.mkdir()
    for step in range(count):
        (directory / f"step_{step:07d}.pt").touch()

    def forbidden(*args, **kwargs):
        pytest.fail("Backlog must skip snapshotting, submission and in-process sampling")

    monkeypatch.setattr(eval_snapshot, "write_eval_snapshot", forbidden)
    monkeypatch.setattr(eval_jobs, "submit_eval", forbidden)
    monkeypatch.setattr(train, "sample_pool", forbidden)
    assert run.evaluate_samples() == {"orbit/eval_skipped_backlog": 1}
    assert run.last_sample_step == run.step
    assert not run._sampling_due()
    assert len(eval_snapshot.pending_eval_snapshots(tmp_path)) == count


@pytest.mark.parametrize("submit_ok", [True, False])
def test_async_training_submits_bounded_snapshots_and_continues(cfg, fixture_data, monkeypatch, submit_ok):
    def forbidden(*args, **kwargs):
        pytest.fail("Async training must not construct an evaluator or sample in-process")

    monkeypatch.setattr(train, "DNGEvaluator", forbidden)
    monkeypatch.setattr(train, "sample_pool", forbidden)
    submitted = []

    def submit(command, **kwargs):
        step = int(command[command.index("--step") + 1])
        run_dir = Path(command[command.index("--run-dir") + 1])
        path = run_dir / "eval_snapshots" / f"step_{step:07d}.pt"
        assert path.is_file()  # Publication precedes submission.
        assert torch.load(path, weights_only=False)["step"] == step
        submitted.append(step)
        if not submit_ok:
            raise subprocess.CalledProcessError(1, command, stderr="Slurm unavailable")
        return SimpleNamespace(stdout=f"{100 + step}\n")

    monkeypatch.setattr(eval_jobs.subprocess, "run", submit)
    cfg = replace(cfg, eval_mode="async", nequip_eval=True, nequip_backend="oeq", thermo_relax_steps=500,
                  sample_count=2, sample_frequency=1, max_steps=3, ckpt_every=0)
    # subprocess.run is shared with check_output; isolate Git lookup from the Slurm stub.
    monkeypatch.setattr(train.subprocess, "check_output", lambda *args, **kwargs: "test-commit\n")
    run = trainer(cfg, fixture_data, f"async-jobs-{submit_ok}")
    assert run.evaluator is None
    saves = []
    save = run.save

    def record_save(**kwargs):
        saves.append(run.step)
        save(**kwargs)

    monkeypatch.setattr(run, "save", record_save)
    run.run()
    assert run.step == run.last_sample_step == 3
    assert submitted == [1, 2]
    assert saves == [3]  # No full resume checkpoint is written solely for async evaluation.
    paths = eval_snapshot.pending_eval_snapshots(run.run_dir)
    assert [path.name for path in paths] == ["step_0000001.pt", "step_0000002.pt"]
    assert not (run.checkpoint_dir / "eval").exists()
    assert not list((run.run_dir / "samples").iterdir())
    metrics = [json.loads(line) for line in (run.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row.get("status/eval_snapshot_step") for row in metrics] == [1, 2, None]
    assert metrics[-1]["status/eval_skipped_backlog"] == 1
    key = "status/eval_job_id" if submit_ok else "status/eval_submit_failed"
    assert [row[key] for row in metrics[:2]] == (["101", "102"] if submit_ok else [1, 1])
    assert "status/eval_submit_failed" not in metrics[-1]
    paths[0].unlink()  # Completed GPU work frees one backlog slot.
    run.step = 4
    assert run.evaluate_samples()["orbit/eval_snapshot_step"] == 4
    assert submitted == [1, 2, 4]
    run.close()


@pytest.mark.parametrize("use_nequip", [False, True])
def test_sync_dispatch_uses_evaluator_and_never_submits(cfg, fixture_data, monkeypatch, use_nequip):
    calls = []

    class Evaluator:
        tag = "sample_ema"

        def __init__(self, data_root, **kwargs):
            calls.append((data_root, kwargs))
            self.symmetry_panel = SimpleNamespace(reference={})

        def evaluate(self, rows, *, step, requested_G):
            assert rows["A0"].shape == (2, 20)
            assert requested_G.shape == (2,)
            calls.append(step)
            return {"sample_ema/valid_rate": .5}

        def relaxation_diagnostics(self):
            return []

    def forbidden(*args, **kwargs):
        pytest.fail("Sync training must not snapshot or submit async work")

    monkeypatch.setattr(train, "DNGEvaluator", Evaluator)
    monkeypatch.setattr(eval_jobs, "submit_eval", forbidden)
    monkeypatch.setattr(eval_snapshot, "write_eval_snapshot", forbidden)
    model_path = fixture_data[1] / "sync-model.pth"
    phase_path = fixture_data[1] / "sync-phase.pkl"
    model_path.write_bytes(b"stub model; no MLIP loaded")
    phase_path.write_bytes(b"stub phase diagram; scoring is stubbed")
    thermo = SimpleNamespace(model_path=str(model_path), make_relaxer=lambda: object())
    monkeypatch.setattr(train.ThermoSettings, "from_train_config", lambda *args, **kwargs: thermo)
    cfg = replace(cfg, sample_count=2, sample_frequency=1, max_steps=1, eval_mode="sync",
                  nequip_eval=use_nequip, extra={**cfg.extra, "phase_diagram": str(phase_path)})
    run = trainer(cfg, fixture_data, f"sync-dispatch-{use_nequip}")
    run.run()
    assert calls[0][1]["sample_count"] == 2
    assert calls[0][1]["thermo"] is (thermo if use_nequip else None)
    assert calls[1:] == [1]
    assert run.last_sample_step == 1 and eval_snapshot.pending_eval_snapshots(run.run_dir) == []
    rows = [json.loads(line) for line in (run.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["eval/valid_rate"] == .5
    assert "status/eval_snapshot_step" not in rows[-1]
    assert "status/eval_error" not in rows[-1]
    assert ("perf/nequip_eval_seconds" in rows[-1]) == use_nequip
    run.close()


@pytest.mark.parametrize("partition,account", [("", ""), ("custom", ""), ("", "custom")])
def test_empty_placement_omits_flags(tmp_path, monkeypatch, partition, account):
    calls = []
    def submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="123")
    monkeypatch.setattr(eval_jobs.subprocess, "run", submit)
    cfg = C.TrainConfig(eval_partition=partition, eval_account=account)
    assert eval_jobs.submit_eval(tmp_path, 1, cfg) == "123"
    assert [arg for arg in calls[0] if arg.startswith("--partition=")] == (
        [f"--partition={partition}"] if partition else [])
    assert [arg for arg in calls[0] if arg.startswith("--account=")] == (
        [f"--account={account}"] if account else [])
