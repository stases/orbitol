"""Stage boundaries with stubbed sampling, relaxation, scoring and W&B."""
import csv
from dataclasses import asdict
import gzip
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from pymatgen.core import Lattice, Structure

from orbitol import contracts as C, eval_gpu, eval_score, nequip_relax, posthoc_eval
from orbitol.posthoc_io import read_json


@pytest.fixture
def stage(tmp_path, monkeypatch):
    cfg = C.TrainConfig(sample_count=3, wandb_name="tiny", wandb_project="fixture-project",
                        extra=dict(sample_seed=17, sample_chunk_size=2))
    snapshot = tmp_path / "eval_snapshots/step_0000020.pt"
    snapshot.parent.mkdir()
    snapshot.write_bytes(b"tiny EMA snapshot")
    calls = dict(load=[], sample=[], relax=[], score=[], submit=[], init=[], log=[], define=[], finish=[])
    rows = dict(A0=torch.tensor([[3], [11], [0]]), F1=torch.zeros(3, 1, 3),
                Y1=torch.zeros(3, 6), pad_mask=torch.tensor([[False], [False], [True]]))
    requested = torch.tensor([1, 2, 3])
    endpoint = Structure(Lattice.cubic(4), ["Li"], [[0, 0, 0]])
    relaxed = [
        dict(index=0, constructed=True, relax_status="ok", reason=None,
             final_structure=endpoint, energy_eV=-1., nsteps=500),
        dict(index=1, constructed=True, relax_status="failed", reason="diverged, bad cell\nretry failed",
             final_structure=None, energy_eV=None, nsteps=None),
        dict(index=2, constructed=False, relax_status="not_constructed", reason="decode failed",
             final_structure=None, energy_eV=None, nsteps=None),
    ]
    for record in relaxed:
        record.update(initial_energy_eV=0. if record["index"] == 0 else None,
                      initial_fmax=.4 if record["index"] == 0 else None,
                      final_fmax=.05 if record["index"] == 0 else None,
                      final_max_force=.01 if record["index"] == 0 else None)
    thermo = posthoc_eval.ThermoSettings(thermo_ppd_mp=str(tmp_path / "hull.pkl"))
    settings = dict(sample_count=3, sun_k=0, sample_seed=17, thermo=asdict(thermo),
                    data_root=str(tmp_path), symmetry_reference=str(tmp_path / "reference.json"),
                    device="cuda", protocol="relaxed_endpoints_v1")

    def load(path):
        calls["load"].append(path)
        return dict(step=20, cfg=asdict(cfg))

    def sample(path, **kwargs):
        calls["sample"].append((path, kwargs))
        return dict(rows=rows, requested_G=requested)

    backend = object()

    def relax(items, relaxer, steps):
        assert isinstance(relaxer, nequip_relax.RobustRelaxer)
        assert relaxer.backend is backend
        calls["relax"].append((items, steps))
        return relaxed

    def score(items, records, **kwargs):
        calls["score"].append((items, records, kwargs))
        return {"sample_ema/SUN": 1 / 3, "sample_ema/SUN_raw_UN": 2 / 3, "empty": float("nan")}, [
            dict(index=i, raw_valid=i == 0, stable=i == 0, e_above_hull=-.1 if i == 0 else None,
                 relaxed_is_unique=i == 0, relaxed_is_novel=i == 0, raw_is_unique=i < 2, raw_is_novel=i < 2,
                 SUN=i == 0, MSUN=i == 0, SUN_raw_UN=i == 0, MSUN_raw_UN=i == 0,
                 initial_energy_eV=0. if i == 0 else None, initial_fmax=.4 if i == 0 else None,
                 final_fmax=.05 if i == 0 else None, rms_disp=.2 if i == 0 else None,
                 e_above_hull_oneshot=0. if i == 0 else None, SUN_oneshot=i == 0,
                 raw_space_group_01=1, relaxed_space_group_01=1 if i == 0 else None)
            for i in range(3)
        ]

    def submit(command, **kwargs):
        assert not snapshot.exists()
        assert (eval_score.stage_directory(tmp_path, 20) / "gpu.done.json").is_file()
        calls["submit"].append((command, kwargs))
        return SimpleNamespace(stdout="987;cluster\n")

    def init(**kwargs):
        calls["init"].append(kwargs)
        return SimpleNamespace(finish=lambda: calls["finish"].append(True))

    monkeypatch.setitem(sys.modules, "orbitol.eval_snapshot", SimpleNamespace(load_eval_snapshot=load))
    monkeypatch.setitem(sys.modules, "orbitol.eval_protocol", SimpleNamespace(relax_all=relax, score=score))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, log=lambda data: calls["log"].append(data),
        define_metric=lambda *a, **k: calls["define"].append((a, k))))
    monkeypatch.setattr(posthoc_eval, "sample_orbitol", sample)
    monkeypatch.setattr(posthoc_eval, "evaluation_settings", lambda args: dict(settings))
    monkeypatch.setattr(nequip_relax, "make_fast_batch_relaxer", lambda *a, **k: backend)
    monkeypatch.setattr(eval_gpu.subprocess, "run", submit)
    reference, novelty, panel, hull = object(), object(), object(), object()
    evaluator = SimpleNamespace(thermo=thermo, ref_structs=reference, novelty_ref_structs=novelty,
                                symmetry_panel=panel, sample_count=3, sample_seed=17, sun_k=0, tag="sample_ema")
    def build(saved):
        assert saved["device"] == "cpu"
        return evaluator
    monkeypatch.setattr(posthoc_eval, "build_evaluator", build)
    import src.eval.stability as stability
    monkeypatch.setattr(stability, "load_phase_diagram", lambda path: hull)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("ORBITOL_WANDB_ENTITY", "fixture-entity")
    monkeypatch.setenv("WANDB_ENTITY", "generic-entity")
    for key in ("SBATCH_PARTITION", "SBATCH_ACCOUNT", "SBATCH_GRES", "SBATCH_OUTPUT"):
        monkeypatch.setenv(key, "inherited")
    return SimpleNamespace(run=tmp_path, snapshot=snapshot, directory=eval_score.stage_directory(tmp_path, 20),
                           calls=calls, rows=rows, relaxed=relaxed, requested=requested,
                           evaluator=evaluator, hull=hull, cfg=cfg)


def test_gpu_stage_schema_deletion_submission_and_idempotency(stage):
    s = stage
    receipt = eval_gpu.main(["--run-dir", str(s.run), "--step", "20"])
    assert receipt == read_json(s.directory / "gpu.done.json")
    assert receipt["job_id"] == "123" and "torch" in receipt["versions"]
    assert all(receipt[key] >= 0 for key in ("seconds", "sample_seconds", "relax_seconds"))
    assert not s.snapshot.exists()
    samples = torch.load(s.directory / "samples.pt", weights_only=False)
    assert torch.equal(samples["rows"]["A0"], s.rows["A0"])
    assert torch.equal(samples["requested_G"], s.requested)
    assert samples["evaluation_settings"]["sampler"]["cfg"]["sample_token_budget"] == 8192
    assert samples["evaluation_settings"]["sampler"]["cfg"]["compile_trunk"] == "reduce-overhead"
    assert s.calls["load"] == [s.snapshot]
    assert s.calls["sample"] == [(s.snapshot, dict(n=3, seed=17, device="cuda", chunk=2))]
    assert s.calls["relax"][0][1] == 500
    with gzip.open(s.directory / "relaxed.csv.gz", "rt") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(eval_score.RELAXED_FIELDS)
        records = list(reader)
    assert [row["relax_status"] for row in records] == ["ok", "failed", "not_constructed"]
    assert json.loads(records[0]["final_structure"])["sites"][0]["label"] == "Li"
    assert records[1]["reason"] == s.relaxed[1]["reason"]
    command, kwargs = s.calls["submit"][0]
    assert command == [str(eval_gpu.ROOT / "slurm/submit"), "--parsable", str(eval_gpu.ROOT / "slurm/eval_score.sbatch"),
                       "--run-dir", str(s.run), "--step", "20"]
    assert kwargs["check"] is True and kwargs["cwd"] == eval_gpu.ROOT
    assert kwargs["env"]["ORBITOL_REPO_ROOT"] == str(eval_gpu.ROOT)
    # The wrapper selects CPU placement and removes the inherited GPU request.
    for key in ("SBATCH_PARTITION", "SBATCH_ACCOUNT", "SBATCH_GRES"):
        assert kwargs["env"][key] == "inherited"
    assert kwargs["env"]["SBATCH_OUTPUT"] == "inherited"
    assert eval_gpu.run_stage(s.run, 20) == receipt
    assert all(len(s.calls[name]) == 1 for name in ("load", "sample", "relax", "submit"))


def test_score_stage_schema_wandb_and_idempotency(stage):
    s = stage
    eval_gpu.run_stage(s.run, 20)
    receipt = eval_score.main(["--run-dir", str(s.run), "--step", "20"])
    assert read_json(s.directory / "metrics.json") == {"sample_ema/SUN": 1 / 3, "sample_ema/SUN_raw_UN": 2 / 3, "empty": None}
    with gzip.open(s.directory / "per_sample.csv.gz", "rt") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 3 and records[0]["SUN"] == "True"
    assert records[2]["e_above_hull"] == ""
    assert records[0]["initial_energy_eV"] == "0.0" and records[0]["rms_disp"] == "0.2"
    assert records[0]["SUN_oneshot"] == "True" and records[2]["final_fmax"] == ""
    _, relaxed, kwargs = s.calls["score"][0]
    assert relaxed == s.relaxed
    assert kwargs["ppd"] is s.hull and kwargs["references"]["ref_structs"] is s.evaluator.ref_structs
    assert kwargs["step"] == 20 and kwargs["tag"] == "sample_ema"
    assert kwargs["settings"]["thermo"]["thermo_relax_steps"] == 500
    assert torch.equal(kwargs["requested_G"], s.requested)
    logged = s.calls["init"][0]
    assert (logged["entity"], logged["project"], logged["name"]) == ("fixture-entity", "fixture-project", "tiny-eval")
    assert logged["resume"] == "allow"
    assert (("*",), {"step_metric": "train_step"}) in s.calls["define"]
    assert s.calls["log"][0]["train_step"] == 20
    assert s.calls["finish"] == [True]
    assert eval_score.run_stage(s.run, 20) == receipt
    assert len(s.calls["score"]) == len(s.calls["log"]) == 1


def test_score_requires_gpu_completion(stage):
    with pytest.raises(FileNotFoundError, match="GPU stage"):
        eval_score.run_stage(stage.run, 20)
    assert not stage.calls["score"] and not stage.calls["log"]


def test_legacy_subset_is_rejected_before_sampling_or_snapshot_deletion(stage, monkeypatch):
    monkeypatch.setattr(sys.modules["orbitol.eval_snapshot"], "load_eval_snapshot", lambda path:
                        dict(step=20, cfg={**asdict(stage.cfg), "sun_k": 512}))
    with pytest.raises(ValueError, match="use sync or posthoc"):
        eval_gpu.run_stage(stage.run, 20)
    assert stage.snapshot.exists() and not stage.calls["sample"] and not stage.calls["submit"]


def test_failed_relaxation_write_keeps_snapshot_and_rerun_overwrites(stage, monkeypatch):
    s = stage
    original = eval_gpu.write_csv
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(eval_gpu, "write_csv", fail)
    with pytest.raises(OSError, match="disk full"):
        eval_gpu.run_stage(s.run, 20)
    assert s.snapshot.exists() and not (s.directory / "gpu.done.json").exists()
    assert not s.calls["submit"]
    (s.directory / "samples.pt").write_bytes(b"partial output")
    monkeypatch.setattr(eval_gpu, "write_csv", original)
    eval_gpu.run_stage(s.run, 20)
    assert torch.load(s.directory / "samples.pt", weights_only=False)["rows"]["A0"].shape == (3, 1)
    assert not s.snapshot.exists() and len(s.calls["submit"]) == 1


def test_failed_wandb_leaves_scoring_retryable(stage, monkeypatch):
    s = stage
    eval_gpu.run_stage(s.run, 20)
    wandb = sys.modules["wandb"]
    log = wandb.log
    def fail(*args):
        raise RuntimeError("offline service")
    monkeypatch.setattr(wandb, "log", fail)
    with pytest.raises(RuntimeError, match="offline service"):
        eval_score.run_stage(s.run, 20)
    assert not (s.directory / "score.done.json").exists()
    assert (s.directory / "metrics.json").is_file() and s.calls["finish"] == [True]
    (s.directory / "metrics.json").write_text("partial output")
    monkeypatch.setattr(wandb, "log", log)
    eval_score.run_stage(s.run, 20)
    assert read_json(s.directory / "metrics.json")["sample_ema/SUN"] == 1 / 3
    assert len(s.calls["log"]) == 1
    assert s.calls["init"][0]["id"] == s.calls["init"][1]["id"]


def test_submission_failure_retains_completed_outputs_for_direct_score_retry(stage, monkeypatch):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="submission unavailable")
    monkeypatch.setattr(eval_gpu.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        eval_gpu.run_stage(stage.run, 20)
    assert (stage.directory / "gpu.done.json").is_file() and not stage.snapshot.exists()
    eval_score.run_stage(stage.run, 20)
    assert (stage.directory / "score.done.json").is_file()


def test_csv_atomic_failure_preserves_existing_output(stage, monkeypatch):
    from orbitol import posthoc_io
    path = stage.run / "relaxed.csv.gz"
    eval_score.write_csv(path, stage.relaxed, fields=eval_score.RELAXED_FIELDS)
    before = path.read_bytes()
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(posthoc_io.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        eval_score.write_csv(path, stage.relaxed[:1], fields=eval_score.RELAXED_FIELDS)
    assert path.read_bytes() == before and not list(stage.run.glob(".*.tmp"))
