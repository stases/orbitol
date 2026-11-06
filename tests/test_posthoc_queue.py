"""Queue state machine tests use tiny fake snapshots, never live runs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from orbitol import posthoc_queue as queue
from orbitol.posthoc_io import identity, read_json, sha256, write_json


@pytest.fixture
def tmp_path():
    # Keep pytest's temporary fixtures inside the checkout.
    parent = Path(__file__).resolve().parents[1] / ".pytest_cache"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="queue-", dir=parent) as name:
        yield Path(name)


@pytest.fixture(autouse=True)
def disabled_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture
def setup(tmp_path):
    runs = [queue.Run("crystalite", tmp_path / "baseline"), queue.Run("orbitol", tmp_path / "orbitol")]
    for run in runs:
        (run.path / "checkpoints").mkdir(parents=True)
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    argv = ["--out-root", str(tmp_path / "out"), "--symmetry-reference", str(reference), "--device", "cpu",
            "--sample-count", "2048", "--sun-k", "512", "--nequip-backend", "torchscript", "--relax-steps", "200"]
    for run in runs:
        argv += ["--run", f"{run.kind}:{run.path}"]
    args = queue.parse_args(argv)
    clock = [10000.]
    calls = dict(sample=[], evaluate=[], log=[])

    def snapshot(run, step, age=1000):
        path = run.path / "checkpoints" / f"step_{step:07d}.pt"
        path.write_bytes(f"fake snapshot {step}".encode())
        os.utime(path, (clock[0] - age, clock[0] - age))
        return path

    def sample(run, snap, directory):
        calls["sample"].append((run.name, snap["step"]))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "samples.pt").write_bytes(f"pool {run.name} {snap['step']}".encode())
        write_json(directory / "evaluation_settings.json", {"thermo": {}})
        pool = dict(kind=run.kind, snapshot=snap["path"], step=snap["step"],
                    snapshot_sha256=sha256(snap["path"]), **identity(snap["path"]),
                    samples_sha256=sha256(directory / "samples.pt"), status="complete",
                    n=args.sample_count, seed=args.seed, sampler={}, sample_seconds=.01)
        write_json(directory / "pool.json", pool)
        return pool

    def evaluate(run, directory):
        step = int(directory.name[5:])
        calls["evaluate"].append((run.name, step))
        return {"train_step": step}

    def log(run, directory, metrics):
        calls["log"].append((run.name, int(directory.name[5:])))
        return {"time": clock[0]}

    def worker(options=None, **kwargs):
        return queue.Queue(options or args, sample_fn=kwargs.get("sample_fn", sample),
                           evaluate_fn=kwargs.get("evaluate_fn", evaluate), log_fn=kwargs.get("log_fn", log),
                           now=lambda: clock[0])

    def pool(run, step):
        snap = next(s for s in queue.snapshots(run) if s["step"] == step)
        directory = args.out_root / run.name / f"step_{step:07d}"
        sample(run, snap, directory)
        return directory

    return SimpleNamespace(args=args, argv=argv, runs=runs, clock=clock, calls=calls,
                           snapshot=snapshot, sample=sample, evaluate=evaluate, log=log,
                           worker=worker, pool=pool)


@pytest.mark.parametrize("keep", [1, 2])
def test_samples_every_ready_snapshot_then_keeps_newest_k(setup, keep):
    s = setup
    s.args.keep_snapshots = keep
    for step in (20000, 40000, 60000, 80000):
        s.snapshot(s.runs[0], step)
    worker = s.worker()
    worker.pass_once()
    assert s.calls["sample"] == []  # a first observation is never readiness
    worker.pass_once()
    assert [step for _, step in s.calls["sample"]] == [20000, 40000, 60000, 80000]
    assert [row["step"] for row in queue.snapshots(s.runs[0])] == [60000, 80000][-keep:]
    assert s.calls["evaluate"] == [("baseline", 20000)]
    status = json.loads((s.args.out_root / "queue_status.json").read_text())
    assert status["runs"]["baseline"]["pooled"] == [20000, 40000, 60000, 80000]
    assert status["runs"]["baseline"]["logged"] == [20000]


def test_prune_requires_all_evidence_and_ignores_nonmatching_names(setup):
    s, run = setup, setup.runs[0]
    paths = {step: s.snapshot(run, step) for step in range(20000, 160000, 20000)}
    for step in paths:
        if step != 60000:  # unpooled
            s.pool(run, step)
    paths[20000].write_bytes(b"changed size")
    info = paths[40000].stat()
    os.utime(paths[40000], ns=(info.st_atime_ns, info.st_mtime_ns + 1))
    info = paths[80000].stat()
    replacement = run.path / "checkpoints/replacement.tmp"
    replacement.write_bytes(paths[80000].read_bytes())
    os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
    os.replace(replacement, paths[80000])  # same size/mtime, different inode
    (s.args.out_root / run.name / "step_0100000/samples.pt").write_bytes(b"corrupt")
    others = ["latest.pt", "epoch_latest.pt", "final.pt", "step_latest.pt", "step_0000001.pt.tmp",
              "latest.pt.tmp", "step_123.pt", "step_12345678.pt"]
    for name in others:
        (run.path / "checkpoints" / name).write_bytes(b"untouchable")
    s.worker()._prune(run)
    assert not paths[120000].exists()
    assert all(path.exists() for step, path in paths.items() if step != 120000)
    assert all((run.path / "checkpoints" / name).read_bytes() == b"untouchable" for name in others)
    records = [json.loads(line) for line in (s.args.out_root / run.name / "retention.log").read_text().splitlines()]
    assert [record["step"] for record in records if record["action"] == "deleted"] == [120000]
    assert {"time", "path", "step", "size", "sha256"} <= records[0].keys()


def test_young_and_growing_snapshots_are_not_sampled(setup):
    s = setup
    young = s.snapshot(s.runs[0], 20000, age=1)
    growing = s.snapshot(s.runs[0], 40000)
    worker = s.worker()
    worker.pass_once()
    growing.write_bytes(b"still writing")
    os.utime(growing, (s.clock[0] - 1000, s.clock[0] - 1000))
    worker.pass_once()
    assert not s.calls["sample"]
    s.clock[0] += 121
    worker.pass_once()
    assert s.calls["sample"] == [("baseline", 20000), ("baseline", 40000)]
    assert growing.exists() and not young.exists()


def test_dry_run_only_records_would_delete(setup):
    s = setup
    paths = [s.snapshot(s.runs[0], step) for step in (20000, 40000)]
    for step in (20000, 40000):
        s.pool(s.runs[0], step)
    s.calls["sample"].clear()
    s.args.dry_run = True
    s.worker().pass_once()
    assert all(path.exists() for path in paths)
    assert s.calls == dict(sample=[], evaluate=[], log=[])
    log = json.loads((s.args.out_root / "baseline/retention.log").read_text())
    assert log["action"] == "would delete"


def test_global_order_errors_and_restart_without_resampling_or_relogging(setup):
    s = setup
    for run, step in ((s.runs[0], 40000), (s.runs[1], 20000), (s.runs[1], 60000)):
        s.snapshot(run, step)
    def evaluate(run, directory):
        metrics = s.evaluate(run, directory)
        if int(directory.name[5:]) == 20000:
            raise ValueError("bad pool")
        return metrics
    s.worker(evaluate_fn=evaluate).pass_once()
    s.worker(evaluate_fn=evaluate).pass_once()
    failure = s.args.out_root / "orbitol/step_0020000/error.json"
    assert "bad pool" in json.loads(failure.read_text())["traceback"]
    for _ in range(3):
        s.worker(evaluate_fn=evaluate).pass_once()
    assert [step for _, step in s.calls["evaluate"]] == [20000, 40000, 60000]
    assert len(s.calls["sample"]) == 3
    assert s.calls["log"] == [("baseline", 40000), ("orbitol", 60000)]


def test_load_failures_are_retried_then_quarantined_and_snapshot_kept(setup):
    s = setup
    snapshot = s.snapshot(s.runs[0], 20000, age=1000)
    calls = []
    def fail(*args):
        calls.append(1)
        raise ValueError("incomplete torch archive")
    for _ in range(6):
        s.worker(sample_fn=fail).pass_once()
    assert len(calls) == 3
    error = json.loads((s.args.out_root / "baseline/step_0020000/pool_error.json").read_text())
    assert error["count"] == 3 and snapshot.exists()
    assert "incomplete torch archive" in error["traceback"]


def test_final_and_idle_exit_conditions_include_pending_and_unlogged(setup):
    s = setup
    s.args.final_step = 20000
    for run in s.runs:
        s.snapshot(run, 20000)
    worker = s.worker()
    assert worker.pass_once() is False
    assert worker.pass_once() is False
    assert worker.pass_once() is True  # both final pools evaluated and logged
    marker = s.args.out_root / "baseline/step_0020000/wandb_logged.json"
    marker.unlink()
    assert worker.finished() is False
    worker.pass_once()
    s.args.final_step = 300000
    assert worker.finished() is False
    s.clock[0] += s.args.idle_exit_minutes * 60
    assert worker.finished() is True
    s.snapshot(s.runs[1], 40000)
    assert worker.pass_once() is False


def test_helper_highest_first_and_lock_exclusivity(setup):
    s = setup
    for step in (20000, 40000, 60000):
        s.snapshot(s.runs[0], step)
        s.pool(s.runs[0], step)
    s.calls["sample"].clear()
    helper_args = queue.parse_args(s.argv + ["--role", "evaluate"])
    highest = s.args.out_root / "baseline/step_0060000"
    with queue.claim(highest / "eval.lock") as acquired:
        assert acquired
        with queue.claim(highest / "eval.lock") as duplicate:
            assert duplicate is False
        s.worker(helper_args).pass_once()
        assert s.calls["evaluate"] == [("baseline", 40000)]
        s.worker()._evaluate_one()
        assert s.calls["evaluate"][-1] == ("baseline", 20000)
    s.worker(helper_args).pass_once()
    assert s.calls["evaluate"][-1] == ("baseline", 60000)
    assert len(queue.snapshots(s.runs[0])) == 3
    assert not s.calls["sample"] and not s.calls["log"]
    assert not list(s.args.out_root.rglob("eval.lock"))


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGUSR1])
def test_signal_exits_zero_and_releases_stage_lock(setup, monkeypatch, sig):
    s = setup
    s.snapshot(s.runs[0], 20000)
    directory = s.pool(s.runs[0], 20000)
    def interrupt(*args):
        os.kill(os.getpid(), sig)
    worker = s.worker(evaluate_fn=interrupt)
    monkeypatch.setattr(queue, "Queue", lambda args: worker)
    assert queue.main(s.argv + ["--once"]) == 0
    assert not (directory / "eval.lock").exists()
    assert not (directory / "error.json").exists()
    assert not (directory / "metrics.json").exists()
    assert not (s.args.out_root / "queue_all.lock").exists()


def test_wandb_history_intent_recovers_remote_success_before_local_marker(setup, monkeypatch):
    monkeypatch.setenv("ORBITOL_WANDB_ENTITY", "fixture-entity")
    monkeypatch.setenv("WANDB_ENTITY", "generic-entity")
    s = setup
    run = s.runs[0]
    s.snapshot(run, 20000)
    directory = s.pool(run, 20000)
    write_json(directory / "metrics.json", {"sample_ema/SUN": .3})
    remote, events, configs = [0], [], []
    class Session:
        def __init__(self):
            self.step = remote[0]
        def finish(self):
            pass
    def init(**kwargs):
        configs.append(kwargs)
        return Session()
    def log(values, *, step, commit):
        events.append(values)
        remote[0] = step + 1
        raise queue.StopRequested()  # server accepted; local marker not written
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, log=log, define_metric=lambda *a, **k: None))
    with pytest.raises(queue.StopRequested):
        queue.log_wandb(s.args, run, directory, {"sample_ema/SUN": .3})
    s.worker(log_fn=lambda r, p, m: queue.log_wandb(s.args, r, p, m))._log_completed()
    assert len(events) == 1
    assert events[0]["eval/SUN"] == .3
    assert events[0]["train_step"] == 20000
    assert events[0]["status/pool_sha256"] == read_json(directory / "pool.json")["samples_sha256"]
    assert "sample_ema/SUN" not in events[0]
    assert (directory / "wandb_logged.json").is_file()
    assert configs[0]["id"] == configs[1]["id"]
    assert configs[0]["entity"] == "fixture-entity"
    assert configs[0]["config"]["run_dir"] == str(run.path)


def test_plan_is_read_only_and_imports_no_torch_even_without_site_packages(setup):
    s = setup
    s.snapshot(s.runs[0], 20000)
    root = s.runs[0].path.parent
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    code = "import sys; from orbitol.posthoc_queue import main; main(sys.argv[1:]); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-S", "-c", code, *s.argv, "--plan"],
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                                 "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["sample"][0]["step"] == 20000 and not plan["sample"][0]["ready"]
    assert not s.args.out_root.exists()
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_runtime_reuses_one_evaluator_across_both_kinds(setup, monkeypatch):
    from orbitol import posthoc_eval
    s = setup
    settings = dict(sample_count=2048, sample_seed=123, sun_k=512, thermo={},
                    data_root="fixture", symmetry_reference="fixture", mlip_model_sha256="fixture")
    built, evaluated = [], []
    evaluator = object()
    monkeypatch.setattr(posthoc_eval, "evaluation_settings", lambda args: settings)
    def build(config):
        built.append(config)
        return evaluator
    monkeypatch.setattr(posthoc_eval, "build_evaluator", build)
    monkeypatch.setattr(posthoc_eval, "evaluate_stage", lambda out, **kwargs: evaluated.append(kwargs["evaluator"]))
    runtime = queue.Runtime(s.args)
    for run in s.runs:
        s.snapshot(run, 20000)
        directory = s.pool(run, 20000)
        write_json(directory / "evaluation_settings.json", settings)
        runtime.evaluate(run, directory)
    assert len(built) == 1 and evaluated == [evaluator, evaluator]


def test_runtime_switches_dataset_references_after_snapshots_are_pruned(setup, monkeypatch):
    from orbitol import contracts as C, posthoc_eval
    s = setup
    s.args.symmetry_reference = None
    model = s.args.out_root.parent / "model.pth"
    model.write_bytes(b"model")
    s.args.nequip_ts_path = model
    built, evaluated = [], []

    def build(settings):
        built.append(settings)
        return settings["data_root"]

    monkeypatch.setattr(posthoc_eval, "build_evaluator", build)
    monkeypatch.setattr(posthoc_eval, "evaluate_stage", lambda out, **kw: evaluated.append(kw["evaluator"]))
    runtime = queue.Runtime(s.args)
    for run, dataset in zip(s.runs, ("mp20", "alex_mp20")):
        snap = s.snapshot(run, 20000)
        directory = s.pool(run, 20000)
        spec = C.dataset_spec(dataset)
        reference = directory / "reference.json"
        reference.write_text("{}")
        hull = directory / "hull.pkl"
        hull.write_bytes(dataset.encode())  # Only the content hash is consumed by this routing test.
        cfg = {"dataset": dataset} if run.kind == "orbitol" else {"dataset_name": dataset}
        args = SimpleNamespace(**{**vars(s.args), "kind": run.kind, "dataset_config": cfg,
                                  "symmetry_reference": reference, "ppd_path": hull})
        settings = posthoc_eval.evaluation_settings(args)
        settings["sampler"] = {"cfg": cfg} if run.kind == "orbitol" else cfg
        write_json(directory / "evaluation_settings.json", settings)
        snap.unlink()
        runtime.evaluate(run, directory)
        assert built[-1]["data_root"] == spec.data_root
    assert len(built) == 2 and evaluated == [C.DATA_ROOT, C.dataset_spec("alex_mp20").data_root]
    # Explicit queue paths still win and are checked against the committed pool.
    s.args.data_root = Path(C.DATA_ROOT)
    with pytest.raises(ValueError, match="data_root"):
        runtime.evaluate(s.runs[-1], directory)


def test_dead_lock_is_reclaimed_but_live_owner_is_preserved(setup, monkeypatch):
    path = setup.args.out_root / "eval.lock"
    write_json(path, dict(host="dead-host", pid=123, SLURM_JOB_ID="123", time=0, token="old"))
    monkeypatch.setattr(queue, "_owner_dead", lambda owner: owner["token"] == "old")
    with queue.claim(path) as acquired:
        assert acquired
        assert json.loads(path.read_text())["pid"] == os.getpid()
        with queue.claim(path) as second:
            assert second is False
    assert not path.exists()


def test_unwritable_audit_log_prevents_checkpoint_deletion(setup, monkeypatch):
    s = setup
    old = s.snapshot(s.runs[0], 20000)
    s.snapshot(s.runs[0], 40000)
    s.pool(s.runs[0], 20000)
    original = Path.open
    def full_disk(path, *args, **kwargs):
        if path.name == "retention.log":
            raise OSError("No space left on device")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", full_disk)
    s.worker()._prune(s.runs[0])
    assert old.exists()


@pytest.mark.parametrize("returncode,stdout,stderr,dead", [
    (0, "", "", True),
    (0, "RUNNING\n", "", False),
    (1, "", "slurm_load_jobs error: Invalid job id specified\n", True),
    (1, "", "slurm_load_jobs error: Socket timed out on send/recv operation\n", False),
])
def test_remote_owner_liveness_from_squeue(monkeypatch, returncode, stdout, stderr, dead):
    result = SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: result)
    owner = dict(host="another-host", pid=1, SLURM_JOB_ID="246326")
    assert queue._owner_dead(owner) is dead


def test_wandb_series_salt_changes_only_the_run_id():
    import hashlib
    legacy = hashlib.sha256("orbitol\0run-posthoc".encode()).hexdigest()[:24]
    assert queue.wandb_run_id("orbitol", "run-posthoc") == legacy
    assert queue.wandb_run_id("orbitol", "run-posthoc", "v2") not in {legacy, queue.wandb_run_id("orbitol", "run-posthoc", "v3")}
    assert queue.parse_args(["--run", "orbitol:/tmp/x", "--wandb-id-salt", "v2"]).wandb_id_salt == "v2"


def test_manual_queue_only_discovers_numbered_training_checkpoints(setup):
    s = setup
    ordinary = s.snapshot(s.runs[0], 20000)
    for path in (s.runs[0].path / "eval_snapshots/step_0040000.pt",
                 s.runs[0].path / "checkpoints/eval/step_0060000.pt"):
        path.parent.mkdir()
        path.write_bytes(b"not a manual queue checkpoint")
    assert [row["path"] for row in queue.snapshots(s.runs[0])] == [str(ordinary)]
    with pytest.raises(SystemExit):
        queue.parse_args(s.argv + ["--training-job-id", "123"])


def test_manual_queue_keeps_posthoc_wandb_identity(setup, monkeypatch):
    s = setup
    s.snapshot(s.runs[0], 20000)
    directory = s.pool(s.runs[0], 20000)
    write_json(directory / "evaluation_settings.json", dict(thermo={}, protocol="relaxed_endpoints_v1"))
    write_json(directory / "metrics.json", {"sample_ema/SUN": .2})
    initializations = []
    def init(**kwargs):
        initializations.append(kwargs)
        return SimpleNamespace(step=0, finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, log=lambda *a, **k: None, define_metric=lambda *a, **k: None))
    queue.log_wandb(s.args, s.runs[0], directory, {"sample_ema/SUN": .2})
    assert initializations[0]["name"] == "baseline-posthoc"
