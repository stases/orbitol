"""Resumable pool-first evaluation queue; --plan uses only the stdlib."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import traceback
from types import SimpleNamespace
import uuid

from orbitol import contracts as C
from orbitol.metric_names import log_view
from orbitol.posthoc_io import (
    POOL_NAME, SNAPSHOT_NAME, fsync_dir, identity, read_json, sha256, verify_pool, write_json,
)


@dataclass(frozen=True)
class Run:
    kind: str
    path: Path

    @property
    def name(self):
        return self.path.name


def _run(value):
    kind, sep, path = value.partition(":")
    if not sep or kind not in {"crystalite", "orbitol"} or not path:
        raise argparse.ArgumentTypeError("expected crystalite:RUN_DIR or orbitol:RUN_DIR")
    return Run(kind, Path(path).expanduser().resolve())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=_run, action="append", required=True)
    parser.add_argument("--out-root", type=Path,
                        default=None)
    parser.add_argument("--keep-snapshots", type=int, default=1)
    parser.add_argument("--final-step", type=int, default=300000)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--min-age-seconds", type=float, default=120)
    parser.add_argument("--idle-exit-minutes", type=float, default=120)
    parser.add_argument("--role", choices=["all", "evaluate"], default="all")
    for flag in ("dry-run", "plan", "once"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--relax-mode", choices=["batch", "sequential"], default=argparse.SUPPRESS)
    parser.add_argument("--relax-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sample-count", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sun-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--nequip-backend", choices=["oeq", "torchscript"], default=argparse.SUPPRESS)
    parser.add_argument("--relax-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--nequip-oeq-path", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--nequip-ts-path", type=Path)
    parser.add_argument("--ppd-path", type=Path)
    parser.add_argument("--symmetry-reference", type=Path)
    parser.add_argument("--wandb-project", help="Default: each pool's dataset project")
    parser.add_argument("--wandb-id-salt", default="",
                        help="start a fresh W&B series under the same run names")
    args = parser.parse_args(argv)
    args.evaluation_overrides = set(vars(args))
    # Keep --plan stdlib-only. Runtime resolves these defaults from each snapshot.
    for name, default in dict(sample_count=10000, sun_k=0, seed=123, relax_mode="batch",
                              relax_batch_size=128, nequip_backend="oeq", relax_steps=500).items():
        if not hasattr(args, name):
            setattr(args, name, default)
    if min(args.keep_snapshots, args.sample_count, args.relax_batch_size, args.final_step, args.relax_steps) < 1:
        parser.error("counts, batch size, final-step and relax-steps must be positive")
    if args.sun_k < 0 or args.seed < 0:
        parser.error("sun-k and seed must be nonnegative")
    if min(args.poll_seconds, args.min_age_seconds, args.idle_exit_minutes) < 0:
        parser.error("time intervals must be nonnegative")
    if len({run.name for run in args.run}) != len(args.run):
        parser.error("run directory names must be unique")
    args.out_root = (args.out_root or Path(C.RUNS_ROOT) / "posthoc").expanduser().resolve()
    for run in args.run:
        output = args.out_root / run.name
        if output.is_relative_to(run.path) or run.path.is_relative_to(output):
            parser.error("pool output must be separate from the live run directory")
    return args


def snapshots(run):
    directory = run.path / "checkpoints"
    if not directory.is_dir():
        return []
    result = []
    for path in directory.iterdir():
        match = SNAPSHOT_NAME.fullmatch(path.name)
        if not match:
            continue
        try:
            info = identity(path)
        except (OSError, ValueError):
            continue
        result.append(dict(path=str(path), step=int(match[1]), **info))
    return sorted(result, key=lambda item: item["step"])


def _same_identity(left, right):
    return all(left.get(key) == right.get(key) for key in ("size", "mtime_ns", "inode"))


def _load(path, default=None):
    try:
        return read_json(path, default)
    except (OSError, ValueError):
        return default


def pool_dirs(args, run):
    root = args.out_root / run.name
    if not root.is_dir():
        return []
    return sorted((path for path in root.iterdir()
                   if POOL_NAME.fullmatch(path.name) and path.is_dir() and not path.is_symlink()
                   and (path / "pool.json").is_file()), key=lambda path: path.name)


class StopRequested(KeyboardInterrupt):
    """Propagate through RobustRelaxer without becoming a structure failure."""


@contextmanager
def _defer_signals():
    old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGUSR1})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


def _owner_dead(owner):
    if not owner:
        return False
    if owner.get("host") == socket.gethostname():
        try:
            os.kill(int(owner["pid"]), 0)
        except ProcessLookupError:
            return True
        except (PermissionError, ValueError, KeyError):
            pass
        return False
    job = owner.get("SLURM_JOB_ID")
    if job and re.fullmatch(r"[0-9_]+", str(job)):
        try:
            result = subprocess.run(["squeue", "--noheader", "--jobs", str(job), "--format=%T"],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return not result.stdout.strip()
            # Slurm purges finished jobs and then reports their ids as invalid.
            return "Invalid job id" in result.stderr
        except (OSError, subprocess.TimeoutExpired):
            pass
    return False  # Never steal an unverified remote or live lock.


def _remove_dead_lock(path):
    # Serialize stale-lock reclaimers on the old inode. Without this, two
    # simultaneous reclaimers could unlink each other's newly acquired lock.
    try:
        # NFS emulates flock with POSIX locks, and an exclusive lock needs a writable fd.
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        with os.fdopen(fd, "r") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            stat = os.fstat(stream.fileno())
            expected = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)
            owner = json.load(stream)
            if _owner_dead(owner) and identity(path) == expected:
                path.unlink()
                fsync_dir(path.parent)
                return True
    except FileNotFoundError:
        return True  # Another owner released it; retry the exclusive create.
    except (OSError, ValueError):
        pass
    return False


@contextmanager
def claim(path):
    """Exclusive, signal-safe lock; reclaim only a demonstrably dead owner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    owner = dict(host=socket.gethostname(), pid=os.getpid(), SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
                 time=time.time(), token=token)
    acquired = False
    created_identity = None
    try:
        with _defer_signals():
            for _ in range(2):
                try:
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    if _remove_dead_lock(path):
                        continue
                    break
                else:
                    acquired = True
                    with os.fdopen(fd, "w") as stream:
                        created_identity = identity(path)
                        json.dump(owner, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    fsync_dir(path.parent)
                    break
        yield acquired
    finally:
        if acquired:
            with _defer_signals():
                current = _load(path)
                owned = current and current.get("token") == token
                if not current and path.exists() and created_identity:
                    owned = identity(path)["inode"] == created_identity["inode"]
                if owned:
                    path.unlink(missing_ok=True)
                    fsync_dir(path.parent)


def training_wandb_id(run):
    directory = run.path / "wandb"
    if not directory.is_dir():
        return None
    latest = directory / "latest-run"
    choices = [latest.resolve()] if latest.exists() else []
    choices.extend(sorted(directory.iterdir(), key=lambda p: p.name, reverse=True))
    for path in choices:
        match = re.fullmatch(r"(?:offline-)?run-\d+_\d+-([A-Za-z0-9]+)", path.name)
        if match:
            return match[1]
    return None


def wandb_run_id(project, name, salt=""):
    """Stable id per (project, name); a salt starts a fresh series under the same name."""
    key = f"{project}\0{name}" + (f"\0{salt}" if salt else "")
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def log_wandb(args, run, directory, metrics):
    """Reserve a history index durably; resumed W&B step is the remote receipt.

    There is exactly one role-all writer. Custom train_step stays independent
    of history order, including pools completed out of order by helpers.
    """
    import wandb
    if os.environ.get("WANDB_MODE", "online") not in {"online", "disabled"}:
        raise RuntimeError("Queue W&B resume requires online mode; offline runs cannot acknowledge remote history")
    from orbitol.posthoc_eval import wandb_project
    pool = verify_pool(directory)
    project = wandb_project(args, pool)
    settings = read_json(directory / "evaluation_settings.json")
    name = f"{run.name}-posthoc"
    run_id = wandb_run_id(project, name, getattr(args, "wandb_id_salt", ""))
    entity = C.local_value("ORBITOL_WANDB_ENTITY")
    metrics_hash = sha256(directory / "metrics.json")
    session = wandb.init(entity=entity, project=project, name=name, id=run_id,
                         resume="allow", dir=str(directory),
                         config=dict(run_dir=str(run.path), training_wandb_id=training_wandb_id(run),
                                     kind=run.kind, sample_count=pool["n"], sun_k=settings.get("sun_k", args.sun_k),
                                     seed=pool["seed"], thermo=settings["thermo"], sampler=pool["sampler"],
                                     protocol=settings.get("protocol", "legacy_raw_un_subset")))
    try:
        wandb.define_metric("train_step")
        wandb.define_metric("*", step_metric="train_step")
        intent_path = directory / "wandb_pending.json"
        intent = read_json(intent_path)
        if intent is None:
            intent = dict(run_id=run_id, entity=entity, project=project,
                          metrics_sha256=metrics_hash, history_step=int(session.step))
            write_json(intent_path, intent)
        if any(intent[key] != value for key, value in dict(
                run_id=run_id, entity=entity, project=project, metrics_sha256=metrics_hash).items()):
            raise ValueError("W&B logging intent conflicts with the completed metrics")
        if int(session.step) <= intent["history_step"]:
            values = log_view(
                {**metrics, "train_step": pool["step"], "posthoc/pool_sha256": pool["samples_sha256"]},
                legacy_subset=(settings.get("sun_k", args.sun_k) or 0) > 0,
            )
            wandb.log(values, step=intent["history_step"], commit=True)
    finally:
        session.finish()
    return {**intent, "time": time.time(), "train_step": pool["step"]}


class Runtime:
    """Lazy heavy imports; reuse the evaluator only while its references agree."""
    def __init__(self, args):
        self.args, self.evaluator, self.settings = args, None, None

    def prepare(self, run, settings):
        from orbitol.posthoc_eval import evaluation_settings, saved_thermo
        # Pools retain their config and resolved paths after snapshot pruning.
        sampler = settings.get("sampler", {})
        cfg = sampler.get("cfg", {}) if run.kind == "orbitol" else sampler
        args = SimpleNamespace(**{**vars(self.args), "kind": run.kind, "dataset_config": cfg})
        if "protocol" not in settings:
            # Resume committed historical pools using their original controls;
            # a queue's new defaults must not silently select another artifact.
            thermo = saved_thermo(settings)
            legacy = dict(sample_count=settings["sample_count"], sun_k=settings["sun_k"],
                          seed=settings["sample_seed"], nequip_backend="torchscript",
                          relax_steps=thermo.thermo_relax_steps, relax_mode=thermo.nequip_relax_mode,
                          relax_batch_size=thermo.thermo_stability_batch,
                          nequip_ts_path=Path(thermo.nequip_ts_path))
            explicit = set(getattr(args, "evaluation_overrides", vars(args)))
            for name, value in legacy.items():
                if name not in explicit or getattr(args, name, None) is None:
                    setattr(args, name, value)
            args.evaluation_overrides = explicit | legacy.keys()
        for flag, saved in (("data_root", settings["data_root"]),
                            ("symmetry_reference", settings["symmetry_reference"]),
                            ("ppd_path", (settings["thermo"] or {}).get("thermo_ppd_mp"))):
            if getattr(args, flag) is None and saved is not None:
                setattr(args, flag, Path(saved))
        expected = evaluation_settings(args)
        if expected != self.settings:
            self.evaluator = None
        self.settings = expected

    def sample(self, run, snapshot, directory):
        from orbitol.posthoc_eval import sample_stage
        return sample_stage(SimpleNamespace(**{**vars(self.args), "kind": run.kind,
                                               "snapshot": Path(snapshot["path"]), "out": directory}))

    def evaluate(self, run, directory):
        from orbitol.posthoc_eval import build_evaluator, evaluate_stage, saved_thermo
        settings = read_json(directory / "evaluation_settings.json")
        self.prepare(run, settings)
        for key in ("sample_count", "sample_seed", "sun_k", "thermo", "data_root",
                    "symmetry_reference", "mlip_model_sha256"):
            actual, expected = ((saved_thermo(settings), saved_thermo(self.settings)) if key == "thermo"
                                else (settings[key], self.settings[key]))
            if actual != expected:
                raise ValueError(f"Pool settings disagree with this queue: {key}")
        if self.evaluator is None:
            self.evaluator = build_evaluator(self.settings)
        return evaluate_stage(directory, evaluator=self.evaluator)


class Queue:
    """One pass is independently testable with injected sample/evaluate/log calls."""
    def __init__(self, args, *, sample_fn=None, evaluate_fn=None, log_fn=None, now=time.time):
        self.args, self.now = args, now
        runtime = Runtime(args)
        self.sample_fn = sample_fn or runtime.sample
        self.evaluate_fn = evaluate_fn or runtime.evaluate
        self.log_fn = log_fn or (lambda run, directory, metrics: log_wandb(args, run, directory, metrics))
        self.state_path = args.out_root / f"queue_{args.role}_state.json"
        self.state = _load(self.state_path, dict(observations={}, last_new=now(), timings={}))
        self.action = {"stage": "idle"}

    def directory(self, run, step):
        return self.args.out_root / run.name / f"step_{step:07d}"

    def _error(self, directory, name, **details):
        write_json(directory / name, dict(time=self.now(), traceback=traceback.format_exc(), **details))

    def _status(self):
        runs = {}
        for run in self.args.run:
            directories = pool_dirs(self.args, run)
            steps = lambda name: [int(p.name[5:]) for p in directories if (p / name).is_file()]
            root = self.args.out_root / run.name
            errors = []
            if root.exists():
                for directory in root.iterdir():
                    if POOL_NAME.fullmatch(directory.name) and directory.is_dir():
                        for name in ("error.json", "pool_error.json", "wandb_error.json"):
                            if (directory / name).is_file():
                                errors.append(dict(step=int(directory.name[5:]), path=str(directory / name)))
            runs[run.name] = dict(snapshots_on_disk=snapshots(run), pooled=steps("pool.json"),
                                  evaluated=steps("metrics.json"), logged=steps("wandb_logged.json"), errors=errors,
                                  locks=[dict(path=str(p / "eval.lock"), owner=_load(p / "eval.lock"))
                                         for p in directories if (p / "eval.lock").exists()])
        write_json(self.args.out_root / "queue_status.json",
                   dict(time=self.now(), role=self.args.role, pid=os.getpid(), runs=runs,
                        current_action=self.action, timings=self.state["timings"]))

    def _act(self, stage, run, step, callback):
        self.action = dict(stage=stage, run=run.name, step=step, started=self.now())
        self._status()
        start = time.monotonic()
        try:
            return callback()
        finally:
            timings = self.state["timings"]
            timings[stage] = timings.get(stage, 0.) + time.monotonic() - start
            self.action = {"stage": "idle"}

    def _observe(self):
        previous = self.state["observations"]
        current, ready = {}, []
        for run in self.args.run:
            for snapshot in snapshots(run):
                path = snapshot["path"]
                current[path] = snapshot
                if path not in previous:
                    self.state["last_new"] = self.now()
                if (path in previous and _same_identity(snapshot, previous[path])
                        and self.now() - snapshot["mtime_ns"] / 1e9 >= self.args.min_age_seconds):
                    ready.append((run, snapshot))
        self.state["observations"] = current
        # Save observations before any long stage: the next process can poll
        # against them, but changed/growing files still have to stabilize again.
        write_json(self.state_path, self.state)
        return sorted(ready, key=lambda pair: (pair[1]["step"], pair[0].name))

    def _sample(self, run, snapshot):
        directory = self.directory(run, snapshot["step"])
        if (directory / "pool.json").exists():
            return
        exhausted = _load(directory / "pool_error.json", {})
        if _same_identity(exhausted, snapshot):
            return
        try:
            self._act("sample", run, snapshot["step"], lambda: self.sample_fn(run, snapshot, directory))
            pool = verify_pool(directory)
            cfg = pool["sampler"].get("cfg", {})
            explicit = getattr(self.args, "evaluation_overrides", vars(self.args))
            count = self.args.sample_count if "sample_count" in explicit else cfg.get("sample_count", self.args.sample_count)
            seed = self.args.seed if "seed" in explicit else cfg.get("extra", {}).get("sample_seed", self.args.seed)
            if (pool["kind"], pool["snapshot"], pool["step"], pool["n"], pool["seed"]) != (
                    run.kind, snapshot["path"], snapshot["step"], count, seed):
                raise ValueError("Sampler committed unexpected pool provenance")
        except Exception:
            attempts = _load(directory / "sample_attempts.json", {})
            count = attempts.get("count", 0) + 1 if _same_identity(attempts, snapshot) else 1
            details = dict(snapshot, count=count)
            self._error(directory, "sample_attempts.json", **details)
            if count >= 3 and self.now() - snapshot["mtime_ns"] / 1e9 > 900:
                self._error(directory, "pool_error.json", **details)

    def _prune(self, run):
        found = snapshots(run)
        protected = {s["step"] for s in found[-self.args.keep_snapshots:]}
        for snapshot in found:
            if snapshot["step"] in protected:
                continue
            directory = self.directory(run, snapshot["step"])
            path = Path(snapshot["path"])
            try:
                pool = verify_pool(directory)
                if (pool["kind"], pool["snapshot"], pool["step"]) != (run.kind, str(path), snapshot["step"]):
                    continue
                if not _same_identity(pool, identity(path)) or sha256(path) != pool["snapshot_sha256"]:
                    continue
                # Recheck both metadata and newest-K immediately before unlink.
                if (not _same_identity(pool, identity(path))
                        or snapshot["step"] in {s["step"] for s in snapshots(run)[-self.args.keep_snapshots:]}):
                    continue
                entry = dict(time=self.now(), path=str(path), step=snapshot["step"], size=pool["size"],
                             sha256=pool["snapshot_sha256"], action="would delete" if self.args.dry_run else "deleted")
                with _defer_signals():
                    log = self.args.out_root / run.name / "retention.log"
                    with log.open("a") as stream:
                        if not self.args.dry_run:
                            # Refuse to delete if the audit log cannot be made
                            # durable (in particular, when storage is full).
                            stream.write(json.dumps({**entry, "action": "delete intent"}) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                            if not _same_identity(pool, identity(path)):
                                continue
                            path.unlink()
                            fsync_dir(path.parent)
                        stream.write(json.dumps(entry) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
            except (OSError, ValueError, KeyError):
                continue  # Missing/changed/corrupt evidence can never authorize deletion.

    def _pending_pools(self):
        pending = [(run, directory) for run in self.args.run for directory in pool_dirs(self.args, run)
                   if not (directory / "metrics.json").exists() and not (directory / "error.json").exists()]
        return sorted(pending, key=lambda pair: (int(pair[1].name[5:]), pair[0].name),
                      reverse=self.args.role == "evaluate")

    def _evaluate_one(self):
        for run, directory in self._pending_pools():
            with claim(directory / "eval.lock") as acquired:
                if not acquired:
                    continue
                if (directory / "metrics.json").exists() or (directory / "error.json").exists():
                    continue
                try:
                    pool = verify_pool(directory)
                    if pool["kind"] != run.kind or pool["step"] != int(directory.name[5:]):
                        raise ValueError("Pool/run identity mismatch")
                    metrics = self._act("evaluate", run, pool["step"], lambda: self.evaluate_fn(run, directory))
                    # Also supports injected evaluators which only return metrics.
                    if not (directory / "metrics.json").exists():
                        write_json(directory / "metrics.json", metrics)
                except Exception:
                    self._error(directory, "error.json", stage="evaluate")
                return  # Revisit discovery after exactly one evaluation attempt.

    def _log_completed(self):
        for run in self.args.run:
            directories = pool_dirs(self.args, run)
            # Finish any reserved history index before assigning another one.
            directories.sort(key=lambda p: (not (p / "wandb_pending.json").exists(), p.name))
            for directory in directories:
                if not (directory / "metrics.json").is_file() or (directory / "wandb_logged.json").exists():
                    continue
                try:
                    receipt = self._act("wandb", run, int(directory.name[5:]),
                                        lambda: self.log_fn(run, directory, read_json(directory / "metrics.json")))
                    write_json(directory / "wandb_logged.json", receipt or {"time": self.now()})
                except Exception:
                    self._error(directory, "wandb_error.json", stage="wandb")
                    break  # Never advance past an uncertain reserved history index.

    def finished(self):
        pending = bool(self._pending_pools())
        if self.args.role == "all":
            for run in self.args.run:
                for snapshot in snapshots(run):
                    directory = self.directory(run, snapshot["step"])
                    exhausted = _load(directory / "pool_error.json", {})
                    pending |= not (directory / "pool.json").exists() and not _same_identity(exhausted, snapshot)
                pending |= any((p / "metrics.json").exists() and not (p / "wandb_logged.json").exists()
                               for p in pool_dirs(self.args, run))
        if pending:
            return False
        final = all((self.directory(run, self.args.final_step) / "metrics.json").is_file() for run in self.args.run)
        return final or self.now() - self.state["last_new"] >= self.args.idle_exit_minutes * 60

    def pass_once(self):
        ready = self._observe()
        self._status()
        try:
            if self.args.role == "all":
                if not self.args.dry_run:
                    for run, snapshot in ready:
                        self._sample(run, snapshot)
                for run in self.args.run:
                    self._act("prune", run, None, lambda: self._prune(run))
            if not self.args.dry_run:
                self._evaluate_one()
                if self.args.role == "all":
                    self._log_completed()
            return self.finished()
        finally:
            write_json(self.state_path, self.state)
            self._status()


def plan(args):
    """Only directory/stat/JSON reads; no hashes, locks, writes or torch imports."""
    state = _load(args.out_root / f"queue_{args.role}_state.json", {"observations": {}})
    result = dict(role=args.role, sample=[], delete=[], evaluate=[])
    for run in args.run:
        found = snapshots(run)
        protected = {s["step"] for s in found[-args.keep_snapshots:]}
        for snapshot in found:
            directory = args.out_root / run.name / f"step_{snapshot['step']:07d}"
            pool = _load(directory / "pool.json", {})
            if not pool and args.role == "all":
                previous = state["observations"].get(snapshot["path"], {})
                age = time.time() - snapshot["mtime_ns"] / 1e9
                ready = _same_identity(previous, snapshot) and age >= args.min_age_seconds
                result["sample"].append(dict(path=snapshot["path"], step=snapshot["step"], ready=ready,
                                             note="ready" if ready else "wait for age and a second unchanged poll"))
            if (args.role == "all" and snapshot["step"] not in protected
                    and pool.get("status") == "complete" and _same_identity(snapshot, pool)):
                result["delete"].append(dict(path=snapshot["path"], note="conditional on pool, snapshot hashes and final identity checks"))
        for directory in pool_dirs(args, run):
            if not (directory / "metrics.json").exists() and not (directory / "error.json").exists():
                result["evaluate"].append(dict(pool=str(directory), step=int(directory.name[5:]),
                                               locked=(directory / "eval.lock").exists()))
    result["sample"].sort(key=lambda item: item["step"])
    result["evaluate"].sort(key=lambda item: item["step"], reverse=args.role == "evaluate")
    result["note"] = "Sample all ready snapshots, prune verified older snapshots, then evaluate ONE unlocked pool."
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.plan:
        plan(args)
        return 0
    if args.symmetry_reference is not None and not args.symmetry_reference.is_file():
        raise FileNotFoundError(f"Prebuilt symmetry reference required: {args.symmetry_reference.resolve()}")

    def stop(signum, frame):
        raise StopRequested(f"Received signal {signum}")

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    try:
        queue = Queue(args)
        if args.role == "all":
            with claim(args.out_root / "queue_all.lock") as acquired:
                if not acquired:
                    raise RuntimeError("Another role-all queue owns this output root")
                while True:
                    if queue.pass_once() or args.once or args.dry_run:
                        return 0
                    time.sleep(args.poll_seconds)
        else:
            while True:
                if queue.pass_once() or args.once or args.dry_run:
                    return 0
                time.sleep(args.poll_seconds)
    except StopRequested:
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
