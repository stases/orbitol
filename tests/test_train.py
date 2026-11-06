"""Small CPU integration gates, including exact interruption and monitoring."""
from dataclasses import asdict, fields, replace
import csv
import gc
import json
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from pymatgen.core import Lattice, Structure
import pytest
import torch

from orbitol import contracts as C
from orbitol.data import build_cache
from orbitol.decode import to_crystalite_rows
from orbitol.edm import compute_orbit_loss, denoise_orbit
from orbitol.sample import rng_state, sample_from_checkpoint, sample_pool
from orbitol.symmetry_panel import SymmetryPanel
from orbitol.tables import build_tables
from orbitol.train import Trainer, atomic_save, lr_multiplier, parser, verify_symmetry_reference

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def disabled_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture(scope="module")
def fixture_data():
    # All fixture writes, including cache construction, stay in this checkout.
    with tempfile.TemporaryDirectory(prefix=".train-tests-", dir=ROOT) as name:
        root = Path(name)
        tables = build_tables(groups=[1, 136])
        tables_dir = root / "tables"
        tables.save(tables_dir)
        rutile = Structure.from_spacegroup(136, Lattice.tetragonal(4.594, 2.959),
                                          ["Ti", "O"], [[0, 0, 0], [.305, .305, 0]])
        p1 = Structure(Lattice.from_parameters(4.1, 5.2, 6.3, 76, 83, 71),
                       ["Li", "Na", "O"], [[.14, .29, .37], [.55, .61, .78], [.82, .08, .21]])
        structures = [p1, rutile] * 16
        source = root / "fixture.csv"
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["material_id", "cif", "spacegroup.number"])
            writer.writeheader()
            for i, structure in enumerate(structures):
                writer.writerow({"material_id": f"fixture-{i}", "cif": structure.to(fmt="cif"),
                                 "spacegroup.number": 1 if i % 2 == 0 else 136})
        cache = root / "cache"
        for split, limit in (("train", 32), ("val", 8)):
            build_cache(split, tables_dir, cache, limit=limit, csv_path=source)
        data = root / "data"
        processed = data / "processed"
        processed.mkdir(parents=True)
        rows = to_crystalite_rows(structures, 20)
        items = [{key: tensor[i] for key, tensor in rows.items()} for i in range(len(structures))]
        for split in ("train", "val"):
            torch.save(items if split == "train" else items[:8], processed / f"mp20_tokens_{split}.pt")
        reference = root / "reference.json"
        SymmetryPanel(processed / "mp20_tokens_train.pt", items, reference).reference
        yield dict(tables_dir=str(tables_dir), cache_dir=str(cache), data_root=str(data),
                   symmetry_reference=str(reference), val_batches=2, sample_seed=123,
                   val_seed=937, sample_chunk_size=4), root


@pytest.fixture
def cfg(fixture_data):
    extra, _ = fixture_data
    return C.TrainConfig(d_model=32, n_layers=2, n_heads=4, batch_size=8, num_workers=0,
                         max_steps=4, warmup_steps=1, precision="fp32", objective="P",
                         sample_count=0, sample_num_steps=5, val_every=0, ckpt_every=2,
                         eval_mode="sync", nequip_eval=False, nequip_backend="torchscript",
                         thermo_relax_steps=200, sun_k=512, compile_trunk="none",
                         sample_token_budget=0, pad_multiple=0,
                         log_every=1, extra=dict(extra))


def trainer(cfg, fixture_data, name, **kwargs):
    return Trainer(cfg, run_name=name, out_root=fixture_data[1] / "runs", device="cpu",
                   no_wandb=True, **kwargs)


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def load_latest(fixture_data, name):
    return torch.load(fixture_data[1] / "runs" / name / "checkpoints/latest.pt",
                      map_location="cpu", weights_only=False)


def test_three_steps_sampling_and_snapshot(cfg, fixture_data):
    cfg = replace(cfg, max_steps=3, sample_count=8, sample_frequency=3, val_every=2)
    run = trainer(cfg, fixture_data, "smoke")
    assert run.evaluator.thermo is None and run.evaluator.sun_k == 0
    run.run()
    latest = load_latest(fixture_data, "smoke")
    assert latest["step"] == latest["last_sample_step"] == 3
    assert latest["sampler"]["cursor"] == 3
    snapshot_path = run.checkpoint_dir / "step_0000003.pt"
    snapshot = torch.load(snapshot_path, weights_only=False)
    assert "optimizer" not in snapshot and "rng" not in snapshot
    assert {"model", "ema", "cfg", "encoding_sha256", "table_sha256", "step"} <= snapshot.keys()
    assert "optimizer" in latest and "scheduler" in latest and "rng" in latest
    samples = torch.load(run.run_dir / "samples/step_0000003.pt", weights_only=False)
    assert samples["rows"]["A0"].shape == (8, 20)
    assert len(samples["skeletons"]) == len(samples["requested_G"]) == 8
    assert samples["raw"]["type"].shape == (8, cfg.K, 16)
    before = rng_state()
    regenerated = sample_from_checkpoint(snapshot_path, n=8, seed=123, device="cpu", chunk=4)
    equal(before, rng_state())
    for key in ("rows", "requested_G", "skeletons", "raw", "contains_a", "failures"):
        equal(samples[key], regenerated[key])
    metrics = [json.loads(line) for line in (run.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert "val/loss_total" in metrics[1]
    assert "decode/contains_a_rate" in metrics[-1]
    assert "eval/valid_rate" in metrics[-1]
    assert any(k.startswith("sym_raw/") for k in metrics[-1])
    # End-of-run monitoring must not repeat a sampling event already run at this step.
    assert len(list((run.run_dir / "samples").glob("*.pt"))) == 1
    run.close()


@pytest.mark.parametrize("batch_size", [8, 16])
def test_bitwise_resume(cfg, fixture_data, batch_size):
    cfg = replace(cfg, batch_size=batch_size, val_every=1)
    full_losses, resumed_losses = [], []
    collect = lambda target: lambda step, losses: target.append(float(losses["total"].detach()))
    full = trainer(cfg, fixture_data, f"full-{batch_size}")
    full.run(on_step=collect(full_losses))
    expected = load_latest(fixture_data, f"full-{batch_size}")
    full.close()
    del full
    gc.collect()
    partial = trainer(cfg, fixture_data, f"split-{batch_size}")
    partial.run(until_step=2, on_step=collect(resumed_losses))
    partial.close()
    del partial
    gc.collect()
    resumed = trainer(cfg, fixture_data, f"split-{batch_size}", resume="auto")
    assert resumed.step == 2
    resumed.run(on_step=collect(resumed_losses))
    actual = load_latest(fixture_data, f"split-{batch_size}")
    assert full_losses == resumed_losses
    for key in ("model", "ema", "optimizer", "scheduler", "sampler", "rng", "loader_rng", "scaler"):
        equal(expected[key], actual[key])
    resumed.close()


def test_fp16_overflow_skips_the_update_and_training_continues(cfg, fixture_data):
    run = trainer(replace(cfg, max_steps=3, ckpt_every=0), fixture_data, "fp16-overflow")
    # The CPU twin of the CUDA fp16 scaler: the same unscale, skip and backoff logic.
    run.scaler = torch.amp.GradScaler("cpu", init_scale=2. ** 16)
    run.run(until_step=1)
    before = {key: value.clone() for key, value in run.model.state_dict().items()}
    loss = run._loss

    def overflow(*args, **kwargs):
        losses, *rest = loss(*args, **kwargs)
        return ({**losses, "total": losses["total"] * 1e38}, *rest)

    run._loss = overflow
    run.run(until_step=2)
    equal(before, run.model.state_dict())
    del run._loss
    run.run()
    assert run.step == 3
    assert not all(torch.equal(before[key], value) for key, value in run.model.state_dict().items())
    metrics = [json.loads(line) for line in (run.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [m["train/grad_scale"] for m in metrics] == [2. ** 16, 2. ** 15, 2. ** 15]
    run.close()


def test_validation_and_sampling_preserve_training_stream(cfg, fixture_data):
    reference = trainer(cfg, fixture_data, "no-monitoring")
    reference.run()
    expected = load_latest(fixture_data, "no-monitoring")
    reference.close()
    del reference
    gc.collect()
    monitored = trainer(cfg, fixture_data, "monitoring")
    monitored.run(until_step=2)
    before = rng_state()
    val = monitored.validate()
    equal(before, rng_state())
    assert val == monitored.validate()
    sample_pool(monitored.model, monitored.ema, cfg, monitored.tables, monitored.tb,
                monitored.encoding, monitored.allowed_mask, monitored.bank, n=2, seed=543,
                device="cpu", chunk=2)
    equal(before, rng_state())
    assert monitored.model.training
    monitored.run()
    actual = load_latest(fixture_data, "monitoring")
    for key in ("model", "ema", "optimizer", "scheduler", "rng", "sampler"):
        equal(expected[key], actual[key])
    monitored.close()


def test_validation_coord_diagnostics(cfg, fixture_data, monkeypatch):
    # Unequal final batch and class counts exercise ratios of accumulated losses.
    cfg = replace(cfg, batch_size=3, extra={**cfg.extra, "val_batches": 3})
    run = trainer(cfg, fixture_data, "coord-diagnostics")
    diagnostics = run._coord_diagnostics
    records, forwards = [], []

    def record(batch, sigma, state, den, class_masks):
        result = diagnostics(batch, sigma, state, den, class_masks)
        if not records:
            # Replay the real heads with zero coordinates, independent of the diagnostic hook.
            class ZeroCoordHead(torch.nn.Module):
                def forward(self, *args, **kwargs):
                    return {**den["raw"], "coord_vel": torch.zeros_like(den["raw"]["coord_vel"])}

            reference = denoise_orbit(ZeroCoordHead(), state, sigma, batch, run.tb,
                                     objective=cfg.objective, sigma_data=cfg.sigma_data)
            expected = float(compute_orbit_loss(
                reference, state, sigma, batch, run.tb, objective=cfg.objective,
                sigma_data=cfg.sigma_data, loss_weights=cfg.loss_weights)["F"])
            assert result[0] == expected
        records.append(result)
        return result

    monkeypatch.setattr(run, "_coord_diagnostics", record)
    hook = run.model.register_forward_hook(lambda *args: forwards.append(None))
    before = rng_state()
    metrics = run.validate()
    equal(before, rng_state())
    assert run.model.training
    assert len(records) == 21
    with_diag = len(forwards)
    forwards.clear()
    monkeypatch.setattr(run, "_coord_diagnostics", lambda *args: (0., {}))
    without = run.validate()
    hook.remove()
    assert with_diag - len(forwards) == 21  # Seven grid points for each of three batches.
    equal(before, rng_state())
    existing = {key: value for key, value in metrics.items() if "/loss_" in key}
    assert len(existing) == 25
    assert existing == {key: value for key, value in without.items() if "/loss_" in key}
    assert metrics["val/sigma_0.02/coord_do_nothing"] != metrics["val/sigma_0.02/loss_coord"]

    for i, s in enumerate((0.02, 0.1, 0.5, 2, 10, 0.2, 0.3)):
        prefix = f"val/sigma_{s}"
        batches = records[i::7]
        reference = sum(value for value, _ in batches)
        assert metrics[f"{prefix}/coord_do_nothing"] == reference * cfg.loss_weights[1] / 3
        for name in ("d1", "d3"):
            pairs = [by_class[name] for _, by_class in batches]
            value, baseline = (sum(pair[j] for pair in pairs) for j in (0, 1))
            key = f"{prefix}/coord_ratio/{name}"
            if baseline > 0:
                assert np.isfinite(metrics[key])
                assert metrics[key] == value / baseline
            else:
                assert key not in metrics
        for absent in ("d2", "wrap", "nondiag"):
            assert f"{prefix}/coord_ratio/{absent}" not in metrics
        if reference > 0:
            assert np.isfinite(metrics[f"{prefix}/coord_ratio"])
            assert metrics[f"{prefix}/coord_ratio"] == pytest.approx(
                metrics[f"{prefix}/loss_coord"] / metrics[f"{prefix}/coord_do_nothing"])
        else:
            assert f"{prefix}/coord_ratio" not in metrics
    assert all(np.isfinite(value) for value in metrics.values())
    run.close()


@pytest.mark.parametrize("objective", C.OBJECTIVES)
def test_validation_zero_coordinate_head_has_unit_ratio(cfg, fixture_data, objective):
    run = trainer(replace(cfg, objective=objective), fixture_data, f"zero-coord-{objective}")
    run.model.eval()
    hook = run.model.register_forward_hook(
        lambda module, args, raw: {**raw, "coord_vel": torch.zeros_like(raw["coord_vel"])})
    metrics = run.validate()
    assert not run.model.training
    assert len(run.model._forward_hooks) == 1
    hook.remove()
    for s in (0.02, 0.1, 0.5, 2, 10):
        prefix = f"val/sigma_{s}"
        assert metrics[f"{prefix}/coord_do_nothing"] == metrics[f"{prefix}/loss_coord"]
        for suffix in ("", "/d1", "/d3"):
            key = f"{prefix}/coord_ratio{suffix}"
            if metrics[f"{prefix}/coord_do_nothing"] > 0:
                assert metrics[key] == 1.0
            else:
                assert key not in metrics
    run.close()


@pytest.mark.parametrize("sig", [signal.SIGUSR1, signal.SIGTERM])
def test_signals_save_completed_steps(cfg, fixture_data, sig):
    run = trainer(cfg, fixture_data, f"signal-{sig}")
    previous = signal.getsignal(sig)

    def interrupt(step, losses):
        if step == 1:
            os.kill(os.getpid(), sig)

    run.run(on_step=interrupt)
    saved = load_latest(fixture_data, f"signal-{sig}")
    assert saved["step"] == (1 if sig == signal.SIGTERM else 4)
    assert saved["sampler"]["cursor"] == saved["step"]
    assert signal.getsignal(sig) == previous
    run.close()


def test_atomic_failure_leaves_previous_checkpoint(fixture_data, monkeypatch):
    path = fixture_data[1] / "atomic.pt"
    atomic_save({"step": 1}, path)

    def fail(payload, stream):
        stream.write(b"incomplete")
        raise OSError("disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        atomic_save({"step": 2}, path)
    assert torch.load(path) == {"step": 1}
    assert not path.with_name(path.name + ".tmp").exists()


def test_pending_final_sampling_is_retried_on_resume(cfg, fixture_data, monkeypatch):
    cfg = replace(cfg, max_steps=2, sample_count=8, sample_frequency=2)
    run = trainer(cfg, fixture_data, "pending-sample")

    def interrupted():
        raise RuntimeError("evaluation interrupted")

    monkeypatch.setattr(run, "evaluate_samples", interrupted)
    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        run.run()
    checkpoint = load_latest(fixture_data, "pending-sample")
    assert checkpoint["step"] == 2 and checkpoint["last_sample_step"] == -1
    run.close()
    del run
    gc.collect()
    resumed = trainer(cfg, fixture_data, "pending-sample")
    resumed.run()
    assert load_latest(fixture_data, "pending-sample")["last_sample_step"] == 2
    assert (resumed.run_dir / "samples/step_0000002.pt").is_file()
    equal(checkpoint["rng"], rng_state())
    resumed.close()


def test_wandb_id_and_rng_survive_resume(cfg, fixture_data, monkeypatch):
    calls = []

    def perturb():
        random.random()
        np.random.random()
        torch.rand(1)

    def init(**kwargs):
        calls.append(kwargs)
        perturb()
        return SimpleNamespace(log=lambda *a, **kw: perturb(), finish=lambda: None)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, util=SimpleNamespace(generate_id=lambda: "test-run-id")))
    kwargs = dict(run_name="wandb", out_root=fixture_data[1] / "runs", device="cpu")
    run = Trainer(cfg, **kwargs)
    run.run(until_step=2)
    run.close()
    del run
    gc.collect()
    assert load_latest(fixture_data, "wandb")["wandb_id"] == "test-run-id"
    resumed = Trainer(cfg, **kwargs)
    resumed.run()
    actual = load_latest(fixture_data, "wandb")
    resumed.close()
    del resumed
    gc.collect()
    reference = trainer(cfg, fixture_data, "wandb-reference")
    reference.run()
    expected = load_latest(fixture_data, "wandb-reference")
    for key in ("model", "optimizer", "rng"):
        equal(expected[key], actual[key])
    assert [c["id"] for c in calls] == ["test-run-id", "test-run-id"]
    assert all(c["resume"] == "allow" for c in calls)
    assert all(c["project"] == "orbitol" for c in calls)  # mp20's DatasetSpec.wandb_project
    reference.close()


def test_stale_reference_never_rebuilt(fixture_data):
    extra, root = fixture_data
    reference = root / "stale.json"
    payload = json.loads(Path(extra["symmetry_reference"]).read_text())
    payload["metadata"]["spglib"] = "stale"
    reference.write_text(json.dumps(payload))
    before = reference.read_bytes()
    with pytest.raises(ValueError, match="Stale symmetry reference"):
        verify_symmetry_reference(extra["data_root"], reference)
    assert reference.read_bytes() == before
    missing = root / "missing.json"
    with pytest.raises(FileNotFoundError, match="Prebuilt symmetry reference"):
        verify_symmetry_reference(extra["data_root"], missing)
    assert not missing.exists()


def test_resume_rejects_changed_schedule(cfg, fixture_data):
    run = trainer(cfg, fixture_data, "mismatch")
    run.run(until_step=2)
    run.close()
    del run
    gc.collect()
    with pytest.raises(ValueError, match="Resume mismatch: cfg"):
        trainer(replace(cfg, max_steps=8), fixture_data, "mismatch")


def test_old_checkpoint_defaults_to_mp20_and_rejects_dataset_switch(cfg, fixture_data):
    run = trainer(cfg, fixture_data, "old-dataset")
    run.save()
    path = run.checkpoint_dir / "latest.pt"
    run.close()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    del checkpoint["cfg"]["dataset"]
    torch.save(checkpoint, path)
    resumed = trainer(cfg, fixture_data, "old-dataset")
    assert resumed.cfg.dataset == "mp20" and resumed.step == checkpoint["step"]
    equal(resumed.model.state_dict(), checkpoint["model"])
    resumed.close()
    # Overrides keep the fixture caches identical, isolating config safety from hashes.
    with pytest.raises(ValueError, match="Resume mismatch: cfg"):
        trainer(replace(cfg, dataset="alex_mp20"), fixture_data, "old-dataset")


def test_historical_sequential_config_resumes(cfg, fixture_data):
    cfg = replace(cfg, nequip_relax_mode="sequential")
    run = trainer(cfg, fixture_data, "old-evaluation")
    run.save()
    path = run.checkpoint_dir / "latest.pt"
    run.close()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for name in ("eval_mode", "nequip_backend", "thermo_relax_steps", "nequip_eval", "compile_trunk",
                 "pad_multiple", "sample_token_budget", "eval_backlog_max", "eval_partition", "eval_account",
                 "eval_gres"):
        del checkpoint["cfg"][name]
    torch.save(checkpoint, path)
    resumed = trainer(cfg, fixture_data, "old-evaluation")
    assert resumed.cfg.nequip_relax_mode == "sequential" and resumed.cfg.nequip_backend == "torchscript"
    equal(resumed.model.state_dict(), checkpoint["model"])
    resumed.close()


def test_cli_resume_normalizes_before_validation(tmp_path, monkeypatch):
    from orbitol import train

    path = tmp_path / "historical/checkpoints/latest.pt"
    path.parent.mkdir(parents=True)
    torch.save({"cfg": {"nequip_relax_mode": "sequential", "sample_count": 8, "sun_k": 2,
                        "use_gem": False}}, path)
    configs = []

    def capture(cfg, **kwargs):
        configs.append(cfg)
        return SimpleNamespace(run=lambda: None, close=lambda: None)

    monkeypatch.setattr(train, "Trainer", capture)
    train.main(["--run-name", "historical", "--out-root", str(tmp_path), "--no-wandb"])
    cfg = configs.pop()
    assert (cfg.nequip_backend, cfg.nequip_relax_mode, cfg.eval_mode) == ("torchscript", "sequential", "sync")
    assert (cfg.sample_count, cfg.sun_k, cfg.thermo_relax_steps) == (8, 2, 200)
    assert (cfg.compile_trunk, cfg.sample_token_budget) == ("none", 0)


def test_cli_config_drops_retired_fields(tmp_path, monkeypatch):
    from orbitol import train

    config = tmp_path / "saved.json"
    config.write_text(json.dumps({"use_gem": False, "gem_mode": "distance"}))
    configs = []
    monkeypatch.setattr(train, "Trainer", lambda cfg, **kwargs: configs.append(cfg)
                        or SimpleNamespace(run=lambda: None, close=lambda: None))
    train.main(["--run-name", "saved", "--out-root", str(tmp_path), "--no-wandb", "--config", str(config)])
    assert configs.pop().gem_mode == "distance"


def test_trainer_defaults_follow_alex_registry(cfg, fixture_data, monkeypatch):
    extra = dict(cfg.extra)
    spec = replace(C.dataset_spec("alex_mp20"), data_root=extra.pop("data_root"),
                   orbit_cache_dir=extra.pop("cache_dir"),
                   symmetry_reference=extra.pop("symmetry_reference"))
    monkeypatch.setitem(C.DATASETS, "alex_mp20", spec)
    run = trainer(replace(cfg, dataset="alex_mp20", extra=extra, sample_count=2),
                  fixture_data, "alex-defaults")
    assert run.evaluator.data_root == Path(spec.data_root)
    assert run.evaluator.symmetry_panel.cache_path == Path(spec.symmetry_reference)
    assert run.allowed_mask.any() and run.bank.table_sha256 == run.tables.sha256
    run.close()


def test_all_config_fields_and_schedule():
    cfg = C.TrainConfig()
    arguments = ["--run-name", "parse-test"]
    for field in fields(cfg):
        value = asdict(cfg)[field.name]
        arguments.append("--" + field.name)
        if isinstance(value, tuple):
            arguments.extend(str(v) for v in value)
        else:
            arguments.append(json.dumps(value) if isinstance(value, (dict, bool)) else str(value))
    parsed = vars(parser().parse_args(arguments))
    for field in fields(cfg):
        value = parsed[field.name]
        if isinstance(getattr(cfg, field.name), tuple):
            value = tuple(value)
        assert value == getattr(cfg, field.name)
    override = parser().parse_args(["--run-name", "parse", "--objective", "M", "--row-embed", "physical",
                                   "--sg-embed", "none", "--gem_compile", "false"])
    assert override.objective == "M" and override.row_embed == "physical" and override.sg_embed == "none"
    assert override.gem_compile is False
    assert lr_multiplier(0, warmup_steps=1000, max_steps=300000) == .001
    assert lr_multiplier(999, warmup_steps=1000, max_steps=300000) == 1.
    assert lr_multiplier(1000, warmup_steps=1000, max_steps=300000) == 1.
    assert lr_multiplier(300000, warmup_steps=1000, max_steps=300000) == 0.


@pytest.mark.parametrize("keep, steps, expected", [(0, 4, [2, 4]), (1, 4, [4]), (2, 6, [4, 6])])
def test_ckpt_keep_prunes_older_numbered_snapshots(cfg, fixture_data, keep, steps, expected):
    run = trainer(replace(cfg, max_steps=steps, ckpt_keep=keep), fixture_data, f"keep-{keep}")
    run.run()
    names = sorted(path.name for path in run.checkpoint_dir.glob("*.pt"))
    assert names == ["latest.pt"] + [f"step_{step:07d}.pt" for step in expected]
    with pytest.raises(ValueError):
        replace(cfg, ckpt_keep=-1)
