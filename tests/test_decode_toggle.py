"""Decoder selection, historical configs, resume boundaries and sweep timing."""
from dataclasses import asdict, replace
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import orbitol
from orbitol import contracts as C, metric_names, sample, train
from orbitol.decode import CrystaliteRows, DecodedBatch
from test_sample_overrides import built_cfg, snapshot  # noqa: F401
from test_decode import encoding, sampler_batch, tables  # noqa: F401


def test_default_historical_and_cli():
    assert C.TrainConfig().decode_impl == "fast"
    assert C.TrainConfig(**C.normalize_historical_config({})).decode_impl == "fast"
    assert C.TrainConfig(**C.normalize_historical_config({"decode_impl": "fast"})).decode_impl == "fast"
    assert "decode_impl" in sample.SAMPLER_OVERRIDES
    for flag in ("--decode-impl", "--decode_impl"):
        assert train.parser().parse_args(["--run-name", "decode", flag, "fast"]).decode_impl == "fast"
    with pytest.raises(SystemExit):
        train.parser().parse_args(["--run-name", "decode", "--decode-impl", "unknown"])
    assert "decode_error" in train.DECODE_FAILURES
    assert set(train.DECODE_FAILURES) == set(metric_names.DECODE_REASONS)
    assert metric_names.log_view({"orbit/failures/decode_error": 1}) == {"decode/fail_decode_error": 1}


@pytest.mark.parametrize("value", [None, True, 1, "", "other"])
def test_invalid_decode_impl(value):
    with pytest.raises(ValueError, match="decode_impl"):
        C.TrainConfig(decode_impl=value)


def test_old_snapshot_and_sampling_override(snapshot):
    assert built_cfg(snapshot).decode_impl == "fast"
    assert built_cfg(snapshot, {"decode_impl": "reference"}).decode_impl == "reference"
    with pytest.raises(ValueError, match="decode_impl"):
        built_cfg(snapshot, {"decode_impl": "other"})


def test_sampling_cli(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sample, "sample_from_checkpoint",
                        lambda path, **kwargs: calls.append(kwargs) or {"failure_counts": {}})
    base = [str(tmp_path / "checkpoint.pt"), "--out", str(tmp_path / "sample.pt"), "--device", "cpu"]
    sample.main(base + ["--decode-impl", "fast"])
    sample.main(base)
    assert [call["overrides"] for call in calls] == [{"decode_impl": "fast"}, {}]


@pytest.mark.parametrize("field,before,after", [
    ("decode_impl", "reference", "fast"), ("decode_impl", "fast", "reference"),
    ("decode_impl", None, "reference"), ("decode_impl", None, "fast"),
    ("decode_overlap", "none", "thread"), ("decode_overlap", "thread", "none"),
    ("decode_overlap", None, "none"), ("decode_overlap", None, "thread"),
])
def test_resume_allows_decoder_only(tmp_path, monkeypatch, field, before, after):
    cfg = C.TrainConfig(sample_count=0, **({field: before} if before is not None else {}))
    saved = asdict(cfg)
    if before is None:
        saved.pop(field)
    path = tmp_path / "run/checkpoints/latest.pt"
    path.parent.mkdir(parents=True)
    torch.save({"cfg": saved}, path)
    monkeypatch.setattr(train.core, "verify_core", lambda: None)
    monkeypatch.setattr(train, "load_tables", lambda *args: SimpleNamespace(sha256="tables"))
    monkeypatch.setattr(train, "load_encoding", lambda: None)
    monkeypatch.setattr(train.shutil, "disk_usage", lambda p: SimpleNamespace(free=100_000_000_000))

    class ConfigAccepted(Exception):
        pass

    def data_boundary(cfg):
        raise ConfigAccepted

    monkeypatch.setattr(C, "resolve_dataset_paths", data_boundary)
    with pytest.raises(ConfigAccepted):
        train.Trainer(replace(cfg, **{field: after}), run_name="run", out_root=tmp_path, device="cpu", no_wandb=True)
    with pytest.raises(ValueError, match="Resume mismatch: cfg"):
        train.Trainer(replace(cfg, **{field: after}, lr=cfg.lr * 2), run_name="run", out_root=tmp_path,
                      device="cpu", no_wandb=True)


@pytest.mark.parametrize("impl, patched, expected", [("reference", True, "reference"), ("fast", False, "fast"),
                                                     ("fast", True, "reference")])
@pytest.mark.parametrize("budget", [0, 8])
def test_sample_pool_dispatch_is_lazy_and_patchable(monkeypatch, tables, encoding, impl, patched, expected, budget):
    """A patched orbitol.sample.decode_batch is used as is; unpatched, "fast" imports the fast decoder."""
    from orbitol import packed

    out, skeleton = sampler_batch(tables, encoding)
    calls = []

    def decoder(label):
        def run(output, skels, tables, encoding, allowed, *, n_max):
            calls.append(label)
            n = len(skels)
            return DecodedBatch(None, CrystaliteRows(n, n_max), {n - 1: "decode_error"},
                                np.zeros(n, dtype=bool), np.array([s.G for s in skels], dtype=np.int64))
        return run

    if patched:
        monkeypatch.setattr(sample, "decode_batch", decoder("reference"))
    fast = ModuleType("orbitol.decode_fast")
    fast.decode_batch = decoder("fast")
    # None makes importing the fast module an error whenever it must not be used.
    monkeypatch.setitem(sys.modules, "orbitol.decode_fast", fast if expected == "fast" else None)
    monkeypatch.setattr(packed, "PackedDenoiser", lambda *a, **kw: SimpleNamespace(denoise=lambda: None))
    model = torch.nn.Linear(1, 1)
    ema = SimpleNamespace(apply=lambda model: None, restore=lambda model, backup: None)
    bank = SimpleNamespace(sample=lambda n, **kw: [skeleton] * n)

    def sampler(model, batch, *args, **kwargs):
        return {k: v[batch["index"]] for k, v in out.items()}

    cfg = C.TrainConfig(decode_impl=impl, sample_token_budget=budget, compile_trunk="none")
    result = sample.sample_pool(model, ema, cfg, tables, tables.torch("cpu"), encoding, None, bank,
                                n=3, seed=42, device="cpu", chunk=2, sampler=sampler)
    assert calls == [expected, expected]
    assert result["failure_counts"] == {"decode_error": 2}
    assert result["failures"] == {1: "decode_error", 2: "decode_error"}
    assert result["rows"]["A0"].shape == (3, cfg.n_prim_max)
    assert model.training


def test_sweep_times_both_decoders_and_warms_each(tmp_path, monkeypatch):
    from tools import sampler_sweep as sweep

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"cfg": {}, "model": {}, "ema": {}, "step": 1}, checkpoint)
    target = SimpleNamespace(load_state_dict=lambda state: None)
    run = SimpleNamespace(model=target, ema=target, tables=None, tb=None, encoding=None, allowed_mask=None,
                          bank=None, device="cpu", evaluator=SimpleNamespace(
                              evaluate=lambda *a, **kw: {}, relaxation_diagnostics=lambda: []))
    monkeypatch.setattr(sweep.train, "Trainer", lambda *a, **kw: run)
    monkeypatch.setattr(sweep, "variant_metrics", lambda *a, **kw: {})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU stub")
    ticks = iter(range(1000))
    monkeypatch.setattr(sweep, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    monkeypatch.setattr(sample, "decode_batch", lambda: None)
    fast = ModuleType("orbitol.decode_fast")
    fast.decode_batch = lambda: None
    monkeypatch.setitem(sys.modules, "orbitol.decode_fast", fast)
    monkeypatch.setattr(orbitol, "decode_fast", fast, raising=False)
    draws = []

    def pool(model, ema, cfg, *args, n, **kwargs):
        draws.append((cfg.decode_impl, n))
        (sample.decode_batch if cfg.decode_impl == "reference" else fast.decode_batch)()
        return dict(rows={}, requested_G=torch.ones(n), failures={}, failure_counts={}, contains_a=torch.ones(n))

    monkeypatch.setattr(sample, "sample_pool", pool)
    # main wraps these functions; let monkeypatch restore them at teardown.
    for owner, name in ((sample, "orbit_edm_sampler"), (sweep.eval_adapter, "relax_all"),
                        (sweep.eval_adapter, "run_in_child")):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    settings = [dict(name="ref", decode_impl="reference"), dict(name="fast", decode_impl="fast"),
                dict(name="ref_again", decode_impl="reference")]
    monkeypatch.setattr(sys, "argv", ["sampler_sweep", "--checkpoint", str(checkpoint), "--out", str(tmp_path),
                                     "--samples", "3", "--warmup", "1", "--settings",
                                     *(json.dumps(s) for s in settings)])
    sweep.main()
    assert draws == [("reference", 1), ("reference", 3), ("fast", 1), ("fast", 3), ("reference", 3)]
    for setting in settings:
        report = json.loads((tmp_path / f"{setting['name']}.json").read_text())
        assert report["sampler"]["decode_impl"] == setting["decode_impl"]
        assert report["timing"]["decode_seconds"] == 1
