"""Runtime-only compile selection and inference-cache integration gates."""
from dataclasses import replace

import pytest
import torch

from orbitol import lattice, train
from test_train import cfg, equal, fixture_data, load_latest  # noqa: F401


@pytest.mark.parametrize("before,after", [("none", "default"), ("default", "none")])
def test_resume_can_change_only_compile_mode(cfg, fixture_data, monkeypatch, before, after):
    compile_fn = torch.compile

    def cpu_compile(fn, *, dynamic, options):
        assert options["emulate_precision_casts"]
        return compile_fn(fn, backend="aot_eager", fullgraph=True, dynamic=dynamic)

    monkeypatch.setattr(torch, "compile", cpu_compile)
    root = fixture_data[1] / "runs"
    name = f"compile-{before}-{after}"
    original = replace(cfg, compile_trunk=before)
    full = train.Trainer(original, run_name=name + "-full", out_root=root, device="cpu", no_wandb=True)
    full.run()
    full.close()
    expected = load_latest(fixture_data, name + "-full")
    split = train.Trainer(original, run_name=name, out_root=root, device="cpu", no_wandb=True)
    split.run(until_step=2)
    split.close()
    if before == "none":
        legacy = load_latest(fixture_data, name)
        legacy["cfg"].pop("compile_trunk")
        train.atomic_save(legacy, split.checkpoint_dir / "latest.pt")
    resumed = train.Trainer(replace(original, compile_trunk=after), run_name=name,
                            out_root=root, device="cpu", no_wandb=True)
    assert resumed.step == 2 and resumed.cfg.compile_trunk == after
    resumed.run()
    resumed.close()
    actual = load_latest(fixture_data, name)
    assert actual["cfg"]["compile_trunk"] == after
    for key in ("rng", "sampler", "loader_rng", "scheduler", "scaler"):
        equal(expected[key], actual[key])
    for key in ("model", "ema", "optimizer"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)


def test_compile_override_does_not_relax_training_config(cfg, fixture_data):
    name = "strict-with-compile"
    root = fixture_data[1] / "runs"
    run = train.Trainer(cfg, run_name=name, out_root=root, device="cpu", no_wandb=True)
    run.run(until_step=1)
    run.close()
    for change in (dict(lr=cfg.lr * 2), dict(batch_size=cfg.batch_size * 2),
                   dict(objective="wrapped"), dict(precision="bf16"),
                   dict(extra={**cfg.extra, "val_seed": 99}), dict(seed=cfg.seed + 1)):
        with pytest.raises(ValueError, match="Resume mismatch: cfg"):
            train.Trainer(replace(cfg, compile_trunk="default", **change), run_name=name,
                          out_root=root, device="cpu", no_wandb=True)


def test_both_lattice_caches_created_in_inference_support_training():
    lattice._free_masks.cache_clear()
    lattice._clamp_bounds.cache_clear()
    family = torch.arange(6)
    with torch.inference_mode():
        lattice.end_clamp(torch.ones(6, 6), family)
    assert not lattice._free_masks(family.device).is_inference()
    assert not lattice._clamp_bounds(family.device, torch.float32).is_inference()
    z = torch.ones(6, 6, requires_grad=True)
    lattice.end_clamp(z, family).sum().backward()
    torch.testing.assert_close(z.grad, lattice.free_mask(family).float(), rtol=0, atol=0)
