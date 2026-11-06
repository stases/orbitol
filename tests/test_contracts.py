"""New-run defaults and historical checkpoint config compatibility."""
from dataclasses import asdict, fields

import pytest

from orbitol.contracts import TrainConfig, normalize_historical_config


@pytest.mark.parametrize("width,compile_mode,budget", [
    (512, "reduce-overhead", 8192), (1024, "max-autotune-no-cudagraphs", 6144),
])
def test_fast_evaluation_defaults(width, compile_mode, budget):
    cfg = TrainConfig(d_model=width)
    defaults = {field.name: field.default for field in fields(TrainConfig)}
    assert defaults["compile_trunk"] == "auto"
    assert defaults["sample_token_budget"] == -1
    assert (cfg.compile_trunk, cfg.sample_token_budget) == (compile_mode, budget)
    assert (cfg.pad_multiple, cfg.sample_count, cfg.sun_k, cfg.nequip_eval,
            cfg.eval_mode, cfg.nequip_backend, cfg.thermo_relax_steps) == (
                4, 10000, 0, True, "async", "oeq", 500)
    assert (cfg.eval_backlog_max, cfg.eval_partition, cfg.eval_account, cfg.eval_gres) == (
        2, "", "", "gpu:1")


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None, "2"])
def test_invalid_backlog(value):
    with pytest.raises(ValueError, match="eval_backlog_max"):
        TrainConfig(eval_backlog_max=value)


@pytest.mark.parametrize("name", ["eval_partition", "eval_account", "eval_gres"])
@pytest.mark.parametrize("value", ["", " ", "gpu\n", "gpu: 1", "bad\0value", None, 1])
def test_invalid_slurm_resource(name, value):
    if value == "" and name != "eval_gres":
        assert getattr(TrainConfig(**{name: value}), name) == ""
        return
    with pytest.raises(ValueError, match=name):
        TrainConfig(**{name: value})


def test_slurm_resource_overrides():
    cfg = TrainConfig(eval_backlog_max=3, eval_partition="long", eval_account="project",
                      eval_gres="gpu:rtx6000:1")
    assert (cfg.eval_backlog_max, cfg.eval_partition, cfg.eval_account, cfg.eval_gres) == (
        3, "long", "project", "gpu:rtx6000:1")


def test_historical_sequential_config_keeps_its_meaning():
    old = dict(nequip_relax_mode="sequential", sample_count=17, sun_k=2, extra={"sample_seed": 31})
    normalized = normalize_historical_config(old)
    cfg = TrainConfig(**normalized)
    assert (cfg.nequip_backend, cfg.thermo_relax_steps, cfg.eval_mode) == ("torchscript", 200, "sync")
    assert (cfg.sample_count, cfg.sun_k, cfg.nequip_relax_mode) == (17, 2, "sequential")
    assert (cfg.compile_trunk, cfg.pad_multiple, cfg.sample_token_budget) == ("none", 0, 0)
    assert cfg.nequip_eval is False
    assert old == dict(nequip_relax_mode="sequential", sample_count=17, sun_k=2, extra={"sample_seed": 31})
    assert normalized is not old
    assert normalize_historical_config(normalized) == normalized


def test_historical_absent_fields_and_explicit_overrides():
    old = TrainConfig(**normalize_historical_config({}))
    assert (old.sample_count, old.sun_k, old.nequip_relax_mode, old.nequip_relax_batch_size) == (
        2048, 512, "batch", 128)
    cfg = asdict(TrainConfig())
    assert normalize_historical_config(cfg) == cfg
    assert TrainConfig(**normalize_historical_config({"sun_k": 0})).sun_k == 0
    with pytest.raises(ValueError, match="oeq requires batch"):
        TrainConfig(**normalize_historical_config(dict(nequip_backend="oeq", nequip_relax_mode="sequential")))
