"""JSON configuration merging through the training entry point, without training."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from orbitol import contracts as C, train


@pytest.fixture
def captured(tmp_path, monkeypatch):
    configs = []

    def capture(cfg, **controls):
        configs.append(cfg)
        return SimpleNamespace(run=lambda: None, close=lambda: None)

    monkeypatch.setattr(train, "Trainer", capture)
    path = tmp_path / "config.json"
    argv = ["--run-name", "test", "--out-root", str(tmp_path), "--config", str(path)]
    return configs, path, argv


def test_config_precedence_and_tuples(captured):
    configs, path, argv = captured
    path.write_text(json.dumps(dict(max_steps=100, batch_size=16, d_model=1024,
                                    sigma_data=[.1, .2, .3], loss_weights=[1, 2, 3])))
    train.main(argv + ["--max_steps", "200", "--loss-weights", "4", "5", "6"])
    cfg = configs.pop()
    assert (cfg.max_steps, cfg.batch_size, cfg.lr) == (200, 16, C.TrainConfig().lr)
    assert cfg.sigma_data == (.1, .2, .3)
    assert cfg.loss_weights == (4., 5., 6.)
    assert (cfg.compile_trunk, cfg.sample_token_budget) == ("max-autotune-no-cudagraphs", 6144)


def test_config_unknown_key(captured):
    configs, path, argv = captured
    path.write_text(json.dumps({"max_step": 200}))
    with pytest.raises(TypeError, match="max_step"):
        train.main(argv)
    assert configs == []


def test_resume_with_same_config(captured, tmp_path):
    configs, path, argv = captured
    path.write_text(json.dumps(dict(max_steps=100, sigma_data=[.1, .2, .3],
                                    loss_weights=[1, 2, 3])))
    train.main(argv + ["--seed", "42"])
    original = configs.pop()
    checkpoint = tmp_path / "test/checkpoints/latest.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"cfg": asdict(original)}, checkpoint)
    train.main(argv)
    assert asdict(configs.pop()) == asdict(original)
    train.main(argv + ["--max-steps", "200"])
    changed = asdict(configs.pop())
    assert changed == {**asdict(original), "max_steps": 200}
