"""Best-MSUN checkpoint tracking in sync evaluation (Trainer.track_best_msun)."""
import json
import math
from types import SimpleNamespace

import torch

from orbitol.train import Trainer


def fake(tmp_path, step):
    return SimpleNamespace(evaluator=SimpleNamespace(tag="sample_ema"), checkpoint_dir=tmp_path, step=step,
                           snapshot=lambda: {"step": step})


def test_keeps_highest_msun_across_evaluations(tmp_path):
    for step, msun in ((100, 0.30), (200, 0.35), (300, 0.33)):
        metrics = {"sample_ema/MSUN": msun}
        Trainer.track_best_msun(fake(tmp_path, step), metrics)
        assert metrics["orbit/best_msun"] == max(0.30, 0.35 if step >= 200 else 0.30)
    assert json.loads((tmp_path / "best_msun.json").read_text()) == {"MSUN": 0.35, "step": 200}
    assert torch.load(tmp_path / "best_msun.pt", weights_only=False) == {"step": 200}


def test_ignores_missing_or_nonfinite_msun(tmp_path):
    for metrics in ({}, {"sample_ema/MSUN": math.nan}):
        Trainer.track_best_msun(fake(tmp_path, 100), metrics)
        assert "orbit/best_msun" not in metrics
    assert not (tmp_path / "best_msun.pt").exists()
