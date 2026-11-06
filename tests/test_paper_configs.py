import json
from pathlib import Path

import pytest

from orbitol import contracts as C

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "configs" / "paper"
MODELS = json.loads((PAPER / "models.json").read_text())
DROP = {
    "wandb_name", "wandb_project", "eval_partition", "eval_account",
    "eval_gres", "eval_backlog_max", "ckpt_every", "ckpt_keep",
}


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model["name"])
def test_paper_config(model):
    config_path = ROOT / model["train_config"]
    sampler_path = ROOT / model["sampler"]
    assert config_path.is_file() and sampler_path.is_file()
    config = json.loads(config_path.read_text())
    assert not DROP.intersection(config)
    assert C.TrainConfig(**config).eval_mode == "sync"
    overrides = json.loads(sampler_path.read_text())
    assert C.TrainConfig(**(config | overrides)).decode_impl == "reference"


def test_paper_models():
    assert [model["name"] for model in MODELS] == ["mp20_x_postper_2500k"]
    assert {path.name for path in PAPER.glob("*.json")} == {
        "models.json", "sampler_mp20_fast.json", *(Path(model["train_config"]).name for model in MODELS),
    }
