"""Atomic, EMA-only snapshots compatible with the standalone sampler."""
from pathlib import Path
import re

import torch

from orbitol.contracts import normalize_historical_config
from orbitol.posthoc_io import atomic_writer


SNAPSHOT_KEYS = ("format_version", "cfg", "step", "table_sha256", "encoding_sha256", "core_manifest", "commit",
                 "bank_sha256", "data_hashes", "allowed_mask", "nequip_model_sha256", "phase_diagram_sha256")


def ema_only(payload):
    """Keep sampling metadata and EMA weights; discard optimizer/trainer state."""
    result = {key: payload[key] for key in SNAPSHOT_KEYS if key in payload}
    result["ema"] = payload["ema"]
    # The sampler loads both model and EMA. Model buffers/frozen parameters still
    # come from the model; trainable parameters alias EMA storage in the archive.
    result["model"] = {
        name: payload["ema"][name] if name in payload["ema"] else value.detach().cpu()
        for name, value in payload["model"].items()
    }
    return result


def write_eval_snapshot(trainer, run_dir, step) -> Path:
    """Reuse sampling metadata and store each EMA tensor only once."""
    payload = ema_only(trainer.snapshot())
    payload["step"] = step
    path = Path(run_dir) / "eval_snapshots" / f"step_{step:07d}.pt"
    with atomic_writer(path) as stream:
        torch.save(payload, stream)
    return path


def pending_eval_snapshots(run_dir) -> list[Path]:
    """Published snapshots remain pending until the GPU stage deletes them."""
    paths = (path for path in (Path(run_dir) / "eval_snapshots").glob("step_*.pt")
             if re.fullmatch(r"step_\d{7,}\.pt", path.name) and path.is_file())
    return sorted(paths, key=lambda path: int(path.stem[5:]))


def load_eval_snapshot(path):
    """Return CPU sampling state and a historically normalized config."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["cfg"] = normalize_historical_config(payload["cfg"])
    return payload
