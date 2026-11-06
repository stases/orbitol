"""Submit one finite GPU evaluation job per published EMA snapshot."""
import logging
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def submit_eval(run_dir, step, cfg, *, wandb=True) -> str:
    """Return the Slurm id, or an empty string after a logged failure.

    wandb=False (training with --no-wandb) disables W&B in the GPU job and, through
    its inherited environment, in the scoring job it submits.
    """
    command = [
        str(ROOT / "slurm/submit"), "--parsable",
        *([f"--partition={cfg.eval_partition}"] if cfg.eval_partition else []),
        *([f"--account={cfg.eval_account}"] if cfg.eval_account else []),
        f"--gres={cfg.eval_gres}",
        str(ROOT / "slurm/eval_gpu.sbatch"), "--run-dir", str(Path(run_dir).resolve()),
        "--step", str(step),
    ]
    try:
        result = subprocess.run(
            command, cwd=ROOT, env={**os.environ, "ORBITOL_REPO_ROOT": str(ROOT),
                                    **({} if wandb else {"WANDB_MODE": "disabled"})},
            check=True, capture_output=True, text=True, timeout=30,
        )
        output = result.stdout.strip()
        if not re.fullmatch(r"[0-9]+(?:;[^;\s]+)?", output):
            raise ValueError(f"Invalid sbatch --parsable output: {output!r}")
        return output.split(";", 1)[0]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.error("orbit/eval_submit_failed=1 run_dir=%s step=%s: %s %s",
                     run_dir, step, exc, getattr(exc, "stderr", "") or "")
        return ""
