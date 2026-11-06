# Training and sampling

## Environment

`pyproject.toml` and `uv.lock` pin the environment: Python 3.12.9 and the exact package versions of the paper runs
(torch 2.8.0 with CUDA 12.8 wheels). Orbitol runs from the checkout with `PYTHONPATH=<repo>`; it is not installed.
`tools/setup_env.sh` creates `.venv` with `uv sync --frozen`, then fetches and verifies the pinned Crystalite core, and
prints the exports to use (`PYTHONPATH`, `ORBITOL_CORE_ROOT`, `ORBITOL_PYTHON`, `PATH`).

- **Crystalite core.** Data handling and evaluation come from github.com/joshrosie/crystalite at commit 03603fe.
  `orbitol/core.py` imports it from `ORBITOL_CORE_ROOT` and checks it against `orbitol/core_manifest.json`.
- **Dependency groups.** The default install includes `dev` (pytest, matplotlib) and `data` (pandas, pyarrow, lmdb;
  only the Alex-MP-20 reference build uses them). `tools/setup_env.sh --no-dev` installs the core packages only.
- **Relaxation.** OpenEquivariance, the default NequIP backend, compiles CUDA kernels at first use. It needs the
  CUDA toolkit (`nvcc`; we use 12.9), gcc ≥ 12 and `ninja` (installed in the environment). The TorchScript backend
  needs none of these.

### Local paths

`orbitol.local.env` (copied from `orbitol.local.env.example`) sets `ORBITOL_DATA_HOME` and `ORBITOL_RUNS_ROOT` to
absolute directories; `python -m orbitol.contracts` prints the resolved paths. The file is gitignored; values are
literal `KEY=value` (optional `export`, no quotes, spaces or shell expansion). Environment variables override it;
`ORBITOL_LOCAL_ENV` selects a different file. Derived dataset and artifact paths are listed in
[datasets.md](datasets.md); individual keys can override them. Python reads this file without sourcing it in your
shell.

### Slurm

Submit from the checkout with `slurm/submit [sbatch options] slurm/<name>.sbatch [args]`.
Set `ORBITOL_SLURM_GPU_PARTITION`, `ORBITOL_SLURM_GPU_ACCOUNT`, `ORBITOL_SLURM_CPU_PARTITION`,
`ORBITOL_SLURM_CPU_ACCOUNT` and `ORBITOL_SLURM_LOG_DIR` in the local file as needed.
Unset placement keys use cluster defaults; explicit submission options win. Nested evaluation jobs use
this wrapper too. The default interpreter is the checkout's `.venv/bin/python`; `ORBITOL_PYTHON`
overrides it. For the OEQ JIT, configure `ORBITOL_CUDA_HOME` and, if needed, `ORBITOL_GCC_ENABLE`
(a script enabling gcc ≥ 12). The wrappers load these values automatically.

## Training

```bash
python -m orbitol.train --run-name mp20 --config configs/paper/mp20_x_postper_2500k.json --no-wandb
slurm/submit slurm/train_orbitol.sbatch mp20 --config configs/paper/mp20_x_postper_2500k.json --no-wandb   # on Slurm
```

Runs write to `$ORBITOL_RUNS_ROOT/<run_name>` and resume automatically from it. Every field of `TrainConfig`
(`orbitol/contracts.py`) is a flag, with dashes or underscores; explicit flags override the config file. Without
`--config`, the defaults differ from the paper model (for example `--objective posterior` and `--gem-mode none`).
The paper configs evaluate in-process (`--eval-mode sync`); the default, `async`, submits evaluation jobs to Slurm
([evaluation.md](evaluation.md)). Without `--no-wandb`, runs log to W&B (run `wandb login` first) under
`ORBITOL_WANDB_ENTITY` (default: your W&B default entity). `SMOKE=1 slurm/submit slurm/train_orbitol.sbatch test_run` runs a four-step check.

Frequently used flags:

| Flag | Default | Meaning |
|---|---|---|
| `--d-model`, `--n-layers`, `--n-heads` | 512, 14, 16 | trunk size |
| `--objective` | `posterior` | coordinate objective; the paper models use `posterior_periodic` (table below) |
| `--gem-mode` | `none` | geometric attention bias: `none`, `multiplicity` or `distance` (the paper model) ([orbitol/gem.py](../orbitol/gem.py)) |
| `--coord-input`, `--coord-readout` | `explicit`, `projection` | how free coordinates enter and leave the trunk |
| `--row-embed`, `--sg-embed` | `hybrid` | Wyckoff-row and space-group embeddings: `categorical`, `physical` or `hybrid` |
| `--compile-trunk` | `auto` | torch.compile mode; `auto` picks `reduce-overhead` up to d512 and `max-autotune-no-cudagraphs` above |
| `--pad-multiple` | 4 | pad each batch to its longest structure, rounded up to a multiple of 4, instead of a fixed K |
| `--batch-size`, `--lr`, `--max-steps` | 128, 1e-4, 300k | optimisation |
| `--sample-frequency`, `--ckpt-every` | 20k, 100k | evaluation and checkpoint cadence |
| `--ckpt-keep` | 1 | numbered checkpoints kept on disk; 0 keeps all |
| `--dataset` | `mp20` | `mp20` or `alex_mp20` ([datasets.md](datasets.md)) |

Paper names and the matching settings:

| Paper | Setting |
|---|---|
| Orbitol-X (representative input, projection readout) | `--coord-input explicit --coord-readout projection` (default) |
| Orbitol-I (chart input, direct readout) | `--coord-input internal --coord-readout direct` |
| target P with sine skip (the reported models) | `--objective posterior_periodic` |
| target P with EDM skip (control) | `--objective posterior` (default) |
| target M, Crystalite's wrapped residual | `--objective wrapped` (alias `M`) |
| hybrid row and group embeddings | `--row-embed hybrid --sg-embed hybrid` (default) |
| distance GEM (the reported models); GEM off | `--gem-mode distance`; `--gem-mode none` (default) |

The alias `--objective P` selects `posterior_unit` (posterior target with a unit skip), which is not a paper setting.

The Orbitol-I variant on Alex-MP-20 (paper appendix), for example:

```bash
slurm/submit slurm/train_orbitol.sbatch alex_idirect \
    --dataset alex_mp20 --objective posterior_periodic --gem-mode distance \
    --coord-input internal --coord-readout direct --loss-weights 0.5 50 5 --max-steps 1000000
```

## Sampling

```bash
python -m orbitol.sample <checkpoint.pt> --out samples.pt --n 10000 --seed 123 [--cif-dir cifs]
```

Sampling uses the EMA weights and the sampler settings stored in the checkpoint; flags such as `--sample-num-steps`,
`--S-churn`, `--sigma-min` and `--sigma-max` override them. The paper's MP-20 pools use the 25-step sampler of
`configs/paper/sampler_mp20_fast.json`. `--cif-dir` also writes each decoded structure as `<attempt>.cif`.

**Packing.** `sample_token_budget` packs several structures into one network call up to a fixed token count, instead
of padding every structure to K. It is faster and computes the same model, but with different kernels and operation
order, so individual samples can differ from padded sampling.

| `sample_token_budget` | Meaning |
|---|---|
| `-1` (default) | auto: 8,192 tokens for d512 and below, 6,144 above |
| `0` | padded chunks |
| positive | that budget |

On an RTX 6000 Ada (paper, Table 13), the released MP-20 model samples 921 structures per second with the paper's
25-step sampler and the fast decoder, and 420 per second with the reference decoder that the paper-pool recipe uses.
The 400k-step checkpoint of the same architecture samples 185 per second with 150 steps. See
[hardware.md](hardware.md) for a comparison across GPUs.
