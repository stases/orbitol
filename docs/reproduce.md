# Reproducing Orbitol

Run commands from a fresh Git checkout. The release contains only Orbitol-X MP-20 at step
1,500,000 of the 2.5M run. This repository covers its training, sampling and in-house evaluation; the
paper's main tables score the samples with LeMat-GenBench. Configuration, provenance and expected results are in
[configs/paper/models.json](../configs/paper/models.json); settings the paper does not list (training
seed 123, pool seeds, `pad_multiple`) are in these files, and the element mask is derived from the
training data. No Alex-MP-20, control or Orbitol-I weights are released.
Release files: [huggingface.co/stases/orbitol](https://huggingface.co/stases/orbitol), revision `0df0e17ce04a39ab78fb6c3a74dcce4a05495a6c`.

## Prerequisites and setup
Use Linux x86_64, git, uv, an NVIDIA GPU supporting bf16, a driver compatible with the locked
PyTorch CUDA 12.8 wheels, and a CUDA toolkit (`nvcc`, tested with 12.9) plus gcc ≥ 12 for the
OpenEquivariance (OEQ) JIT. Setup installs ninja. Keep the checkout for training's Git provenance.
Budget space for the environment, extracted bundle, public inputs and JIT caches separately;
retain 20 GiB free after downloads. Training and the Table 12 trainer require at least 15 GB free at
runtime under the output root. Inspect fetch's dry run before downloading; the MP-20 hull alone
expands to about 2 GB. Allow additional space for checkpoints and evaluation outputs.

```bash
tools/setup_env.sh
# Apply the exports it prints, including PATH, PYTHONPATH and ORBITOL_PYTHON.
cp orbitol.local.env.example orbitol.local.env
# Edit the file: set absolute ORBITOL_DATA_HOME and ORBITOL_RUNS_ROOT directories.
python -m orbitol.contracts
export ORBITOL_DATA_HOME="$(python -m orbitol.contracts ORBITOL_DATA_HOME)"
export ORBITOL_RUNS_ROOT="$(python -m orbitol.contracts ORBITOL_RUNS_ROOT)"
```

The local file accepts literal `KEY=value`, optionally prefixed by `export`; no quotes, spaces
or shell expansion. Environment values win. `ORBITOL_LOCAL_ENV` selects a replacement file.
[Dataset layout and overrides](datasets.md) come from `C.DATA_LAYOUT`; Python reads the file
itself. The two exports above make the roots available to the shell examples as well.

## Fetch, relaxer and verification

The fetch interface places bundle and public inputs at configured paths and checks every file
against its SHA-256. Supply the bundle's download URL at the paper revision, or a local copy of it,
with `--bundle`:

```bash
BUNDLE=https://huggingface.co/stases/orbitol/resolve/0df0e17ce04a39ab78fb6c3a74dcce4a05495a6c
python -m tools.release.fetch --bundle "$BUNDLE" --public mp20-raw,mp20-hull --dry-run
python -m tools.release.fetch --bundle "$BUNDLE" --public mp20-raw,mp20-hull
```

A local copy works the same way: `uvx --from huggingface_hub hf download stases/orbitol --revision 0df0e17ce04a39ab78fb6c3a74dcce4a05495a6c --local-dir <dir>`,
then `--bundle <dir>`.

The bundle contains the data layout, `checkpoints/`, `configs/paper/` and `MANIFEST.json`.
MP-20 raw CSVs and its hull are fetched separately. Processed train/validation tokens, orbit caches and
the symmetry reference are bundled; processed test splits, generated pools and Alex-MP-20 data are not.
The public MP-20 raw download includes the test CSV, which `verify data` checks; the released training and
evaluation recipes do not use it, and the paper uses the test split only for skeleton coverage. The optional
public input `nequip-source` rebuilds the relaxer: `python -m tools.release.fetch --public nequip-source`.

The shipped OEQ artifact runs on Ampere and Ada GPUs. On Turing, build it on that GPU, with
CUDA_HOME and gcc configured in the shell (or use the Slurm wrapper below):

```bash
python tools/build_nequip.py --only oeq --force
```

This replaces the configured OEQ output, checks its weights and energy parity against the
bundled TorchScript artifact, and writes a build receipt. A rebuilt file need not have the paper's
file hash. The bundled relaxers do not have it either: their TorchScript debug tables were removed, so
`verify nequip` reports `paper_identical: false`, while their weights digest, which it checks, is the
paper's. Keep the original bundle if you need both versions. Then check the configured installation:

```bash
python -m tools.release.verify data
python -m tools.release.verify checkpoints
python -m tools.release.verify nequip
```

Verification returns nonzero on failure. Resolve failures before expensive reproduction.

## MP-20 paper pools

These settings match the recorded pool/evaluation settings: 2,500 attempts, seeds 3001–3003,
EMA weights, reference decoding, all-attempt OEQ relaxation, 500 steps, batch size 128. Even
`--stage sample` requires the symmetry reference, hull and relaxer because it hashes them.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled
B="$ORBITOL_DATA_HOME/checkpoints"
OUT="$ORBITOL_RUNS_ROOT/reproduce"
for S in 3001 3002 3003; do
  ORBITOL_NUMBA=0 python -m orbitol.posthoc_eval --kind orbitol --snapshot "$B/mp20_x_postper_2500k/step_1500000.pt" \
    --out "$OUT/mp20_fast_s$S" --stage all --sample-count 2500 --seed "$S" \
    --sun-k 0 --nequip-backend oeq --relax-steps 500 --relax-mode batch --relax-batch-size 128 \
    --sampler-overrides "$(cat configs/paper/sampler_mp20_fast.json)"
done
```

`ORBITOL_NUMBA=0` (plain-Python kernels) applies to this command only; the later commands, including
Table 12's fast decoder, keep the compiled Numba kernels.

`--stage all` adds in-house relaxation/scoring; use `--stage sample` for raw pools only.
Compare two pools with `python -m tools.release.verify pools LEFT RIGHT`. By default it requires bitwise
equality of everything except relocatable metadata. With `--max-differing N`, requested groups, skeletons,
key sets, shapes and dtypes must still match; only the generated tensors and decode outcomes of at most N
attempts may differ, and failure counts must agree with failures.
Relaxation varies between runs (GPU atomics), so compare MSUN, not files: seeds 3001-3003 of the released
checkpoint score in-house MSUN 0.4584 / 0.4492 / 0.4724 (mean 0.4600). On other GPU architectures every
sample differs, so each seed is a new draw (on an RTX 3090, mean 0.4504).

## Table 12: in-house evaluation

Table 12 uses the fast decoder and 10,000 attempts per seed. The registry holds the two rows
with three seeds (chosen and stored); run them on a GPU:

```bash
for S in 2026 2027 2028; do
  python -m tools.sampler_sweep --checkpoint "$B/mp20_x_postper_2500k/step_1500000.pt" \
    --out "$OUT/table12" --samples 10000 --seed "$S" \
    --settings "{\"name\":\"chosen_seed$S\",\"sample_num_steps\":25,\"S_churn\":10,\"sigma_min\":0.01,\"sigma_max\":3.0,\"decode_impl\":\"fast\"}" \
               "{\"name\":\"stored_seed$S\",\"sigma_max\":80.0,\"sample_num_steps\":150,\"S_churn\":60,\"decode_impl\":\"fast\"}"
done
python -m tools.release.verify table12 "$OUT/table12"
```

`verify table12` accepts a difference of 0.01 per seed and rate (`--tolerance`) and needs all
three seeds of each setting it finds (`--partial` accepts fewer). The table's six single-seed rows
use seed 2026 and the settings printed there, with `S_churn` = 0.4 × steps; run them with the same
command (for example `{"name":"n50","sample_num_steps":50,"S_churn":20,"sigma_min":0.01,"sigma_max":3.0,"decode_impl":"fast"}`)
and compare with the table by hand.
Registry values below are fractions, in seed order 2026/2027/2028. Relaxed endpoints are not bitwise
reproducible, even on the same GPU type, so compare statistically.

| Setting | MSUN | UN | One-shot MSUN |
|---|---|---|---|
| Chosen | 0.4506 / 0.4440 / 0.4469 | 0.77065 / 0.7651 / 0.7601 | 0.3158 / 0.3185 / 0.3141 |
| Stored | 0.4398 / 0.4428 / 0.4475 | not recorded in registry | not recorded in registry |

On one RTX 6000 Ada, each setting/seed took about 30 minutes (evaluation: 1,719–1,784 s).
Six setting/seed combinations cost roughly three GPU-hours, plus setup and compilation.

## Retraining
Train from scratch with the paper configuration; CLI flags override the JSON. The released
EMA-only weights lack optimizer state and are for inference, not exact training continuation.

```bash
python -m orbitol.train --run-name reproduce_mp20 --out-root "$ORBITOL_RUNS_ROOT" --no-wandb \
  --config configs/paper/mp20_x_postper_2500k.json --eval-mode sync --ckpt-every 100000 --ckpt-keep 0
```

Keeping periodic checkpoints preserves step 1.5M after later training. The JSON runs to 2.5M.

## Any Slurm cluster

Submit from the checkout. Put local placement in `ORBITOL_SLURM_GPU_PARTITION`,
`ORBITOL_SLURM_GPU_ACCOUNT`, `ORBITOL_SLURM_CPU_PARTITION`, `ORBITOL_SLURM_CPU_ACCOUNT`; unset keys
use cluster defaults. `ORBITOL_SLURM_LOG_DIR` selects logs. `ORBITOL_PYTHON` overrides the project
interpreter; `ORBITOL_CUDA_HOME` and `ORBITOL_GCC_ENABLE` configure OEQ's toolkit/compiler.

```bash
slurm/submit slurm/build_nequip.sbatch --only oeq --force
slurm/submit slurm/train_orbitol.sbatch paper_mp20 --config configs/paper/mp20_x_postper_2500k.json --no-wandb \
  --ckpt-keep 0
```

Syntax: `slurm/submit [sbatch options] slurm/<name>.sbatch [args]`; explicit options win.
`--ckpt-keep 0` keeps every 100k checkpoint, including step 1.5M; the default keeps only the newest.
Nested GPU evaluation and CPU scoring jobs also use it. The evaluation mode comes from the config
(the paper JSONs use sync, in the training allocation; TrainConfig defaults to async); `EVAL_MODE` or
`--eval-mode` overrides it. Old checkpoints can retain placement:
override `--eval-partition`/`--eval-account`, or use sync. Never run training/relaxation on a login node.

`slurm/release_check.sbatch` runs these steps in one GPU job from a fresh clone: environment, bundle fetch and
verification, MP-20 pools, Table 12 and a 200-step training run. It compares against the stored paper pools and
the full training checkpoint, so it takes them as inputs.

```bash
RELEASE_WORK_ROOT=/node/local/scratch slurm/submit slurm/release_check.sbatch --bundle DIR \
  --mp20-pool DIR --table12-samples FILE --full-checkpoint FILE
```

## Licences

Code is [MIT](../LICENSE), with third-party notices in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md);
the bundle is CC BY 4.0. See [licences and source attribution](../tools/release/ATTRIBUTION.md) and cite the
paper and data sources.
