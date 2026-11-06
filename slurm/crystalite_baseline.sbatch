#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
# Auxiliary Crystalite baseline recipe: Crystalite's own training script from the pinned
# 03603fe export, with the flags of Crystalite's main model except GEM off, aa_rho 0,
# PCA16 typing, 300k steps.
# No MLIP inside training: in-training sampling metrics every 20k (validity, uniqueness, novelty, UN,
# Wasserstein); NequIP SUN/MSUN runs afterwards on the 20k snapshots with orbitol.eval_adapter, the same
# code path as the Orbitol run.
# Usage: slurm/submit --job-name NAME slurm/crystalite_baseline.sbatch <run_name> [extra flags overriding the defaults]
set -euo pipefail
source "${ORBITOL_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit through slurm/submit}}/slurm/common.sh" ||
    { echo "Submit through slurm/submit from the Orbitol checkout" >&2; exit 2; }
RUN_NAME=$1; shift
PIN=$(PYTHONPATH="$REPO" "$ORBITOL_BASE_PYTHON" -m orbitol.contracts ORBITOL_CORE_ROOT)
PY="$ORBITOL_BASE_PYTHON"
DR=$(PYTHONPATH="$REPO" "$ORBITOL_BASE_PYTHON" -m orbitol.contracts ORBITOL_MP20_DATA_ROOT)
OUT=${ORBITOL_RUNS_ROOT:?}/$RUN_NAME
mkdir -p "$OUT" && cd "$OUT"
# One BLAS thread: the PCA16 type-encoding fit has degenerate eigenvalues (13-16), and multi-threaded BLAS
# rotates those columns. One thread reproduces the frozen table (state sha256 a1049dde...).
export PYTHONPATH=$PIN PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "host $(hostname) start $(date -Is)"; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
if curl -sS -m 10 -o /dev/null https://api.wandb.ai/ 2>/dev/null; then export WANDB_MODE=online; else export WANDB_MODE=offline; fi
echo "WANDB_MODE=$WANDB_MODE"
$PY $PIN/src/train_crystalite.py \
  --dataset_name mp20 --data_root $DR --metrics_dataset_name mp20 --metrics_data_root $DR \
  --type_encoding subatomic_tokenizer_pca_16 --nmax 20 \
  --d_model 512 --n_heads 16 --n_layers 14 --dropout 0 --attn_dropout 0 \
  --coord_embed_mode fourier --coord_n_freqs 32 --coord_head_mode direct --coord_loss_mode frac_mse \
  --lattice_repr ltri --lattice_embed_mode mlp \
  --sigma_data_type 0.3 --sigma_data_coord 0.3 --sigma_data_lattice 0.3 --loss_weights 1 50 5 \
  --edm_P_mean -1.2 --edm_P_std 1.2 --sigma_min 0.002 --sigma_max 80 \
  --rho 7 --S_churn 60 --S_min 0 --S_max 999 --S_noise 1.003 --aa_rho_coords 0 --aa_rho_lattice 0 --aa_rho_types 0 \
  --batch_size 128 --lr 1e-4 --weight_decay 0 --lr_warmup_steps 1000 --max_steps 300000 --ema_decay 0.9999 \
  --num_workers 4 --seed 123 --bf16 --log_every 50 --val_every 5000 --val_batches 10 \
  --sample_frequency 20000 --sample_count 2048 --sample_num_steps 150 --sample_mode ema --sample_seed 123 \
  --sample_chunk_size 2048 --sample_vis_count 5 --sample_novelty_limit -1 --atom_count_strategy empirical \
  --sun_k 0 \
  --ckpt_every 20000 \
  --wandb_project orbitol --wandb_name "$RUN_NAME" --output_dir "$OUT" "$@"
