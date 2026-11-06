#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
# Manual posthoc evaluation uses the same relaxation/scoring protocol as training.
# Usage: slurm/submit slurm/posthoc_eval.sbatch KIND OUT_ROOT SNAPSHOT [SNAPSHOT ...]
# Optional env: EVAL_WANDB_NAME, EVAL_WANDB_PROJECT, EVAL_RELAX_MODE,
# EVAL_SAMPLE_COUNT, EVAL_SUN_K, EVAL_SEED, EVAL_RELAX_BATCH_SIZE,
# EVAL_NEQUIP_BACKEND, EVAL_RELAX_STEPS. SMOKE=1 evaluates 64 samples.
set -euo pipefail
source "${ORBITOL_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit through slurm/submit}}/slurm/common.sh" ||
    { echo "Submit through slurm/submit from the Orbitol checkout" >&2; exit 2; }
if (( $# < 3 )); then
    echo "Usage: slurm/submit slurm/posthoc_eval.sbatch KIND OUT_ROOT SNAPSHOT [SNAPSHOT ...]" >&2
    exit 2
fi
KIND=$1
OUT_ROOT=$2
shift 2
source "$REPO/slurm/oeq_env.sh"
: "${ORBITOL_PYTHON:?slurm/oeq_env.sh must export ORBITOL_PYTHON}"
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
# Snapshot configs can enable trunk compilation; Dynamo invokes nvcc by name.
if [[ -n ${ORBITOL_CUDA_HOME:-} ]]; then export PATH="$ORBITOL_CUDA_HOME/bin:$PATH"; fi
# Small Frechet matrix operations and deterministic sampling use one BLAS thread.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WB_ARGS=()
if [[ -n "${EVAL_WANDB_NAME:-}" ]]; then
    # Without EVAL_WANDB_PROJECT, posthoc_eval logs to the pool's dataset project.
    WB_ARGS=(${EVAL_WANDB_PROJECT:+--wandb-project "$EVAL_WANDB_PROJECT"} --wandb-name "$EVAL_WANDB_NAME")
fi
for SNAPSHOT in "$@"; do
    SNAPSHOT_NAME=$(basename "$SNAPSHOT" .pt)
    EVAL_ARGS=()
    if [[ "${SMOKE:-0}" == 1 ]]; then EVAL_ARGS+=(--sample-count 64); fi
    for SETTING in SAMPLE_COUNT SUN_K SEED RELAX_MODE RELAX_BATCH_SIZE NEQUIP_BACKEND RELAX_STEPS; do
        ENV_NAME=EVAL_$SETTING
        if [[ -n "${!ENV_NAME:-}" ]]; then
            FLAG=${SETTING,,}
            EVAL_ARGS+=(--"${FLAG//_/-}" "${!ENV_NAME}")
        fi
    done
    "$ORBITOL_PYTHON" -m orbitol.posthoc_eval \
        --kind "$KIND" --snapshot "$SNAPSHOT" --out "$OUT_ROOT/$SNAPSHOT_NAME" \
        "${EVAL_ARGS[@]}" "${WB_ARGS[@]}"
done
