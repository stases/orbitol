#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --signal=B:USR1@600
# EVAL_MODE=async or sync overrides --config and TrainConfig (default async). SMOKE=1: four steps, 64 samples at
# steps 2/4.
# Async training submits one GPU evaluation job per accepted snapshot.
set -euo pipefail
source "${ORBITOL_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit through slurm/submit}}/slurm/common.sh" ||
    { echo "Submit through slurm/submit from the Orbitol checkout" >&2; exit 2; }
: "${1:?Usage: slurm/submit slurm/train_orbitol.sbatch RUN_NAME [flags]}"
source "$REPO/slurm/oeq_env.sh"
: "${ORBITOL_PYTHON:?slurm/oeq_env.sh must export ORBITOL_PYTHON}"
export PYTHONPATH="$REPO"
# Dynamo's diagnostic generator invokes nvcc by name, independently of CUDACXX.
if [[ -n ${ORBITOL_CUDA_HOME:-} ]]; then export PATH="$ORBITOL_CUDA_HOME/bin:$PATH"; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$REPO"
if curl -sS -m 10 -o /dev/null https://api.wandb.ai/ 2>/dev/null; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi
echo "WANDB_MODE=$WANDB_MODE"
child_pid=
trap '[[ -z "$child_pid" ]] || kill -USR1 "$child_pid" 2>/dev/null || true' USR1
trap '[[ -z "$child_pid" ]] || kill -TERM "$child_pid" 2>/dev/null || true' TERM
EVAL_ARGS=()
if [[ -n ${EVAL_MODE:-} ]]; then EVAL_ARGS=(--eval-mode "$EVAL_MODE"); fi
SMOKE_ARGS=()
if [[ "${SMOKE:-0}" == 1 ]]; then
    SMOKE_ARGS=(--max-steps 4 --sample-count 64 --sample-frequency 2 --val-every 0 --ckpt-every 2 --num-workers 0)
fi
"$ORBITOL_PYTHON" -m orbitol.train --run-name "$1" --resume auto \
    "${EVAL_ARGS[@]}" "${SMOKE_ARGS[@]}" "${@:2}" &
child_pid=$!
# A trapped signal interrupts wait; keep the allocation alive until Python exits.
while true; do
    set +e
    wait "$child_pid"
    status=$?
    set -e
    if ! kill -0 "$child_pid" 2>/dev/null; then
        exit "$status"
    fi
done
