#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@300
# Manual queue for saved checkpoints; all CLI arguments are forwarded.
# Usage: slurm/submit slurm/posthoc_queue.sbatch [args ...]
set -euo pipefail
source "${ORBITOL_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit through slurm/submit}}/slurm/common.sh" ||
    { echo "Submit through slurm/submit from the Orbitol checkout" >&2; exit 2; }
source "$REPO/slurm/oeq_env.sh"
: "${ORBITOL_PYTHON:?slurm/oeq_env.sh must export ORBITOL_PYTHON}"
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
# Snapshot configs can enable trunk compilation; Dynamo invokes nvcc by name.
if [[ -n ${ORBITOL_CUDA_HOME:-} ]]; then export PATH="$ORBITOL_CUDA_HOME/bin:$PATH"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
child_pid=
trap '[[ -z "$child_pid" ]] || kill -USR1 "$child_pid" 2>/dev/null || true' USR1
trap '[[ -z "$child_pid" ]] || kill -TERM "$child_pid" 2>/dev/null || true' TERM
"$ORBITOL_PYTHON" -m orbitol.posthoc_queue "$@" &
child_pid=$!
while true; do
    set +e
    wait "$child_pid"
    status=$?
    set -e
    if ! kill -0 "$child_pid" 2>/dev/null; then
        exit "$status"
    fi
done
