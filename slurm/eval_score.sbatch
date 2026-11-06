#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# Score the completed relaxation endpoints; no GPU is requested.
# Usage: slurm/submit slurm/eval_score.sbatch [args ...]
set -euo pipefail
source "${ORBITOL_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit through slurm/submit}}/slurm/common.sh" ||
    { echo "Submit through slurm/submit from the Orbitol checkout" >&2; exit 2; }
source "$REPO/slurm/oeq_env.sh"
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$REPO"
exec "$ORBITOL_PYTHON" -m orbitol.eval_score "$@"
