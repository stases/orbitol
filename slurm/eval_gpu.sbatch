#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# Sample and relax one eval snapshot, then submit CPU scoring.
# Usage: slurm/submit slurm/eval_gpu.sbatch [args ...]
set -euo pipefail
source "${ORBITOL_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit through slurm/submit}}/slurm/common.sh" ||
    { echo "Submit through slurm/submit from the Orbitol checkout" >&2; exit 2; }
source "$REPO/slurm/oeq_env.sh"
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$REPO"
exec "$ORBITOL_PYTHON" -m orbitol.eval_gpu "$@"
