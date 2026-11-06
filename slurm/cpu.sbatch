#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# Usage: slurm/submit --job-name NAME slurm/cpu.sbatch <command ...>
set -euo pipefail
source "${ORBITOL_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit through slurm/submit}}/slurm/common.sh" ||
    { echo "Submit through slurm/submit from the Orbitol checkout" >&2; exit 2; }
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$REPO"
echo "host $(hostname) start $(date -Is) commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
"$ORBITOL_BASE_PYTHON" "$@"
