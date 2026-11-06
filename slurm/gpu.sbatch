#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
# Usage: slurm/submit [--partition PARTITION --account ACCOUNT] --job-name NAME slurm/gpu.sbatch <command ...>
set -euo pipefail
source "${ORBITOL_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit through slurm/submit}}/slurm/common.sh" ||
    { echo "Submit through slurm/submit from the Orbitol checkout" >&2; exit 2; }
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1
# Dynamo's diagnostic generator invokes nvcc by name, independently of CUDACXX.
if [[ -n ${ORBITOL_CUDA_HOME:-} ]]; then export PATH="$ORBITOL_CUDA_HOME/bin:$PATH"; fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$REPO"
echo "host $(hostname) start $(date -Is) commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader || true
"$ORBITOL_BASE_PYTHON" "$@"
