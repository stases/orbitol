#!/bin/bash
#SBATCH --job-name=orbitol_build_nequip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# Build on the target GPU class; pass fresh output paths to preserve paper artifacts.
# Usage: slurm/submit slurm/build_nequip.sbatch [args ...]
set -euo pipefail
source "${ORBITOL_REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit through slurm/submit}}/slurm/common.sh" ||
    { echo "Submit through slurm/submit from the Orbitol checkout" >&2; exit 2; }
cd "$ORBITOL_REPO_ROOT"
export PYTHONPATH="$ORBITOL_REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Keep caller-selected Python/toolchain while sourcing the cluster defaults.
declare -A overrides
for key in ORBITOL_PYTHON CUDA_HOME CC CXX TORCH_CUDA_ARCH_LIST; do
    if [[ -n ${!key:-} ]]; then overrides[$key]=${!key}; fi
done
source slurm/oeq_env.sh
for key in "${!overrides[@]}"; do export "$key=${overrides[$key]}"; done
export PATH="$(dirname "$ORBITOL_PYTHON"):${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
# A bare --reference-oeq opts into the locally configured reference. An explicit path wins.
args=()
while (( $# )); do
    if [[ $1 == --reference-oeq && ( $# == 1 || $2 == --* ) ]]; then
        args+=(--reference-oeq "$("$ORBITOL_PYTHON" -m orbitol.contracts ORBITOL_NEQUIP_OEQ)")
    else
        args+=("$1")
    fi
    shift
done
"$ORBITOL_PYTHON" -m tools.build_nequip "${args[@]}"
