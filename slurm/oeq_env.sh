#!/usr/bin/env bash
# Source before OEQ import: compilers, CUDA and JIT caches for the OpenEquivariance build.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script to configure the current shell." >&2
    exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/common.sh" || return $?
# Host compiler for the OEQ JIT build (gcc >= 12).
if [[ -n ${ORBITOL_GCC_ENABLE:-} ]]; then source "$ORBITOL_GCC_ENABLE" || return $?; fi
export CC="$(command -v gcc)" CXX="$(command -v g++)"
if [[ -n ${ORBITOL_CUDA_HOME:-} ]]; then export CUDA_HOME=$ORBITOL_CUDA_HOME; fi
# OEQ silently fails to build without ninja: the interpreter's bin must lead PATH.
export PATH="$(dirname "$ORBITOL_PYTHON"):${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
if [[ -n ${ORBITOL_TORCH_CUDA_ARCH_LIST:-} ]]; then export TORCH_CUDA_ARCH_LIST=$ORBITOL_TORCH_CUDA_ARCH_LIST; fi
jit=${ORBITOL_DATA_HOME:?set ORBITOL_DATA_HOME in orbitol.local.env}/jit_cache
export TORCH_EXTENSIONS_DIR=$jit/torch_extensions TORCHINDUCTOR_CACHE_DIR=$jit/torchinductor
export TRITON_CACHE_DIR=$jit/triton NUMBA_CACHE_DIR=$jit/numba
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
# Building the NequIP relaxer switches torch to the "file_system" sharing strategy, which routes every DataLoader
# batch through torch_shm_manager; on a CPU-saturated node it times out and kills training. Keep torch's default.
export NEQUIP_MULTIPROCESSING_SHARING_STRATEGY=file_descriptor
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$NUMBA_CACHE_DIR" || return $?
[[ -x "$ORBITOL_PYTHON" && -x "$(dirname "$ORBITOL_PYTHON")/ninja" ]] || {
    echo "Missing ORBITOL_PYTHON ($ORBITOL_PYTHON) or ninja next to it; run tools/setup_env.sh or set ORBITOL_PYTHON." >&2
    return 1
}
