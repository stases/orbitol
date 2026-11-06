#!/usr/bin/env bash
# Reproduce the locked runtime; no cluster paths or compiler setup required.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CORE_SHA=03603fe0184cbe76db20a641c8cb20e93bc1bd41
CORE_DIR=${ORBITOL_CORE_DIR:-$REPO/external/crystalite-03603fe}
SYNC_ARGS=()
while (($#)); do
    case "$1" in
        --no-dev) SYNC_ARGS+=(--no-dev); shift ;;
        --core-dir) : "${2:?--core-dir requires a directory}"; CORE_DIR=$2; shift 2 ;;
        -h|--help)
            echo "Usage: $REPO/tools/setup_env.sh [--no-dev] [--core-dir DIRECTORY]"
            echo 'Environment: UV, UV_PROJECT_ENVIRONMENT, UV_CACHE_DIR, ORBITOL_CORE_ROOT, ORBITOL_CORE_DIR'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
cd "$REPO"
UV=${UV:-uv}
command -v "$UV" >/dev/null || { echo 'uv is required; install uv and put it on PATH.' >&2; exit 1; }
echo "uv: $(command -v "$UV") ($("$UV" --version))"
[[ -f pyproject.toml && -f uv.lock && -f .python-version ]] || {
    echo "Missing project environment specification in $REPO" >&2; exit 1;
}
if [[ $(uname -s) == Linux ]] && command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo 'Warning: nvidia-smi could not query a GPU.' >&2
    NVCC=${CUDA_HOME:+$CUDA_HOME/bin/nvcc}
    NVCC=${NVCC:-$(command -v nvcc || true)}
    if [[ -n "$NVCC" && -x "$NVCC" ]]; then "$NVCC" --version; else
        echo 'Warning: nvcc missing; OEQ relaxation needs a CUDA toolkit (CUDA_HOME/bin/nvcc). TorchScript does not.' >&2
    fi
    if command -v gcc >/dev/null; then
        gcc --version | head -1
        (( $(gcc -dumpversion | cut -d. -f1) >= 12 )) ||
            echo 'Warning: OEQ was built with gcc 12; older host compilers fail to build it. TorchScript does not need one.' >&2
    else
        echo 'Warning: gcc missing; OEQ relaxation needs a host compiler (gcc 12). TorchScript does not.' >&2
    fi
fi
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}
[[ "$UV_PROJECT_ENVIRONMENT" == /* ]] || export UV_PROJECT_ENVIRONMENT="$REPO/$UV_PROJECT_ENVIRONMENT"
# uv's own CPython from .python-version (3.12.9, the paper runtime), downloaded if needed; never a system
# Python, which may differ in patch version and lack the headers Triton compiles against.
"$UV" sync --frozen --python "$(cat .python-version)" --python-preference only-managed "${SYNC_ARGS[@]}"
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
if [[ -z ${ORBITOL_CORE_ROOT:-} ]]; then
    command -v git >/dev/null || { echo 'git is required to obtain the pinned Crystalite core.' >&2; exit 1; }
    [[ "$CORE_DIR" == /* ]] || CORE_DIR="$REPO/$CORE_DIR"
    if [[ ! -e "$CORE_DIR" ]]; then
        mkdir -p "$(dirname "$CORE_DIR")"
        git clone --no-checkout https://github.com/joshrosie/crystalite "$CORE_DIR"
        git -C "$CORE_DIR" checkout --detach "$CORE_SHA"
    elif [[ $(git -C "$CORE_DIR" rev-parse HEAD) != "$CORE_SHA" ]]; then
        echo "Existing core at $CORE_DIR is not pinned to $CORE_SHA; choose an empty --core-dir." >&2
        exit 1
    fi
    export ORBITOL_CORE_ROOT="$CORE_DIR"
fi
ORBITOL_CORE_ROOT=$(cd "$ORBITOL_CORE_ROOT" && pwd)
export ORBITOL_CORE_ROOT PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m orbitol.core
"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import os
from pathlib import Path
import sys
import torch
import orbitol
assert Path(orbitol.__file__).resolve().parent.parent == Path(os.environ['PYTHONPATH']).resolve()
print(f'Python: {sys.executable} ({sys.version.split()[0]})')
for name in ('torch', 'numpy', 'pymatgen', 'nequip', 'openequivariance'):
    print(f'{name}: {metadata.version(name)}')
print(f'torch.version.cuda: {torch.version.cuda}')
PY
printf 'export PYTHONPATH=%q\nexport ORBITOL_CORE_ROOT=%q\nexport ORBITOL_PYTHON=%q\n' "$REPO" "$ORBITOL_CORE_ROOT" "$PYTHON"
printf 'export PATH=%q:"$PATH"\n' "$UV_PROJECT_ENVIRONMENT/bin"
echo 'OEQ JIT uses ninja from the environment bin directory; configure CUDA_HOME and a compatible host compiler.'
