#!/usr/bin/env bash
# Shared local configuration and interpreters. Submit jobs through slurm/submit.
# Cluster values belong in orbitol.local.env; nonempty environment values win.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script; submit jobs through slurm/submit." >&2
    exit 1
fi
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export ORBITOL_REPO_ROOT=$REPO

orbitol_load_local() {
    local file=$1 line key number=0
    local -A values=()
    [[ -f $file ]] || return 0
    while IFS= read -r line || [[ -n $line ]]; do
        number=$((number + 1))
        [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
        [[ $line =~ ^[[:space:]]*(export[[:space:]]+)?([A-Z][A-Z0-9_]*)=([^[:space:]\'\"\`\$\\]*)[[:space:]]*$ ]] || {
            echo "$file:$number: expected KEY=value with a literal value (no spaces, quotes, \$ or backslashes)" >&2
            return 1
        }
        values[${BASH_REMATCH[2]}]=${BASH_REMATCH[3]}
    done < "$file"
    # Like the Python reader, the last file entry wins; the initial environment wins over all entries.
    for key in "${!values[@]}"; do
        [[ -n ${!key:-} ]] || export "$key=${values[$key]}"
    done
}
orbitol_load_local "${ORBITOL_LOCAL_ENV:-$REPO/orbitol.local.env}" || return $?
export ORBITOL_PYTHON=${ORBITOL_PYTHON:-$REPO/.venv/bin/python}
export ORBITOL_BASE_PYTHON=${ORBITOL_BASE_PYTHON:-$ORBITOL_PYTHON}
if [[ -n ${ORBITOL_WANDB_ENTITY:-} ]]; then export WANDB_ENTITY=$ORBITOL_WANDB_ENTITY; fi
