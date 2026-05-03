#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

DATA_GEN_ROOT="${DATA_GEN_ROOT:-$WORKSPACE_ROOT}"

if [ -f "$WORKSPACE_ROOT/config.env" ]; then
    # shellcheck source=/dev/null
    source "$WORKSPACE_ROOT/config.env"
fi

# Fallback for running outside the Docker image.
if [ ! -d "${ISAACLAB_ROOT:-}" ] && [ -d /home/ntruong/Truong/IsaacLab ]; then
    ISAACLAB_ROOT=/home/ntruong/Truong/IsaacLab
fi

ISAACLAB_SH="${ISAACLAB_SH:-$ISAACLAB_ROOT/isaaclab.sh}"
DATA_ROOT="${DATA_ROOT:-$WORKSPACE_ROOT/data}"
LOG_ROOT="${LOG_ROOT:-$WORKSPACE_ROOT/logs}"
TASK_INFO_SCRIPT="$DATA_GEN_ROOT/scripts/task_spec_info.py"
ISAAC_ENV_CHECK_SCRIPT="$DATA_GEN_ROOT/scripts/check_isaaclab_env.py"

choose_python() {
    if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
        printf '%s\n' "$PYTHON_BIN"
        return
    fi
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
        printf '%s\n' "$VIRTUAL_ENV/bin/python"
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    return 1
}

PYTHON_BIN="$(choose_python)"
ISAAC_PYTHON_BIN="${ISAAC_PYTHON_BIN:-$PYTHON_BIN}"

require_file() {
    local path=$1
    if [ ! -f "$path" ]; then
        echo "Error: missing file: $path" >&2
        exit 1
    fi
}

require_dir() {
    local path=$1
    if [ ! -d "$path" ]; then
        echo "Error: missing directory: $path" >&2
        exit 1
    fi
}

ensure_data_env() {
    require_dir "$DATA_GEN_ROOT"
    require_file "$TASK_INFO_SCRIPT"
    require_file "$ISAAC_ENV_CHECK_SCRIPT"
    require_file "$ISAACLAB_SH"
    mkdir -p "$DATA_ROOT/source" "$DATA_ROOT/mimic" "$LOG_ROOT"
}

task_field() {
    local spec=$1
    local field=$2
    PYTHONPATH="$DATA_GEN_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$TASK_INFO_SCRIPT" --spec "$spec" --field "$field"
}

check_isaac_runtime() {
    PYTHONPATH="$DATA_GEN_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$ISAAC_PYTHON_BIN" "$ISAAC_ENV_CHECK_SCRIPT"
}

print_env_summary() {
    echo "[env] WORKSPACE_ROOT=$WORKSPACE_ROOT"
    echo "[env] DATA_GEN_ROOT=$DATA_GEN_ROOT"
    echo "[env] ISAACLAB_ROOT=$ISAACLAB_ROOT"
    echo "[env] ISAACLAB_SH=$ISAACLAB_SH"
    echo "[env] PYTHON_BIN=$PYTHON_BIN"
    echo "[env] ISAAC_PYTHON_BIN=$ISAAC_PYTHON_BIN"
    echo "[env] DATA_ROOT=$DATA_ROOT"
}
