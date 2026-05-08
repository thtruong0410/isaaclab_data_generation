#!/usr/bin/env bash
set -euo pipefail

# Convert MimicGen budget outputs to GR00T-flavored LeRobot v2 datasets.
#
# Usage:
#   bash scripts/convert_mimic_budgets_to_lerobot.sh BASE_SPEC [SETUP_NAME] [MIMIC_ROOT] [LEROBOT_ROOT]
#
# Example:
#   bash scripts/convert_mimic_budgets_to_lerobot.sh open_drawer_normal open_drawer_normal_top
#
# Expected input layout:
#   data/mimic/<SETUP_NAME>_10/<generated_hdf5_name>
#   data/mimic/<SETUP_NAME>_20/<generated_hdf5_name>
#   ...
#
# Output layout:
#   data/lerobot/<SETUP_NAME>_10/
#   data/lerobot/<SETUP_NAME>_20/
#   ...
#
# Optional env vars:
#   BUDGETS="10 20 30 40 50"
#   TASK_TEXT="open the drawer"
#   PYTHON_BIN=/path/to/python
#   FPS=30
#   CRF=23
#   SKIP_EXISTING=1
#   FAIL_ON_MISSING=1
#   DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

BASE_SPEC="${1:-}"
SETUP_NAME="${2:-}"
MIMIC_ROOT_ARG="${3:-}"
LEROBOT_ROOT_ARG="${4:-}"

if [ -z "$BASE_SPEC" ]; then
    echo "Usage: bash scripts/convert_mimic_budgets_to_lerobot.sh BASE_SPEC [SETUP_NAME] [MIMIC_ROOT] [LEROBOT_ROOT]" >&2
    exit 1
fi

SETUP_NAME="${SETUP_NAME:-$BASE_SPEC}"
MIMIC_ROOT="${MIMIC_ROOT_ARG:-$DATA_ROOT/mimic}"
LEROBOT_ROOT="${LEROBOT_ROOT_ARG:-$DATA_ROOT/lerobot}"
BUDGETS="${BUDGETS:-10 20 30 40 50}"
FPS="${FPS:-30}"
CRF="${CRF:-23}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
FAIL_ON_MISSING="${FAIL_ON_MISSING:-0}"
DRY_RUN="${DRY_RUN:-0}"

CONVERT_SCRIPT="$DATA_GEN_ROOT/scripts/convert_isaaclab_hdf5_to_lerobot.py"

require_file "$TASK_INFO_SCRIPT"
require_file "$CONVERT_SCRIPT"
require_dir "$MIMIC_ROOT"
mkdir -p "$LEROBOT_ROOT"

GENERATED_NAME="$(task_field "$BASE_SPEC" dataset_naming.generated_hdf5_name)"
TASK_TEXT="${TASK_TEXT:-$(task_field "$BASE_SPEC" eval.task_description)}"

if [ "$DRY_RUN" != "1" ]; then
    if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import av
import h5py
import numpy
import pandas
PY
    then
        echo "Error: selected Python cannot import converter dependencies: av, h5py, numpy, pandas." >&2
        echo "Set PYTHON_BIN to the Python environment that can run convert_isaaclab_hdf5_to_lerobot.py." >&2
        exit 1
    fi
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "Base spec   : $BASE_SPEC"
log "Setup name  : $SETUP_NAME"
log "Task text   : $TASK_TEXT"
log "HDF5 name   : $GENERATED_NAME"
log "Budgets     : $BUDGETS"
log "Mimic root  : $MIMIC_ROOT"
log "LeRobot root: $LEROBOT_ROOT"
log "Python      : $PYTHON_BIN"

converted=0
skipped=0
failed=0

for budget in $BUDGETS; do
    if ! [[ "$budget" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid budget '$budget'. BUDGETS must contain integers." >&2
        exit 1
    fi

    variant="${SETUP_NAME}_${budget}"
    input_path="$MIMIC_ROOT/$variant/$GENERATED_NAME"
    output_dir="$LEROBOT_ROOT/$variant"

    if [ ! -f "$input_path" ]; then
        log "[missing] $variant: $input_path"
        if [ "$FAIL_ON_MISSING" = "1" ]; then
            failed=$((failed + 1))
        else
            skipped=$((skipped + 1))
        fi
        continue
    fi

    if [ "$SKIP_EXISTING" = "1" ] && [ -d "$output_dir/meta" ]; then
        log "[skip] $variant already converted at $output_dir"
        skipped=$((skipped + 1))
        continue
    fi

    log "[convert] $variant"
    log "  input : $input_path"
    log "  output: $output_dir"

    if [ "$DRY_RUN" = "1" ]; then
        log "  [dry-run] $PYTHON_BIN $CONVERT_SCRIPT --input $input_path --output $output_dir --task \"$TASK_TEXT\" --fps $FPS --crf $CRF"
        skipped=$((skipped + 1))
        continue
    fi

    if "$PYTHON_BIN" "$CONVERT_SCRIPT" \
        --input "$input_path" \
        --output "$output_dir" \
        --task "$TASK_TEXT" \
        --fps "$FPS" \
        --crf "$CRF"; then
        converted=$((converted + 1))
    else
        failed=$((failed + 1))
    fi
done

log "Summary: converted=$converted skipped=$skipped failed=$failed"

if [ "$failed" -gt 0 ]; then
    exit 1
fi
