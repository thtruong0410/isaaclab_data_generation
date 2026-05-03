#!/usr/bin/env bash
set -euo pipefail

# Run merge -> annotate -> MimicGen for one registered task spec.
# Usage:
#   bash scripts/mimic_generate.sh open_door_normal [num_trials] [raw_dir] [mimic_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

SPEC="${1:-}"
if [ -z "$SPEC" ]; then
    echo "Usage: bash scripts/mimic_generate.sh SPEC [NUM_MIMIC_DEMOS] [RAW_DIR] [MIMIC_DIR]" >&2
    exit 1
fi

NUM_MIMIC_DEMOS="${2:-${DEFAULT_MIMIC_DEMOS:-1000}}"
RAW_DIR="${3:-$DATA_ROOT/source/$SPEC}"
MIMIC_DIR="${4:-$DATA_ROOT/mimic/$SPEC}"
OVERWRITE="${OVERWRITE:-0}"

ensure_data_env
print_env_summary
check_isaac_runtime

MERGE_SCRIPT="$ISAACLAB_ROOT/scripts/tools/merge_hdf5_datasets.py"
ANNOTATE_WRAPPER="$WORKSPACE_ROOT/tools/annotate_task_demos_local.py"
GENERATE_WRAPPER="$WORKSPACE_ROOT/tools/generate_task_dataset_local.py"

require_file "$MERGE_SCRIPT"
require_file "$ANNOTATE_WRAPPER"
require_file "$GENERATE_WRAPPER"

DEMO_PREFIX="$(task_field "$SPEC" dataset_naming.raw_demo_file_prefix)"
MERGED_NAME="$(task_field "$SPEC" dataset_naming.merged_hdf5_name)"
ANNOTATED_NAME="$(task_field "$SPEC" dataset_naming.annotated_hdf5_name)"
GENERATED_NAME="$(task_field "$SPEC" dataset_naming.generated_hdf5_name)"
MIMIC_ENV_VAR_PREFIX="$(task_field "$SPEC" mimic_env_var_prefix)"

MERGED="$MIMIC_DIR/$MERGED_NAME"
ANNOTATED="$MIMIC_DIR/$ANNOTATED_NAME"
GENERATED="$MIMIC_DIR/$GENERATED_NAME"
GENERATED_FAILED="${GENERATED%.hdf5}_failed.hdf5"

mkdir -p "$MIMIC_DIR"

if [ -f "$GENERATED" ] && [ "$OVERWRITE" != "1" ]; then
    echo "Error: generated output already exists: $GENERATED" >&2
    echo "Set OVERWRITE=1 to regenerate this spec." >&2
    exit 1
fi

APP_LAUNCH_ARGS=(--enable_cameras)
case "${HEADLESS:-1}" in
    1|true|TRUE|yes|YES|on|ON)
        APP_LAUNCH_ARGS=(--headless --enable_cameras)
        ;;
esac

echo ""
echo "============================================"
echo "  Step 1/3 - Merge raw demos"
echo "============================================"
echo "[mimic] spec=$SPEC raw=$RAW_DIR mimic=$MIMIC_DIR trials=$NUM_MIMIC_DEMOS"

rm -f "$MERGED" "$ANNOTATED" "$GENERATED" "$GENERATED_FAILED"

mapfile -t RAW_HDF5_FILES < <(find -L "$RAW_DIR" -maxdepth 1 -type f -name "${DEMO_PREFIX}_*.hdf5" ! -name '*smoke*.hdf5' | sort -V)
if [ ${#RAW_HDF5_FILES[@]} -eq 0 ]; then
    echo "Error: no ${DEMO_PREFIX}_*.hdf5 files found in $RAW_DIR" >&2
    exit 1
fi

VALID_HDF5_FILES=()
CORRUPT_HDF5_FILES=()
for file in "${RAW_HDF5_FILES[@]}"; do
    if "$PYTHON_BIN" - "$file" <<'PY' >/dev/null 2>&1
import sys
import h5py
with h5py.File(sys.argv[1], "r"):
    pass
PY
    then
        VALID_HDF5_FILES+=("$file")
    else
        CORRUPT_HDF5_FILES+=("$file")
    fi
done

if [ ${#CORRUPT_HDF5_FILES[@]} -gt 0 ]; then
    echo "[mimic] skipping corrupted source demos:"
    printf '  - %s\n' "${CORRUPT_HDF5_FILES[@]}"
fi
if [ ${#VALID_HDF5_FILES[@]} -eq 0 ]; then
    echo "Error: all matched source demo files are corrupted." >&2
    exit 1
fi

"$PYTHON_BIN" "$MERGE_SCRIPT" --input_files "${VALID_HDF5_FILES[@]}" --output_file "$MERGED"
echo "[mimic] merged -> $MERGED"

bash_env=(env DATA_GEN_ROOT="$DATA_GEN_ROOT" ISAACLAB_ROOT="$ISAACLAB_ROOT" PYTHONPATH="$DATA_GEN_ROOT")

if [ -n "${ACTION_NOISE-}" ]; then
    bash_env+=("${MIMIC_ENV_VAR_PREFIX}_ACTION_NOISE=$ACTION_NOISE")
fi
if [ -n "${INTERP_STEPS-}" ]; then
    bash_env+=("${MIMIC_ENV_VAR_PREFIX}_INTERP_STEPS=$INTERP_STEPS")
fi
if [ -n "${GRASP_OFFSET_MIN-}" ]; then
    bash_env+=("${MIMIC_ENV_VAR_PREFIX}_GRASP_OFFSET_MIN=$GRASP_OFFSET_MIN")
fi
if [ -n "${GRASP_OFFSET_MAX-}" ]; then
    bash_env+=("${MIMIC_ENV_VAR_PREFIX}_GRASP_OFFSET_MAX=$GRASP_OFFSET_MAX")
fi
if [ -n "${NN_K-}" ]; then
    bash_env+=("${MIMIC_ENV_VAR_PREFIX}_NN_K=$NN_K")
fi

echo ""
echo "============================================"
echo "  Step 2/3 - Annotate subtask boundaries"
echo "============================================"

"${bash_env[@]}" bash "$ISAACLAB_SH" -p "$ANNOTATE_WRAPPER" \
    --spec "$SPEC" \
    --input_file "$MERGED" \
    --output_file "$ANNOTATED" \
    --auto \
    "${APP_LAUNCH_ARGS[@]}"

ANNOTATED_EPISODE_COUNT=$("$PYTHON_BIN" - "$ANNOTATED" <<'PY'
import sys
import h5py
with h5py.File(sys.argv[1], "r") as handle:
    data_group = handle.get("data")
    print(0 if data_group is None else len(data_group.keys()))
PY
)
if [ "$ANNOTATED_EPISODE_COUNT" -eq 0 ]; then
    echo "Error: annotation produced 0 usable source demos in $ANNOTATED" >&2
    exit 1
fi
echo "[mimic] annotated -> $ANNOTATED ($ANNOTATED_EPISODE_COUNT demos)"

echo ""
echo "============================================"
echo "  Step 3/3 - Generate MimicGen demos"
echo "============================================"

"${bash_env[@]}" bash "$ISAACLAB_SH" -p "$GENERATE_WRAPPER" \
    --spec "$SPEC" \
    --input_file "$ANNOTATED" \
    --output_file "$GENERATED" \
    --generation_num_trials "$NUM_MIMIC_DEMOS" \
    --num_envs "${NUM_ENVS:-1}" \
    "${APP_LAUNCH_ARGS[@]}"

echo ""
echo "[mimic] done"
echo "  merged:    $MERGED"
echo "  annotated: $ANNOTATED"
echo "  generated: $GENERATED"
