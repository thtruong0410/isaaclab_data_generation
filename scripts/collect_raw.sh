#!/usr/bin/env bash
set -euo pipefail

# Collect raw teleop demos for one registered task spec.
# Usage:
#   bash scripts/collect_raw.sh open_door_normal [num_demos] [output_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

SPEC="${1:-}"
if [ -z "$SPEC" ]; then
    echo "Usage: bash scripts/collect_raw.sh SPEC [NUM_DEMOS] [OUTPUT_DIR]" >&2
    exit 1
fi

NUM_DEMOS="${2:-${DEFAULT_RAW_DEMOS:-10}}"
OUTPUT_DIR="${3:-$DATA_ROOT/source/$SPEC}"
BUCKET_PRESET="${BUCKET_PRESET:-}"
OVERWRITE="${OVERWRITE:-0}"
DRAW_EE_TARGET_LINE="${DRAW_EE_TARGET_LINE:-0}"
EE_TARGET_VIS_MODE="${EE_TARGET_VIS_MODE:-line}"
EE_TARGET_LINE_THICKNESS="${EE_TARGET_LINE_THICKNESS:-}"
START_RECORD_ON_SUBTASK="${START_RECORD_ON_SUBTASK:-}"
START_RECORD_ON_SUBTASK_STEPS="${START_RECORD_ON_SUBTASK_STEPS:-}"

if [[ "$SPEC" == place_cup* ]]; then
    DRAW_EE_TARGET_LINE=0
fi

ensure_data_env
print_env_summary
check_isaac_runtime

COLLECT_SCRIPT="$DATA_GEN_ROOT/isaac_lab_tasks/collect_task_demos.py"
require_file "$COLLECT_SCRIPT"

COLLECTION_MODE="$(task_field "$SPEC" collection_mode)"
DEMO_PREFIX="$(task_field "$SPEC" dataset_naming.raw_demo_file_prefix)"
if [ "$COLLECTION_MODE" = "sphere" ] && [ -z "$BUCKET_PRESET" ]; then
    BUCKET_PRESET="$(task_field "$SPEC" sphere.default_bucket_preset)"
fi

mkdir -p "$OUTPUT_DIR"

DEBUG_VIS_ARGS=()
case "$DRAW_EE_TARGET_LINE" in
    1|true|TRUE|yes|YES|on|ON)
        DEBUG_VIS_ARGS+=(--draw_ee_target_line)
        DEBUG_VIS_ARGS+=(--ee_target_vis_mode "$EE_TARGET_VIS_MODE")
        ;;
esac
if [ -n "$EE_TARGET_LINE_THICKNESS" ]; then
    DEBUG_VIS_ARGS+=(--ee_target_line_thickness "$EE_TARGET_LINE_THICKNESS")
fi

RECORD_GATE_ARGS=()
if [ -n "$START_RECORD_ON_SUBTASK" ]; then
    RECORD_GATE_ARGS+=(--start_record_on_subtask "$START_RECORD_ON_SUBTASK")
fi
if [ -n "$START_RECORD_ON_SUBTASK_STEPS" ]; then
    RECORD_GATE_ARGS+=(--start_record_on_subtask_steps "$START_RECORD_ON_SUBTASK_STEPS")
fi

echo "[collect] spec=$SPEC mode=$COLLECTION_MODE demos=$NUM_DEMOS output=$OUTPUT_DIR"
if [ ${#DEBUG_VIS_ARGS[@]} -gt 0 ]; then
    echo "[collect] debug visualization: ${DEBUG_VIS_ARGS[*]}"
fi
if [ ${#RECORD_GATE_ARGS[@]} -gt 0 ]; then
    echo "[collect] record gate: ${RECORD_GATE_ARGS[*]}"
fi

for i in $(seq 1 "$NUM_DEMOS"); do
    FILE="$OUTPUT_DIR/${DEMO_PREFIX}_${i}.hdf5"
    if [ -f "$FILE" ] && [ "$OVERWRITE" != "1" ]; then
        echo "[collect] skip existing: $FILE"
        continue
    fi

    echo ""
    echo "============================================"
    echo "  [$SPEC] Demo $i / $NUM_DEMOS -> $FILE"
    echo "============================================"

    if [ "$COLLECTION_MODE" = "sphere" ]; then
        PYTHONPATH="$DATA_GEN_ROOT${PYTHONPATH:+:$PYTHONPATH}" TERM="${TERM:-xterm}" bash "$ISAACLAB_SH" -p "$COLLECT_SCRIPT" \
            --spec "$SPEC" \
            --teleop_device "${TELEOP_DEVICE:-keyboard}" \
            --dataset_file "$FILE" \
            --num_demos 1 \
            --bucket_preset "${BUCKET_PRESET:-coverage20}" \
            --bucket_index $(( (i - 1) % 20 )) \
            --demo_seed $((1000 + i)) \
            --front_cam_pos_jitter 0.015 \
            --side_cam_pos_jitter 0.020 \
            --front_cam_rot_jitter_deg 2.0 \
            --side_cam_rot_jitter_deg 3.0 \
            --light_intensity_range 2200 4200 \
            --light_color_jitter 0.08 \
            "${DEBUG_VIS_ARGS[@]}" \
            --enable_cameras
    else
        PYTHONPATH="$DATA_GEN_ROOT${PYTHONPATH:+:$PYTHONPATH}" TERM="${TERM:-xterm}" bash "$ISAACLAB_SH" -p "$COLLECT_SCRIPT" \
            --spec "$SPEC" \
            --teleop_device "${TELEOP_DEVICE:-keyboard}" \
            --dataset_file "$FILE" \
            --num_demos 1 \
            "${DEBUG_VIS_ARGS[@]}" \
            "${RECORD_GATE_ARGS[@]}" \
            --enable_cameras
    fi

    echo "[collect] saved: $FILE"
    sleep 3
done

echo "[collect] done: $OUTPUT_DIR"
