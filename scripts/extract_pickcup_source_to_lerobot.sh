#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/may3/VNX/Truong/quad-rl-wmp/logs/rsl_rl/WMP/2026-04-03_09-52-27/params/isaaclab_data_generation"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/may3/IsaacLab}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASK_TEXT="${TASK_TEXT:-pick up the cup}"
FPS="${FPS:-30}"
CRF="${CRF:-23}"
OVERWRITE="${OVERWRITE:-0}"

cd "$ROOT"

MERGE_SCRIPT="$ISAACLAB_ROOT/scripts/tools/merge_hdf5_datasets.py"
CONVERT_SCRIPT="scripts/convert_isaaclab_hdf5_to_lerobot.py"

TMP_ROOT="data/tmp_pickcup_extract"
LEROBOT_ROOT="data/lerobot"

mkdir -p "$TMP_ROOT" "$LEROBOT_ROOT"

convert_one() {
  local name="$1"
  local src_dir="$2"
  local out_dir="$3"
  local merged="$TMP_ROOT/${name}_merged.hdf5"

  echo ""
  echo "============================================"
  echo "Convert $name"
  echo "source : $src_dir"
  echo "merged : $merged"
  echo "output : $out_dir"
  echo "============================================"

  if [ ! -d "$src_dir" ]; then
    echo "Missing source dir: $src_dir" >&2
    exit 1
  fi

  mapfile -t files < <(
    find -L "$src_dir" -maxdepth 1 -type f -name "*.hdf5" \
      ! -name "*merged*.hdf5" \
      ! -name "*annotated*.hdf5" \
      ! -name "*mimic*.hdf5" \
      | sort -V
  )

  if [ "${#files[@]}" -eq 0 ]; then
    echo "No .hdf5 files found in $src_dir" >&2
    exit 1
  fi

  echo "Found ${#files[@]} hdf5 files"

  rm -f "$merged"
  "$PYTHON_BIN" "$MERGE_SCRIPT" \
    --input_files "${files[@]}" \
    --output_file "$merged"

  if [ -d "$out_dir" ]; then
    if [ "$OVERWRITE" = "1" ]; then
      rm -rf "$out_dir"
    else
      echo "Output exists: $out_dir"
      echo "Set OVERWRITE=1 to replace it."
      exit 1
    fi
  fi

  "$PYTHON_BIN" "$CONVERT_SCRIPT" \
    --input "$merged" \
    --output "$out_dir" \
    --task "$TASK_TEXT" \
    --fps "$FPS" \
    --crf "$CRF"

  echo "Done -> $out_dir"
}

convert_one \
  "pick_cup_normal" \
  "data/source/pickcup_demos/pick_cup_normal" \
  "$LEROBOT_ROOT/pick_cup_normal"

convert_one \
  "pick_cup_sphere" \
  "data/source/pickcup_demos/pick_cup_sphere" \
  "$LEROBOT_ROOT/pick_cup_sphere"

echo ""
echo "All done."
echo "Outputs:"
echo "  $LEROBOT_ROOT/pick_cup_normal"
echo "  $LEROBOT_ROOT/pick_cup_sphere"