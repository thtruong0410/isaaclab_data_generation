#!/usr/bin/env bash
set -euo pipefail

# Run MimicGen in non-overlapping source chunks, then merge the generated demos
# cumulatively. With the defaults:
#
#   10 -> chunk 01-10                         -> 1000 generated demos
#   20 -> chunk 01-10 + chunk 11-20           -> 2000 generated demos
#   30 -> chunks 01-10 + 11-20 + 21-30        -> 3000 generated demos
#   50 -> five 10-demo chunks                 -> 5000 generated demos
#
# Usage:
#   bash scripts/mimic_budget_cumulative.sh SPEC [RAW_DIR] [MIMIC_ROOT] [NUM_MIMIC_DEMOS_PER_CHUNK] [SETUP_NAME]
#
# Example:
#   HEADLESS=1 NUM_ENVS=1 bash scripts/mimic_budget_cumulative.sh place_cup_normal
#
# Optional env vars:
#   BUDGETS="10 20 30 40 50"
#   CHUNK_SIZE=10
#   SUBSET_ROOT=data/source_subsets
#   SKIP_EXISTING=1
#   OVERWRITE=1              forwarded to mimic_generate.sh and cumulative output
#   ISAAC_DEVICE=cuda:1      forwarded to IsaacLab AppLauncher
#   DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

SPEC="${1:-}"
RAW_DIR_ARG="${2:-}"
MIMIC_ROOT_ARG="${3:-}"
NUM_MIMIC_DEMOS="${4:-${DEFAULT_MIMIC_DEMOS:-1000}}"
SETUP_NAME="${5:-}"

if [ -z "$SPEC" ]; then
    echo "Usage: bash scripts/mimic_budget_cumulative.sh SPEC [RAW_DIR] [MIMIC_ROOT] [NUM_MIMIC_DEMOS_PER_CHUNK] [SETUP_NAME]" >&2
    exit 1
fi

if [ -n "$RAW_DIR_ARG" ]; then
    RAW_DIR="$RAW_DIR_ARG"
elif [ -d "$DATA_ROOT/source/$SPEC" ]; then
    RAW_DIR="$DATA_ROOT/source/$SPEC"
else
    RAW_DIR="$DATA_ROOT/source/${SPEC}_top_bottom"
fi

MIMIC_ROOT="${MIMIC_ROOT_ARG:-$DATA_ROOT/mimic}"
SETUP_NAME="${SETUP_NAME:-$(basename "$RAW_DIR")}"
BUDGETS="${BUDGETS:-10 20 30 40 50}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
SUBSET_ROOT="${SUBSET_ROOT:-$DATA_ROOT/source_subsets}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$SETUP_NAME" == */* ]] || [ -z "$SETUP_NAME" ]; then
    echo "Error: SETUP_NAME must be a single folder name, not a path." >&2
    exit 1
fi
if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || [ "$CHUNK_SIZE" -le 0 ]; then
    echo "Error: CHUNK_SIZE must be a positive integer." >&2
    exit 1
fi
if ! [[ "$NUM_MIMIC_DEMOS" =~ ^[0-9]+$ ]] || [ "$NUM_MIMIC_DEMOS" -le 0 ]; then
    echo "Error: NUM_MIMIC_DEMOS_PER_CHUNK must be a positive integer." >&2
    exit 1
fi

ensure_data_env
require_dir "$RAW_DIR"

MERGE_SCRIPT="$ISAACLAB_ROOT/scripts/tools/merge_hdf5_datasets.py"
require_file "$MERGE_SCRIPT"

DEMO_PREFIX="$(task_field "$SPEC" dataset_naming.raw_demo_file_prefix)"
GENERATED_NAME="$(task_field "$SPEC" dataset_naming.generated_hdf5_name)"

mapfile -t RAW_HDF5_FILES < <(
    find -L "$RAW_DIR" -maxdepth 1 -type f -name "${DEMO_PREFIX}_*.hdf5" ! -name '*smoke*.hdf5' | sort -V
)

if [ "${#RAW_HDF5_FILES[@]}" -eq 0 ]; then
    echo "Error: no ${DEMO_PREFIX}_*.hdf5 files found in $RAW_DIR" >&2
    exit 1
fi

max_budget=0
for budget in $BUDGETS; do
    if ! [[ "$budget" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid budget '$budget'. BUDGETS must contain integers." >&2
        exit 1
    fi
    if [ "$budget" -le 0 ]; then
        echo "Error: budget must be positive: $budget" >&2
        exit 1
    fi
    if [ $((budget % CHUNK_SIZE)) -ne 0 ]; then
        echo "Error: budget $budget must be divisible by CHUNK_SIZE=$CHUNK_SIZE." >&2
        exit 1
    fi
    if [ "${#RAW_HDF5_FILES[@]}" -lt "$budget" ]; then
        echo "Error: budget $budget requested but only ${#RAW_HDF5_FILES[@]} source demos exist in $RAW_DIR" >&2
        exit 1
    fi
    if [ "$budget" -gt "$max_budget" ]; then
        max_budget="$budget"
    fi
done

chunk_count=$((max_budget / CHUNK_SIZE))
chunk_root="$MIMIC_ROOT/${SETUP_NAME}_chunks"

echo "[cumulative] spec=$SPEC"
echo "[cumulative] raw_dir=$RAW_DIR"
echo "[cumulative] setup_name=$SETUP_NAME"
echo "[cumulative] mimic_root=$MIMIC_ROOT"
echo "[cumulative] chunk_root=$chunk_root"
echo "[cumulative] source demos=${#RAW_HDF5_FILES[@]}"
echo "[cumulative] budgets=$BUDGETS"
echo "[cumulative] chunk_size=$CHUNK_SIZE"
echo "[cumulative] mimic demos per chunk=$NUM_MIMIC_DEMOS"
echo "[cumulative] generated hdf5=$GENERATED_NAME"

chunk_generated_files=()

for chunk_idx in $(seq 1 "$chunk_count"); do
    start=$(( (chunk_idx - 1) * CHUNK_SIZE + 1 ))
    end=$(( chunk_idx * CHUNK_SIZE ))
    chunk_label="$(printf '%03d_%03d' "$start" "$end")"
    subset_dir="$SUBSET_ROOT/${SETUP_NAME}_chunk_${chunk_label}"
    chunk_output_dir="$chunk_root/${SETUP_NAME}_chunk_${chunk_label}"
    chunk_generated="$chunk_output_dir/$GENERATED_NAME"
    chunk_generated_files+=("$chunk_generated")

    echo ""
    echo "============================================"
    echo "  Chunk $chunk_idx/$chunk_count: source $start-$end -> $chunk_generated"
    echo "============================================"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$chunk_generated" ]; then
        echo "[cumulative] skip existing chunk: $chunk_generated"
        continue
    fi

    echo "[cumulative] preparing subset: $subset_dir"
    if [ "$DRY_RUN" != "1" ]; then
        rm -rf "$subset_dir"
        mkdir -p "$subset_dir"
        manifest="$subset_dir/manifest.tsv"
        printf 'subset_file\tsource_path\n' > "$manifest"
        for local_i in $(seq 1 "$CHUNK_SIZE"); do
            raw_index=$((start + local_i - 2))
            src="${RAW_HDF5_FILES[$raw_index]}"
            dst="$subset_dir/${DEMO_PREFIX}_${local_i}.hdf5"
            ln -s "$(realpath "$src")" "$dst"
            printf '%s\t%s\n' "$(basename "$dst")" "$src" >> "$manifest"
        done
    else
        for local_i in $(seq 1 "$CHUNK_SIZE"); do
            raw_index=$((start + local_i - 2))
            src="${RAW_HDF5_FILES[$raw_index]}"
            echo "[dry-run] link $src -> $subset_dir/${DEMO_PREFIX}_${local_i}.hdf5"
        done
    fi

    echo "[cumulative] running MimicGen chunk: raw=$subset_dir mimic=$chunk_output_dir"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] bash $SCRIPT_DIR/mimic_generate.sh $SPEC $NUM_MIMIC_DEMOS $subset_dir $chunk_output_dir"
        continue
    fi

    bash "$SCRIPT_DIR/mimic_generate.sh" "$SPEC" "$NUM_MIMIC_DEMOS" "$subset_dir" "$chunk_output_dir"
done

for budget in $BUDGETS; do
    needed_chunks=$((budget / CHUNK_SIZE))
    output_dir="$MIMIC_ROOT/${SETUP_NAME}_${budget}"
    output_file="$output_dir/$GENERATED_NAME"

    echo ""
    echo "============================================"
    echo "  Cumulative budget $budget -> $output_file"
    echo "============================================"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$output_file" ]; then
        echo "[cumulative] skip existing cumulative file: $output_file"
        continue
    fi
    if [ -f "$output_file" ] && [ "${OVERWRITE:-0}" != "1" ]; then
        echo "Error: cumulative output already exists: $output_file" >&2
        echo "Set OVERWRITE=1 to regenerate it, or SKIP_EXISTING=1 to keep it." >&2
        exit 1
    fi

    input_files=()
    for chunk_idx in $(seq 1 "$needed_chunks"); do
        chunk_file="${chunk_generated_files[$((chunk_idx - 1))]}"
        input_files+=("$chunk_file")
        if [ "$DRY_RUN" != "1" ]; then
            require_file "$chunk_file"
        fi
    done

    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] mkdir -p $output_dir"
        echo "[dry-run] $PYTHON_BIN $MERGE_SCRIPT --input_files ${input_files[*]} --output_file $output_file"
        continue
    fi

    mkdir -p "$output_dir"
    rm -f "$output_file"
    "$PYTHON_BIN" "$MERGE_SCRIPT" --input_files "${input_files[@]}" --output_file "$output_file"

    manifest="$output_dir/cumulative_manifest.tsv"
    printf 'budget\tchunk_file\n' > "$manifest"
    for chunk_file in "${input_files[@]}"; do
        printf '%s\t%s\n' "$budget" "$chunk_file" >> "$manifest"
    done

    demo_count=$("$PYTHON_BIN" - "$output_file" <<'PY'
import sys
import h5py

with h5py.File(sys.argv[1], "r") as handle:
    data = handle.get("data")
    print(0 if data is None else len(data.keys()))
PY
)
    echo "[cumulative] merged $needed_chunks chunks -> $demo_count demos"
done

echo ""
echo "[cumulative] done"
