#!/usr/bin/env bash
set -euo pipefail

# Run MimicGen budget variants from a single raw source folder.
# For each budget N, the script links the first N raw demos into
# data/source_subsets/<SETUP_NAME>_<N>/ and calls scripts/mimic_generate.sh.
#
# Usage:
#   bash scripts/mimic_budget_comparison.sh SPEC RAW_DIR [MIMIC_ROOT] [NUM_MIMIC_DEMOS] [SETUP_NAME]
#
# Example:
#   OPEN_DRAWER_TARGET_DRAWER=both HEADLESS=1 bash scripts/mimic_budget_comparison.sh \
#       open_drawer_sphere \
#       data/source/open_drawer_sphere_top_bottom \
#       data/mimic \
#       1000 \
#       open_drawer_sphere_top_bottom
#
# Optional env vars:
#   BUDGETS="10 20 30 40 50"
#   SUBSET_ROOT=data/source_subsets
#   SKIP_EXISTING=1
#   OVERWRITE=1              forwarded to mimic_generate.sh
#   DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

SPEC="${1:-}"
RAW_DIR="${2:-}"
MIMIC_ROOT_ARG="${3:-}"
NUM_MIMIC_DEMOS="${4:-${DEFAULT_MIMIC_DEMOS:-1000}}"
SETUP_NAME="${5:-}"

if [ -z "$SPEC" ] || [ -z "$RAW_DIR" ]; then
    echo "Usage: bash scripts/mimic_budget_comparison.sh SPEC RAW_DIR [MIMIC_ROOT] [NUM_MIMIC_DEMOS] [SETUP_NAME]" >&2
    exit 1
fi

MIMIC_ROOT="${MIMIC_ROOT_ARG:-$DATA_ROOT/mimic}"
SETUP_NAME="${SETUP_NAME:-$(basename "$RAW_DIR")}"
BUDGETS="${BUDGETS:-10 20 30 40 50}"
SUBSET_ROOT="${SUBSET_ROOT:-$DATA_ROOT/source_subsets}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$SETUP_NAME" == */* ]] || [ -z "$SETUP_NAME" ]; then
    echo "Error: SETUP_NAME must be a single folder name, not a path." >&2
    exit 1
fi

ensure_data_env
require_dir "$RAW_DIR"

DEMO_PREFIX="$(task_field "$SPEC" dataset_naming.raw_demo_file_prefix)"
GENERATED_NAME="$(task_field "$SPEC" dataset_naming.generated_hdf5_name)"

mapfile -t RAW_HDF5_FILES < <(
    find -L "$RAW_DIR" -maxdepth 1 -type f -name "${DEMO_PREFIX}_*.hdf5" ! -name '*smoke*.hdf5' | sort -V
)

if [ "${#RAW_HDF5_FILES[@]}" -eq 0 ]; then
    echo "Error: no ${DEMO_PREFIX}_*.hdf5 files found in $RAW_DIR" >&2
    exit 1
fi

echo "[budget] spec=$SPEC"
echo "[budget] raw_dir=$RAW_DIR"
echo "[budget] setup_name=$SETUP_NAME"
echo "[budget] mimic_root=$MIMIC_ROOT"
echo "[budget] source demos=${#RAW_HDF5_FILES[@]}"
echo "[budget] budgets=$BUDGETS"
echo "[budget] mimic demos per run=$NUM_MIMIC_DEMOS"

for budget in $BUDGETS; do
    if ! [[ "$budget" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid budget '$budget'. BUDGETS must contain integers." >&2
        exit 1
    fi
    if [ "$budget" -le 0 ]; then
        echo "Error: budget must be positive: $budget" >&2
        exit 1
    fi
    if [ "${#RAW_HDF5_FILES[@]}" -lt "$budget" ]; then
        echo "Error: budget $budget requested but only ${#RAW_HDF5_FILES[@]} source demos exist in $RAW_DIR" >&2
        exit 1
    fi

    subset_dir="$SUBSET_ROOT/${SETUP_NAME}_${budget}"
    output_dir="$MIMIC_ROOT/${SETUP_NAME}_${budget}"
    generated_file="$output_dir/$GENERATED_NAME"

    echo ""
    echo "============================================"
    echo "  [$SETUP_NAME] budget=$budget -> $output_dir"
    echo "============================================"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$generated_file" ]; then
        echo "[budget] skip existing generated file: $generated_file"
        continue
    fi

    echo "[budget] preparing subset: $subset_dir"
    if [ "$DRY_RUN" != "1" ]; then
        rm -rf "$subset_dir"
        mkdir -p "$subset_dir"
        manifest="$subset_dir/manifest.tsv"
        printf 'subset_file\tsource_path\n' > "$manifest"
        for i in $(seq 1 "$budget"); do
            src="${RAW_HDF5_FILES[$((i - 1))]}"
            dst="$subset_dir/${DEMO_PREFIX}_${i}.hdf5"
            ln -s "$(realpath "$src")" "$dst"
            printf '%s\t%s\n' "$(basename "$dst")" "$src" >> "$manifest"
        done
    else
        for i in $(seq 1 "$budget"); do
            src="${RAW_HDF5_FILES[$((i - 1))]}"
            echo "[dry-run] link $src -> $subset_dir/${DEMO_PREFIX}_${i}.hdf5"
        done
    fi

    echo "[budget] running MimicGen: raw=$subset_dir mimic=$output_dir"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] bash $SCRIPT_DIR/mimic_generate.sh $SPEC $NUM_MIMIC_DEMOS $subset_dir $output_dir"
        continue
    fi

    bash "$SCRIPT_DIR/mimic_generate.sh" "$SPEC" "$NUM_MIMIC_DEMOS" "$subset_dir" "$output_dir"
done

echo ""
echo "[budget] done"
