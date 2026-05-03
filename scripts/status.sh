#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

ensure_data_env

if [ "$#" -gt 0 ]; then
    TASK_LIST=("$@")
else
    # shellcheck disable=SC2206
    TASK_LIST=(${TASKS:-})
fi

printf '%-24s %-8s %-12s %-12s %-12s\n' "SPEC" "RAW" "MERGED" "ANNOTATED" "GENERATED"
printf '%-24s %-8s %-12s %-12s %-12s\n' "----" "---" "------" "---------" "---------"

for spec in "${TASK_LIST[@]}"; do
    prefix="$(task_field "$spec" dataset_naming.raw_demo_file_prefix 2>/dev/null || true)"
    merged_name="$(task_field "$spec" dataset_naming.merged_hdf5_name 2>/dev/null || true)"
    annotated_name="$(task_field "$spec" dataset_naming.annotated_hdf5_name 2>/dev/null || true)"
    generated_name="$(task_field "$spec" dataset_naming.generated_hdf5_name 2>/dev/null || true)"

    if [ -z "$prefix" ]; then
        printf '%-24s %-8s %-12s %-12s %-12s\n' "$spec" "unknown" "-" "-" "-"
        continue
    fi

    raw_dir="$DATA_ROOT/source/$spec"
    mimic_dir="$DATA_ROOT/mimic/$spec"
    raw_count=0
    if [ -d "$raw_dir" ]; then
        raw_count=$(find "$raw_dir" -maxdepth 1 -type f -name "${prefix}_*.hdf5" ! -name '*smoke*.hdf5' | wc -l)
    fi

    mark_file() {
        local path=$1
        if [ -f "$path" ]; then
            printf 'yes'
        else
            printf 'no'
        fi
    }

    printf '%-24s %-8s %-12s %-12s %-12s\n' \
        "$spec" \
        "$raw_count" \
        "$(mark_file "$mimic_dir/$merged_name")" \
        "$(mark_file "$mimic_dir/$annotated_name")" \
        "$(mark_file "$mimic_dir/$generated_name")"
done
