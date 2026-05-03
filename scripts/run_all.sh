#!/usr/bin/env bash
set -euo pipefail

# Run collect, mimic, or both across a task list.
# Usage:
#   bash scripts/run_all.sh collect [spec...]
#   bash scripts/run_all.sh mimic [spec...]
#   bash scripts/run_all.sh all [spec...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "Usage: bash scripts/run_all.sh {collect|mimic|all} [SPEC...]" >&2
    exit 1
fi
shift || true

if [ "$#" -gt 0 ]; then
    TASK_LIST=("$@")
else
    # shellcheck disable=SC2206
    TASK_LIST=(${TASKS:-})
fi

if [ ${#TASK_LIST[@]} -eq 0 ]; then
    echo "Error: no tasks provided and TASKS is empty." >&2
    exit 1
fi

for spec in "${TASK_LIST[@]}"; do
    case "$MODE" in
        collect)
            "$SCRIPT_DIR/collect_raw.sh" "$spec"
            ;;
        mimic)
            "$SCRIPT_DIR/mimic_generate.sh" "$spec"
            ;;
        all)
            "$SCRIPT_DIR/collect_raw.sh" "$spec"
            "$SCRIPT_DIR/mimic_generate.sh" "$spec"
            ;;
        *)
            echo "Error: unknown mode '$MODE'." >&2
            exit 1
            ;;
    esac
done
