#!/usr/bin/env bash
set -euo pipefail

# Copy two raw demo folders into one interleaved raw folder.
# Original raw files are never modified.
#
# Usage:
#   bash scripts/interleave_raw_demos.sh SPEC [FIRST_DIR SECOND_DIR OUTPUT_DIR] [MAX_DEMOS]
#
# Example:
#   bash scripts/interleave_raw_demos.sh open_drawer_sphere
#
# The default convention is:
#   first  = data/source/<SPEC>_top
#   second = data/source/<SPEC>_bottom
#   output = data/source/<SPEC>_top_bottom
#
# Optional env vars:
#   FIRST_LABEL=top       label written to manifest.tsv for FIRST_DIR
#   SECOND_LABEL=bottom   label written to manifest.tsv for SECOND_DIR
#   COPY_MODE=copy        copy | symlink | hardlink
#   OVERWRITE=1           replace existing output demo files
#   DRY_RUN=1             print the plan without writing files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

SPEC="${1:-}"
FIRST_DIR="${2:-}"
SECOND_DIR="${3:-}"
OUTPUT_DIR="${4:-}"
MAX_DEMOS="${5:-${MAX_DEMOS:-}}"

if [ -z "$SPEC" ]; then
    echo "Usage: bash scripts/interleave_raw_demos.sh SPEC [FIRST_DIR SECOND_DIR OUTPUT_DIR] [MAX_DEMOS]" >&2
    exit 1
fi

FIRST_DIR="${FIRST_DIR:-$DATA_ROOT/source/${SPEC}_top}"
SECOND_DIR="${SECOND_DIR:-$DATA_ROOT/source/${SPEC}_bottom}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/source/${SPEC}_top_bottom}"

FIRST_LABEL="${FIRST_LABEL:-top}"
SECOND_LABEL="${SECOND_LABEL:-bottom}"
COPY_MODE="${COPY_MODE:-copy}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

require_file "$TASK_INFO_SCRIPT"
require_dir "$FIRST_DIR"
require_dir "$SECOND_DIR"

DEMO_PREFIX="$(task_field "$SPEC" dataset_naming.raw_demo_file_prefix)"

case "$COPY_MODE" in
    copy|symlink|hardlink) ;;
    *)
        echo "Error: COPY_MODE must be one of: copy, symlink, hardlink" >&2
        exit 1
        ;;
esac

if [ -n "$MAX_DEMOS" ] && ! [[ "$MAX_DEMOS" =~ ^[0-9]+$ ]]; then
    echo "Error: MAX_DEMOS must be a positive integer." >&2
    exit 1
fi

mapfile -t FIRST_FILES < <(
    find -L "$FIRST_DIR" -maxdepth 1 -type f -name "${DEMO_PREFIX}_*.hdf5" ! -name '*smoke*.hdf5' | sort -V
)
mapfile -t SECOND_FILES < <(
    find -L "$SECOND_DIR" -maxdepth 1 -type f -name "${DEMO_PREFIX}_*.hdf5" ! -name '*smoke*.hdf5' | sort -V
)

if [ "${#FIRST_FILES[@]}" -eq 0 ]; then
    echo "Error: no ${DEMO_PREFIX}_*.hdf5 files found in $FIRST_DIR" >&2
    exit 1
fi
if [ "${#SECOND_FILES[@]}" -eq 0 ]; then
    echo "Error: no ${DEMO_PREFIX}_*.hdf5 files found in $SECOND_DIR" >&2
    exit 1
fi

copy_one() {
    local src="$1"
    local dst="$2"
    case "$COPY_MODE" in
        copy)
            cp -p "$src" "$dst"
            ;;
        symlink)
            ln -s "$(realpath "$src")" "$dst"
            ;;
        hardlink)
            ln "$src" "$dst"
            ;;
    esac
}

write_one() {
    local src="$1"
    local label="$2"
    local out_idx="$3"
    local out_file="$OUTPUT_DIR/${DEMO_PREFIX}_${out_idx}.hdf5"

    printf '[interleave] %s -> %s (%s)\n' "$src" "$out_file" "$label"
    if [ "$DRY_RUN" = "1" ]; then
        return
    fi

    copy_one "$src" "$out_file"
    printf '%s\t%s\t%s\n' "$(basename "$out_file")" "$label" "$src" >> "$MANIFEST"
}

echo "[interleave] spec=$SPEC"
echo "[interleave] first=$FIRST_DIR (${#FIRST_FILES[@]} files, label=$FIRST_LABEL)"
echo "[interleave] second=$SECOND_DIR (${#SECOND_FILES[@]} files, label=$SECOND_LABEL)"
echo "[interleave] output=$OUTPUT_DIR"
echo "[interleave] copy_mode=$COPY_MODE"
[ -n "$MAX_DEMOS" ] && echo "[interleave] max_demos=$MAX_DEMOS"

if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$OUTPUT_DIR"
    shopt -s nullglob
    existing=("$OUTPUT_DIR"/"${DEMO_PREFIX}"_*.hdf5)
    shopt -u nullglob
    if [ "${#existing[@]}" -gt 0 ]; then
        if [ "$OVERWRITE" != "1" ]; then
            echo "Error: output already contains ${DEMO_PREFIX}_*.hdf5 files: $OUTPUT_DIR" >&2
            echo "Set OVERWRITE=1 or choose a new output directory." >&2
            exit 1
        fi
        rm -f "$OUTPUT_DIR"/"${DEMO_PREFIX}"_*.hdf5 "$OUTPUT_DIR/manifest.tsv"
    fi
    MANIFEST="$OUTPUT_DIR/manifest.tsv"
    printf 'output_file\tsource_label\tsource_path\n' > "$MANIFEST"
else
    MANIFEST=/dev/null
fi

first_idx=0
second_idx=0
out_idx=1
take_first=1

while [ "$first_idx" -lt "${#FIRST_FILES[@]}" ] || [ "$second_idx" -lt "${#SECOND_FILES[@]}" ]; do
    if [ -n "$MAX_DEMOS" ] && [ "$out_idx" -gt "$MAX_DEMOS" ]; then
        break
    fi

    if [ "$take_first" -eq 1 ] && [ "$first_idx" -lt "${#FIRST_FILES[@]}" ]; then
        write_one "${FIRST_FILES[$first_idx]}" "$FIRST_LABEL" "$out_idx"
        first_idx=$((first_idx + 1))
    elif [ "$second_idx" -lt "${#SECOND_FILES[@]}" ]; then
        write_one "${SECOND_FILES[$second_idx]}" "$SECOND_LABEL" "$out_idx"
        second_idx=$((second_idx + 1))
    elif [ "$first_idx" -lt "${#FIRST_FILES[@]}" ]; then
        write_one "${FIRST_FILES[$first_idx]}" "$FIRST_LABEL" "$out_idx"
        first_idx=$((first_idx + 1))
    else
        break
    fi

    out_idx=$((out_idx + 1))
    if [ "$take_first" -eq 1 ]; then
        take_first=0
    else
        take_first=1
    fi
done

created=$((out_idx - 1))
echo "[interleave] done: $created demos"
if [ "$DRY_RUN" != "1" ]; then
    echo "[interleave] manifest: $MANIFEST"
fi
