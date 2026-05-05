#!/usr/bin/env bash
set -euo pipefail

# Append raw demo files from one folder into another folder with new numbering.
# Original source files are never modified.
#
# Usage:
#   bash scripts/append_raw_demos.sh SPEC SOURCE_DIR DEST_DIR [START_INDEX] [MAX_DEMOS]
#
# Example:
#   bash scripts/append_raw_demos.sh open_drawer_sphere \
#       data/source/open_drawer_sphere_top_2 \
#       data/source/open_drawer_sphere_top \
#       26 25
#
# Optional env vars:
#   COPY_MODE=copy        copy | symlink | hardlink
#   OVERWRITE=1           replace destination files with the same new numbers
#   DRY_RUN=1             print the plan without writing files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

SPEC="${1:-}"
SOURCE_DIR="${2:-}"
DEST_DIR="${3:-}"
START_INDEX="${4:-}"
MAX_DEMOS="${5:-${MAX_DEMOS:-}}"

if [ -z "$SPEC" ] || [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ]; then
    echo "Usage: bash scripts/append_raw_demos.sh SPEC SOURCE_DIR DEST_DIR [START_INDEX] [MAX_DEMOS]" >&2
    exit 1
fi

COPY_MODE="${COPY_MODE:-copy}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

require_file "$TASK_INFO_SCRIPT"
require_dir "$SOURCE_DIR"

DEMO_PREFIX="$(task_field "$SPEC" dataset_naming.raw_demo_file_prefix)"

case "$COPY_MODE" in
    copy|symlink|hardlink) ;;
    *)
        echo "Error: COPY_MODE must be one of: copy, symlink, hardlink" >&2
        exit 1
        ;;
esac

if [ -n "$START_INDEX" ] && ! [[ "$START_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Error: START_INDEX must be a positive integer." >&2
    exit 1
fi

if [ -n "$MAX_DEMOS" ] && ! [[ "$MAX_DEMOS" =~ ^[0-9]+$ ]]; then
    echo "Error: MAX_DEMOS must be a positive integer." >&2
    exit 1
fi

mapfile -t SOURCE_FILES < <(
    find -L "$SOURCE_DIR" -maxdepth 1 -type f -name "${DEMO_PREFIX}_*.hdf5" ! -name '*smoke*.hdf5' | sort -V
)

if [ "${#SOURCE_FILES[@]}" -eq 0 ]; then
    echo "Error: no ${DEMO_PREFIX}_*.hdf5 files found in $SOURCE_DIR" >&2
    exit 1
fi

next_index() {
    local max_index=0
    local path base index

    if [ ! -d "$DEST_DIR" ]; then
        printf '1\n'
        return
    fi

    shopt -s nullglob
    for path in "$DEST_DIR"/"${DEMO_PREFIX}"_*.hdf5; do
        base="$(basename "$path" .hdf5)"
        index="${base##${DEMO_PREFIX}_}"
        if [[ "$index" =~ ^[0-9]+$ ]] && [ "$index" -gt "$max_index" ]; then
            max_index="$index"
        fi
    done
    shopt -u nullglob

    printf '%s\n' "$((max_index + 1))"
}

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

START_INDEX="${START_INDEX:-$(next_index)}"
MANIFEST="$DEST_DIR/append_manifest.tsv"

echo "[append] spec=$SPEC"
echo "[append] source=$SOURCE_DIR (${#SOURCE_FILES[@]} files)"
echo "[append] dest=$DEST_DIR"
echo "[append] start_index=$START_INDEX"
echo "[append] copy_mode=$COPY_MODE"
[ -n "$MAX_DEMOS" ] && echo "[append] max_demos=$MAX_DEMOS"

if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$DEST_DIR"
    if [ ! -f "$MANIFEST" ]; then
        printf 'output_file\tsource_path\n' > "$MANIFEST"
    fi
fi

out_idx="$START_INDEX"
created=0

for src in "${SOURCE_FILES[@]}"; do
    if [ -n "$MAX_DEMOS" ] && [ "$created" -ge "$MAX_DEMOS" ]; then
        break
    fi

    out_file="$DEST_DIR/${DEMO_PREFIX}_${out_idx}.hdf5"
    printf '[append] %s -> %s\n' "$src" "$out_file"

    if [ "$DRY_RUN" != "1" ]; then
        if [ -e "$out_file" ] && [ "$OVERWRITE" != "1" ]; then
            echo "Error: destination exists: $out_file" >&2
            echo "Set OVERWRITE=1 or choose another START_INDEX." >&2
            exit 1
        fi
        copy_one "$src" "$out_file"
        printf '%s\t%s\n' "$(basename "$out_file")" "$src" >> "$MANIFEST"
    fi

    out_idx=$((out_idx + 1))
    created=$((created + 1))
done

echo "[append] done: $created demos"
if [ "$DRY_RUN" != "1" ]; then
    echo "[append] manifest: $MANIFEST"
fi
