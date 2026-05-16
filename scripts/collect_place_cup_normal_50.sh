#!/usr/bin/env bash
set -euo pipefail

# Collect 50 normal place-cup source demos in sequence.
#
# Usage:
#   bash scripts/collect_place_cup_normal_50.sh [OUTPUT_DIR]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

OUTPUT_DIR="${1:-$PROJECT_ROOT/data/source/place_cup_normal}"

bash "$SCRIPT_DIR/collect_raw.sh" \
    place_cup_normal \
    50 \
    "$OUTPUT_DIR"
