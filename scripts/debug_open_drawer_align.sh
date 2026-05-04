#!/usr/bin/env bash
set -euo pipefail

# Quick replay debug for open-drawer MimicGen alignment.
# It uses a small source budget and one generation trial so the useful logs come
# from annotation without spending time on a full 10..50 MimicGen sweep.
#
# Usage:
#   ISAAC_DEVICE=cuda:1 bash scripts/debug_open_drawer_align.sh [SPEC]
#
# Useful overrides:
#   OPEN_DRAWER_MIMIC_ALIGN_THRESHOLD=0.0
#   OPEN_DRAWER_MIMIC_DEBUG_INTERVAL=5
#   BUDGETS=10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

SPEC="${1:-open_drawer_sphere}"

if [ "${KEEP_CUDA_VISIBLE_DEVICES:-0}" != "1" ]; then
    unset CUDA_VISIBLE_DEVICES
fi
if [ "${KEEP_DISPLAY:-0}" != "1" ]; then
    unset DISPLAY
fi

export HEADLESS="${HEADLESS:-1}"
export NUM_ENVS="${NUM_ENVS:-1}"
export OVERWRITE="${OVERWRITE:-1}"
export BUDGETS="${BUDGETS:-10}"
export OPEN_DRAWER_TARGET_DRAWER="${OPEN_DRAWER_TARGET_DRAWER:-both}"
export OPEN_DRAWER_MIMIC_GRASP_MODE="${OPEN_DRAWER_MIMIC_GRASP_MODE:-joint}"
export OPEN_DRAWER_MIMIC_ALIGN_THRESHOLD="${OPEN_DRAWER_MIMIC_ALIGN_THRESHOLD:-0.0}"
export OPEN_DRAWER_MIMIC_DEBUG="${OPEN_DRAWER_MIMIC_DEBUG:-1}"
export OPEN_DRAWER_MIMIC_DEBUG_FIRST_N="${OPEN_DRAWER_MIMIC_DEBUG_FIRST_N:-10}"
export OPEN_DRAWER_MIMIC_DEBUG_INTERVAL="${OPEN_DRAWER_MIMIC_DEBUG_INTERVAL:-5}"

exec bash "$SCRIPT_DIR/mimic_budget_comparison.sh" "$SPEC" "" "" 1
