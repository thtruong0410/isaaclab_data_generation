#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaac_lab_tasks.task_suite.isaac_env_compat import (  # noqa: E402
    ensure_isaaclab_runtime_compatible,
    format_isaaclab_runtime_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Isaac Lab runtime package compatibility.")
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warnings as failures.",
    )
    args = parser.parse_args()

    try:
        report = ensure_isaaclab_runtime_compatible(strict_warnings=args.strict_warnings)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(format_isaaclab_runtime_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
