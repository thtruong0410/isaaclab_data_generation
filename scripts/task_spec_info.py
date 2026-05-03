#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaac_lab_tasks.task_suite import get_task_spec, spec_to_dict


def resolve_field(payload: dict, field_path: str):
    value = payload
    for key in field_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Field '{field_path}' not found.")
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Print task-suite spec metadata.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--field", default=None, help="Optional dotted field path, e.g. dataset_naming.raw_demo_file_prefix")
    parser.add_argument("--json", action="store_true", help="Print the whole spec as JSON.")
    args = parser.parse_args()

    spec = get_task_spec(args.spec)
    payload = spec_to_dict(spec)

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    if args.field:
        value = resolve_field(payload, args.field)
    else:
        value = payload

    if isinstance(value, (dict, list)):
        print(json.dumps(value))
    else:
        print(value)


if __name__ == "__main__":
    main()
