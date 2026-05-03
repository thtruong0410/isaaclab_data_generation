from __future__ import annotations

import argparse
import importlib
import os
import sys

from .registry import get_task_spec
from .types import TaskSuiteSpec


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_project_root_on_path() -> None:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def import_registration_modules(spec: TaskSuiteSpec) -> None:
    ensure_project_root_on_path()
    for module_name in spec.registration_modules:
        importlib.import_module(module_name)


def resolve_task_spec_from_argv(
    argv: list[str] | None = None,
    *,
    default_spec_key: str | None = None,
) -> tuple[TaskSuiteSpec, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    if default_spec_key is None:
        parser.add_argument("--spec", required=True)
    else:
        parser.add_argument("--spec", default=default_spec_key)
    known_args, remaining = parser.parse_known_args(argv)
    spec = get_task_spec(known_args.spec)
    return spec, remaining


def inject_default_arg(argv: list[str], flag: str, value: str) -> list[str]:
    if flag in argv:
        return list(argv)
    return [flag, value, *argv]


def import_object_by_path(path: str):
    module_name, object_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)
