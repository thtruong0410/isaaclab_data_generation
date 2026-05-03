#!/usr/bin/env python3
"""Data-only wrapper around IsaacLab Mimic annotate_demos.py.

This avoids hardcoded IsaacLab paths and stays local to this data workspace.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def _data_gen_root() -> Path:
    env_path = os.environ.get("DATA_GEN_ROOT")
    if env_path:
        return Path(env_path).resolve()
    return Path(__file__).resolve().parents[1]


def _isaaclab_root() -> Path:
    env_path = os.environ.get("ISAACLAB_ROOT")
    if env_path:
        return Path(env_path).resolve()
    for candidate in (Path("/opt/IsaacLab"), Path("/home/ntruong/Truong/IsaacLab")):
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError("Set ISAACLAB_ROOT to the IsaacLab repository path.")


DATA_GEN_ROOT = _data_gen_root()
if str(DATA_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_ROOT))

from isaac_lab_tasks.task_suite.isaac_env_compat import ensure_isaaclab_runtime_compatible  # noqa: E402
from isaac_lab_tasks.task_suite.runtime import (  # noqa: E402
    ensure_project_root_on_path,
    import_registration_modules,
    inject_default_arg,
    resolve_task_spec_from_argv,
)


def main() -> None:
    ensure_project_root_on_path()
    ensure_isaaclab_runtime_compatible()
    spec, forwarded_argv = resolve_task_spec_from_argv(sys.argv[1:])
    import_registration_modules(spec)
    forwarded_argv = inject_default_arg(forwarded_argv, "--task", spec.mimic_task_id)
    sys.argv = [sys.argv[0], *forwarded_argv]

    script = _isaaclab_root() / "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing IsaacLab annotate script: {script}")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
