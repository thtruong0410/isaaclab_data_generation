"""
Registry-backed demo collection entrypoint.

Examples:
    # Normal demos
    bash isaaclab.sh -p /path/to/collect_task_demos.py \
        --spec pick_cup_normal \
        --dataset_file /tmp/pick_demo_1.hdf5 \
        --num_demos 1 \
        --teleop_device keyboard

    # Sphere-gated demos
    bash isaaclab.sh -p /path/to/collect_task_demos.py \
        --spec pick_cup_sphere \
        --dataset_file /tmp/sphere_demo_1.hdf5 \
        --num_demos 1 \
        --teleop_device keyboard
"""

from __future__ import annotations

import os
import runpy
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaac_lab_tasks.task_suite.runtime import (
    ensure_project_root_on_path,
    import_registration_modules,
    inject_default_arg,
    resolve_task_spec_from_argv,
)
from isaac_lab_tasks.task_suite.isaac_env_compat import ensure_isaaclab_runtime_compatible


def main(default_spec_key: str | None = None) -> None:
    ensure_project_root_on_path()
    ensure_isaaclab_runtime_compatible()
    spec, forwarded_argv = resolve_task_spec_from_argv(sys.argv[1:], default_spec_key=default_spec_key)

    if spec.collection_mode == "sphere":
        from isaac_lab_tasks.task_suite.collect_sphere_app import run_cli

        run_cli(spec, forwarded_argv)
        return

    from isaac_lab_tasks.task_suite.collect_demo_app import run_cli

    run_cli(spec, forwarded_argv)


if __name__ == "__main__":
    main()
