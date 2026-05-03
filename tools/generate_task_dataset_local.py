#!/usr/bin/env python3
"""Data-only wrapper around IsaacLab Mimic generate_dataset.py.

This avoids hardcoded IsaacLab paths and stays local to this data workspace.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import os
from pathlib import Path
import runpy
import subprocess
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


def _normalize_hdf5_path(path: str) -> str:
    return path if path.endswith(".hdf5") else f"{path}.hdf5"


def _failed_output_path(output_file: str) -> str:
    normalized = _normalize_hdf5_path(output_file)
    root, ext = os.path.splitext(normalized)
    return f"{root}_failed{ext}"


def _describe_open_holders(path: str) -> str:
    try:
        output = subprocess.check_output(["lsof", path], text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return output


def _assert_output_files_unlocked(forwarded_argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_file", type=str, required=True)
    args, _ = parser.parse_known_args(forwarded_argv)

    for path in (_normalize_hdf5_path(args.output_file), _failed_output_path(args.output_file)):
        if not os.path.exists(path):
            continue
        holders = _describe_open_holders(path)
        if holders:
            raise RuntimeError(f"Output file is already in use: {path}\n{holders}")
        with open(path, "a+b") as handle:
            try:
                fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                raise RuntimeError(f"Output file is already in use: {path}") from exc
            finally:
                try:
                    fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def main() -> None:
    ensure_project_root_on_path()
    ensure_isaaclab_runtime_compatible()
    spec, forwarded_argv = resolve_task_spec_from_argv(sys.argv[1:])
    _assert_output_files_unlocked(forwarded_argv)
    import_registration_modules(spec)
    forwarded_argv = inject_default_arg(forwarded_argv, "--task", spec.mimic_task_id)
    sys.argv = [sys.argv[0], *forwarded_argv]

    script = _isaaclab_root() / "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing IsaacLab generate script: {script}")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
