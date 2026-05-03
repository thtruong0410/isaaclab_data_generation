from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import shlex
import sys


EXPECTED_NUMPY_PREFIX = "1.26."
EXPECTED_NUMPY_EXACT = "1.26.0"
EXPECTED_TYPING_EXTENSIONS = "4.12.2"


@dataclass
class IsaacLabRuntimeReport:
    errors: list[str]
    warnings: list[str]
    summary: list[str]
    repair_command: str


def _find_dist_info_dirs(pattern: str) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for raw_path in sys.path:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_dir():
            continue
        for dist_info in sorted(path.glob(pattern)):
            dist_path = str(dist_info)
            if dist_path not in seen:
                seen.add(dist_path)
                results.append(dist_path)
    return results


def collect_isaaclab_runtime_report() -> IsaacLabRuntimeReport:
    errors: list[str] = []
    warnings: list[str] = []
    summary: list[str] = [f"python={sys.executable}"]
    python_quoted = shlex.quote(sys.executable)
    repair_command = (
        f"{python_quoted} -m pip install --no-deps --force-reinstall "
        f"numpy=={EXPECTED_NUMPY_EXACT} typing_extensions=={EXPECTED_TYPING_EXTENSIONS} && "
        f"{python_quoted} -m pip check"
    )

    try:
        isaacsim_kernel_version = metadata.version("isaacsim-kernel")
    except metadata.PackageNotFoundError:
        errors.append(
            "isaacsim-kernel is not installed in the active interpreter. "
            "Activate env_isaaclab before launching Isaac Lab tasks."
        )
    else:
        summary.append(f"isaacsim-kernel={isaacsim_kernel_version}")

    try:
        isaaclab_version = metadata.version("isaaclab")
    except metadata.PackageNotFoundError:
        errors.append(
            "isaaclab is not installed in the active interpreter. "
            "Activate env_isaaclab before launching Isaac Lab tasks."
        )
    else:
        summary.append(f"isaaclab={isaaclab_version}")

    numpy_imported_version: str | None = None
    try:
        import numpy as np
    except Exception as exc:
        errors.append(f"Failed to import numpy: {exc!r}")
    else:
        numpy_imported_version = np.__version__
        summary.append(f"numpy={np.__version__}")
        summary.append(f"numpy_path={np.__file__}")
        if not np.__version__.startswith(EXPECTED_NUMPY_PREFIX):
            errors.append(
                f"NumPy {np.__version__} is incompatible with Isaac Sim 5.1 / Isaac Lab; "
                f"expected {EXPECTED_NUMPY_EXACT}."
            )
        elif np.__version__ != EXPECTED_NUMPY_EXACT:
            warnings.append(
                f"NumPy {np.__version__} differs from Isaac Sim's pinned {EXPECTED_NUMPY_EXACT}. "
                "This may still work, but it is outside the validated configuration."
            )

    try:
        numpy_meta_version = metadata.version("numpy")
    except metadata.PackageNotFoundError:
        if numpy_imported_version is None:
            errors.append("The active interpreter does not have NumPy metadata installed.")
    else:
        summary.append(f"numpy_dist={numpy_meta_version}")
        if numpy_imported_version is not None and numpy_meta_version != numpy_imported_version:
            warnings.append(
                f"NumPy import version ({numpy_imported_version}) does not match package metadata "
                f"({numpy_meta_version}). This usually means the environment has stale files."
            )

    numpy_dist_infos = _find_dist_info_dirs("numpy-*.dist-info")
    if len(numpy_dist_infos) > 1:
        warnings.append(
            "Multiple NumPy dist-info directories are present: "
            + ", ".join(numpy_dist_infos)
        )

    try:
        typing_extensions_version = metadata.version("typing-extensions")
    except metadata.PackageNotFoundError:
        warnings.append("typing-extensions is missing from the active interpreter.")
    else:
        summary.append(f"typing-extensions={typing_extensions_version}")
        if typing_extensions_version != EXPECTED_TYPING_EXTENSIONS:
            warnings.append(
                f"typing-extensions {typing_extensions_version} differs from Isaac Sim's pinned "
                f"{EXPECTED_TYPING_EXTENSIONS}. pip check will report this mismatch."
            )

    return IsaacLabRuntimeReport(
        errors=errors,
        warnings=warnings,
        summary=summary,
        repair_command=repair_command,
    )


def format_isaaclab_runtime_report(
    report: IsaacLabRuntimeReport,
    *,
    include_warnings: bool = True,
) -> str:
    lines = ["[isaac-env] Runtime summary:"]
    for item in report.summary:
        lines.append(f"  - {item}")
    if report.errors:
        lines.append("[isaac-env] Errors:")
        for item in report.errors:
            lines.append(f"  - {item}")
    if include_warnings and report.warnings:
        lines.append("[isaac-env] Warnings:")
        for item in report.warnings:
            lines.append(f"  - {item}")
    if report.errors:
        lines.append("[isaac-env] Repair command:")
        lines.append(f"  {report.repair_command}")
        lines.append(
            "[isaac-env] Note: if you installed cuRobo manually, prefer a no-deps install "
            "so Isaac Sim keeps its pinned NumPy."
        )
    return "\n".join(lines)


def ensure_isaaclab_runtime_compatible(*, strict_warnings: bool = False) -> IsaacLabRuntimeReport:
    report = collect_isaaclab_runtime_report()
    if report.errors or (strict_warnings and report.warnings):
        raise RuntimeError(format_isaaclab_runtime_report(report))
    return report
