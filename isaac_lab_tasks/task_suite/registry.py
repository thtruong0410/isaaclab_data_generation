from __future__ import annotations

from dataclasses import asdict

from .types import TaskSuiteSpec


_REGISTRY: dict[str, TaskSuiteSpec] = {}
_SPECS_LOADED = False


def register_task_spec(spec: TaskSuiteSpec) -> None:
    if spec.key in _REGISTRY:
        raise ValueError(f"Task spec '{spec.key}' is already registered.")
    _REGISTRY[spec.key] = spec


def _ensure_specs_loaded() -> None:
    global _SPECS_LOADED
    if _SPECS_LOADED:
        return
    from .specs import close_door, close_drawer, franka_bin_stack, open_door, open_drawer, place_cup  # noqa: F401
    from .specs import place_cup_in_box  # noqa: F401

    _SPECS_LOADED = True


def get_task_spec(key: str) -> TaskSuiteSpec:
    _ensure_specs_loaded()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown task spec '{key}'. Available specs: {available}") from exc


def list_task_specs() -> list[TaskSuiteSpec]:
    _ensure_specs_loaded()
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def spec_to_dict(spec: TaskSuiteSpec) -> dict:
    return asdict(spec)
