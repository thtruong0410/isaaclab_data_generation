from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


CollectionMode = Literal["normal", "sphere"]


@dataclass(frozen=True)
class DatasetNamingSpec:
    raw_demo_file_prefix: str
    merged_hdf5_name: str
    annotated_hdf5_name: str
    generated_hdf5_name: str


@dataclass(frozen=True)
class SphereCollectionSpec:
    radius_m: float
    object_half_height_m: float
    object_name: str = "object"
    ee_frame_name: str = "ee_frame"
    keyboard_step_hz: int = 20
    keyboard_pos_sensitivity: float = 0.06
    keyboard_rot_sensitivity: float = 0.15
    keyboard_action_smoothing_alpha: float = 0.30
    other_step_hz: int = 30
    other_pos_sensitivity: float = 0.10
    other_rot_sensitivity: float = 0.18
    other_action_smoothing_alpha: float = 1.0
    default_bucket_preset: str = "coverage20"
    recorder_manager_cfg_path: str = (
        "isaac_lab_tasks.franka_pick_cup.sphere_gated_recorders:SphereGatedRecorderManagerCfg"
    )


@dataclass(frozen=True)
class EvalSpec:
    task_description: str
    max_steps: int = 500
    report_success_at: tuple[int, ...] = (300, 500)


@dataclass(frozen=True)
class TaskSuiteSpec:
    key: str
    task_name: str
    object_name: str
    variant: str
    collection_mode: CollectionMode
    gym_task_id: str
    mimic_task_id: str
    mimic_env_var_prefix: str | None
    registration_modules: tuple[str, ...]
    dataset_prefix: str
    dataset_naming: DatasetNamingSpec
    eval: EvalSpec
    sphere: SphereCollectionSpec | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    normal_start_record_on_subtask: str | None = None
    normal_start_record_on_subtask_steps: int = 1
