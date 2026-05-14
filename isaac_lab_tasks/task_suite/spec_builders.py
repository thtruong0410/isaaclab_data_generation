from __future__ import annotations

from .types import DatasetNamingSpec, EvalSpec, SphereCollectionSpec, TaskSuiteSpec


def make_task_variant_specs(
    *,
    key_prefix: str,
    task_name: str,
    object_name: str,
    gym_task_id: str,
    mimic_task_id: str,
    mimic_env_var_prefix: str,
    registration_modules: tuple[str, ...],
    dataset_prefix: str,
    normal_language_prompt: str,
    sphere_language_prompt: str,
    sphere_radius_m: float,
    object_half_height_m: float,
    normal_demo_prefix: str = "demo",
    sphere_demo_prefix: str = "sphere_demo",
    sphere_object_name: str = "object",
    sphere_ee_frame_name: str = "ee_frame",
    sphere_default_bucket_preset: str = "coverage20",
    sphere_recorder_manager_cfg_path: str = (
        "isaac_lab_tasks.franka_pick_cup.sphere_gated_recorders:SphereGatedRecorderManagerCfg"
    ),
    eval_max_steps: int = 500,
    eval_report_success_at: tuple[int, ...] = (300, 500),
    normal_notes: tuple[str, ...] = ("Base task variant.",),
    sphere_notes: tuple[str, ...] = ("Near-affordance sphere-gated skill variant.",),
    normal_start_record_on_subtask: str | None = None,
    normal_start_record_on_subtask_steps: int = 1,
) -> tuple[TaskSuiteSpec, TaskSuiteSpec]:
    normal_spec = TaskSuiteSpec(
        key=f"{key_prefix}_normal",
        task_name=task_name,
        object_name=object_name,
        variant="normal",
        collection_mode="normal",
        gym_task_id=gym_task_id,
        mimic_task_id=mimic_task_id,
        mimic_env_var_prefix=mimic_env_var_prefix,
        registration_modules=registration_modules,
        dataset_prefix=f"{dataset_prefix}_normal",
        dataset_naming=DatasetNamingSpec(
            raw_demo_file_prefix=normal_demo_prefix,
            merged_hdf5_name=f"{normal_demo_prefix}s_merged.hdf5",
            annotated_hdf5_name=f"{normal_demo_prefix}s_annotated.hdf5",
            generated_hdf5_name=f"{normal_demo_prefix}s_mimic.hdf5",
        ),
        eval=EvalSpec(
            task_description=normal_language_prompt,
            max_steps=eval_max_steps,
            report_success_at=eval_report_success_at,
        ),
        notes=normal_notes,
        normal_start_record_on_subtask=normal_start_record_on_subtask,
        normal_start_record_on_subtask_steps=normal_start_record_on_subtask_steps,
    )
    sphere_spec = TaskSuiteSpec(
        key=f"{key_prefix}_sphere",
        task_name=task_name,
        object_name=object_name,
        variant="sphere",
        collection_mode="sphere",
        gym_task_id=gym_task_id,
        mimic_task_id=mimic_task_id,
        mimic_env_var_prefix=mimic_env_var_prefix,
        registration_modules=registration_modules,
        dataset_prefix=f"{dataset_prefix}_sphere",
        dataset_naming=DatasetNamingSpec(
            raw_demo_file_prefix=sphere_demo_prefix,
            merged_hdf5_name=f"{sphere_demo_prefix}s_merged.hdf5",
            annotated_hdf5_name=f"{sphere_demo_prefix}s_annotated.hdf5",
            generated_hdf5_name=f"{sphere_demo_prefix}s_mimic.hdf5",
        ),
        eval=EvalSpec(
            task_description=sphere_language_prompt,
            max_steps=eval_max_steps,
            report_success_at=eval_report_success_at,
        ),
        sphere=SphereCollectionSpec(
            radius_m=sphere_radius_m,
            object_half_height_m=object_half_height_m,
            object_name=sphere_object_name,
            ee_frame_name=sphere_ee_frame_name,
            default_bucket_preset=sphere_default_bucket_preset,
            recorder_manager_cfg_path=sphere_recorder_manager_cfg_path,
        ),
        notes=sphere_notes,
    )
    return normal_spec, sphere_spec


def make_pick_variant_specs(
    *,
    key_prefix: str,
    task_name: str,
    object_name: str,
    gym_task_id: str,
    mimic_task_id: str,
    mimic_env_var_prefix: str,
    registration_modules: tuple[str, ...],
    dataset_prefix: str,
    normal_language_prompt: str,
    sphere_language_prompt: str,
    sphere_radius_m: float,
    object_half_height_m: float,
    normal_demo_prefix: str = "pick_demo",
    sphere_demo_prefix: str = "sphere_demo",
) -> tuple[TaskSuiteSpec, TaskSuiteSpec]:
    return make_task_variant_specs(
        key_prefix=key_prefix,
        task_name=task_name,
        object_name=object_name,
        gym_task_id=gym_task_id,
        mimic_task_id=mimic_task_id,
        mimic_env_var_prefix=mimic_env_var_prefix,
        registration_modules=registration_modules,
        dataset_prefix=dataset_prefix,
        normal_language_prompt=normal_language_prompt,
        sphere_language_prompt=sphere_language_prompt,
        sphere_radius_m=sphere_radius_m,
        object_half_height_m=object_half_height_m,
        normal_demo_prefix=normal_demo_prefix,
        sphere_demo_prefix=sphere_demo_prefix,
        normal_notes=(
            "Free-object pick baseline.",
        ),
        sphere_notes=(
            "Near-object sphere-gated skill variant.",
        ),
    )
