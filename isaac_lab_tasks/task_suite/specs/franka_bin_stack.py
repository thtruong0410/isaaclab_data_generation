from __future__ import annotations

from ..registry import register_task_spec
from ..types import DatasetNamingSpec, EvalSpec, TaskSuiteSpec


FRANKA_BIN_STACK_NORMAL_SPEC = TaskSuiteSpec(
    key="franka_bin_stack_normal",
    task_name="stack",
    object_name="cube_bin",
    variant="normal",
    collection_mode="normal",
    gym_task_id="Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
    mimic_task_id="Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
    mimic_env_var_prefix="FRANKA_BIN_STACK_MIMIC",
    registration_modules=("isaaclab_mimic.envs",),
    dataset_prefix="franka_bin_stack_normal",
    dataset_naming=DatasetNamingSpec(
        raw_demo_file_prefix="franka_bin_stack_demo",
        merged_hdf5_name="franka_bin_stack_demos_merged.hdf5",
        annotated_hdf5_name="franka_bin_stack_demos_annotated.hdf5",
        generated_hdf5_name="franka_bin_stack_demos_mimic.hdf5",
    ),
    eval=EvalSpec(
        task_description="stack cubes into the bin with the Franka arm",
        max_steps=600,
        report_success_at=(300, 600),
    ),
    notes=(
        "Native IsaacLab Franka bin-stack task with relative IK and Mimic config.",
        "Closest upstream data-generation task to place-into-container on Franka; "
        "it is not a pre-grasped cup placement task.",
    ),
)

register_task_spec(FRANKA_BIN_STACK_NORMAL_SPEC)
