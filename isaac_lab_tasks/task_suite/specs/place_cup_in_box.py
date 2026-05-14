from __future__ import annotations

from ..registry import register_task_spec
from ..types import DatasetNamingSpec, EvalSpec, TaskSuiteSpec


FRANKA_PLACE_CUP_IN_BOX_NORMAL_SPEC = TaskSuiteSpec(
    key="franka_place_cup_in_box_normal",
    task_name="place",
    object_name="cup_in_box",
    variant="normal",
    collection_mode="normal",
    gym_task_id="Isaac-Place-Cup-In-Box-GR00T-Franka-IK-Rel-v0",
    mimic_task_id="Isaac-Place-Cup-In-Box-GR00T-Franka-IK-Rel-Mimic-v0",
    mimic_env_var_prefix="FRANKA_PLACE_CUP_IN_BOX_MIMIC",
    registration_modules=("isaac_lab_tasks.franka_place_cup_in_box",),
    dataset_prefix="franka_place_cup_in_box_normal",
    dataset_naming=DatasetNamingSpec(
        raw_demo_file_prefix="franka_place_cup_in_box_demo",
        merged_hdf5_name="franka_place_cup_in_box_demos_merged.hdf5",
        annotated_hdf5_name="franka_place_cup_in_box_demos_annotated.hdf5",
        generated_hdf5_name="franka_place_cup_in_box_demos_mimic.hdf5",
    ),
    eval=EvalSpec(
        task_description="place the cup into the box",
        max_steps=600,
        report_success_at=(300, 600),
    ),
    notes=(
        "Franka cup-to-box task adapted from IsaacLab Franka bin-stack components.",
        "The cup starts on the table; the demo should grasp top-down, move to the box, open, and retreat.",
    ),
)

register_task_spec(FRANKA_PLACE_CUP_IN_BOX_NORMAL_SPEC)
