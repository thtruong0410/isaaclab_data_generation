from __future__ import annotations

from ..registry import register_task_spec
from ..types import DatasetNamingSpec, EvalSpec, TaskSuiteSpec


PLACE_TOY2BOX_NORMAL_SPEC = TaskSuiteSpec(
    key="place_toy2box_normal",
    task_name="place",
    object_name="toy2box",
    variant="normal",
    collection_mode="normal",
    gym_task_id="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
    mimic_task_id="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0",
    mimic_env_var_prefix="PLACE_TOY2BOX_MIMIC",
    registration_modules=(
        "isaaclab_tasks.manager_based.manipulation.place.config.agibot",
        "isaaclab_mimic.envs",
    ),
    dataset_prefix="place_toy2box_normal",
    dataset_naming=DatasetNamingSpec(
        raw_demo_file_prefix="place_toy2box_demo",
        merged_hdf5_name="place_toy2box_demos_merged.hdf5",
        annotated_hdf5_name="place_toy2box_demos_annotated.hdf5",
        generated_hdf5_name="place_toy2box_demos_mimic.hdf5",
    ),
    eval=EvalSpec(
        task_description="place the toy into the box",
        max_steps=600,
        report_success_at=(300, 600),
    ),
    notes=(
        "Native IsaacLab Agibot place-to-box task.",
        "Closest upstream analogue for a place-only cup-in-box task; no custom mock backend is registered here.",
    ),
)

register_task_spec(PLACE_TOY2BOX_NORMAL_SPEC)
