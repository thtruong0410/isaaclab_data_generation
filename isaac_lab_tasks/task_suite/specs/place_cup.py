from __future__ import annotations

from ..registry import register_task_spec
from ..spec_builders import make_task_variant_specs


PLACE_CUP_NORMAL_SPEC, PLACE_CUP_SPHERE_SPEC = make_task_variant_specs(
    key_prefix="place_cup",
    task_name="place",
    object_name="cup",
    gym_task_id="Isaac-Place-Cup-Franka-IK-Rel-v0",
    mimic_task_id="Isaac-Place-Cup-Franka-IK-Rel-Mimic-v0",
    mimic_env_var_prefix="PLACE_CUP_MIMIC",
    registration_modules=("isaac_lab_tasks.franka_place_cup",),
    dataset_prefix="place_cup",
    normal_language_prompt="place the cup into the box",
    sphere_language_prompt="place the cup into the box",
    sphere_radius_m=0.18,
    object_half_height_m=0.08,
    normal_demo_prefix="place_cup_demo",
    sphere_demo_prefix="place_cup_sphere_demo",
    sphere_object_name="box",
    sphere_default_bucket_preset="none",
    sphere_recorder_manager_cfg_path="isaac_lab_tasks.franka_place_cup.sphere_gated_recorders:BoxGatedRecorderManagerCfg",
    eval_max_steps=600,
    eval_report_success_at=(300, 600),
    normal_notes=(
        "Cup starts already grasped in the Franka gripper.",
        "Normal collection records the place-only motion from reset.",
    ),
    sphere_notes=(
        "Box-centered sphere-gated placement skill.",
        "Recording starts when the already-held cup reaches the box area.",
    ),
)

register_task_spec(PLACE_CUP_NORMAL_SPEC)
register_task_spec(PLACE_CUP_SPHERE_SPEC)
