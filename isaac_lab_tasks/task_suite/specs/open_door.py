from __future__ import annotations

from ..registry import register_task_spec
from ..spec_builders import make_task_variant_specs
# open door task

OPEN_DOOR_NORMAL_SPEC, OPEN_DOOR_SPHERE_SPEC = make_task_variant_specs(
    key_prefix="open_door",
    task_name="open",
    object_name="cabinet",
    gym_task_id="Isaac-Open-Door-GR00T-Franka-IK-Rel-v0",
    mimic_task_id="Isaac-Open-Door-GR00T-Franka-IK-Rel-Mimic-v0",
    mimic_env_var_prefix="OPEN_DOOR_MIMIC",
    registration_modules=("isaac_lab_tasks.franka_open_door",),
    dataset_prefix="open_door",
    normal_language_prompt="open the cabinet door",
    sphere_language_prompt="open the cabinet door",
    sphere_radius_m=0.15,
    object_half_height_m=0.0,
    normal_demo_prefix="open_door_demo",
    sphere_demo_prefix="open_door_sphere_demo",
    sphere_object_name="cabinet",
    sphere_default_bucket_preset="none",
    sphere_recorder_manager_cfg_path=(
        "isaac_lab_tasks.franka_open_door.handle_gated_recorders:HandleGatedRecorderManagerCfg"
    ),
    eval_max_steps=600,
    eval_report_success_at=(300, 600),
    normal_notes=(
        "Door-opening baseline from the full reset state.",
    ),
    sphere_notes=(
        "Handle-centered local door-opening skill variant.",
    ),
)

register_task_spec(OPEN_DOOR_NORMAL_SPEC)
register_task_spec(OPEN_DOOR_SPHERE_SPEC)
