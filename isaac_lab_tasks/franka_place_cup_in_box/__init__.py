# SPDX-License-Identifier: BSD-3-Clause
"""Register the Franka grasp-cup-then-place-in-box data-generation task."""

import gymnasium as gym


gym.register(
    id="Isaac-Place-Cup-In-Box-GR00T-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_place_cup_in_box.franka_place_cup_in_box_env_cfg:"
            "FrankaPlaceCupInBoxEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Place-Cup-In-Box-GR00T-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pick_place_mimic_env:PickPlaceRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_place_cup_in_box.franka_place_cup_in_box_mimic_env_cfg:"
            "FrankaPlaceCupInBoxMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
