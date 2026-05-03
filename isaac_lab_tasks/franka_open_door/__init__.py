# SPDX-License-Identifier: BSD-3-Clause
"""
Registers the Franka open-door data-generation task variants with the Gymnasium registry.
"""

from __future__ import annotations

import gymnasium as gym


gym.register(
    id="Isaac-Open-Door-GR00T-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_open_door.franka_open_door_env_cfg:FrankaOpenDoorEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Open-Door-GR00T-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pick_place_mimic_env:PickPlaceRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_open_door.franka_open_door_mimic_env_cfg:FrankaOpenDoorMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
