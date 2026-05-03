# SPDX-License-Identifier: BSD-3-Clause
"""Registers the Franka close-door data-generation task variants with the Gymnasium registry."""

from __future__ import annotations

import gymnasium as gym


gym.register(
    id="Isaac-Close-Door-DataGen-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_close_door.franka_close_door_env_cfg:FrankaCloseDoorEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Close-Door-DataGen-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pick_place_mimic_env:PickPlaceRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_close_door.franka_close_door_mimic_env_cfg:FrankaCloseDoorMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
