# SPDX-License-Identifier: BSD-3-Clause
"""Registers the Franka close-drawer data-generation task variants with the Gymnasium registry."""

from __future__ import annotations

import gymnasium as gym


gym.register(
    id="Isaac-Close-Drawer-DataGen-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_close_drawer.franka_close_drawer_env_cfg:FrankaCloseDrawerEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Close-Drawer-DataGen-Franka-IK-Rel-Mimic-v0",
    entry_point="isaac_lab_tasks.franka_close_drawer.franka_close_drawer_mimic_env:FrankaCloseDrawerMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_close_drawer.franka_close_drawer_mimic_env_cfg:FrankaCloseDrawerMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
