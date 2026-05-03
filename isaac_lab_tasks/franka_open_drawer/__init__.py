# SPDX-License-Identifier: BSD-3-Clause
"""
Registers the Franka open-drawer data-generation task variants with the Gymnasium registry.

Import this module before calling gym.make() or before launching the generic
task-suite wrappers:

    import isaac_lab_tasks.franka_open_drawer  # noqa: F401
"""

import gymnasium as gym


gym.register(
    id="Isaac-Open-Drawer-GR00T-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_open_drawer.franka_open_drawer_env_cfg:FrankaOpenDrawerEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Open-Drawer-GR00T-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pick_place_mimic_env:PickPlaceRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_open_drawer.franka_open_drawer_mimic_env_cfg:FrankaOpenDrawerMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
