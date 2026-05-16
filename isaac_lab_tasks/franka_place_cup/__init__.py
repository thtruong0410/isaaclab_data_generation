# SPDX-License-Identifier: BSD-3-Clause
"""Register the Franka place-cup data-generation task."""

import gymnasium as gym


gym.register(
    id="Isaac-Place-Cup-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "isaac_lab_tasks.franka_place_cup.franka_place_cup_env_cfg:FrankaPlaceCupEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Place-Cup-Franka-IK-Rel-Mimic-v0",
    entry_point="isaac_lab_tasks.franka_place_cup.place_cup_mimic_env:PlaceCupRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaac_lab_tasks.franka_place_cup.franka_place_cup_mimic_env_cfg:FrankaPlaceCupMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)
