# SPDX-License-Identifier: BSD-3-Clause
"""MimicGen config for placing an already-grasped cup into a box."""

from __future__ import annotations

import os

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .franka_place_cup_env_cfg import FrankaPlaceCupEnvCfg


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@configclass
class FrankaPlaceCupMimicEnvCfg(FrankaPlaceCupEnvCfg, MimicEnvCfg):
    """Place-only MimicGen config: release in the box, then retreat."""

    def __post_init__(self):
        super().__post_init__()

        action_noise = _get_env_float("PLACE_CUP_MIMIC_ACTION_NOISE", 0.01)
        interp_steps = _get_env_int("PLACE_CUP_MIMIC_INTERP_STEPS", 5)
        release_offset_min = _get_env_int("PLACE_CUP_MIMIC_RELEASE_OFFSET_MIN", 0)
        release_offset_max = _get_env_int("PLACE_CUP_MIMIC_RELEASE_OFFSET_MAX", 0)
        selection_nn_k = _get_env_int("PLACE_CUP_MIMIC_NN_K", 3)

        self.datagen_config.name = "demo_src_place_cup_franka_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        self.subtask_configs["franka"] = [
            SubTaskConfig(
                object_ref="box",
                subtask_term_signal="release",
                subtask_term_offset_range=(release_offset_min, release_offset_max),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": selection_nn_k},
                action_noise=action_noise,
                num_interpolation_steps=interp_steps,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place and release cup into the box",
                next_subtask_description="Retreat after release",
            ),
            SubTaskConfig(
                object_ref="box",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": selection_nn_k},
                action_noise=action_noise,
                num_interpolation_steps=interp_steps,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Retreat after release",
            ),
        ]
