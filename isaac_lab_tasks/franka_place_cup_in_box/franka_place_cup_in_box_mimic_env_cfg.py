# SPDX-License-Identifier: BSD-3-Clause
"""MimicGen config for the Franka place-cup-in-box task."""

from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .franka_place_cup_in_box_env_cfg import FrankaPlaceCupInBoxEnvCfg


@configclass
class FrankaPlaceCupInBoxMimicEnvCfg(FrankaPlaceCupInBoxEnvCfg, MimicEnvCfg):
    """Place-only Mimic config: the source demo starts with the cup already held."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_place_cup_in_box_franka_D0"
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
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place the held cup into the box",
            )
        ]
