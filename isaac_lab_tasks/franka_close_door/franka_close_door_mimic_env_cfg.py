# SPDX-License-Identifier: BSD-3-Clause
"""MimicGen-compatible environment config for the Franka close-door task."""

from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp as cabinet_mdp

from isaac_lab_tasks.franka_close_door.franka_close_door_env_cfg import FrankaCloseDoorEnvCfg
from isaac_lab_tasks.franka_open_door.franka_open_door_mimic_env_cfg import (
    _OpenDoorSubtaskTermsCfg,
    _get_env_float,
    _get_env_int,
)


@configclass
class FrankaCloseDoorMimicEnvCfg(FrankaCloseDoorEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        action_noise = _get_env_float("OPEN_DOOR_MIMIC_ACTION_NOISE", 0.03)
        interp_steps = _get_env_int("OPEN_DOOR_MIMIC_INTERP_STEPS", 5)
        grasp_offset_min = _get_env_int("OPEN_DOOR_MIMIC_GRASP_OFFSET_MIN", 0)
        grasp_offset_max = _get_env_int("OPEN_DOOR_MIMIC_GRASP_OFFSET_MAX", 10)
        selection_nn_k = _get_env_int("OPEN_DOOR_MIMIC_NN_K", 3)

        self.observations.policy.eef_pos = ObsTerm(func=cabinet_mdp.ee_pos)
        self.observations.policy.eef_quat = ObsTerm(func=cabinet_mdp.ee_quat)
        self.observations.policy.concatenate_terms = False
        self.observations.subtask_terms = _OpenDoorSubtaskTermsCfg()

        self.datagen_config.name = "demo_src_close_door_isaac_lab"
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
                object_ref="cabinet",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(grasp_offset_min, grasp_offset_max),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": selection_nn_k},
                action_noise=action_noise,
                num_interpolation_steps=interp_steps,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp cabinet door handle",
                next_subtask_description="Close cabinet door",
            ),
            SubTaskConfig(
                object_ref="cabinet",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": selection_nn_k},
                action_noise=action_noise,
                num_interpolation_steps=interp_steps,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
