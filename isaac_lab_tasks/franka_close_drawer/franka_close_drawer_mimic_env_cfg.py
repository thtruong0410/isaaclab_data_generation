# SPDX-License-Identifier: BSD-3-Clause
"""MimicGen-compatible environment config for the Franka close-drawer task."""

from __future__ import annotations

import os

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp as cabinet_mdp

from isaac_lab_tasks.franka_close_drawer.franka_close_drawer_env_cfg import FrankaCloseDrawerEnvCfg
from isaac_lab_tasks.franka_open_drawer.franka_open_drawer_mimic_env_cfg import (
    _get_env_float,
    _get_env_int,
    handle_is_grasped as open_drawer_handle_is_grasped,
)


def _get_close_drawer_env_float(name: str, fallback_name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        value = os.getenv(fallback_name)
    return default if value is None else float(value)


def _get_close_drawer_env_int(name: str, fallback_name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        value = os.getenv(fallback_name)
    return default if value is None else int(value)


def _get_close_drawer_env_bool(name: str, fallback_name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        value = os.getenv(fallback_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def close_handle_is_grasped(
    env,
    dist_threshold: float | None = None,
    gripper_threshold: float | None = None,
    fingertip_dist_threshold: float | None = None,
    fingertip_gap_threshold: float | None = None,
    require_both_fingertips: bool | None = None,
    align_threshold: float | None = None,
    require_wrap_alignment: bool | None = None,
    grasp_mode: str | None = None,
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    cabinet_frame_name: str = "cabinet_frame",
):
    """Require a real handle engagement before advancing the close-drawer subtask."""
    if dist_threshold is None:
        dist_threshold = _get_env_float("CLOSE_DRAWER_MIMIC_GRASP_DIST_THRESHOLD", 0.10)
    if gripper_threshold is None:
        gripper_threshold = _get_env_float("CLOSE_DRAWER_MIMIC_GRIPPER_THRESHOLD", 0.06)
    if align_threshold is None:
        align_threshold = _get_env_float("CLOSE_DRAWER_MIMIC_ALIGN_THRESHOLD", -1.0)
    if require_wrap_alignment is None:
        require_wrap_alignment = os.getenv("CLOSE_DRAWER_MIMIC_REQUIRE_WRAP_ALIGNMENT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if fingertip_dist_threshold is None:
        fingertip_dist_threshold = _get_close_drawer_env_float(
            "CLOSE_DRAWER_MIMIC_FINGERTIP_DIST_THRESHOLD",
            "OPEN_DRAWER_MIMIC_FINGERTIP_DIST_THRESHOLD",
            0.08,
        )
    if fingertip_gap_threshold is None:
        fingertip_gap_threshold = _get_close_drawer_env_float(
            "CLOSE_DRAWER_MIMIC_FINGERTIP_GAP_THRESHOLD",
            "OPEN_DRAWER_MIMIC_FINGERTIP_GAP_THRESHOLD",
            0.07,
        )
    if require_both_fingertips is None:
        require_both_fingertips = _get_close_drawer_env_bool(
            "CLOSE_DRAWER_MIMIC_REQUIRE_BOTH_FINGERTIPS",
            "OPEN_DRAWER_MIMIC_REQUIRE_BOTH_FINGERTIPS",
            False,
        )
    if grasp_mode is None:
        grasp_mode = os.getenv("CLOSE_DRAWER_MIMIC_GRASP_MODE", "joint")
    return open_drawer_handle_is_grasped(
        env,
        dist_threshold=dist_threshold,
        gripper_threshold=gripper_threshold,
        fingertip_dist_threshold=fingertip_dist_threshold,
        fingertip_gap_threshold=fingertip_gap_threshold,
        require_both_fingertips=require_both_fingertips,
        align_threshold=align_threshold,
        require_wrap_alignment=require_wrap_alignment,
        grasp_mode=grasp_mode,
        robot_name=robot_name,
        ee_frame_name=ee_frame_name,
        cabinet_frame_name=cabinet_frame_name,
    )


@configclass
class _CloseDrawerSubtaskTermsCfg(ObsGroup):
    grasp = ObsTerm(func=close_handle_is_grasped)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class FrankaCloseDrawerMimicEnvCfg(FrankaCloseDrawerEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Close-drawer is contact-heavy and much less tolerant of noisy subtask stitching
        # than open-drawer. Default to conservative generation settings, while still
        # honoring the legacy OPEN_DRAWER_MIMIC_* overrides for compatibility.
        action_noise = _get_close_drawer_env_float(
            "CLOSE_DRAWER_MIMIC_ACTION_NOISE", "OPEN_DRAWER_MIMIC_ACTION_NOISE", 0.01
        )
        interp_steps = _get_close_drawer_env_int(
            "CLOSE_DRAWER_MIMIC_INTERP_STEPS", "OPEN_DRAWER_MIMIC_INTERP_STEPS", 8
        )
        grasp_offset_min = _get_close_drawer_env_int(
            "CLOSE_DRAWER_MIMIC_GRASP_OFFSET_MIN", "OPEN_DRAWER_MIMIC_GRASP_OFFSET_MIN", 0
        )
        grasp_offset_max = _get_close_drawer_env_int(
            "CLOSE_DRAWER_MIMIC_GRASP_OFFSET_MAX", "OPEN_DRAWER_MIMIC_GRASP_OFFSET_MAX", 4
        )
        selection_nn_k = _get_close_drawer_env_int("CLOSE_DRAWER_MIMIC_NN_K", "OPEN_DRAWER_MIMIC_NN_K", 1)
        keep_failed = _get_close_drawer_env_bool(
            "CLOSE_DRAWER_MIMIC_KEEP_FAILED", "OPEN_DRAWER_MIMIC_KEEP_FAILED", False
        )
        select_src_per_subtask = _get_close_drawer_env_bool(
            "CLOSE_DRAWER_MIMIC_SELECT_SRC_PER_SUBTASK", "OPEN_DRAWER_MIMIC_SELECT_SRC_PER_SUBTASK", False
        )

        self.observations.policy.eef_pos = ObsTerm(func=cabinet_mdp.ee_pos)
        self.observations.policy.eef_quat = ObsTerm(func=cabinet_mdp.ee_quat)
        self.observations.policy.concatenate_terms = False
        self.observations.subtask_terms = _CloseDrawerSubtaskTermsCfg()

        self.datagen_config.name = "demo_src_close_drawer_isaac_lab"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = keep_failed
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = select_src_per_subtask
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
                description="Grasp drawer handle",
                next_subtask_description="Close drawer",
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
