# SPDX-License-Identifier: BSD-3-Clause
"""
MimicGen-compatible environment config for the Franka open-drawer task.

Subtasks:
  1. grasp  - gripper closes around the drawer handle
  2. open   - pull the drawer open
"""

from __future__ import annotations

import os

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp as cabinet_mdp

from .franka_open_drawer_env_cfg import FrankaOpenDrawerEnvCfg, resolve_drawer_target_variants


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _selected_handle_pose(env: ManagerBasedRLEnv, cabinet_name: str = "cabinet") -> tuple[torch.Tensor, torch.Tensor]:
    cabinet: Articulation = env.scene[cabinet_name]
    target = os.getenv("OPEN_DRAWER_TARGET_DRAWER", "both")
    handle_positions = []
    handle_quaternions = []
    for handle_target in resolve_drawer_target_variants(target):
        body_ids, _ = cabinet.find_bodies([str(handle_target["handle_body_name"])], preserve_order=True)
        body_id = body_ids[0]
        body_pos = cabinet.data.body_pos_w[:, body_id, :3]
        body_quat = cabinet.data.body_quat_w[:, body_id, :]
        offset_pos = torch.tensor(
            handle_target["handle_offset_pos"], device=body_pos.device, dtype=body_pos.dtype
        ).unsqueeze(0).expand_as(body_pos)
        offset_quat = torch.tensor(
            handle_target["handle_offset_rot"], device=body_quat.device, dtype=body_quat.dtype
        ).unsqueeze(0).expand_as(body_quat)
        handle_pos, handle_quat = combine_frame_transforms(body_pos, body_quat, offset_pos, offset_quat)
        handle_positions.append(handle_pos)
        handle_quaternions.append(handle_quat)
    return torch.stack(handle_positions, dim=1), torch.stack(handle_quaternions, dim=1)


def _nearest_handle_pose(
    ee_pos: torch.Tensor, handle_pos: torch.Tensor, handle_quat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = torch.linalg.vector_norm(handle_pos - ee_pos.unsqueeze(1), dim=-1)
    nearest_ids = distances.argmin(dim=1)
    batch_ids = torch.arange(ee_pos.shape[0], device=ee_pos.device)
    nearest_pos = handle_pos[batch_ids, nearest_ids]
    nearest_quat = handle_quat[batch_ids, nearest_ids]
    nearest_dist = distances[batch_ids, nearest_ids]
    return nearest_pos, nearest_quat, nearest_dist


def _align_ee_to_handle(ee_quat: torch.Tensor, handle_quat: torch.Tensor) -> torch.Tensor:
    ee_rot_mat = matrix_from_quat(ee_quat)
    handle_rot_mat = matrix_from_quat(handle_quat)

    handle_x, handle_y = handle_rot_mat[..., 0], handle_rot_mat[..., 1]
    ee_x, ee_z = ee_rot_mat[..., 0], ee_rot_mat[..., 2]

    align_z = torch.bmm(ee_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def handle_is_grasped(
    env: ManagerBasedRLEnv,
    dist_threshold: float | None = None,
    gripper_threshold: float | None = None,
    align_threshold: float | None = None,
    require_wrap_alignment: bool | None = None,
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    cabinet_frame_name: str = "cabinet_frame",
) -> torch.Tensor:
    """Return True when the gripper is plausibly engaged with the drawer handle.

    The stock Panda hand often opens the Isaac Lab drawer with a partial hook/contact
    rather than a perfect wrap. We therefore accept either the strict wrap check or a
    softer pose-alignment score, together with proximity and partial gripper closure.
    """
    if dist_threshold is None:
        dist_threshold = _get_env_float("OPEN_DRAWER_MIMIC_GRASP_DIST_THRESHOLD", 0.11)
    if gripper_threshold is None:
        gripper_threshold = _get_env_float("OPEN_DRAWER_MIMIC_GRIPPER_THRESHOLD", 0.055)
    if align_threshold is None:
        align_threshold = _get_env_float("OPEN_DRAWER_MIMIC_ALIGN_THRESHOLD", 0.15)
    if require_wrap_alignment is None:
        require_wrap_alignment = os.getenv("OPEN_DRAWER_MIMIC_REQUIRE_WRAP_ALIGNMENT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    robot: Articulation = env.scene[robot_name]
    ee_pos = env.scene[ee_frame_name].data.target_pos_w[:, 0, :]
    ee_quat = env.scene[ee_frame_name].data.target_quat_w[:, 0, :]
    handle_pos_all, handle_quat_all = _selected_handle_pose(env)
    handle_pos, handle_quat, dist = _nearest_handle_pose(ee_pos, handle_pos_all, handle_quat_all)

    close_enough = dist < dist_threshold

    finger_ids, _ = robot.find_joints(["panda_finger_joint1", "panda_finger_joint2"])
    finger_pos = robot.data.joint_pos[:, finger_ids]
    gripper_closed = finger_pos.max(dim=1).values < gripper_threshold

    ee_fingertips_w = env.scene[ee_frame_name].data.target_pos_w[:, 1:, :]
    lfinger_pos = ee_fingertips_w[:, 0, :]
    rfinger_pos = ee_fingertips_w[:, 1, :]
    wrap_aligned = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])
    pose_aligned = _align_ee_to_handle(ee_quat, handle_quat) > align_threshold
    aligned = wrap_aligned if require_wrap_alignment else (wrap_aligned | pose_aligned)
    return (close_enough & gripper_closed & aligned).unsqueeze(-1).float()


@configclass
class _OpenDrawerSubtaskTermsCfg(ObsGroup):
    grasp = ObsTerm(func=handle_is_grasped)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class FrankaOpenDrawerMimicEnvCfg(FrankaOpenDrawerEnvCfg, MimicEnvCfg):
    """Adds MimicGen subtask definitions and EEF observations for drawer opening."""

    def __post_init__(self):
        super().__post_init__()

        action_noise = _get_env_float("OPEN_DRAWER_MIMIC_ACTION_NOISE", 0.03)
        interp_steps = _get_env_int("OPEN_DRAWER_MIMIC_INTERP_STEPS", 5)
        grasp_offset_min = _get_env_int("OPEN_DRAWER_MIMIC_GRASP_OFFSET_MIN", 0)
        grasp_offset_max = _get_env_int("OPEN_DRAWER_MIMIC_GRASP_OFFSET_MAX", 10)
        selection_nn_k = _get_env_int("OPEN_DRAWER_MIMIC_NN_K", 3)

        self.observations.policy.eef_pos = ObsTerm(func=cabinet_mdp.ee_pos)
        self.observations.policy.eef_quat = ObsTerm(func=cabinet_mdp.ee_quat)
        self.observations.policy.concatenate_terms = False
        self.observations.subtask_terms = _OpenDrawerSubtaskTermsCfg()

        self.datagen_config.name = "demo_src_open_drawer_isaac_lab"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = [
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
                next_subtask_description="Open drawer",
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
        self.subtask_configs["franka"] = subtask_configs
