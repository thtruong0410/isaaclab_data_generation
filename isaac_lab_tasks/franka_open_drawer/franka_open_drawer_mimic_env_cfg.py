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


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tensor_range(value: torch.Tensor) -> str:
    value = value.detach().flatten()
    if value.numel() == 0:
        return "empty"
    return f"{float(value.min().item()):.4f}..{float(value.max().item()):.4f}"


def _bool_count(value: torch.Tensor) -> str:
    value = value.detach().flatten().bool()
    return f"{int(value.sum().item())}/{value.numel()}"


def _debug_handle_grasp(
    env: ManagerBasedRLEnv,
    *,
    grasp_mode: str,
    dist_threshold: float,
    gripper_threshold: float,
    fingertip_dist_threshold: float,
    fingertip_gap_threshold: float,
    align_threshold: float,
    require_wrap_alignment: bool,
    require_both_fingertips: bool,
    dist: torch.Tensor,
    finger_max: torch.Tensor,
    lfinger_handle_dist: torch.Tensor,
    rfinger_handle_dist: torch.Tensor,
    fingertip_gap: torch.Tensor,
    close_enough: torch.Tensor,
    joint_closed: torch.Tensor,
    geometry_closed: torch.Tensor,
    handle_engaged: torch.Tensor,
    aligned: torch.Tensor,
    grasp_signal: torch.Tensor,
) -> None:
    if not _get_env_bool("OPEN_DRAWER_MIMIC_DEBUG", False):
        return

    count = int(getattr(env, "_open_drawer_mimic_debug_count", 0)) + 1
    setattr(env, "_open_drawer_mimic_debug_count", count)

    first_n = _get_env_int("OPEN_DRAWER_MIMIC_DEBUG_FIRST_N", 5)
    interval = max(_get_env_int("OPEN_DRAWER_MIMIC_DEBUG_INTERVAL", 25), 1)
    grasp_seen = bool(getattr(env, "_open_drawer_mimic_debug_seen_grasp", False))
    has_grasp = bool(grasp_signal.detach().bool().any().item())
    if has_grasp:
        setattr(env, "_open_drawer_mimic_debug_seen_grasp", True)

    should_print = count <= first_n or count % interval == 0 or (has_grasp and not grasp_seen)
    if not should_print:
        return

    sim_step = getattr(env, "common_step_counter", "NA")
    target = os.getenv("OPEN_DRAWER_TARGET_DRAWER", "both")
    print(
        "[open_drawer_mimic][grasp_debug] "
        f"call={count} sim_step={sim_step} target={target} mode={grasp_mode} "
        f"thr(dist={dist_threshold:.3f}, grip={gripper_threshold:.3f}, "
        f"ft_dist={fingertip_dist_threshold:.3f}, ft_gap={fingertip_gap_threshold:.3f}, "
        f"align={align_threshold:.3f}, wrap={int(require_wrap_alignment)}, "
        f"both_ft={int(require_both_fingertips)}) "
        f"counts(close={_bool_count(close_enough)}, joint={_bool_count(joint_closed)}, "
        f"geom={_bool_count(geometry_closed)}, engaged={_bool_count(handle_engaged)}, "
        f"aligned={_bool_count(aligned)}, grasp={_bool_count(grasp_signal)}) "
        f"dist={_tensor_range(dist)} finger_max={_tensor_range(finger_max)} "
        f"lfinger_dist={_tensor_range(lfinger_handle_dist)} "
        f"rfinger_dist={_tensor_range(rfinger_handle_dist)} "
        f"finger_gap={_tensor_range(fingertip_gap)}",
        flush=True,
    )


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
    fingertip_dist_threshold: float | None = None,
    fingertip_gap_threshold: float | None = None,
    require_both_fingertips: bool | None = None,
    align_threshold: float | None = None,
    require_wrap_alignment: bool | None = None,
    grasp_mode: str | None = None,
    robot_name: str = "robot",
    ee_frame_name: str = "ee_frame",
    cabinet_frame_name: str = "cabinet_frame",
) -> torch.Tensor:
    """Return True when the gripper is plausibly engaged with the drawer handle.

    The default detector mirrors the pick-cup grasp detector: the gripper must be
    near the handle and the fingers must be closed enough. Set
    ``OPEN_DRAWER_MIMIC_GRASP_MODE=geometry`` to try fingertip geometry instead,
    or ``either``/``both`` to combine both detectors.
    """
    if dist_threshold is None:
        dist_threshold = _get_env_float("OPEN_DRAWER_MIMIC_GRASP_DIST_THRESHOLD", 0.12)
    if gripper_threshold is None:
        gripper_threshold = _get_env_float("OPEN_DRAWER_MIMIC_GRIPPER_THRESHOLD", -0.01)
    if fingertip_dist_threshold is None:
        fingertip_dist_threshold = _get_env_float("OPEN_DRAWER_MIMIC_FINGERTIP_DIST_THRESHOLD", 0.08)
    if fingertip_gap_threshold is None:
        fingertip_gap_threshold = _get_env_float("OPEN_DRAWER_MIMIC_FINGERTIP_GAP_THRESHOLD", 0.07)
    if require_both_fingertips is None:
        require_both_fingertips = _get_env_bool("OPEN_DRAWER_MIMIC_REQUIRE_BOTH_FINGERTIPS", False)
    if align_threshold is None:
        align_threshold = _get_env_float("OPEN_DRAWER_MIMIC_ALIGN_THRESHOLD", -1.0)
    if require_wrap_alignment is None:
        require_wrap_alignment = _get_env_bool("OPEN_DRAWER_MIMIC_REQUIRE_WRAP_ALIGNMENT", False)
    if grasp_mode is None:
        grasp_mode = os.getenv("OPEN_DRAWER_MIMIC_GRASP_MODE", "joint")
    grasp_mode = grasp_mode.strip().lower()

    robot: Articulation = env.scene[robot_name]
    ee_pos = env.scene[ee_frame_name].data.target_pos_w[:, 0, :]
    ee_quat = env.scene[ee_frame_name].data.target_quat_w[:, 0, :]
    ee_fingertips_w = env.scene[ee_frame_name].data.target_pos_w[:, 1:, :]
    lfinger_pos = ee_fingertips_w[:, 0, :]
    rfinger_pos = ee_fingertips_w[:, 1, :]
    handle_pos_all, handle_quat_all = _selected_handle_pose(env)
    handle_pos, handle_quat, dist = _nearest_handle_pose(ee_pos, handle_pos_all, handle_quat_all)

    close_enough = dist < dist_threshold

    finger_ids, _ = robot.find_joints(["panda_finger_joint1", "panda_finger_joint2"])
    finger_pos = robot.data.joint_pos[:, finger_ids]
    finger_max = finger_pos.max(dim=1).values
    joint_closed = finger_max < gripper_threshold

    lfinger_handle_dist = torch.linalg.vector_norm(lfinger_pos - handle_pos, dim=1)
    rfinger_handle_dist = torch.linalg.vector_norm(rfinger_pos - handle_pos, dim=1)
    if require_both_fingertips:
        fingertips_close = (lfinger_handle_dist < fingertip_dist_threshold) & (
            rfinger_handle_dist < fingertip_dist_threshold
        )
    else:
        fingertips_close = torch.minimum(lfinger_handle_dist, rfinger_handle_dist) < fingertip_dist_threshold
    fingertip_gap = torch.linalg.vector_norm(lfinger_pos - rfinger_pos, dim=1)
    geometry_closed = fingertips_close & (fingertip_gap < fingertip_gap_threshold)

    if grasp_mode in {"geometry", "fingertip", "fingertips"}:
        handle_engaged = geometry_closed
    elif grasp_mode in {"joint", "gripper"}:
        handle_engaged = joint_closed
    elif grasp_mode in {"either", "or", "hybrid"}:
        handle_engaged = geometry_closed | joint_closed
    elif grasp_mode in {"both", "and"}:
        handle_engaged = geometry_closed & joint_closed
    else:
        raise ValueError(
            "OPEN_DRAWER_MIMIC_GRASP_MODE must be one of: geometry, joint, either, both. "
            f"Got: {grasp_mode!r}"
        )

    if align_threshold <= -1.0 and not require_wrap_alignment:
        aligned = torch.ones_like(close_enough, dtype=torch.bool)
    else:
        wrap_aligned = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])
        pose_aligned = _align_ee_to_handle(ee_quat, handle_quat) > align_threshold
        aligned = wrap_aligned if require_wrap_alignment else (wrap_aligned | pose_aligned)
    grasp_signal = close_enough & handle_engaged & aligned
    _debug_handle_grasp(
        env,
        grasp_mode=grasp_mode,
        dist_threshold=dist_threshold,
        gripper_threshold=gripper_threshold,
        fingertip_dist_threshold=fingertip_dist_threshold,
        fingertip_gap_threshold=fingertip_gap_threshold,
        align_threshold=align_threshold,
        require_wrap_alignment=require_wrap_alignment,
        require_both_fingertips=require_both_fingertips,
        dist=dist,
        finger_max=finger_max,
        lfinger_handle_dist=lfinger_handle_dist,
        rfinger_handle_dist=rfinger_handle_dist,
        fingertip_gap=fingertip_gap,
        close_enough=close_enough,
        joint_closed=joint_closed,
        geometry_closed=geometry_closed,
        handle_engaged=handle_engaged,
        aligned=aligned,
        grasp_signal=grasp_signal,
    )
    return grasp_signal.unsqueeze(-1).float()


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
