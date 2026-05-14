# SPDX-License-Identifier: BSD-3-Clause
"""Reset helpers for the Franka place-cup-in-box task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_object_to_gripper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hand_body_name: str = "panda_hand",
    relative_pos: tuple[float, float, float] = (0.0, 0.0, 0.105),
    relative_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    """Place the object at a fixed pose relative to the Franka hand on reset."""

    if env_ids is None:
        return

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    body_ids, _ = robot.find_bodies(hand_body_name)
    if not body_ids:
        raise ValueError(f"Could not find robot body '{hand_body_name}'.")

    hand_pos_w = robot.data.body_pos_w[env_ids, body_ids[0]]
    hand_quat_w = robot.data.body_quat_w[env_ids, body_ids[0]]
    offset_pos = torch.tensor(relative_pos, device=env.device, dtype=hand_pos_w.dtype).repeat(len(env_ids), 1)
    offset_quat = torch.tensor(relative_rot, device=env.device, dtype=hand_pos_w.dtype).repeat(len(env_ids), 1)

    obj_pos_w, obj_quat_w = math_utils.combine_frame_transforms(hand_pos_w, hand_quat_w, offset_pos, offset_quat)
    obj.write_root_pose_to_sim(torch.cat((obj_pos_w, obj_quat_w), dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)
