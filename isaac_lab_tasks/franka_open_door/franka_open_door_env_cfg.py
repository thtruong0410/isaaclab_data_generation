# SPDX-License-Identifier: BSD-3-Clause
"""
Franka open-door task with data-generation-compatible RGB observations.

This mirrors the drawer task pipeline, but targets the cabinet doors instead of
the drawers. Success defaults to a modest door joint opening threshold that
works with the Panda gripper on the built-in cabinet asset.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.observations import image as obs_image
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp as cabinet_mdp
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.ik_rel_env_cfg import (
    FrankaCabinetEnvCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _resolve_success_threshold() -> tuple[float | None, float]:
    """Resolve the door-open success threshold.

    We keep backward compatibility with the earlier degree-based environment
    variable, while defaulting to a more practical joint-position threshold of
    ``0.18`` rad for this cabinet/hand combination.
    """
    success_joint_pos = os.getenv("OPEN_DOOR_SUCCESS_MIN_OPEN_JOINT_POS")
    success_degrees = os.getenv("OPEN_DOOR_SUCCESS_MIN_OPEN_DEG")

    if success_joint_pos is not None:
        return abs(float(success_joint_pos)), 30.0

    if success_degrees is not None:
        parsed = float(success_degrees)
        # Treat small values as radians to support the expected 0.18-style
        # threshold the user is tuning against in this task.
        if abs(parsed) <= math.pi:
            return abs(parsed), 30.0
        return None, parsed

    return 0.18, 30.0


def _left_door_target() -> dict[str, object]:
    return {
        "name": "left",
        "joint_name": "door_left_joint",
        "joint_names": ("door_left_joint",),
        "primary_joint_name": "door_left_joint",
        "handle_prim_path": "{ENV_REGEX_NS}/Cabinet/door_left_nob_link",
        "handle_name": "door_left_nob_link",
        # Derived from sektion_cabinet_2.urdf visual origin + door_left_nob.obj mesh center.
        "handle_offset_pos": (0.0235945, 0.3510085, 0.1845515),
        "handle_offset_rot": (1.0, 0.0, 0.0, 0.0),
        "handle_body_name": "door_left_nob_link",
        "door_body_name": "door_left",
    }


def _right_door_target() -> dict[str, object]:
    return {
        "name": "right",
        "joint_name": "door_right_joint",
        "joint_names": ("door_right_joint",),
        "primary_joint_name": "door_right_joint",
        "handle_prim_path": "{ENV_REGEX_NS}/Cabinet/door_right_nob_link",
        "handle_name": "door_right_nob_link",
        # Derived from sektion_cabinet_2.urdf visual origin + door_right_nob.obj mesh center.
        "handle_offset_pos": (0.0236055, -0.3517835, 0.1850665),
        "handle_offset_rot": (1.0, 0.0, 0.0, 0.0),
        "handle_body_name": "door_right_nob_link",
        "door_body_name": "door_right",
    }


def resolve_door_target_variants(target: str) -> list[dict[str, object]]:
    target = target.strip().lower()
    if target in {"both", "either", "auto"}:
        return [_left_door_target(), _right_door_target()]
    if target in {"right", "r", "2"}:
        return [_right_door_target()]
    return [_left_door_target()]


def _resolve_door_target(target: str) -> dict[str, object]:
    variants = resolve_door_target_variants(target)
    if len(variants) == 1:
        return variants[0]
    primary = variants[0]
    return {
        "name": "both",
        "joint_names": tuple(variant["joint_name"] for variant in variants),
        "primary_joint_name": primary["primary_joint_name"],
        "handle_prim_path": primary["handle_prim_path"],
        "handle_name": primary["handle_name"],
        "handle_offset_pos": primary["handle_offset_pos"],
        "handle_offset_rot": primary["handle_offset_rot"],
        "handle_body_name": primary["handle_body_name"],
        "door_body_name": primary["door_body_name"],
    }


def door_joint_position(
    env: ManagerBasedRLEnv,
    cabinet_name: str = "cabinet",
    door_joint_name: str | Sequence[str] = "door_left_joint",
) -> torch.Tensor:
    cabinet: Articulation = env.scene[cabinet_name]
    joint_names = [door_joint_name] if isinstance(door_joint_name, str) else list(door_joint_name)
    joint_ids, _ = cabinet.find_joints(joint_names)
    positions = cabinet.data.joint_pos[:, joint_ids]
    return positions[:, 0] if len(joint_names) == 1 else positions


def door_joint_abs_angle(
    env: ManagerBasedRLEnv,
    cabinet_name: str = "cabinet",
    door_joint_name: str | Sequence[str] = "door_left_joint",
) -> torch.Tensor:
    joint_pos = door_joint_position(env, cabinet_name, door_joint_name)
    return joint_pos.abs()


def door_is_open(
    env: ManagerBasedRLEnv,
    min_open_joint_pos: float | None = None,
    min_open_degrees: float = 30.0,
    cabinet_name: str = "cabinet",
    door_joint_name: str | Sequence[str] = "door_left_joint",
) -> torch.Tensor:
    open_angle = door_joint_abs_angle(env, cabinet_name, door_joint_name)
    threshold = math.radians(min_open_degrees) if min_open_joint_pos is None else abs(min_open_joint_pos)
    return (open_angle >= threshold).any(dim=1) if open_angle.ndim > 1 else (open_angle >= threshold)


def door_is_closed(
    env: ManagerBasedRLEnv,
    max_closed_joint_pos: float | None = None,
    max_closed_degrees: float = 5.0,
    cabinet_name: str = "cabinet",
    door_joint_name: str | Sequence[str] = "door_left_joint",
) -> torch.Tensor:
    closed_angle = door_joint_abs_angle(env, cabinet_name, door_joint_name)
    threshold = math.radians(max_closed_degrees) if max_closed_joint_pos is None else abs(max_closed_joint_pos)
    return (closed_angle <= threshold).all(dim=1) if closed_angle.ndim > 1 else (closed_angle <= threshold)


def open_door_bonus_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    door_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]].abs()
    is_graspable = cabinet_mdp.align_grasp_around_handle(env).float()
    return (is_graspable + 1.0) * door_pos


def multi_stage_open_door_abs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    door_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]].abs()
    is_graspable = cabinet_mdp.align_grasp_around_handle(env).float()
    open_easy = (door_pos > math.radians(5.0)) * 0.5
    open_medium = (door_pos > math.radians(20.0)) * is_graspable
    open_hard = (door_pos > math.radians(30.0)) * is_graspable
    return open_easy + open_medium + open_hard


@configclass
class FrankaOpenDoorEnvCfg(FrankaCabinetEnvCfg):
    """data-generation-ready Franka door-opening environment with three RGB cameras."""

    def __post_init__(self):
        super().__post_init__()

        door_target = _resolve_door_target(os.getenv("OPEN_DOOR_TARGET_DOOR", "both"))
        door_joint_names = tuple(door_target["joint_names"])
        door_joint_name = str(door_target["primary_joint_name"])
        handle_body_name = str(door_target["handle_body_name"])
        success_joint_pos, success_degrees = _resolve_success_threshold()

        robot_base_x = _get_env_float("OPEN_DOOR_ROBOT_BASE_X", 0.0)
        robot_base_y = _get_env_float("OPEN_DOOR_ROBOT_BASE_Y", 0.0)
        robot_base_z = _get_env_float("OPEN_DOOR_ROBOT_BASE_Z", 0.0)
        self.scene.robot.init_state.pos = (robot_base_x, robot_base_y, robot_base_z)

        cabinet_x = _get_env_float("OPEN_DOOR_CABINET_X", 1.05)
        cabinet_y = _get_env_float("OPEN_DOOR_CABINET_Y", 0.0)
        cabinet_z = _get_env_float("OPEN_DOOR_CABINET_Z", 0.40)
        self.scene.cabinet.init_state.pos = (cabinet_x, cabinet_y, cabinet_z)

        target_frame = self.scene.cabinet_frame.target_frames[0]
        target_frame.prim_path = str(door_target["handle_prim_path"])
        target_frame.name = str(door_target["handle_name"])
        target_frame.offset.pos = tuple(door_target["handle_offset_pos"])
        target_frame.offset.rot = tuple(door_target["handle_offset_rot"])

        self.observations.policy.cabinet_joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "cabinet", joint_names=[door_joint_name]
        )
        self.observations.policy.cabinet_joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "cabinet", joint_names=[door_joint_name]
        )
        self.rewards.open_drawer_bonus.params["asset_cfg"] = SceneEntityCfg("cabinet", joint_names=[door_joint_name])
        self.rewards.multi_stage_open_drawer.params["asset_cfg"] = SceneEntityCfg(
            "cabinet", joint_names=[door_joint_name]
        )
        self.rewards.open_drawer_bonus.func = open_door_bonus_abs
        self.rewards.multi_stage_open_drawer.func = multi_stage_open_door_abs
        self.events.cabinet_physics_material.params["asset_cfg"].body_names = handle_body_name

        self.terminations.success = DoneTerm(
            func=door_is_open,
            params={
                "min_open_joint_pos": success_joint_pos,
                "min_open_degrees": success_degrees,
                "door_joint_name": door_joint_names,
            },
        )
        self.episode_length_s = 10.0

        _pinhole = sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 4.0),
        )

        self.scene.front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=_pinhole,
            offset=CameraCfg.OffsetCfg(
                pos=(1.60, 0.0, 0.95),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
        )

        self.scene.side_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/side_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=_pinhole,
            offset=CameraCfg.OffsetCfg(
                pos=(0.90, -1.00, 0.90),
                rot=(0.5, -0.866, 0.0, 0.0),
                convention="ros",
            ),
        )

        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=_pinhole,
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        self.observations.policy.front_cam = ObsTerm(
            func=obs_image,
            params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False},
        )
        self.observations.policy.side_cam = ObsTerm(
            func=obs_image,
            params={"sensor_cfg": SceneEntityCfg("side_cam"), "data_type": "rgb", "normalize": False},
        )
        self.observations.policy.wrist_cam = ObsTerm(
            func=obs_image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        self.observations.policy.concatenate_terms = False
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
