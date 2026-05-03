# SPDX-License-Identifier: BSD-3-Clause
"""
Franka open-drawer task with data-generation-compatible RGB observations.

This keeps Isaac Lab's native cabinet asset and Panda gripper setup, adds a
success termination on drawer motion, and exposes the same `joint_pos` +
`front_cam/side_cam/wrist_cam` observation keys used by the current pick-cup
pipeline so the rest of the data path can stay unchanged.
"""

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING, Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs.mdp.observations import image as obs_image
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.ik_rel_env_cfg import (
    FrankaCabinetEnvCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _top_drawer_target() -> dict[str, object]:
    return {
        "name": "top",
        "joint_name": "drawer_top_joint",
        "joint_names": ("drawer_top_joint",),
        "primary_joint_name": "drawer_top_joint",
        "handle_prim_path": "{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
        "handle_name": "drawer_handle_top",
        "handle_offset_pos": (0.305, 0.0, 0.01),
        "handle_offset_rot": (0.5, 0.5, -0.5, -0.5),
        "handle_body_name": "drawer_handle_top",
        "drawer_body_name": "drawer_top",
    }


def _bottom_drawer_target() -> dict[str, object]:
    return {
        "name": "bottom",
        "joint_name": "drawer_bottom_joint",
        "joint_names": ("drawer_bottom_joint",),
        "primary_joint_name": "drawer_bottom_joint",
        "handle_prim_path": "{ENV_REGEX_NS}/Cabinet/drawer_handle_bottom",
        "handle_name": "drawer_handle_bottom",
        "handle_offset_pos": (0.222, 0.0, 0.005),
        "handle_offset_rot": (0.5, 0.5, -0.5, -0.5),
        "handle_body_name": "drawer_handle_bottom",
        "drawer_body_name": "drawer_bottom",
    }


def resolve_drawer_target_variants(target: str) -> list[dict[str, object]]:
    target = target.strip().lower()
    if target in {"both", "either", "auto"}:
        return [_top_drawer_target(), _bottom_drawer_target()]
    if target in {"bottom", "lower", "second", "2"}:
        return [_bottom_drawer_target()]
    return [_top_drawer_target()]


def _resolve_drawer_target(target: str) -> dict[str, object]:
    variants = resolve_drawer_target_variants(target)
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
        "drawer_body_name": primary["drawer_body_name"],
    }


def _make_loop_handle_part(
    prim_path: str,
    *,
    pos: tuple[float, float, float],
    axis: str,
    radius: float,
    height: float,
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            axis=axis,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def drawer_joint_position(
    env: ManagerBasedRLEnv,
    cabinet_name: str = "cabinet",
    drawer_joint_name: str | Sequence[str] = "drawer_top_joint",
) -> torch.Tensor:
    """Return drawer joint position for each environment.

    If multiple joint names are provided, returns one column per joint with shape
    ``(num_envs, num_joints)``.
    """
    cabinet: Articulation = env.scene[cabinet_name]
    joint_names = [drawer_joint_name] if isinstance(drawer_joint_name, str) else list(drawer_joint_name)
    joint_ids, _ = cabinet.find_joints(joint_names)
    positions = cabinet.data.joint_pos[:, joint_ids]
    return positions[:, 0] if len(joint_names) == 1 else positions


def drawer_open_fraction(
    env: ManagerBasedRLEnv,
    cabinet_name: str = "cabinet",
    drawer_joint_name: str | Sequence[str] = "drawer_top_joint",
) -> torch.Tensor:
    """Return normalized drawer openness in ``[0, 1]`` based on joint limits."""
    cabinet: Articulation = env.scene[cabinet_name]
    joint_names = [drawer_joint_name] if isinstance(drawer_joint_name, str) else list(drawer_joint_name)
    joint_ids, _ = cabinet.find_joints(joint_names)
    joint_pos = cabinet.data.joint_pos[:, joint_ids]
    joint_limits = cabinet.data.soft_joint_pos_limits[:, joint_ids, :]
    joint_range = torch.clamp(joint_limits[..., 1] - joint_limits[..., 0], min=1e-6)
    openness = ((joint_pos - joint_limits[..., 0]) / joint_range).clamp(0.0, 1.0)
    return openness[:, 0] if len(joint_names) == 1 else openness


def drawer_is_open(
    env: ManagerBasedRLEnv,
    min_open_joint_pos: float | None = None,
    min_open_fraction: float = 0.80,
    cabinet_name: str = "cabinet",
    drawer_joint_name: str | Sequence[str] = "drawer_top_joint",
) -> torch.Tensor:
    """Return True when the drawer is sufficiently open.

    Standard references for cabinet-opening tasks use a near-full-open criterion:
    the Isaac Gym Franka cabinet task resets at roughly ``0.39`` joint position,
    and ManipGen's open task marks success once the remaining DOF error is below
    ``20%`` of the target. We therefore use normalized openness by default.
    """
    if min_open_joint_pos is not None:
        joint_pos = drawer_joint_position(env, cabinet_name, drawer_joint_name)
        return (joint_pos > min_open_joint_pos).any(dim=1) if joint_pos.ndim > 1 else (joint_pos > min_open_joint_pos)
    openness = drawer_open_fraction(env, cabinet_name, drawer_joint_name)
    return (openness >= min_open_fraction).any(dim=1) if openness.ndim > 1 else (openness >= min_open_fraction)


def drawer_is_closed(
    env: ManagerBasedRLEnv,
    max_closed_joint_pos: float = 0.04,
    cabinet_name: str = "cabinet",
    drawer_joint_name: str | Sequence[str] = "drawer_top_joint",
) -> torch.Tensor:
    """Return True for each env where the drawer is back near the closed state."""
    joint_pos = drawer_joint_position(env, cabinet_name, drawer_joint_name)
    return (joint_pos < max_closed_joint_pos).all(dim=1) if joint_pos.ndim > 1 else (joint_pos < max_closed_joint_pos)


@configclass
class FrankaOpenDrawerEnvCfg(FrankaCabinetEnvCfg):
    """data-generation-ready Franka drawer-opening environment with three RGB cameras."""

    def __post_init__(self):
        super().__post_init__()

        handle_style = os.getenv("OPEN_DRAWER_HANDLE_STYLE", "native").strip().lower()
        drawer_target = _resolve_drawer_target(os.getenv("OPEN_DRAWER_TARGET_DRAWER", "both"))
        drawer_joint_names = tuple(drawer_target["joint_names"])
        drawer_joint_name = str(drawer_target["primary_joint_name"])
        drawer_body_name = str(drawer_target["drawer_body_name"])
        handle_body_name = str(drawer_target["handle_body_name"])
        success_fraction = _get_env_float("OPEN_DRAWER_SUCCESS_MIN_OPEN_FRACTION", 0.39)
        success_joint_pos = os.getenv("OPEN_DRAWER_SUCCESS_MIN_OPEN_JOINT_POS")
        robot_base_x = _get_env_float("OPEN_DRAWER_ROBOT_BASE_X", 0.0)
        robot_base_y = _get_env_float("OPEN_DRAWER_ROBOT_BASE_Y", 0.0)
        robot_base_z = _get_env_float("OPEN_DRAWER_ROBOT_BASE_Z", 0.0)
        self.scene.robot.init_state.pos = (robot_base_x, robot_base_y, robot_base_z)
        # Move the cabinet slightly farther from the robot than the Isaac Lab default
        # so teleop feels less cramped while keeping the handle reachable.
        cabinet_x = _get_env_float("OPEN_DRAWER_CABINET_X", 1.05)
        cabinet_y = _get_env_float("OPEN_DRAWER_CABINET_Y", 0.0)
        cabinet_z = _get_env_float("OPEN_DRAWER_CABINET_Z", 0.40)
        self.scene.cabinet.init_state.pos = (cabinet_x, cabinet_y, cabinet_z)

        target_frame = self.scene.cabinet_frame.target_frames[0]
        target_frame.prim_path = str(drawer_target["handle_prim_path"])
        target_frame.name = str(drawer_target["handle_name"])
        target_frame.offset.pos = tuple(drawer_target["handle_offset_pos"])
        target_frame.offset.rot = tuple(drawer_target["handle_offset_rot"])

        self.observations.policy.cabinet_joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "cabinet", joint_names=[drawer_joint_name]
        )
        self.observations.policy.cabinet_joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "cabinet", joint_names=[drawer_joint_name]
        )
        self.rewards.open_drawer_bonus.params["asset_cfg"] = SceneEntityCfg("cabinet", joint_names=[drawer_joint_name])
        self.rewards.multi_stage_open_drawer.params["asset_cfg"] = SceneEntityCfg(
            "cabinet", joint_names=[drawer_joint_name]
        )
        self.events.cabinet_physics_material.params["asset_cfg"].body_names = handle_body_name

        if handle_style == "loop":
            handle_attach_x = _get_env_float("OPEN_DRAWER_LOOP_HANDLE_ATTACH_X", 0.305)
            handle_center_x = _get_env_float("OPEN_DRAWER_LOOP_HANDLE_CENTER_X", 0.365)
            handle_center_z = _get_env_float("OPEN_DRAWER_LOOP_HANDLE_CENTER_Z", 0.01)
            handle_width = _get_env_float("OPEN_DRAWER_LOOP_HANDLE_WIDTH", 0.16)
            handle_radius = _get_env_float("OPEN_DRAWER_LOOP_HANDLE_RADIUS", 0.012)
            handle_color = (
                _get_env_float("OPEN_DRAWER_LOOP_HANDLE_COLOR_R", 0.78),
                _get_env_float("OPEN_DRAWER_LOOP_HANDLE_COLOR_G", 0.68),
                _get_env_float("OPEN_DRAWER_LOOP_HANDLE_COLOR_B", 0.50),
            )
            post_length = max(handle_center_x - handle_attach_x, 0.02)
            post_center_x = handle_attach_x + 0.5 * post_length
            half_width = 0.5 * handle_width

            self.scene.loop_handle_left_post = _make_loop_handle_part(
                f"{{ENV_REGEX_NS}}/Cabinet/{drawer_body_name}/loop_handle_left_post",
                pos=(post_center_x, -half_width, handle_center_z),
                axis="X",
                radius=handle_radius,
                height=post_length,
                color=handle_color,
            )
            self.scene.loop_handle_right_post = _make_loop_handle_part(
                f"{{ENV_REGEX_NS}}/Cabinet/{drawer_body_name}/loop_handle_right_post",
                pos=(post_center_x, half_width, handle_center_z),
                axis="X",
                radius=handle_radius,
                height=post_length,
                color=handle_color,
            )
            self.scene.loop_handle_crossbar = _make_loop_handle_part(
                f"{{ENV_REGEX_NS}}/Cabinet/{drawer_body_name}/loop_handle_crossbar",
                pos=(handle_center_x, 0.0, handle_center_z),
                axis="Y",
                radius=handle_radius,
                height=handle_width,
                color=handle_color,
            )

            target_frame.prim_path = f"{{ENV_REGEX_NS}}/Cabinet/{drawer_body_name}"
            target_frame.name = f"drawer_loop_handle_{drawer_target['name']}"
            target_frame.offset.pos = (handle_center_x, 0.0, handle_center_z)
            target_frame.offset.rot = (0.5, 0.5, -0.5, -0.5)

            self.events.cabinet_physics_material.params["asset_cfg"].body_names = drawer_body_name

        self.terminations.success = DoneTerm(
            func=drawer_is_open,
            params={
                "min_open_joint_pos": None if success_joint_pos is None else float(success_joint_pos),
                "min_open_fraction": success_fraction,
                "drawer_joint_name": drawer_joint_names,
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
