# SPDX-License-Identifier: BSD-3-Clause
"""Franka place-cup task derived from the pick-cup setup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.observations import image as obs_image
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CUP_INIT_POS = (0.50, 0.0, 0.055)
BOX_INIT_POS = (0.82, 0.22, 0.0203)


def cup_is_placed_in_box_and_released(
    env: ManagerBasedRLEnv,
    *,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    xy_threshold: float = 0.11,
    height_diff: float = 0.08,
    height_threshold: float = 0.08,
    no_contact_distance: float = 0.065,
) -> torch.Tensor:
    """Success: gripper is open, cup is in the box, and fingers have retreated from the cup."""

    placed_and_open = place_mdp.object_a_is_into_b(
        env,
        robot_cfg=robot_cfg,
        object_a_cfg=cup_cfg,
        object_b_cfg=box_cfg,
        xy_threshold=xy_threshold,
        height_diff=height_diff,
        height_threshold=height_threshold,
    )

    cup: RigidObject = env.scene[cup_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cup_pos = cup.data.root_pos_w
    target_pos = ee_frame.data.target_pos_w
    if target_pos.shape[1] >= 3:
        finger_pos = target_pos[:, 1:3, :3]
        min_finger_dist = torch.linalg.vector_norm(finger_pos - cup_pos.unsqueeze(1), dim=-1).min(dim=1).values
    else:
        min_finger_dist = torch.linalg.vector_norm(target_pos[:, 0, :3] - cup_pos, dim=1)
    return torch.logical_and(placed_and_open, min_finger_dist > no_contact_distance)


def cup_is_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    dist_threshold: float = 0.08,
    gripper_threshold: float = 0.025,
) -> torch.Tensor:
    """Subtask signal: cup is close to the gripper and both fingers are closed."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cup: RigidObject = env.scene[cup_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :3]
    close_enough = torch.linalg.vector_norm(cup.data.root_pos_w - ee_pos, dim=1) < dist_threshold
    finger_ids, _ = robot.find_joints(["panda_finger_joint1", "panda_finger_joint2"])
    gripper_closed = robot.data.joint_pos[:, finger_ids].max(dim=1).values < gripper_threshold
    return (close_enough & gripper_closed).unsqueeze(-1).float()


@configclass
class EventCfg:
    reset_all = EventTerm(func=lift_mdp.reset_scene_to_default, mode="reset")

    reset_cup_pose = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (CUP_INIT_POS[0], CUP_INIT_POS[0]),
                "y": (CUP_INIT_POS[1], CUP_INIT_POS[1]),
                "z": (CUP_INIT_POS[2], CUP_INIT_POS[2]),
                "yaw": (0.0, 0.0),
            },
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("object")],
        },
    )

    reset_box_pose = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (BOX_INIT_POS[0], BOX_INIT_POS[0]),
                "y": (BOX_INIT_POS[1], BOX_INIT_POS[1]),
                "z": (BOX_INIT_POS[2], BOX_INIT_POS[2]),
                "yaw": (0.0, 0.0),
            },
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("box")],
        },
    )


@configclass
class _PlaceCupSubtaskTermsCfg(ObsGroup):
    grasp = ObsTerm(func=cup_is_grasped)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class FrankaPlaceCupEnvCfg(FrankaCubeLiftEnvCfg):
    """Pick the nearby cup, place it into a farther box, then retreat."""

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        self.scene.plane.semantic_tags = [("class", "ground")]

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            init_state=RigidObjectCfg.InitialStateCfg(pos=CUP_INIT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.05),
            ),
        )

        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_INIT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
                scale=(1.1, 1.6, 3.3),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
        )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.107)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
            ],
        )

        self.terminations.success = DoneTerm(func=cup_is_placed_in_box_and_released)
        self.episode_length_s = 10.0

        pinhole = sim_utils.PinholeCameraCfg(
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
            spawn=pinhole,
            offset=CameraCfg.OffsetCfg(
                pos=(1.20, 0.0, 0.65),
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
            spawn=pinhole,
            offset=CameraCfg.OffsetCfg(pos=(0.65, -0.95, 0.70), rot=(0.5, -0.866, 0.0, 0.0), convention="ros"),
        )
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=pinhole,
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )

        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"

        self.observations.policy.cup_pos = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("object"), "return_key": "pos"},
        )
        self.observations.policy.cup_quat = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("object"), "return_key": "quat"},
        )
        self.observations.policy.box_pos = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("box"), "return_key": "pos"},
        )
        self.observations.policy.box_quat = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("box"), "return_key": "quat"},
        )
        self.observations.policy.eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
        self.observations.policy.eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)
        self.observations.policy.gripper_pos = ObsTerm(func=stack_mdp.gripper_pos)
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
        self.observations.subtask_terms = _PlaceCupSubtaskTermsCfg()
