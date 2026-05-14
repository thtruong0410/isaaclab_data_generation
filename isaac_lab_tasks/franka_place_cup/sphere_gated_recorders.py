# SPDX-License-Identifier: BSD-3-Clause
"""Box-centered sphere-gated recorder terms for the Franka place-cup task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


_ENTERED_KEY = "_sphere_entered"
_MIN_DIST_KEY = "_sphere_min_dist"


class BoxSphereEntryDetector(RecorderTerm):
    """Start recording once the end-effector reaches a sphere centered near the box opening."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        self._env.extras[_ENTERED_KEY] = False
        self._env.extras[_MIN_DIST_KEY] = float("inf")
        return None, None

    def record_pre_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            ee_pos = self._env.scene[self.cfg.ee_frame_name].data.target_pos_w[:, 0, :3]
            box_pos = self._env.scene[self.cfg.object_name].data.root_pos_w[:, :3].clone()
            box_pos[:, 2] += self.cfg.object_half_height
            dist = torch.linalg.vector_norm(ee_pos - box_pos, dim=-1)
            min_dist = float(dist.min().item())
            self._env.extras[_MIN_DIST_KEY] = min(min_dist, self._env.extras.get(_MIN_DIST_KEY, float("inf")))

            in_box_sphere = dist < self.cfg.sphere_radius
            if self.cfg.require_cup_held:
                robot: Articulation = self._env.scene[self.cfg.robot_name]
                cup: RigidObject = self._env.scene[self.cfg.cup_name]
                cup_near_ee = torch.linalg.vector_norm(cup.data.root_pos_w[:, :3] - ee_pos, dim=-1) < self.cfg.held_dist
                finger_ids, _ = robot.find_joints(["panda_finger_joint1", "panda_finger_joint2"])
                gripper_closed = robot.data.joint_pos[:, finger_ids].max(dim=1).values < self.cfg.gripper_closed_threshold
                in_box_sphere = in_box_sphere & cup_near_ee & gripper_closed

            if bool(in_box_sphere.any()):
                self._env.extras[_ENTERED_KEY] = True
                print(
                    f"[BoxSphereGate] Gripper entered box sphere "
                    f"(min_dist={min_dist:.3f} m, r={self.cfg.sphere_radius} m) "
                    "- saving sphere-entry state as initial_state and starting recording."
                )
                entry_state = self._env.scene.get_state(is_relative=True)
                for asset_data in entry_state.get("articulation", {}).values():
                    asset_data["joint_velocity"] = torch.zeros_like(asset_data["joint_velocity"])
                    asset_data["root_velocity"] = torch.zeros_like(asset_data["root_velocity"])
                for asset_data in entry_state.get("rigid_object", {}).values():
                    asset_data["root_velocity"] = torch.zeros_like(asset_data["root_velocity"])
                return "initial_state", entry_state

        return None, None


class BoxGatedActionsRecorder(RecorderTerm):
    def record_pre_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            return None, None
        return "actions", self._env.action_manager.action.clone()


class BoxGatedObsRecorder(RecorderTerm):
    def record_pre_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            return None, None
        return "obs", self._env.obs_buf["policy"]


class BoxGatedStatesRecorder(RecorderTerm):
    def record_post_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            return None, None
        return "states", self._env.scene.get_state(is_relative=True)


class BoxGatedProcessedActionsRecorder(RecorderTerm):
    def record_post_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            return None, None
        processed = None
        for term_name in self._env.action_manager.active_terms:
            term_actions = self._env.action_manager.get_term(term_name).processed_actions.clone()
            processed = term_actions if processed is None else torch.cat([processed, term_actions], dim=-1)
        return "processed_actions", processed


@configclass
class BoxSphereEntryCfg(RecorderTermCfg):
    sphere_radius: float = 0.18
    object_half_height: float = 0.08
    ee_frame_name: str = "ee_frame"
    object_name: str = "box"
    cup_name: str = "object"
    robot_name: str = "robot"
    require_cup_held: bool = True
    held_dist: float = 0.10
    gripper_closed_threshold: float = 0.025


@configclass
class BoxSphereEntryDetectorCfg(BoxSphereEntryCfg):
    class_type: type[RecorderTerm] = BoxSphereEntryDetector


@configclass
class BoxGatedActionsRecorderCfg(BoxSphereEntryCfg):
    class_type: type[RecorderTerm] = BoxGatedActionsRecorder


@configclass
class BoxGatedObsRecorderCfg(BoxSphereEntryCfg):
    class_type: type[RecorderTerm] = BoxGatedObsRecorder


@configclass
class BoxGatedStatesRecorderCfg(BoxSphereEntryCfg):
    class_type: type[RecorderTerm] = BoxGatedStatesRecorder


@configclass
class BoxGatedProcessedActionsRecorderCfg(BoxSphereEntryCfg):
    class_type: type[RecorderTerm] = BoxGatedProcessedActionsRecorder


@configclass
class BoxGatedRecorderManagerCfg(RecorderManagerBaseCfg):
    detect_sphere_entry = BoxSphereEntryDetectorCfg()
    record_pre_step_actions = BoxGatedActionsRecorderCfg()
    record_pre_step_observations = BoxGatedObsRecorderCfg()
    record_post_step_states = BoxGatedStatesRecorderCfg()
    record_post_step_processed_acts = BoxGatedProcessedActionsRecorderCfg()
