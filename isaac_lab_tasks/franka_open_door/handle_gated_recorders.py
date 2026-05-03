# SPDX-License-Identifier: BSD-3-Clause
"""
Handle-centered sphere-gated recorder terms for the Franka door task.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms

from .franka_open_door_env_cfg import resolve_door_target_variants


_ENTERED_KEY = "_sphere_entered"
_MIN_DIST_KEY = "_sphere_min_dist"


def _hand_probe_positions(env) -> torch.Tensor:
    robot = env.scene["robot"]
    probe_body_names = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
    body_ids, _ = robot.find_bodies(probe_body_names, preserve_order=True)
    body_pos = robot.data.body_pos_w[:, body_ids, :3]

    ee_targets = env.scene["ee_frame"].data.target_pos_w[..., :3]
    if ee_targets.ndim == 2:
        ee_targets = ee_targets.unsqueeze(1)

    return torch.cat([body_pos, ee_targets], dim=1)


def _selected_handle_positions(env) -> torch.Tensor:
    target = os.getenv("OPEN_DOOR_TARGET_DOOR", "both").strip().lower()

    if target not in {"both", "either", "auto"}:
        try:
            target_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :3]
            return target_pos.unsqueeze(1)
        except Exception:
            pass

    cabinet = env.scene["cabinet"]
    positions = []
    for handle_target in resolve_door_target_variants(target):
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
        target_pos, _ = combine_frame_transforms(body_pos, body_quat, offset_pos, offset_quat)
        positions.append(target_pos)
    return torch.stack(positions, dim=1)


class HandleSphereEntryDetector(RecorderTerm):
    def record_post_reset(self, env_ids: Sequence[int] | None):
        self._env.extras[_ENTERED_KEY] = False
        self._env.extras[_MIN_DIST_KEY] = float("inf")
        return None, None

    def record_pre_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            hand_probe_pos = _hand_probe_positions(self._env)
            handle_pos = _selected_handle_positions(self._env)
            dist = torch.linalg.vector_norm(handle_pos.unsqueeze(2) - hand_probe_pos.unsqueeze(1), dim=-1)
            min_dist = dist.amin(dim=(1, 2))
            self._env.extras[_MIN_DIST_KEY] = float(min_dist.min().item())

            if bool((min_dist < self.cfg.sphere_radius).any()):
                self._env.extras[_ENTERED_KEY] = True
                print(
                    f"[DoorSphereGate] Gripper entered handle sphere "
                    f"(min_dist={min_dist.min().item():.3f} m, r={self.cfg.sphere_radius} m) "
                    f"- saving sphere-entry state as initial_state and starting recording."
                )
                entry_state = self._env.scene.get_state(is_relative=True)
                for asset_data in entry_state.get("articulation", {}).values():
                    asset_data["joint_velocity"] = torch.zeros_like(asset_data["joint_velocity"])
                    asset_data["root_velocity"] = torch.zeros_like(asset_data["root_velocity"])
                for asset_data in entry_state.get("rigid_object", {}).values():
                    asset_data["root_velocity"] = torch.zeros_like(asset_data["root_velocity"])
                return "initial_state", entry_state

        return None, None


class HandleGatedActionsRecorder(RecorderTerm):
    def record_pre_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            return None, None
        return "actions", self._env.action_manager.action.clone()


class HandleGatedObsRecorder(RecorderTerm):
    def record_pre_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            return None, None
        return "obs", self._env.obs_buf["policy"]


class HandleGatedStatesRecorder(RecorderTerm):
    def record_post_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            return None, None
        return "states", self._env.scene.get_state(is_relative=True)


class HandleGatedProcessedActionsRecorder(RecorderTerm):
    def record_post_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            return None, None
        processed = None
        for term_name in self._env.action_manager.active_terms:
            term_actions = self._env.action_manager.get_term(term_name).processed_actions.clone()
            processed = term_actions if processed is None else torch.cat([processed, term_actions], dim=-1)
        return "processed_actions", processed


@configclass
class HandleSphereEntryCfg(RecorderTermCfg):
    sphere_radius: float = 0.15
    object_half_height: float = 0.0
    ee_frame_name: str = "ee_frame"
    object_name: str = "cabinet"
    cabinet_frame_name: str = "cabinet_frame"


@configclass
class HandleSphereEntryDetectorCfg(HandleSphereEntryCfg):
    class_type: type[RecorderTerm] = HandleSphereEntryDetector


@configclass
class HandleGatedActionsRecorderCfg(HandleSphereEntryCfg):
    class_type: type[RecorderTerm] = HandleGatedActionsRecorder


@configclass
class HandleGatedObsRecorderCfg(HandleSphereEntryCfg):
    class_type: type[RecorderTerm] = HandleGatedObsRecorder


@configclass
class HandleGatedStatesRecorderCfg(HandleSphereEntryCfg):
    class_type: type[RecorderTerm] = HandleGatedStatesRecorder


@configclass
class HandleGatedProcessedActionsRecorderCfg(HandleSphereEntryCfg):
    class_type: type[RecorderTerm] = HandleGatedProcessedActionsRecorder


@configclass
class HandleGatedRecorderManagerCfg(RecorderManagerBaseCfg):
    detect_sphere_entry = HandleSphereEntryDetectorCfg()
    record_pre_step_actions = HandleGatedActionsRecorderCfg()
    record_pre_step_observations = HandleGatedObsRecorderCfg()
    record_post_step_states = HandleGatedStatesRecorderCfg()
    record_post_step_processed_acts = HandleGatedProcessedActionsRecorderCfg()
