# SPDX-License-Identifier: BSD-3-Clause
"""
Handle-centered sphere-gated recorder terms for the Franka drawer task.

Recording begins only after the end-effector enters a sphere around the drawer
handle. This mirrors the near-object pick pipeline, but uses the cabinet handle
frame instead of a rigid object's root position.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils import configclass

from .franka_open_drawer_env_cfg import resolve_drawer_target_variants


_ENTERED_KEY = "_sphere_entered"


def _selected_handle_positions(env) -> torch.Tensor:
    cabinet = env.scene["cabinet"]
    target = os.getenv("OPEN_DRAWER_TARGET_DRAWER", "both")
    positions = []
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
        target_pos, _ = combine_frame_transforms(body_pos, body_quat, offset_pos, offset_quat)
        positions.append(target_pos)
    return torch.stack(positions, dim=1)


class HandleSphereEntryDetector(RecorderTerm):
    """Latch entry into a handle-centered sphere and save that simulator state as initial_state."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        self._env.extras[_ENTERED_KEY] = False
        return None, None

    def record_pre_step(self):
        if not self._env.extras.get(_ENTERED_KEY, False):
            ee_pos = self._env.scene[self.cfg.ee_frame_name].data.target_pos_w[:, 0, :3]
            handle_pos = _selected_handle_positions(self._env)
            dist = torch.linalg.vector_norm(handle_pos - ee_pos.unsqueeze(1), dim=-1)
            min_dist = dist.min(dim=1).values

            if bool((min_dist < self.cfg.sphere_radius).any()):
                self._env.extras[_ENTERED_KEY] = True
                print(
                    f"[HandleSphereGate] Gripper entered handle sphere "
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
    sphere_radius: float = 0.18
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
