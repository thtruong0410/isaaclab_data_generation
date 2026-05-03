from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import Articulation
from isaaclab_mimic.envs.pick_place_mimic_env import PickPlaceRelMimicEnv


class FrankaCloseDrawerMimicEnv(PickPlaceRelMimicEnv):
    """Close-drawer Mimic env that keys object pose off the active drawer handle."""

    _HANDLE_BODY_NAMES = ("drawer_handle_top", "drawer_handle_bottom")

    def _env_ids_tensor(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _active_handle_indices(self, env_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "_close_drawer_active_handle_indices"):
            active_indices = self._close_drawer_active_handle_indices
            if active_indices.shape[0] == self.num_envs:
                return active_indices[env_ids]

        cabinet: Articulation = self.scene["cabinet"]
        joint_ids, _ = cabinet.find_joints(["drawer_top_joint", "drawer_bottom_joint"], preserve_order=True)
        drawer_joint_pos = cabinet.data.joint_pos[env_ids][:, joint_ids]
        return drawer_joint_pos.argmax(dim=1)

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        object_pose_matrix = super().get_object_poses(env_ids)

        env_ids_tensor = self._env_ids_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return object_pose_matrix

        cabinet: Articulation = self.scene["cabinet"]
        body_ids, _ = cabinet.find_bodies(list(self._HANDLE_BODY_NAMES), preserve_order=True)

        robot = self.scene["robot"]
        root_pos = robot.data.root_pos_w[env_ids_tensor]
        root_quat = robot.data.root_quat_w[env_ids_tensor]

        handle_poses = []
        for body_id in body_ids:
            handle_pos_w = cabinet.data.body_pos_w[env_ids_tensor, body_id, :3]
            handle_quat_w = cabinet.data.body_quat_w[env_ids_tensor, body_id, :]
            handle_pos_b, handle_quat_b = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat, handle_pos_w, handle_quat_w
            )
            handle_poses.append(PoseUtils.make_pose(handle_pos_b, PoseUtils.matrix_from_quat(handle_quat_b)))

        handle_poses = torch.stack(handle_poses, dim=1)
        active_indices = self._active_handle_indices(env_ids_tensor)
        batch_indices = torch.arange(len(env_ids_tensor), device=self.device)

        object_pose_matrix["cabinet"] = handle_poses[batch_indices, active_indices]
        return object_pose_matrix
