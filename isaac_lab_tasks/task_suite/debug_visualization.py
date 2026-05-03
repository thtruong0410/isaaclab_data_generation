from __future__ import annotations

import os

import torch
from isaaclab.utils.math import combine_frame_transforms

from .types import TaskSuiteSpec


def _as_xyz(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor[:, 0, :3]
    return tensor[:, :3]


class EeTargetLineVisualizer:
    """Viewport-only EE-to-target line or axis component visualization."""

    def __init__(
        self,
        spec: TaskSuiteSpec,
        *,
        enabled: bool,
        mode: str = "line",
        color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0),
        thickness: float = 5.0,
    ):
        self.spec = spec
        self.enabled = enabled
        self.mode = mode
        self.color = color
        self.thickness = thickness
        self._draw_interface = None
        self._warned = False

        if self.enabled:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            self._draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    def clear(self) -> None:
        if self._draw_interface is not None:
            self._draw_interface.clear_lines()

    def update(self, env) -> float | None:
        if self._draw_interface is None:
            return None

        try:
            ee_pos = _as_xyz(env.scene["ee_frame"].data.target_pos_w).detach()
            handle_target_pos = self._nearest_target_pos(env, ee_pos).detach()
        except Exception as exc:  # noqa: BLE001 - debug visualization must not break collection.
            if not self._warned:
                print(f"[TaskSuite] EE-target debug line disabled: {exc}")
                self._warned = True
            self.clear()
            return None

        source_pos, line_target_pos, colors = self._line_segments(ee_pos, handle_target_pos)
        self.clear()
        source = source_pos.cpu().tolist()
        target = line_target_pos.cpu().tolist()
        thicknesses = [self.thickness] * len(source)
        self._draw_interface.draw_lines(source, target, colors, thicknesses)
        return float(torch.linalg.vector_norm(handle_target_pos - ee_pos, dim=-1).min().item())

    def _line_segments(self, ee_pos: torch.Tensor, target_pos: torch.Tensor):
        if self.mode == "axes":
            x_end = torch.stack((target_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2]), dim=-1)
            y_end = torch.stack((ee_pos[:, 0], target_pos[:, 1], ee_pos[:, 2]), dim=-1)
            z_end = torch.stack((ee_pos[:, 0], ee_pos[:, 1], target_pos[:, 2]), dim=-1)

            sources = ee_pos.repeat(3, 1)
            targets = torch.cat((x_end, y_end, z_end), dim=0)
            num_envs = ee_pos.shape[0]
            colors = (
                [[1.0, 0.0, 0.0, 1.0]] * num_envs
                + [[0.0, 1.0, 0.0, 1.0]] * num_envs
                + [[0.1, 0.35, 1.0, 1.0]] * num_envs
            )
            return sources, targets, colors

        colors = [list(self.color)] * ee_pos.shape[0]
        return ee_pos, target_pos, colors

    def _nearest_target_pos(self, env, ee_pos: torch.Tensor) -> torch.Tensor:
        target_positions = self._candidate_target_positions(env)
        distances = torch.linalg.vector_norm(target_positions - ee_pos.unsqueeze(1), dim=-1)
        nearest_ids = distances.argmin(dim=1)
        batch_ids = torch.arange(ee_pos.shape[0], device=ee_pos.device)
        return target_positions[batch_ids, nearest_ids]

    def _candidate_target_positions(self, env) -> torch.Tensor:
        if "drawer" in self.spec.key:
            return self._drawer_handle_positions(env)
        if "door" in self.spec.key:
            return self._door_handle_positions(env)
        return self._cabinet_frame_target_pos(env).unsqueeze(1)

    def _door_handle_positions(self, env) -> torch.Tensor:
        from isaac_lab_tasks.franka_open_door.franka_open_door_env_cfg import resolve_door_target_variants

        target = os.getenv("OPEN_DOOR_TARGET_DOOR", "both")
        return self._articulation_handle_positions(env, resolve_door_target_variants(target))

    def _drawer_handle_positions(self, env) -> torch.Tensor:
        from isaac_lab_tasks.franka_open_drawer.franka_open_drawer_env_cfg import resolve_drawer_target_variants

        if self.spec.key.startswith("close_drawer"):
            target = os.getenv("CLOSE_DRAWER_TARGET_DRAWER", os.getenv("OPEN_DRAWER_TARGET_DRAWER", "both"))
        else:
            target = os.getenv("OPEN_DRAWER_TARGET_DRAWER", "both")
        return self._articulation_handle_positions(env, resolve_drawer_target_variants(target))

    def _articulation_handle_positions(self, env, handle_targets: list[dict[str, object]]) -> torch.Tensor:
        cabinet = env.scene["cabinet"]
        positions = []
        for handle_target in handle_targets:
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

    def _cabinet_frame_target_pos(self, env) -> torch.Tensor:
        return _as_xyz(env.scene["cabinet_frame"].data.target_pos_w)
