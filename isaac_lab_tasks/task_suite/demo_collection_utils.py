from __future__ import annotations

import random
from typing import Any

from scipy.spatial.transform import Rotation
import torch


class ActionSmoother:
    """Low-pass filter for teleop actions to reduce keyboard quantization."""

    def __init__(self, alpha: float, zero_snap_tolerance: float = 1e-6):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.zero_snap_tolerance = zero_snap_tolerance
        self._state: torch.Tensor | None = None

    def reset(self) -> None:
        self._state = None

    def filter(self, action: torch.Tensor) -> torch.Tensor:
        if self._state is None:
            self._state = action.clone()
            return self._state.clone()

        filtered = action.clone()
        if torch.max(torch.abs(action[:6])) <= self.zero_snap_tolerance:
            filtered[:6] = 0.0
        else:
            filtered[:6] = self.alpha * action[:6] + (1.0 - self.alpha) * self._state[:6]
        filtered[6:] = action[6:]
        self._state = filtered
        return filtered.clone()


def build_teleop_device(
    device_name: str,
    *,
    sim_device: str,
    pos_sensitivity: float | None,
    rot_sensitivity: float | None,
    request_reset: Any,
):
    """Create a teleop device with conservative defaults for demo collection."""
    device_name = device_name.lower()

    if device_name == "keyboard":
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

        pos_sensitivity = 0.06 if pos_sensitivity is None else pos_sensitivity
        rot_sensitivity = 0.15 if rot_sensitivity is None else rot_sensitivity
        teleop = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=pos_sensitivity,
                rot_sensitivity=rot_sensitivity,
                sim_device=sim_device,
            )
        )
        teleop.add_callback("R", request_reset)
        controls = (
            "W/S=X  A/D=Y  Q/E=Z  Z/X=roll  T/G=pitch  C/V=yaw  "
            "K=gripper  R=reset"
        )
    elif device_name == "spacemouse":
        from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg

        pos_sensitivity = 0.10 if pos_sensitivity is None else pos_sensitivity
        rot_sensitivity = 0.18 if rot_sensitivity is None else rot_sensitivity
        teleop = Se3SpaceMouse(
            Se3SpaceMouseCfg(
                pos_sensitivity=pos_sensitivity,
                rot_sensitivity=rot_sensitivity,
                sim_device=sim_device,
            )
        )
        teleop.add_callback("R", request_reset)
        controls = (
            "Push/pull/tilt cap=SE(3) delta pose  left button=gripper  "
            "right button=reset"
        )
    elif device_name == "gamepad":
        import carb
        from isaaclab.devices import Se3Gamepad, Se3GamepadCfg

        pos_sensitivity = 0.10 if pos_sensitivity is None else pos_sensitivity
        rot_sensitivity = 0.25 if rot_sensitivity is None else rot_sensitivity
        teleop = Se3Gamepad(
            Se3GamepadCfg(
                pos_sensitivity=pos_sensitivity,
                rot_sensitivity=rot_sensitivity,
                sim_device=sim_device,
            )
        )
        reset_button = getattr(carb.input.GamepadInput, "B", None)
        if reset_button is not None:
            teleop.add_callback(reset_button, request_reset)
        controls = (
            "left stick=XY  right stick=Z+yaw  D-pad=roll/pitch  "
            "X=gripper  B=reset"
        )
    else:
        raise ValueError(
            f"Unsupported teleop device '{device_name}'. "
            "Supported devices: keyboard, spacemouse, gamepad."
        )

    return teleop, controls, pos_sensitivity, rot_sensitivity


def apply_collection_diversity(
    env_cfg,
    *,
    demo_seed: int | None,
    bucket_preset: str,
    bucket_index: int,
    object_x_range: tuple[float, float] | None,
    object_y_range: tuple[float, float] | None,
    object_yaw_range: tuple[float, float] | None,
    front_cam_pos_jitter: float,
    side_cam_pos_jitter: float,
    front_cam_rot_jitter_deg: float,
    side_cam_rot_jitter_deg: float,
    light_intensity_range: tuple[float, float] | None,
    light_color_jitter: float,
) -> dict[str, Any]:
    """Apply collection-time diversity for one demo collection run."""
    rng = random.Random(demo_seed)
    bucket = _resolve_bucket(bucket_preset, bucket_index)

    x_range = object_x_range if object_x_range is not None else bucket["object_x_range"]
    y_range = object_y_range if object_y_range is not None else bucket["object_y_range"]
    yaw_range = object_yaw_range if object_yaw_range is not None else bucket["object_yaw_range"]

    pose_reset_event_name = None
    pose_reset_event = None
    for event_name in ("reset_object_position", "reset_object_pose", "reset_target_position"):
        candidate = getattr(getattr(env_cfg, "events", None), event_name, None)
        if candidate is not None and hasattr(candidate, "params"):
            pose_reset_event_name = event_name
            pose_reset_event = candidate
            break

    if pose_reset_event is not None:
        pose_reset_event.params["pose_range"] = {
            "x": x_range,
            "y": y_range,
            "z": (0.0, 0.0),
            "yaw": yaw_range,
        }

    if front_cam_pos_jitter > 0.0 or front_cam_rot_jitter_deg > 0.0:
        _jitter_camera(
            env_cfg.scene.front_cam.offset,
            pos_jitter_m=front_cam_pos_jitter,
            rot_jitter_deg=front_cam_rot_jitter_deg,
            rng=rng,
        )
    if side_cam_pos_jitter > 0.0 or side_cam_rot_jitter_deg > 0.0:
        _jitter_camera(
            env_cfg.scene.side_cam.offset,
            pos_jitter_m=side_cam_pos_jitter,
            rot_jitter_deg=side_cam_rot_jitter_deg,
            rng=rng,
        )

    base_color = tuple(float(x) for x in env_cfg.scene.light.spawn.color)
    if light_intensity_range is not None:
        env_cfg.scene.light.spawn.intensity = rng.uniform(*light_intensity_range)
    if light_color_jitter > 0.0:
        env_cfg.scene.light.spawn.color = _sample_balanced_color(
            base=base_color,
            variation=light_color_jitter,
            rng=rng,
        )

    return {
        "bucket_preset": bucket_preset,
        "bucket_index": bucket_index,
        "object_x_range": x_range,
        "object_y_range": y_range,
        "object_yaw_range": yaw_range,
        "pose_randomization_applied": pose_reset_event is not None,
        "pose_reset_event_name": pose_reset_event_name,
        "front_cam_pos": tuple(env_cfg.scene.front_cam.offset.pos),
        "side_cam_pos": tuple(env_cfg.scene.side_cam.offset.pos),
        "light_intensity": float(env_cfg.scene.light.spawn.intensity),
        "light_color": tuple(float(x) for x in env_cfg.scene.light.spawn.color),
        "demo_seed": demo_seed,
    }


def _resolve_bucket(bucket_preset: str, bucket_index: int) -> dict[str, tuple[float, float]]:
    if bucket_preset == "none":
        return {
            "object_x_range": (-0.10, 0.10),
            "object_y_range": (-0.25, 0.25),
            "object_yaw_range": (0.0, 0.0),
        }
    if bucket_preset == "coverage20":
        x_centers = (-0.10, -0.05, 0.00, 0.05, 0.10)
        y_centers = (-0.18, -0.06, 0.06, 0.18)
        yaw_centers = (-0.70, -0.35, 0.00, 0.35, 0.70)
        x_idx = bucket_index % len(x_centers)
        y_idx = (bucket_index // len(x_centers)) % len(y_centers)
        yaw_center = yaw_centers[(x_idx + 2 * y_idx) % len(yaw_centers)]
        return {
            "object_x_range": _make_range(x_centers[x_idx], half_width=0.02),
            "object_y_range": _make_range(y_centers[y_idx], half_width=0.04),
            "object_yaw_range": _make_range(yaw_center, half_width=0.20),
        }
    if bucket_preset == "hard12":
        x_centers = (-0.10, -0.04, 0.04, 0.10)
        y_centers = (-0.20, -0.08, 0.08)
        yaw_centers = (-0.85, -0.45, 0.00, 0.45, 0.85)
        x_idx = bucket_index % len(x_centers)
        y_idx = (bucket_index // len(x_centers)) % len(y_centers)
        yaw_center = yaw_centers[(2 * x_idx + y_idx) % len(yaw_centers)]
        return {
            "object_x_range": _make_range(x_centers[x_idx], half_width=0.018),
            "object_y_range": _make_range(y_centers[y_idx], half_width=0.03),
            "object_yaw_range": _make_range(yaw_center, half_width=0.18),
        }
    raise ValueError(
        f"Unsupported bucket preset '{bucket_preset}'. "
        "Supported presets: none, coverage20, hard12."
    )


def _make_range(center: float, *, half_width: float) -> tuple[float, float]:
    return (center - half_width, center + half_width)


def _jitter_camera(offset_cfg, *, pos_jitter_m: float, rot_jitter_deg: float, rng: random.Random) -> None:
    pos = list(offset_cfg.pos)
    if pos_jitter_m > 0.0:
        pos = [value + rng.uniform(-pos_jitter_m, pos_jitter_m) for value in pos]
        offset_cfg.pos = tuple(pos)

    if rot_jitter_deg > 0.0:
        quat_wxyz = offset_cfg.rot
        base_rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        delta_rot = Rotation.from_euler(
            "xyz",
            [
                rng.uniform(-rot_jitter_deg, rot_jitter_deg),
                rng.uniform(-rot_jitter_deg, rot_jitter_deg),
                rng.uniform(-rot_jitter_deg, rot_jitter_deg),
            ],
            degrees=True,
        )
        jittered = (delta_rot * base_rot).as_quat()
        offset_cfg.rot = (float(jittered[3]), float(jittered[0]), float(jittered[1]), float(jittered[2]))


def _sample_balanced_color(
    *,
    base: tuple[float, float, float],
    variation: float,
    rng: random.Random,
) -> tuple[float, float, float]:
    offsets = [rng.uniform(-variation, variation) for _ in range(3)]
    avg_offset = sum(offsets) / 3.0
    balanced = [offset - avg_offset for offset in offsets]
    return tuple(
        max(0.0, min(1.0, component + offset))
        for component, offset in zip(base, balanced, strict=True)
    )
