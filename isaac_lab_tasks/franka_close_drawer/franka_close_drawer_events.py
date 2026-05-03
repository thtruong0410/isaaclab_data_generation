from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv


def _normalize_drawer_target(target: str) -> str:
    target = target.strip().lower()
    if target in {"bottom", "lower", "second", "2"}:
        return "bottom"
    if target in {"top", "upper", "first", "1"}:
        return "top"
    return "both"


def reset_close_drawer_active_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    start_open_joint_pos: float,
    target: str = "both",
    cabinet_name: str = "cabinet",
) -> None:
    """Reset the close-drawer task so exactly one drawer starts open."""
    cabinet = env.scene[cabinet_name]
    if len(env_ids) == 0:
        return

    env_ids = env_ids.to(device=cabinet.device, dtype=torch.long)
    joint_pos = cabinet.data.default_joint_pos[env_ids].clone()
    joint_vel = cabinet.data.default_joint_vel[env_ids].clone()

    joint_ids, _ = cabinet.find_joints(["drawer_top_joint", "drawer_bottom_joint"], preserve_order=True)
    top_joint_id, bottom_joint_id = joint_ids

    joint_pos[:, top_joint_id] = 0.0
    joint_pos[:, bottom_joint_id] = 0.0
    joint_vel[:, top_joint_id] = 0.0
    joint_vel[:, bottom_joint_id] = 0.0

    normalized_target = _normalize_drawer_target(target)
    if normalized_target == "top":
        active_indices = torch.zeros(len(env_ids), device=cabinet.device, dtype=torch.long)
    elif normalized_target == "bottom":
        active_indices = torch.ones(len(env_ids), device=cabinet.device, dtype=torch.long)
    else:
        active_indices = torch.randint(0, 2, (len(env_ids),), device=cabinet.device)

    active_joint_ids = torch.where(active_indices == 0, top_joint_id, bottom_joint_id)
    joint_pos[torch.arange(len(env_ids), device=cabinet.device), active_joint_ids] = start_open_joint_pos

    cabinet.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    cabinet.set_joint_position_target(joint_pos, env_ids=env_ids)
    cabinet.set_joint_velocity_target(joint_vel, env_ids=env_ids)

    if not hasattr(env, "_close_drawer_active_handle_indices"):
        env._close_drawer_active_handle_indices = torch.zeros(
            env.num_envs, device=cabinet.device, dtype=torch.long
        )
    env._close_drawer_active_handle_indices[env_ids] = active_indices
