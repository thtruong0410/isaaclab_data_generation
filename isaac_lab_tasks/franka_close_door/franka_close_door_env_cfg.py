# SPDX-License-Identifier: BSD-3-Clause
"""Franka close-door task built on the working open-door environment."""

from __future__ import annotations

import os

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaac_lab_tasks.franka_open_door.franka_open_door_env_cfg import (
    FrankaOpenDoorEnvCfg,
    _get_env_float,
    _resolve_door_target,
    door_is_closed,
    resolve_door_target_variants,
)


def _resolve_close_success_threshold() -> float:
    return _get_env_float("CLOSE_DOOR_SUCCESS_MAX_CLOSED_JOINT_POS", 0.18)


def _resolve_start_open_joint_pos() -> float:
    return _get_env_float("CLOSE_DOOR_START_OPEN_JOINT_POS", 0.45)


def _resolve_door_passive_stiffness() -> float:
    return _get_env_float("CLOSE_DOOR_ACTUATOR_STIFFNESS", 0.0)


def _resolve_door_passive_damping() -> float:
    return _get_env_float("CLOSE_DOOR_ACTUATOR_DAMPING", 0.0)


def _signed_open_start(joint_name: str, start_open_joint_pos: float) -> float:
    return -abs(start_open_joint_pos) if "left" in joint_name else abs(start_open_joint_pos)


@configclass
class FrankaCloseDoorEnvCfg(FrankaOpenDoorEnvCfg):
    """Door-closing task with the same visual setup as the open-door task."""

    def __post_init__(self):
        super().__post_init__()

        target = os.getenv("OPEN_DOOR_TARGET_DOOR", "both")
        door_target = _resolve_door_target(target)
        start_open_joint_pos = _resolve_start_open_joint_pos()
        max_closed_joint_pos = _resolve_close_success_threshold()

        # Make the cabinet door joints passive for close-task collection/training.
        # Otherwise the built-in implicit actuators pull the door shut without
        # requiring the robot to actually perform the close action.
        passive_stiffness = _resolve_door_passive_stiffness()
        passive_damping = _resolve_door_passive_damping()
        self.scene.cabinet.actuators["doors"].stiffness = passive_stiffness
        self.scene.cabinet.actuators["doors"].damping = passive_damping

        for variant in resolve_door_target_variants(target):
            joint_name = str(variant["joint_name"])
            self.scene.cabinet.init_state.joint_pos[joint_name] = _signed_open_start(joint_name, start_open_joint_pos)

        self.terminations.success = DoneTerm(
            func=door_is_closed,
            params={
                "max_closed_joint_pos": max_closed_joint_pos,
                "door_joint_name": tuple(door_target["joint_names"]),
            },
        )
