# SPDX-License-Identifier: BSD-3-Clause
"""Franka close-drawer task built on the working open-drawer environment."""

from __future__ import annotations

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaac_lab_tasks.franka_close_drawer.franka_close_drawer_events import reset_close_drawer_active_target
from isaac_lab_tasks.franka_open_drawer.franka_open_drawer_env_cfg import (
    FrankaOpenDrawerEnvCfg,
    _get_env_float,
    _resolve_drawer_target,
    drawer_is_closed,
    resolve_drawer_target_variants,
)


def _resolve_close_success_threshold() -> float:
    return _get_env_float("CLOSE_DRAWER_SUCCESS_MAX_CLOSED_JOINT_POS", 0.08)


def _resolve_start_open_joint_pos() -> float:
    return _get_env_float("CLOSE_DRAWER_START_OPEN_JOINT_POS", 0.22)


def _resolve_close_drawer_target() -> str:
    return os.getenv("CLOSE_DRAWER_TARGET_DRAWER", os.getenv("OPEN_DRAWER_TARGET_DRAWER", "both"))


def _resolve_drawer_passive_stiffness() -> float:
    return _get_env_float("CLOSE_DRAWER_ACTUATOR_STIFFNESS", 0.0)


def _resolve_drawer_passive_damping() -> float:
    return _get_env_float("CLOSE_DRAWER_ACTUATOR_DAMPING", 0.0)


@configclass
class FrankaCloseDrawerEnvCfg(FrankaOpenDrawerEnvCfg):
    """Drawer-closing task with the same visual setup as the open-drawer task."""

    def __post_init__(self):
        super().__post_init__()

        target = _resolve_close_drawer_target()
        drawer_target = _resolve_drawer_target(target)
        start_open_joint_pos = _resolve_start_open_joint_pos()
        max_closed_joint_pos = _resolve_close_success_threshold()

        # Make the cabinet drawer joints passive for close-task collection/training.
        # Otherwise the built-in implicit actuators pull the drawer back toward the
        # closed pose even before the robot makes contact.
        passive_stiffness = _resolve_drawer_passive_stiffness()
        passive_damping = _resolve_drawer_passive_damping()
        self.scene.cabinet.actuators["drawers"].stiffness = passive_stiffness
        self.scene.cabinet.actuators["drawers"].damping = passive_damping

        for variant in resolve_drawer_target_variants("both"):
            self.scene.cabinet.init_state.joint_pos[str(variant["joint_name"])] = 0.0

        self.events.reset_all.params = {"reset_joint_targets": True}
        self.events.reset_active_drawer = EventTerm(
            func=reset_close_drawer_active_target,
            mode="reset",
            params={
                "start_open_joint_pos": start_open_joint_pos,
                "target": target,
            },
        )

        self.terminations.success = DoneTerm(
            func=drawer_is_closed,
            params={
                "max_closed_joint_pos": max_closed_joint_pos,
                "drawer_joint_name": tuple(drawer_target["joint_names"]),
            },
        )
