# SPDX-License-Identifier: BSD-3-Clause
"""Mimic environment wrapper for the Franka place-cup task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab_mimic.envs.pick_place_mimic_env import PickPlaceRelMimicEnv


class PlaceCupRelMimicEnv(PickPlaceRelMimicEnv):
    """Pick/place-style relative Mimic env that exposes configured subtask terms.

    IsaacLab's generic ``PickPlaceRelMimicEnv`` only forwards ``grasp``-named
    subtask terms. Place-cup uses ``release`` as the split between placement and
    retreat, so annotation needs this wrapper to write the release signal into
    the annotated HDF5 datagen info.
    """

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        subtask_terms = self.obs_buf["subtask_terms"]
        signals: dict[str, torch.Tensor] = {}

        for eef_subtask_configs in self.cfg.subtask_configs.values():
            for subtask_config in eef_subtask_configs:
                signal_name = subtask_config.subtask_term_signal
                if signal_name is None:
                    continue
                if signal_name not in subtask_terms:
                    raise KeyError(
                        f"Configured subtask signal {signal_name!r} is missing from obs_buf['subtask_terms']. "
                        f"Available terms: {list(subtask_terms.keys())}"
                    )
                signals[signal_name] = subtask_terms[signal_name][env_ids]

        return signals
