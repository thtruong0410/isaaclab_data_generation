from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time

from isaaclab.app import AppLauncher

from .demo_collection_utils import ActionSmoother, build_teleop_device
from .runtime import ensure_project_root_on_path, import_registration_modules
from .types import TaskSuiteSpec


def run_cli(spec: TaskSuiteSpec, forwarded_argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=f"Demo collection for {spec.key}.")
    parser.add_argument("--task", type=str, default=spec.gym_task_id)
    parser.add_argument("--teleop_device", type=str, default="keyboard")
    parser.add_argument("--dataset_file", type=str, default=f"./datasets/{spec.key}.hdf5")
    parser.add_argument("--step_hz", type=int, default=None)
    parser.add_argument("--pos_sensitivity", type=float, default=None)
    parser.add_argument("--rot_sensitivity", type=float, default=None)
    parser.add_argument(
        "--action_smoothing_alpha",
        type=float,
        default=None,
        help="Low-pass filter coefficient for teleop actions. Smaller = smoother.",
    )
    parser.add_argument("--num_demos", type=int, default=0, help="Stop after N successful demos. 0 = run indefinitely.")
    parser.add_argument(
        "--num_success_steps",
        type=int,
        default=10,
        help="Consecutive success steps required before exporting a demo.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(forwarded_argv)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import omni.ui as ui
        import torch

        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.envs.ui import EmptyWindow
        from isaaclab.managers import DatasetExportMode
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        ensure_project_root_on_path()
        import_registration_modules(spec)

        logger = logging.getLogger(__name__)

        class RateLimiter:
            def __init__(self, hz: int):
                self.hz = hz
                self.last_time = time.time()
                self.sleep_duration = 1.0 / hz
                self.render_period = min(0.033, self.sleep_duration)

            def sleep(self, env):
                next_wakeup = self.last_time + self.sleep_duration
                while time.time() < next_wakeup:
                    time.sleep(self.render_period)
                    env.sim.render()
                self.last_time += self.sleep_duration
                if self.last_time < time.time():
                    while self.last_time < time.time():
                        self.last_time += self.sleep_duration

        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]

        success_term = None
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            logger.warning("No 'success' termination term found — demos won't be auto-marked.")

        env_cfg.terminations.time_out = None
        env_cfg.observations.policy.concatenate_terms = False
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args_cli.dataset_file))
        env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

        os.makedirs(env_cfg.recorders.dataset_export_dir_path, exist_ok=True)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        def request_reset():
            nonlocal should_reset
            should_reset = True

        if args_cli.teleop_device.lower() == "keyboard":
            pos_sensitivity = 0.06 if args_cli.pos_sensitivity is None else args_cli.pos_sensitivity
            rot_sensitivity = 0.15 if args_cli.rot_sensitivity is None else args_cli.rot_sensitivity
            step_hz = args_cli.step_hz or 20
            smoothing_alpha = 0.30 if args_cli.action_smoothing_alpha is None else args_cli.action_smoothing_alpha
        else:
            pos_sensitivity = 0.10 if args_cli.pos_sensitivity is None else args_cli.pos_sensitivity
            rot_sensitivity = 0.18 if args_cli.rot_sensitivity is None else args_cli.rot_sensitivity
            step_hz = args_cli.step_hz or 30
            smoothing_alpha = 1.0 if args_cli.action_smoothing_alpha is None else args_cli.action_smoothing_alpha

        teleop, controls_text, pos_sensitivity, rot_sensitivity = build_teleop_device(
            args_cli.teleop_device,
            sim_device=env.device,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
            request_reset=request_reset,
        )
        smoother = ActionSmoother(smoothing_alpha)
        rate_limiter = RateLimiter(step_hz)

        should_reset = False
        recorded = 0
        success_count = 0

        window = EmptyWindow(env, f"Task Demo Recorder — {spec.key}")
        label_ref = [None]
        with window.ui_window_elements["main_vstack"]:
            label_ref[0] = ui.Label(f"Recorded: {recorded} demos")

        def update_label():
            label_ref[0].text = f"Recorded: {recorded} demos"

        env.sim.reset()
        env.reset()
        teleop.reset()
        smoother.reset()

        print(
            f"\n[TaskSuite] Spec={spec.key} device={args_cli.teleop_device} "
            f"pos_sens={pos_sensitivity:.3f} rot_sens={rot_sensitivity:.3f}"
        )
        print(f"[TaskSuite] Controls: {controls_text}")
        print(f"[TaskSuite] Action smoothing alpha={smoothing_alpha:.2f}")

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                action = smoother.filter(teleop.advance())
                actions = action.repeat(env.num_envs, 1)
                env.step(actions)
                update_label()

                if success_term is not None and not should_reset:
                    if bool(success_term.func(env, **success_term.params)[0]):
                        success_count += 1
                        if success_count >= args_cli.num_success_steps:
                            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                            env.recorder_manager.set_success_to_episodes(
                                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                            )
                            env.recorder_manager.export_episodes([0])
                            print("[TaskSuite] Success! Demo saved.")
                            should_reset = True
                    else:
                        success_count = 0

                new_count = env.recorder_manager.exported_successful_episode_count
                if new_count > recorded:
                    recorded = new_count
                    print(f"[TaskSuite] Total recorded: {recorded}")

                if args_cli.num_demos > 0 and recorded >= args_cli.num_demos:
                    print(f"[TaskSuite] All {recorded} demos collected. Exiting.")
                    break

                if should_reset or env.sim.is_stopped():
                    if env.sim.is_stopped():
                        env.sim.reset()
                    env.recorder_manager.reset()
                    env.reset()
                    teleop.reset()
                    smoother.reset()
                    success_count = 0
                    should_reset = False

                rate_limiter.sleep(env)

        env.close()
        print(f"\n[TaskSuite] Session complete — {recorded} demos saved to {args_cli.dataset_file}")
    finally:
        simulation_app.close()
