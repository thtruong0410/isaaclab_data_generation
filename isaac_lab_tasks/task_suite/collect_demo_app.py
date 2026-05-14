from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time

from isaaclab.app import AppLauncher

from .debug_visualization import EeTargetLineVisualizer
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
        "--draw_ee_target_line",
        action="store_true",
        help="Draw a viewport-only debug line from ee_frame to the nearest task handle.",
    )
    parser.add_argument("--ee_target_vis_mode", choices=("line", "axes"), default="line")
    parser.add_argument("--ee_target_line_thickness", type=float, default=5.0)
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
    parser.add_argument(
        "--start_record_on_subtask",
        type=str,
        default=None,
        help="Clear the recorder and start the exported episode when this subtask term first becomes true.",
    )
    parser.add_argument(
        "--start_record_on_subtask_steps",
        type=int,
        default=1,
        help="Consecutive true steps required for --start_record_on_subtask.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(forwarded_argv)
    start_record_on_subtask = args_cli.start_record_on_subtask
    if start_record_on_subtask is None:
        start_record_on_subtask = spec.normal_start_record_on_subtask
    start_record_on_subtask_steps = args_cli.start_record_on_subtask_steps
    if args_cli.start_record_on_subtask is None and spec.normal_start_record_on_subtask is not None:
        start_record_on_subtask_steps = spec.normal_start_record_on_subtask_steps

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
        line_visualizer = EeTargetLineVisualizer(
            spec,
            enabled=args_cli.draw_ee_target_line and not args_cli.headless,
            mode=args_cli.ee_target_vis_mode,
            thickness=args_cli.ee_target_line_thickness,
        )

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
        gate_count = 0
        recording_started = start_record_on_subtask is None

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
                line_visualizer.update(env)
                update_label()

                if start_record_on_subtask is not None and not recording_started and not should_reset:
                    subtask_terms = env.obs_buf.get("subtask_terms", {})
                    if start_record_on_subtask not in subtask_terms:
                        raise KeyError(
                            f"Subtask term '{start_record_on_subtask}' not found. "
                            f"Available terms: {list(subtask_terms.keys())}"
                        )
                    if bool(subtask_terms[start_record_on_subtask][0]):
                        gate_count += 1
                        if gate_count >= start_record_on_subtask_steps:
                            env.recorder_manager.reset([0])
                            env.recorder_manager.record_post_reset([0])
                            recording_started = True
                            success_count = 0
                            print(
                                "[TaskSuite] Recording started at subtask "
                                f"'{start_record_on_subtask}'."
                            )
                    else:
                        gate_count = 0

                if success_term is not None and recording_started and not should_reset:
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
                    gate_count = 0
                    recording_started = start_record_on_subtask is None
                    should_reset = False

                rate_limiter.sleep(env)

        line_visualizer.clear()
        env.close()
        print(f"\n[TaskSuite] Session complete — {recorded} demos saved to {args_cli.dataset_file}")
    finally:
        simulation_app.close()
