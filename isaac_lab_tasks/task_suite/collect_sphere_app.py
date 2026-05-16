from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time

from isaaclab.app import AppLauncher

from .debug_visualization import EeTargetLineVisualizer, ee_to_cup_vector_text
from .demo_collection_utils import ActionSmoother, apply_collection_diversity, build_teleop_device, reset_teleop_device
from .runtime import ensure_project_root_on_path, import_object_by_path, import_registration_modules
from .types import TaskSuiteSpec


def run_cli(spec: TaskSuiteSpec, forwarded_argv: list[str]) -> None:
    if spec.collection_mode != "sphere":
        raise ValueError(f"Spec '{spec.key}' is not a sphere-gated collection spec.")
    if spec.sphere is None:
        raise ValueError(f"Spec '{spec.key}' is missing sphere collection settings.")

    parser = argparse.ArgumentParser(description=f"Sphere-gated demo collection for {spec.key}.")
    parser.add_argument("--task", type=str, default=spec.gym_task_id)
    parser.add_argument("--teleop_device", type=str, default="keyboard")
    parser.add_argument("--dataset_file", type=str, default=f"./datasets/{spec.key}.hdf5")
    parser.add_argument("--step_hz", type=int, default=None)
    parser.add_argument("--pos_sensitivity", type=float, default=None)
    parser.add_argument("--rot_sensitivity", type=float, default=None)
    parser.add_argument("--draw_ee_target_line", action="store_true",
                        help="Draw a viewport-only debug line from ee_frame to the nearest task handle.")
    parser.add_argument("--ee_target_vis_mode", choices=("line", "axes"), default="line")
    parser.add_argument("--ee_target_line_thickness", type=float, default=5.0)
    parser.add_argument("--action_smoothing_alpha", type=float, default=None,
                        help="Low-pass filter coefficient for teleop actions. Smaller = smoother.")
    parser.add_argument("--num_demos", type=int, default=0,
                        help="Stop after N successful demos. 0 = run indefinitely.")
    parser.add_argument("--num_success_steps", type=int, default=10,
                        help="Consecutive steps above lift threshold to count as success.")
    parser.add_argument("--demo_seed", type=int, default=None,
                        help="Optional seed used for demo-time diversity sampling.")
    parser.add_argument("--sphere_radius", type=float, default=spec.sphere.radius_m,
                        help="Sphere radius around the resolved affordance center (metres).")
    parser.add_argument("--object_half_height", type=float, default=spec.sphere.object_half_height_m,
                        help="Half-height of the object (metres) used by the current sphere recorder.")
    parser.add_argument("--bucket_preset", type=str, default=spec.sphere.default_bucket_preset,
                        help="Collection diversity preset: none, coverage20, hard12.")
    parser.add_argument("--bucket_index", type=int, default=0,
                        help="Index into the chosen diversity preset.")
    parser.add_argument("--object_x_range", type=float, nargs=2, default=None,
                        help="Override object x reset range relative to the default pose.")
    parser.add_argument("--object_y_range", type=float, nargs=2, default=None,
                        help="Override object y reset range relative to the default pose.")
    parser.add_argument("--object_yaw_range", type=float, nargs=2, default=None,
                        help="Override object yaw reset range in radians.")
    parser.add_argument("--front_cam_pos_jitter", type=float, default=0.0,
                        help="Uniform XYZ position jitter applied to the front camera at env creation.")
    parser.add_argument("--side_cam_pos_jitter", type=float, default=0.0,
                        help="Uniform XYZ position jitter applied to the side camera at env creation.")
    parser.add_argument("--front_cam_rot_jitter_deg", type=float, default=0.0,
                        help="Uniform XYZ rotation jitter applied to the front camera in degrees.")
    parser.add_argument("--side_cam_rot_jitter_deg", type=float, default=0.0,
                        help="Uniform XYZ rotation jitter applied to the side camera in degrees.")
    parser.add_argument("--light_intensity_range", type=float, nargs=2, default=None,
                        help="Optional dome-light intensity range for collection-time diversity.")
    parser.add_argument("--light_color_jitter", type=float, default=0.0,
                        help="Balanced RGB jitter magnitude applied to the dome light color.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(forwarded_argv)
    show_ee_cup_vector = spec.key.startswith("place_cup")
    if show_ee_cup_vector:
        args_cli.draw_ee_target_line = False

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import omni.ui as ui
        import torch

        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.envs.ui import EmptyWindow
        from isaaclab.managers import DatasetExportMode
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        ensure_project_root_on_path()
        import_registration_modules(spec)

        logger = logging.getLogger(__name__)

        class RateLimiter:
            def __init__(self, hz):
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

        def build_env(output_dir: str, output_filename: str):
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

            recorder_cfg_cls = import_object_by_path(spec.sphere.recorder_manager_cfg_path)
            recorder_cfg: ActionStateRecorderManagerCfg = recorder_cfg_cls()
            for attr in vars(recorder_cfg):
                term_cfg = getattr(recorder_cfg, attr)
                if hasattr(term_cfg, "sphere_radius"):
                    term_cfg.sphere_radius = args_cli.sphere_radius
                if hasattr(term_cfg, "object_half_height"):
                    term_cfg.object_half_height = args_cli.object_half_height
                if hasattr(term_cfg, "object_name"):
                    term_cfg.object_name = spec.sphere.object_name
                if hasattr(term_cfg, "ee_frame_name"):
                    term_cfg.ee_frame_name = spec.sphere.ee_frame_name

            recorder_cfg.dataset_export_dir_path = output_dir
            recorder_cfg.dataset_filename = output_filename
            recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
            env_cfg.recorders = recorder_cfg

            diversity_summary = apply_collection_diversity(
                env_cfg,
                demo_seed=args_cli.demo_seed,
                bucket_preset=args_cli.bucket_preset,
                bucket_index=args_cli.bucket_index,
                object_x_range=tuple(args_cli.object_x_range) if args_cli.object_x_range is not None else None,
                object_y_range=tuple(args_cli.object_y_range) if args_cli.object_y_range is not None else None,
                object_yaw_range=tuple(args_cli.object_yaw_range) if args_cli.object_yaw_range is not None else None,
                front_cam_pos_jitter=args_cli.front_cam_pos_jitter,
                side_cam_pos_jitter=args_cli.side_cam_pos_jitter,
                front_cam_rot_jitter_deg=args_cli.front_cam_rot_jitter_deg,
                side_cam_rot_jitter_deg=args_cli.side_cam_rot_jitter_deg,
                light_intensity_range=tuple(args_cli.light_intensity_range) if args_cli.light_intensity_range is not None else None,
                light_color_jitter=args_cli.light_color_jitter,
            )

            env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
            return env, success_term, diversity_summary

        def run(env, success_term, rate_limiter, diversity_summary):
            recorded = 0
            success_count = 0
            should_reset = False
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
                pos_sensitivity = (
                    args_cli.pos_sensitivity
                    if args_cli.pos_sensitivity is not None
                    else spec.sphere.keyboard_pos_sensitivity
                )
                rot_sensitivity = (
                    args_cli.rot_sensitivity
                    if args_cli.rot_sensitivity is not None
                    else spec.sphere.keyboard_rot_sensitivity
                )
                step_hz = args_cli.step_hz or spec.sphere.keyboard_step_hz
                smoothing_alpha = args_cli.action_smoothing_alpha
                if smoothing_alpha is None:
                    smoothing_alpha = spec.sphere.keyboard_action_smoothing_alpha
            else:
                pos_sensitivity = (
                    args_cli.pos_sensitivity
                    if args_cli.pos_sensitivity is not None
                    else spec.sphere.other_pos_sensitivity
                )
                rot_sensitivity = (
                    args_cli.rot_sensitivity
                    if args_cli.rot_sensitivity is not None
                    else spec.sphere.other_rot_sensitivity
                )
                step_hz = args_cli.step_hz or spec.sphere.other_step_hz
                smoothing_alpha = args_cli.action_smoothing_alpha
                if smoothing_alpha is None:
                    smoothing_alpha = spec.sphere.other_action_smoothing_alpha

            teleop, controls_text, pos_sensitivity, rot_sensitivity = build_teleop_device(
                args_cli.teleop_device,
                sim_device=env.device,
                pos_sensitivity=pos_sensitivity,
                rot_sensitivity=rot_sensitivity,
                request_reset=request_reset,
            )

            smoother = ActionSmoother(smoothing_alpha)
            start_with_closed_gripper = spec.key.startswith("place_cup")
            rate_limiter.hz = step_hz
            rate_limiter.sleep_duration = 1.0 / step_hz
            rate_limiter.render_period = min(0.033, rate_limiter.sleep_duration)

            window = EmptyWindow(env, f"Task Demo Recorder — {spec.key}")
            label_ref = [None]
            with window.ui_window_elements["main_vstack"]:
                label_ref[0] = ui.Label(
                    f"Recorded: {recorded} demos | sphere r={args_cli.sphere_radius:.3f}m"
                )

            def update_label():
                in_sphere = env.extras.get("_sphere_entered", False)
                min_dist = env.extras.get("_sphere_min_dist", float("inf"))
                min_dist_text = "inf" if min_dist == float("inf") else f"{min_dist:.3f}m"
                cup_vector_text = ee_to_cup_vector_text(env) if show_ee_cup_vector else ""
                label_ref[0].text = (
                    f"Recorded: {recorded} demos | "
                    f"Recording: {'YES ●' if in_sphere else 'no ○'} | "
                    f"In sphere: {'YES' if in_sphere else 'no'} | "
                    f"r={args_cli.sphere_radius:.3f}m | "
                    f"min_dist={min_dist_text}"
                    f"{cup_vector_text}"
                )

            env.sim.reset()
            env.reset()
            reset_teleop_device(teleop, close_gripper=start_with_closed_gripper)
            smoother.reset()

            print(
                f"\n[TaskSuite] Spec={spec.key} device={args_cli.teleop_device} "
                f"pos_sens={pos_sensitivity:.3f} rot_sens={rot_sensitivity:.3f}"
            )
            print(f"[TaskSuite] Controls: {controls_text}")
            print(f"[TaskSuite] Action smoothing alpha={smoothing_alpha:.2f}")
            print(
                f"[TaskSuite] Sphere radius={args_cli.sphere_radius}m "
                f"object_half_height={args_cli.object_half_height}m\n"
            )
            print(
                "[TaskSuite] Sphere gate active: not recording yet. "
                "Recording starts when the gripper enters the target sphere."
            )
            print(
                "[TaskSuite] Diversity:"
                f" bucket={diversity_summary['bucket_preset']}#{diversity_summary['bucket_index']}"
                f" x={diversity_summary['object_x_range']}"
                f" y={diversity_summary['object_y_range']}"
                f" yaw={diversity_summary['object_yaw_range']}"
                f" light={diversity_summary['light_intensity']:.1f}"
            )

            with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
                while simulation_app.is_running():
                    action = smoother.filter(teleop.advance())
                    actions = action.repeat(env.num_envs, 1)

                    env.step(actions)
                    line_visualizer.update(env)
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
                        reset_teleop_device(teleop, close_gripper=start_with_closed_gripper)
                        smoother.reset()
                        success_count = 0
                        print("[TaskSuite] Reset: not recording yet. Waiting for sphere entry.")
                        should_reset = False

                    rate_limiter.sleep(env)

            line_visualizer.clear()
            return recorded

        output_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
        output_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
        os.makedirs(output_dir, exist_ok=True)

        env, success_term, diversity_summary = build_env(output_dir, output_filename)
        initial_hz = args_cli.step_hz or (
            spec.sphere.keyboard_step_hz if args_cli.teleop_device.lower() == "keyboard" else spec.sphere.other_step_hz
        )
        rate_limiter = RateLimiter(initial_hz)

        recorded = run(env, success_term, rate_limiter, diversity_summary)
        env.close()
        print(f"\n[TaskSuite] Session complete — {recorded} demos saved to {args_cli.dataset_file}")
    finally:
        simulation_app.close()
