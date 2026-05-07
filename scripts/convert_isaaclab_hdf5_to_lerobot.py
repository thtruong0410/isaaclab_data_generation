#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Convert Isaac Lab HDF5 demo files to GR00T-flavored LeRobot v2 format.

The output dataset is ready for fine-tuning with `gr00t/experiment/launch_finetune.py`.

Expected HDF5 structure (produced by our pick-cup mimic pipeline):
  data/demo_N/actions              (T, 7)  float32  [dx,dy,dz,drx,dry,drz,gripper]
  data/demo_N/obs/joint_pos        (T, 9)  float32  [7 arm + 2 finger joints]
  data/demo_N/obs/front_cam        (T, 224, 224, 3) uint8
  data/demo_N/obs/side_cam         (T, 224, 224, 3) uint8
  data/demo_N/obs/wrist_cam        (T, 224, 224, 3) uint8

Output (GR00T LeRobot v2):
  <output_dir>/
    data/chunk-000/episode_000000.parquet   ← state + action per step
    videos/chunk-000/
      observation.images.front/episode_000000.mp4
      observation.images.side/episode_000000.mp4
      observation.images.wrist/episode_000000.mp4
    meta/{info,episodes,tasks,modality}.json(l)

Usage:
    # 1. Merge individual loop demos first (if not already merged):
    python /home/eric-nguyen/IsaacLab/scripts/tools/merge_hdf5_datasets.py \\
        --input_files demo_data/option_b/mimic_pick/pick_demos_mimic.hdf5 \\
        --output_file demo_data/option_b/mimic_pick/merged.hdf5

    # 2. Convert:
    uv run python scripts/convert_isaaclab_hdf5_to_lerobot.py \\
        --input  demo_data/option_b/mimic_pick/pick_demos_mimic.hdf5 \\
        --output demo_data/lerobot/franka_pick_cup \\
        --task   "pick up the cup"

    # 3. Compute dataset statistics (must match modality config):
    uv run python gr00t/data/stats.py demo_data/lerobot/franka_pick_cup new_embodiment
"""

import argparse
import json
import shutil
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Convert Isaac Lab HDF5 → GR00T LeRobot v2")
parser.add_argument("--input",  required=True,
                    help="Path to Isaac Lab HDF5 file (merged multi-demo)")
parser.add_argument("--output", required=True,
                    help="Output directory for the LeRobot v2 dataset")
parser.add_argument("--task",   default="pick up the cup",
                    help="Task description string written into tasks.jsonl")
parser.add_argument("--fps",    type=int, default=30,
                    help="Frames-per-second for output MP4 files (default: 30)")
parser.add_argument("--crf",    type=int, default=23,
                    help="H.264 CRF quality (lower = better, default: 23)")
args = parser.parse_args()

# Camera streams: (HDF5 obs key, LeRobot video stream name)
CAMERAS = [
    ("front_cam", "front"),
    ("side_cam",  "side"),
    ("wrist_cam", "wrist"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_demo(f: h5py.File, demo_key: str):
    """Return (actions, joint_pos, camera_frames_dict) for one demo.

    camera_frames_dict: {stream_name: np.ndarray(T, H, W, 3) uint8} | None per key
    """
    demo = f[f"data/{demo_key}"]
    actions = demo["actions"][:]        # (T, 7) float32

    obs = demo["obs"]
    if "joint_pos" not in obs:
        raise KeyError(
            f"'joint_pos' not found in obs of {demo_key}. "
            f"Keys present: {list(obs.keys())}"
        )
    joint_pos = obs["joint_pos"][:]  # (T, 9)

    frames = {}
    for hdf5_key, stream_name in CAMERAS:
        if hdf5_key in obs:
            frames[stream_name] = obs[hdf5_key][:]  # (T, H, W, 3) uint8
        else:
            frames[stream_name] = None

    return actions, joint_pos, frames


def encode_mp4(frames: np.ndarray, path: Path, fps: int, crf: int):
    """Encode (T, H, W, 3) uint8 array to H.264 MP4 using PyAV."""
    T, H, W, _ = frames.shape
    path.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width   = W
    stream.height  = H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "fast"}

    for t in range(T):
        frame = av.VideoFrame.from_ndarray(frames[t], format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert():
    input_path  = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists():
        print(f"[warn] '{output_path}' already exists — overwriting.")
        shutil.rmtree(output_path)

    # Create directory tree
    (output_path / "data" / "chunk-000").mkdir(parents=True)
    (output_path / "meta").mkdir(parents=True)
    for _, stream_name in CAMERAS:
        (output_path / "videos" / "chunk-000" / f"observation.images.{stream_name}").mkdir(
            parents=True
        )

    with h5py.File(input_path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        print(f"Found {len(demo_keys)} demos in {input_path.name}")

        global_idx   = 0
        episode_meta = []

        for ep_idx, demo_key in enumerate(demo_keys):
            print(f"  [{ep_idx+1:4d}/{len(demo_keys)}] {demo_key}", end="", flush=True)

            try:
                actions, joint_pos, cam_frames = read_demo(f, demo_key)
            except Exception as exc:
                print(f"  [skip] {exc}")
                continue

            T = len(actions)

            # ── Parquet (state + action) ─────────────────────────────────────
            rows = []
            for t in range(T):
                rows.append({
                    "observation.state":   joint_pos[t].tolist(),  # 9-D
                    "action":              actions[t].tolist(),     # 7-D
                    "timestamp":           t / args.fps,
                    "annotation.human.action.task_description": 0,
                    "task_index":          0,
                    "annotation.human.validity": 1,
                    "episode_index":       ep_idx,
                    "index":               global_idx + t,
                    "next.reward":         1.0 if t == T - 1 else 0.0,
                    "next.done":           (t == T - 1),
                })
            pd.DataFrame(rows).to_parquet(
                str(output_path / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"),
                index=False,
            )

            # ── MP4 videos ───────────────────────────────────────────────────
            missing = []
            for stream_name, frames in cam_frames.items():
                if frames is None:
                    missing.append(stream_name)
                    continue
                encode_mp4(
                    frames,
                    output_path / "videos" / "chunk-000"
                    / f"observation.images.{stream_name}"
                    / f"episode_{ep_idx:06d}.mp4",
                    args.fps,
                    args.crf,
                )
            if missing:
                print(f"  [warn] missing cameras: {missing}", end="")

            episode_meta.append({
                "episode_index": ep_idx,
                "tasks":         [args.task],
                "length":        T,
            })
            global_idx += T
            print(f"  T={T}")

    num_ep = len(episode_meta)
    print(f"\nConverted {num_ep} episodes, {global_idx} total steps.")

    # ── meta/tasks.jsonl ─────────────────────────────────────────────────────
    with open(output_path / "meta" / "tasks.jsonl", "w") as fp:
        fp.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

    # ── meta/episodes.jsonl ──────────────────────────────────────────────────
    with open(output_path / "meta" / "episodes.jsonl", "w") as fp:
        for ep in episode_meta:
            fp.write(json.dumps(ep) + "\n")

    # ── meta/info.json ───────────────────────────────────────────────────────
    video_paths = {
        f"observation.images.{sn}": (
            f"videos/chunk-{{episode_chunk:03d}}/observation.images.{sn}"
            f"/episode_{{episode_index:06d}}.mp4"
        )
        for _, sn in CAMERAS
    }
    info = {
        "codebase_version": "v2.0",
        "robot_type":       "franka",
        "fps":              args.fps,
        "total_episodes":   num_ep,
        "total_frames":     global_idx,
        "chunks_size":      1000,
        "splits":           {"train": f"0:{num_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": list(video_paths.values())[0],  # primary path (loader uses modality.json)
        "features": {
            "observation.state": {"dtype": "float32", "shape": [9]},
            "action":            {"dtype": "float32", "shape": [7]},
            **{k: {"dtype": "video", "shape": [224, 224, 3]} for k in video_paths},
        },
    }
    with open(output_path / "meta" / "info.json", "w") as fp:
        json.dump(info, fp, indent=2)

    # ── meta/modality.json ───────────────────────────────────────────────────
    # Must match franka_pick_cup_gr00t_config.py modality keys exactly.
    modality = {
        "state": {
            "arm":     {"start": 0, "end": 7},   # 7 Franka arm joints
            "gripper": {"start": 7, "end": 9},   # 2 finger joints
        },
        "action": {
            "arm":     {"start": 0, "end": 6},   # 6-D delta EEF pose
            "gripper": {"start": 6, "end": 7},   # 1-D binary gripper
        },
        "video": {
            sn: {"original_key": f"observation.images.{sn}"}
            for _, sn in CAMERAS
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
    with open(output_path / "meta" / "modality.json", "w") as fp:
        json.dump(modality, fp, indent=2)

    print(f"\nDataset ready at: {output_path}")
    print("\nNext — compute statistics (required before training):")
    print(f"  uv run python gr00t/data/stats.py {output_path} new_embodiment")


if __name__ == "__main__":
    convert()
