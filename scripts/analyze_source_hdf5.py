#!/usr/bin/env python3
"""Summarize raw IsaacLab HDF5 demos and flag distribution outliers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - user environment guard
    raise SystemExit("Missing h5py. Install once with: python3 -m pip install --user h5py") from exc

import numpy as np


def _demo_index(path: Path) -> int:
    tail = path.stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _safe_dataset(group: h5py.Group, key: str) -> np.ndarray | None:
    return group[key][()] if key in group else None


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q3": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _format_stats(name: str, values: list[float]) -> str:
    stats = _stats(values)
    return (
        f"{name:14s} min={stats['min']:.4f} p05={stats['p05']:.4f} "
        f"q1={stats['q1']:.4f} med={stats['median']:.4f} "
        f"q3={stats['q3']:.4f} p95={stats['p95']:.4f} "
        f"max={stats['max']:.4f} mean={stats['mean']:.4f} std={stats['std']:.4f}"
    )


def _iqr_bounds(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def _load_row(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        obs = demo["obs"]
        n = int(demo.attrs.get("num_samples", obs["actions"].shape[0]))
        success = bool(demo.attrs.get("success", False))

        env_args_raw = handle["data"].attrs.get("env_args", "{}")
        env_args = json.loads(env_args_raw) if isinstance(env_args_raw, str) else {}
        sim_args = env_args.get("sim_args", {})
        dt = float(sim_args.get("dt", 0.0)) * float(sim_args.get("decimation", 1.0))

        cabinet_obs = _safe_dataset(obs, "cabinet_joint_pos")
        drawer_start = drawer_final = drawer_max = np.nan
        if cabinet_obs is not None:
            drawer = cabinet_obs.reshape(n, -1)[:, 0]
            drawer_start = float(drawer[0])
            drawer_final = float(drawer[-1])
            drawer_max = float(np.max(drawer))

        rel_obs = _safe_dataset(obs, "rel_ee_drawer_distance")
        rel_start = rel_min = rel_final = np.nan
        rel_argmin = -1
        if rel_obs is not None:
            rel = rel_obs.reshape(n, -1)
            rel_norm = np.linalg.norm(rel, axis=1)
            rel_start = float(rel_norm[0])
            rel_min = float(np.min(rel_norm))
            rel_final = float(rel_norm[-1])
            rel_argmin = int(np.argmin(rel_norm))

        joint_pos = _safe_dataset(obs, "joint_pos")
        finger_start = finger_min = finger_final = np.nan
        if joint_pos is not None and joint_pos.shape[-1] >= 2:
            finger = joint_pos.reshape(n, -1)[:, -2:].max(axis=1)
            finger_start = float(finger[0])
            finger_min = float(np.min(finger))
            finger_final = float(finger[-1])

        active_joint = -1
        active_joint_delta = np.nan
        cabinet_state = _safe_dataset(demo, "states/articulation/cabinet/joint_position")
        if cabinet_state is not None:
            cabinet_state = cabinet_state.reshape(n, -1)
            delta = cabinet_state[-1] - cabinet_state[0]
            active_joint = int(np.argmax(np.abs(delta)))
            active_joint_delta = float(delta[active_joint])

        actions = demo["actions"][()].reshape(n, -1)
        action_arm_norm = np.linalg.norm(actions[:, : min(6, actions.shape[1])], axis=1)

    return {
        "file": path.name,
        "idx": _demo_index(path),
        "n": n,
        "seconds": n * dt if dt > 0 else np.nan,
        "success": success,
        "drawer_start": drawer_start,
        "drawer_final": drawer_final,
        "drawer_max": drawer_max,
        "drawer_delta": drawer_final - drawer_start,
        "rel_start": rel_start,
        "rel_min": rel_min,
        "rel_final": rel_final,
        "rel_argmin": rel_argmin,
        "finger_start": finger_start,
        "finger_min": finger_min,
        "finger_final": finger_final,
        "active_joint": active_joint,
        "active_joint_delta": active_joint_delta,
        "action_arm_mean": float(np.mean(action_arm_norm)),
        "action_arm_max": float(np.max(action_arm_norm)),
    }


def _valid_metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = float(row[metric])
        if np.isfinite(value):
            values.append(value)
    return values


def _print_summary(rows: list[dict[str, Any]], metrics: list[str]) -> None:
    print(f"files: {len(rows)}")
    print(f"success: {sum(row['success'] for row in rows)}/{len(rows)}")
    print(f"active_joint_counts: {dict(Counter(row['active_joint'] for row in rows))}")
    print()
    print("summary:")
    for metric in metrics:
        values = _valid_metric_values(rows, metric)
        if values:
            print("  " + _format_stats(metric, values))


def _print_group_summary(rows: list[dict[str, Any]], metrics: list[str]) -> None:
    print()
    print("group summary by active_joint:")
    for active_joint in sorted(set(row["active_joint"] for row in rows)):
        group = [row for row in rows if row["active_joint"] == active_joint]
        demo_ids = [row["idx"] for row in group]
        print(f"  active_joint={active_joint} count={len(group)} demos={demo_ids}")
        for metric in metrics:
            values = _valid_metric_values(group, metric)
            if values:
                stats = _stats(values)
                print(
                    f"    {metric:12s} min={stats['min']:.4f} "
                    f"med={stats['median']:.4f} max={stats['max']:.4f} mean={stats['mean']:.4f}"
                )


def _print_outliers(rows: list[dict[str, Any]], metrics: list[str]) -> None:
    print()
    print("iqr outliers within each active_joint group:")
    flagged: dict[str, list[str]] = {}
    for active_joint in sorted(set(row["active_joint"] for row in rows)):
        group = [row for row in rows if row["active_joint"] == active_joint]
        print(f"  group active_joint={active_joint}")
        for metric in metrics:
            values = _valid_metric_values(group, metric)
            if len(values) < 4:
                continue
            lo, hi = _iqr_bounds(values)
            outliers = [
                row for row in group if np.isfinite(float(row[metric])) and (float(row[metric]) < lo or float(row[metric]) > hi)
            ]
            print(f"    {metric:12s} bounds=[{lo:.4f}, {hi:.4f}] count={len(outliers)}")
            for row in outliers:
                flagged.setdefault(row["file"], []).append(metric)
                print(
                    f"      {row['file']}: {metric}={float(row[metric]):.4f} "
                    f"n={row['n']} drawer={row['drawer_final']:.4f} "
                    f"rel_min={row['rel_min']:.4f} rel_final={row['rel_final']:.4f} "
                    f"finger={row['finger_final']:.4f}"
                )
    if flagged:
        print()
        print("recommended inspect list:")
        for file_name, reasons in sorted(flagged.items(), key=lambda item: _demo_index(Path(item[0]))):
            print(f"  {file_name}: {', '.join(reasons)}")


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print()
    print("per-demo rows:")
    for row in sorted(rows, key=lambda item: item["idx"]):
        print(
            f"  {row['idx']:03d} {row['file']} "
            f"joint={row['active_joint']} n={row['n']} "
            f"drawer={row['drawer_final']:.4f} rel_min={row['rel_min']:.4f} "
            f"rel_final={row['rel_final']:.4f} finger={row['finger_final']:.4f} "
            f"success={row['success']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path, help="Folder containing one-demo raw HDF5 files.")
    parser.add_argument("--prefix", default="", help="Optional filename prefix, e.g. open_drawer_sphere_demo.")
    parser.add_argument("--list", action="store_true", help="Print one row per demo.")
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    pattern = f"{args.prefix}_*.hdf5" if args.prefix else "*.hdf5"
    files = sorted(raw_dir.glob(pattern), key=_demo_index)
    if not files:
        raise SystemExit(f"No HDF5 files found: {raw_dir}/{pattern}")

    rows = [_load_row(path) for path in files]
    print(f"dataset: {raw_dir}")
    print(f"pattern: {pattern}")

    summary_metrics = [
        "n",
        "seconds",
        "drawer_final",
        "drawer_max",
        "drawer_delta",
        "rel_start",
        "rel_min",
        "rel_final",
        "rel_argmin",
        "finger_min",
        "finger_final",
        "action_arm_mean",
        "action_arm_max",
    ]
    group_metrics = ["n", "drawer_final", "rel_min", "rel_final", "finger_final"]

    _print_summary(rows, summary_metrics)
    _print_group_summary(rows, group_metrics)
    _print_outliers(rows, group_metrics)
    if args.list:
        _print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
