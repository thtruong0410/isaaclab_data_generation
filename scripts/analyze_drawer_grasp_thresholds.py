#!/usr/bin/env python3
"""Analyze drawer MimicGen grasp thresholds from raw HDF5 demos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def _demo_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _load_manifest_labels(raw_dir: Path) -> dict[str, str]:
    manifest = raw_dir / "manifest.tsv"
    if not manifest.exists():
        return {}

    labels: dict[str, str] = {}
    with manifest.open("r", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            output_file = row.get("output_file")
            source_label = row.get("source_label")
            if output_file and source_label:
                labels[output_file] = source_label
    return labels


def _infer_label(path: Path) -> str:
    parent_name = path.parent.name.lower()
    if parent_name.endswith("_top"):
        return "top"
    if parent_name.endswith("_bottom"):
        return "bottom"

    index = _demo_index(path)
    return "top" if index % 2 == 1 else "bottom"


def _load_demo(path: Path, label: str | None = None) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        finger_max = demo["obs/joint_pos"][()][:, -2:].max(axis=1)
        rel_dist = np.linalg.norm(demo["obs/rel_ee_drawer_distance"][()], axis=-1)
        drawer_pos = demo["obs/cabinet_joint_pos"][()][:, 0]
    return {
        "path": path,
        "label": label or _infer_label(path),
        "finger_max": finger_max,
        "rel_dist": rel_dist,
        "drawer_pos": drawer_pos,
    }


def _stats(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.min():.4f}..{arr.max():.4f} avg={arr.mean():.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_dir",
        type=Path,
        nargs="?",
        default=Path("data/source/open_drawer_sphere_top_bottom"),
        help="Raw HDF5 folder to analyze.",
    )
    parser.add_argument("--prefix", default="open_drawer_sphere_demo", help="Raw demo filename prefix.")
    parser.add_argument(
        "--finger-thresholds",
        type=float,
        nargs="+",
        default=[-0.002, -0.005, -0.008, -0.010, -0.012, -0.015, -0.018, -0.020, -0.022],
    )
    parser.add_argument(
        "--distance-thresholds",
        type=float,
        nargs="+",
        default=[0.09, 0.10, 0.11, 0.12, 0.15, 0.18],
    )
    args = parser.parse_args()

    files = sorted(args.raw_dir.glob(f"{args.prefix}_*.hdf5"), key=_demo_index)
    if not files:
        raise FileNotFoundError(f"No {args.prefix}_*.hdf5 files found in {args.raw_dir}")

    manifest_labels = _load_manifest_labels(args.raw_dir)
    demos = [_load_demo(path, manifest_labels.get(path.name)) for path in files]
    print(f"raw_dir: {args.raw_dir}")
    print(f"demos: {len(demos)}")

    for label in ("top", "bottom", "all"):
        subset = demos if label == "all" else [demo for demo in demos if demo["label"] == label]
        print(f"\n[{label}] n={len(subset)}")
        print("finger start:", _stats([float(demo["finger_max"][0]) for demo in subset]))
        print("finger min:  ", _stats([float(demo["finger_max"].min()) for demo in subset]))
        print("finger final:", _stats([float(demo["finger_max"][-1]) for demo in subset]))
        print("dist min:    ", _stats([float(demo["rel_dist"].min()) for demo in subset]))
        print("dist final:  ", _stats([float(demo["rel_dist"][-1]) for demo in subset]))
        print("drawer final:", _stats([float(demo["drawer_pos"][-1]) for demo in subset]))

    print("\nTransition count by finger threshold only:")
    for finger_threshold in args.finger_thresholds:
        crossings = []
        dists = []
        for demo in demos:
            signal = demo["finger_max"] < finger_threshold
            if not bool(signal[0]) and bool(signal.any()):
                idx = int(np.where(signal)[0][0])
                crossings.append(idx)
                dists.append(float(demo["rel_dist"][idx]))
        print(
            f"finger < {finger_threshold: .3f}: {len(crossings)}/{len(demos)} "
            f"first_idx={min(crossings) if crossings else 'NA'}..{max(crossings) if crossings else 'NA'} "
            f"dist_at_cross={min(dists) if dists else float('nan'):.4f}..{max(dists) if dists else float('nan'):.4f}"
        )

    print("\nTransition count by finger + distance thresholds:")
    header = "finger".ljust(12) + " ".join(f"dist<{threshold:.2f}".rjust(10) for threshold in args.distance_thresholds)
    print(header)
    for finger_threshold in args.finger_thresholds:
        counts = []
        for distance_threshold in args.distance_thresholds:
            count = 0
            for demo in demos:
                signal = (demo["finger_max"] < finger_threshold) & (demo["rel_dist"] < distance_threshold)
                if not bool(signal[0]) and bool(signal.any()):
                    count += 1
            counts.append(count)
        print(f"{finger_threshold: .3f}".ljust(12) + " ".join(f"{count}/{len(demos)}".rjust(10) for count in counts))

    print("\nRecommended default for this dataset:")
    print("OPEN_DRAWER_MIMIC_GRIPPER_THRESHOLD=-0.01")
    print("OPEN_DRAWER_MIMIC_GRASP_DIST_THRESHOLD=0.12")
    print(
        "Reason: all demos start with grasp=false, later cross finger<-0.01, "
        "and the max distance at that crossing is about 0.09m; 0.12m leaves "
        "replay margin while still requiring gripper closure."
    )
    print(
        "Note: open-drawer MimicGen defaults to OPEN_DRAWER_MIMIC_GRASP_MODE=joint. "
        "Optional geometry/either/both modes are still available for experiments."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
