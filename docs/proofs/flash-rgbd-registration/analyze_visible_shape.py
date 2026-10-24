"""Separate visible-shape alignment, static jitter and held-object twist changes.

The chosen intervals are review annotations of this recording, not runtime
thresholds. Depth PCA is a consistency diagnostic; gripper motion is a proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from compare import RECORDING, frame_data, reference
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.result / "trace.npz")
    frames, poses, source = data["frames"], data["poses"], data["reference_points"]
    _, _, basis = np.linalg.svd(source - source.mean(0), full_matrices=False)
    axes = np.einsum("nij,j->ni", poses[:, :3, :3], basis[0])
    centers = np.einsum("nij,j->ni", poses[:, :3, :3], source.mean(0)) + poses[:, :3, 3]
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    angles = []
    for index, frame in enumerate(frames):
        points = frame_data(int(frame), k)["obs"]["points"]
        _, _, current = np.linalg.svd(points - points.mean(0), full_matrices=False)
        angles.append(float(np.degrees(np.arccos(np.clip(abs(axes[index] @ current[0]), 0, 1)))))
    angles = np.array(angles)
    result = {
        "angle_reference": "PCA of largest connected cleaned depth patch, same as short experiment",
        "intervals": [],
    }
    for start, stop in ((280, 349), (350, 529), (360, 399), (530, 712), (560, 712)):
        selected = (frames >= start) & (frames <= stop)
        result["intervals"].append(
            {
                "start": start,
                "stop": stop,
                "median_axis_vs_depth_deg": float(np.median(angles[selected])),
                "max_axis_vs_depth_deg": float(np.max(angles[selected])),
            }
        )
    selected = np.flatnonzero(frames >= 560)
    result["stationary_after_release"] = {
        "start": 560,
        "stop": 712,
        "max_center_displacement_mm": float(
            np.max(np.linalg.norm(centers[selected] - centers[selected[0]], axis=1)) * 1000
        ),
        "max_axis_change_deg": float(
            np.max(np.degrees(np.arccos(np.clip(np.abs(axes[selected] @ axes[selected[0]]), 0, 1))))
        ),
    }
    ee, _ = reference()
    rotations = Rotation.from_matrix(poses[1:, :3, :3] @ poses[:-1, :3, :3].transpose(0, 2, 1))
    vectors = rotations.as_rotvec() * 180 / np.pi
    twist = np.abs(np.sum(vectors * axes[:-1], axis=1))
    held = np.flatnonzero((frames[1:] >= 350) & (frames[1:] <= 529))
    result["largest_held_twist_steps"] = []
    for index in held[np.argsort(twist[held])[-5:]][::-1]:
        frame = int(frames[index + 1])
        result["largest_held_twist_steps"].append(
            {
                "frame": frame,
                "twist_component_deg": float(twist[index]),
                "axis_change_deg": float(
                    np.degrees(np.arccos(np.clip(abs(axes[index] @ axes[index + 1]), 0, 1)))
                ),
                "gripper_rotation_deg": float(
                    Rotation.from_matrix(ee[frame, :3, :3] @ ee[frame - 1, :3, :3].T).magnitude()
                    * 180
                    / np.pi
                ),
            }
        )
    result["warning"] = (
        "Neither depth consistency nor the rigid-grasp proxy is independent ground-truth pose accuracy."
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
