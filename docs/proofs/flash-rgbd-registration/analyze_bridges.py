"""Compare motion continuity with same-sensor surface alignment, separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from compare import RECORDING, frame_data
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-reference", type=Path, required=True)
    args = parser.parse_args()
    traces = {path.name: np.load(path / "trace.npz") for path in args.results}
    first = next(iter(traces.values()))
    frames, source = first["frames"], first["reference_points"]
    if any(not np.array_equal(frames, data["frames"]) for data in traces.values()):
        raise ValueError("comparison intervals must be identical")
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    points = [frame_data(int(frame), k)["obs"]["points"] for frame in frames]
    depth_axes = np.array([np.linalg.svd(p - p.mean(0), full_matrices=False)[2][0] for p in points])
    source_axis = np.linalg.svd(source - source.mean(0), full_matrices=False)[2][0]
    result = {
        "warning": "Continuity is not accuracy. Depth agreement uses the same sensor as fitting. No independent pose ground truth.",
        "intervals": {"held_review": [350, 514], "clear_held": [360, 399], "stationary": [560, 712]},
        "methods": {},
    }
    methods = {name: data["poses"] for name, data in traces.items()}
    methods["production_arbitration"] = list(traces.values())[-1]["runtime_poses"]
    motion_reference = np.load(args.motion_reference / "trace.npz")
    if not np.array_equal(frames, motion_reference["frames"]):
        raise ValueError("motion reference must cover the same frames")
    methods["motion_prediction_only"] = motion_reference["prior_poses"]
    for name, poses in methods.items():
        valid = np.isfinite(poses).all((1, 2))
        axes = poses[:, :3, :3] @ source_axis
        centers = poses[:, :3, :3] @ source.mean(0) + poses[:, :3, 3]
        angles = np.degrees(np.arccos(np.clip(abs(np.sum(axes * depth_axes, axis=1)), 0, 1)))
        held = np.flatnonzero((frames[1:] >= 350) & (frames[1:] <= 514) & valid[:-1] & valid[1:])
        vectors = (
            Rotation.from_matrix(poses[held + 1, :3, :3] @ poses[held, :3, :3].transpose(0, 2, 1)).as_rotvec()
            * 180
            / np.pi
        )
        twist = np.abs(np.sum(vectors * axes[held], axis=1))
        axis_steps = np.degrees(np.arccos(np.clip(np.sum(axes[held + 1] * axes[held], axis=1), -1, 1)))
        clear = (frames >= 360) & (frames <= 399) & valid
        static = np.flatnonzero((frames >= 560) & valid)
        errors = []
        for i in np.flatnonzero(clear):
            surface = source @ poses[i, :3, :3].T + poses[i, :3, 3]
            errors.append(float(np.median(cKDTree(surface).query(points[i])[0]) * 1000))
        held_frames = np.flatnonzero((frames >= 350) & (frames <= 514) & valid)
        relative = (
            motion_reference["prior_poses"][held_frames, :3, :3].transpose(0, 2, 1)
            @ poses[held_frames, :3, :3]
        )
        attachment_change = Rotation.from_matrix(relative[0].T @ relative[-1]).as_rotvec() * 180 / np.pi
        twist_change = float(attachment_change @ source_axis)
        result["methods"][name] = {
            "held_twist_step_p95_deg": float(np.percentile(twist, 95)),
            "held_twist_step_max_deg": float(twist.max()),
            "held_axis_step_p95_deg": float(np.percentile(axis_steps, 95)),
            "held_axis_step_max_deg": float(axis_steps.max()),
            "net_twist_change_relative_to_rigid_grasp_deg": twist_change,
            "net_nonaxial_change_relative_to_rigid_grasp_deg": float(
                np.linalg.norm(attachment_change - twist_change * source_axis)
            ),
            "clear_held_axis_vs_depth_median_deg": float(np.median(angles[clear])),
            "clear_held_axis_vs_depth_max_deg": float(np.max(angles[clear])),
            "clear_held_observed_to_reference_median_mm": float(np.median(errors)),
            "stationary_center_max_displacement_mm": float(
                np.linalg.norm(centers[static] - centers[static[0]], axis=1).max() * 1000
            ),
            "stationary_axis_max_change_deg": float(
                np.degrees(np.arccos(np.clip(np.abs(axes[static] @ axes[static[0]]), 0, 1))).max()
            ),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
