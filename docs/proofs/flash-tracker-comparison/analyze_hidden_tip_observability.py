"""Test how much a partly visible elongated object constrains its hidden end.

The RGB-D branch uses only the current segmented depth, a length learned from
the first view, and a prior gripper orientation to identify which end is which.
It is compared with Point2Pose, not with independently labeled tip truth.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_multiview import frame_points, move


def depth_end_hypothesis(
    points: np.ndarray, length_m: float, prior_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a visible center axis, then extend from the exposed axial end.

    The outer depth points lie on the visible surface, not necessarily the
    cylinder centerline. This is deliberately a diagnostic, not a pose update.
    """
    centered = points - points.mean(axis=0)
    raw_axis = np.linalg.eigh(np.cov(centered.T))[1][:, -1]
    axial = centered @ raw_axis
    lower, upper = np.quantile(axial, (0.2, 0.8))
    middle = points[(axial >= lower) & (axial <= upper)]
    values, directions = np.linalg.eigh(np.cov(middle.T))
    axis = directions[:, -1]
    if axis @ prior_axis < 0:
        axis = -axis
    center = middle.mean(axis=0)
    observed_end = center + (np.quantile(points @ axis, 0.05) - center @ axis) * axis
    return observed_end, observed_end + length_m * axis, axis, float(values[-1] / values[-2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()

    root = Path(__file__).parent
    traces = np.load(root / "pose_traces.npz")
    p2p = np.load(root / "p2p_dowel_internal_control_100_550.npz")
    base_from_camera = traces["base_from_camera"]
    camera_from_base = np.linalg.inv(base_from_camera)
    ee = traces["demo_ee_pose_base"]
    grasp_frame = int(traces["rigid_grasp_seed_frame"])
    reference_frame = int(traces["grasp_reference_frame"])
    reference_pose = p2p["poses"][reference_frame - 100]
    reference_center = traces["dowel_reference_center_camera_m"]
    reference_axis = traces["dowel_reference_shaft_camera"]
    near_offset, far_offset = traces["dowel_reference_end_offsets_m"]
    reference_near = reference_center + near_offset * reference_axis
    reference_tip = reference_center + far_offset * reference_axis
    length_m = float(far_offset - near_offset)
    rows = []
    for frame in range(400, 501):
        if p2p["lost"][frame - 100]:
            continue
        points, _ = frame_points(
            args.dataset, args.masks, "dowel", frame, traces["intrinsics"], edge_limit_m=0.01
        )
        if len(points) < 3:
            continue
        gripper_pose = (
            camera_from_base @ ee[frame] @ np.linalg.inv(ee[grasp_frame]) @ base_from_camera @ reference_pose
        )
        gripper_axis = gripper_pose[:3, :3] @ reference_axis
        near, tip, depth_axis, axis_eigenratio = depth_end_hypothesis(points, length_m, gripper_axis)
        direct = p2p["poses"][frame - 100]
        direct_near = move(reference_near[None], direct)[0]
        direct_tip = move(reference_tip[None], direct)[0]
        gripper_tip = move(reference_tip[None], gripper_pose)[0]
        direct_axis = direct[:3, :3] @ reference_axis
        rows.append(
            (
                frame,
                len(points),
                axis_eigenratio,
                np.degrees(np.arccos(np.clip(depth_axis @ direct_axis, -1, 1))),
                np.linalg.norm(near - direct_near) * 1000,
                np.linalg.norm(tip - direct_tip) * 1000,
                np.linalg.norm(gripper_tip - direct_tip) * 1000,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "depth_points",
                "middle_axis_eigenratio",
                "depth_to_p2p_axis_deg",
                "depth_to_p2p_visible_end_mm",
                "depth_to_p2p_hidden_tip_mm",
                "gripper_to_p2p_hidden_tip_mm",
            )
        )
        writer.writerows(rows)
    figure, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, constrained_layout=True)
    frames = [row[0] for row in rows]
    axes[0].plot(frames, [row[5] for row in rows], label="RGB-D end hypothesis vs Point2Pose")
    axes[0].plot(frames, [row[6] for row in rows], label="Gripper prediction vs Point2Pose")
    axes[0].set(ylabel="Hidden-tip disagreement (mm)", ylim=(0, 55))
    axes[0].legend()
    axes[1].plot(frames, [row[2] for row in rows], color="#9a62bb")
    axes[1].set(xlabel="Demo frame", ylabel="Visible-axis eigenvalue ratio")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(args.figure, dpi=160)
    plt.close(figure)
    for label, index in (
        ("depth axis gap deg", 3),
        ("visible end gap mm", 4),
        ("hidden tip gap mm", 5),
        ("gripper tip gap mm", 6),
    ):
        values = np.array([row[index] for row in rows])
        print(
            label,
            "median",
            round(float(np.median(values)), 2),
            "p90",
            round(float(np.quantile(values, 0.9)), 2),
        )


if __name__ == "__main__":
    main()
