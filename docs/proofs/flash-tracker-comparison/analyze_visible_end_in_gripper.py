"""Check whether the RGB-D-visible dowel end moves inside the gripper.

The observed end is an axial percentile of the segmented depth cloud, not a
manually labeled cap center. It can move under changing occlusion even if the
physical grasp is rigid. The prior trace only resolves shaft end identity.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from analyze_hidden_tip_observability import depth_end_hypothesis
from analyze_multiview import frame_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--prior-trace", required=True, type=Path)
    parser.add_argument("--reference-frame", required=True, type=int)
    parser.add_argument("--first-frame", required=True, type=int)
    parser.add_argument("--last-frame", required=True, type=int)
    parser.add_argument("--depth-edge-limit-m", type=float, default=0.01)
    parser.add_argument("--report-frame", type=int, action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.first_frame > args.reference_frame
        or args.reference_frame > args.last_frame
        or args.depth_edge_limit_m <= 0
    ):
        parser.error("invalid frame interval or depth edge limit")

    with np.load(Path(__file__).with_name("pose_traces.npz")) as reference:
        intrinsics = reference["intrinsics"].copy()
        camera_from_base = np.linalg.inv(reference["base_from_camera"])
        gripper_poses = reference["demo_ee_pose_base"].copy()
        axis = reference["dowel_reference_shaft_camera"].copy()
        length = float(np.diff(reference["dowel_reference_end_offsets_m"])[0])
    with np.load(args.prior_trace) as prior:
        poses = {
            int(frame): pose.copy()
            for frame, pose, lost in zip(prior["frames"], prior["poses"], prior["lost"], strict=True)
            if not bool(lost)
        }
    if args.last_frame >= len(gripper_poses) or args.reference_frame not in poses:
        parser.error("missing recorded gripper or prior pose")

    rows = []
    for frame in range(args.first_frame, args.last_frame + 1):
        pose = poses.get(frame)
        if pose is None:
            continue
        points, _ = frame_points(
            args.dataset,
            args.masks,
            "dowel",
            frame,
            intrinsics,
            edge_limit_m=args.depth_edge_limit_m,
        )
        if len(points) < 3:
            continue
        observed_end, _, observed_axis, eigenratio = depth_end_hypothesis(points, length, pose[:3, :3] @ axis)
        camera_from_ee = camera_from_base @ gripper_poses[frame]
        ee_from_camera = np.linalg.inv(camera_from_ee)
        end_in_ee = ee_from_camera[:3, :3] @ observed_end + ee_from_camera[:3, 3]
        axis_in_ee = ee_from_camera[:3, :3] @ observed_axis
        rows.append((frame, len(points), eigenratio, end_in_ee, axis_in_ee))
    anchor = next((row for row in rows if row[0] == args.reference_frame), None)
    if anchor is None:
        raise ValueError("the reference frame has no usable segmented depth")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "depth_points",
                "middle_axis_eigenratio",
                "visible_end_ee_x_m",
                "visible_end_ee_y_m",
                "visible_end_ee_z_m",
                "visible_axis_ee_x",
                "visible_axis_ee_y",
                "visible_axis_ee_z",
                "visible_end_gap_from_reference_mm",
                "visible_axis_gap_from_reference_deg",
            )
        )
        for frame, count, ratio, end, direction in rows:
            writer.writerow(
                (
                    frame,
                    count,
                    ratio,
                    *end,
                    *direction,
                    float(np.linalg.norm(end - anchor[3]) * 1000),
                    float(np.degrees(np.arccos(np.clip(direction @ anchor[4], -1, 1)))),
                )
            )
    for frame in dict.fromkeys((args.reference_frame, *args.report_frame)):
        selected = next((row for row in rows if row[0] == frame), None)
        if selected is not None:
            print(
                frame,
                "visible-end gap mm",
                round(float(np.linalg.norm(selected[3] - anchor[3]) * 1000), 2),
                "axis gap deg",
                round(float(np.degrees(np.arccos(np.clip(selected[4] @ anchor[4], -1, 1)))), 2),
                "axis eigenratio",
                round(float(selected[2]), 2),
            )


if __name__ == "__main__":
    main()
