"""Measure held-pose continuity in recorded gripper coordinates.

The diagnostic assumes a rigid attachment over the selected interval. Real
slip, robot/camera timestamp error or calibration error can also cause drift;
this is neither a hidden-tip label nor an automatic reason to discard a pose.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, action="append", help="NAME=POSE_TRACE.npz")
    parser.add_argument("--first-frame", type=int, default=400)
    parser.add_argument("--last-frame", type=int, default=500)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.first_frame > args.last_frame:
        parser.error("invalid frame interval")

    with np.load(Path(__file__).with_name("pose_traces.npz")) as reference:
        camera_from_base = np.linalg.inv(reference["base_from_camera"])
        gripper_poses = reference["demo_ee_pose_base"].copy()
        center = reference["dowel_reference_center_camera_m"].copy()
        axis = reference["dowel_reference_shaft_camera"].copy()
        tip = center + float(reference["dowel_reference_end_offsets_m"][1]) * axis
    if args.last_frame >= len(gripper_poses):
        parser.error("the recording has no gripper pose for the last frame")

    rows = []
    seen = set()
    for specification in args.candidate:
        name, separator, filename = specification.partition("=")
        if not separator or not name or name in seen:
            parser.error("each candidate must be a unique NAME=POSE_TRACE.npz")
        seen.add(name)
        with np.load(filename) as trace:
            candidate = {
                int(frame): pose.copy()
                for frame, pose, lost in zip(trace["frames"], trace["poses"], trace["lost"], strict=True)
                if not bool(lost)
            }
        values = []
        for frame in range(args.first_frame, args.last_frame + 1):
            pose = candidate.get(frame)
            if pose is None:
                continue
            camera_from_ee = camera_from_base @ gripper_poses[frame]
            ee_from_camera = np.linalg.inv(camera_from_ee)
            contact_end_in_camera = pose[:3, :3] @ tip + pose[:3, 3]
            contact_end_in_ee = ee_from_camera[:3, :3] @ contact_end_in_camera + ee_from_camera[:3, 3]
            axis_in_ee = ee_from_camera[:3, :3] @ pose[:3, :3] @ axis
            values.append((frame, contact_end_in_ee, axis_in_ee))
        if not values:
            continue
        median_end = np.median(np.stack([value[1] for value in values]), axis=0)
        previous = None
        jumps = []
        for frame, end, direction in values:
            jump = float("nan")
            angle = float("nan")
            if previous is not None and previous[0] == frame - 1:
                jump = float(np.linalg.norm(end - previous[1]) * 1000)
                angle = float(np.degrees(np.arccos(np.clip(direction @ previous[2], -1, 1))))
                jumps.append(jump)
            rows.append(
                (
                    frame,
                    name,
                    *end,
                    *direction,
                    float(np.linalg.norm(end - median_end) * 1000),
                    jump,
                    angle,
                )
            )
            previous = frame, end, direction
        deviations = np.asarray([row[8] for row in rows if row[1] == name])
        print(
            name,
            "frames",
            len(values),
            "EE-tip deviation median/p90 mm",
            np.round(np.percentile(deviations, [50, 90]), 2),
            "consecutive tip jump p90 mm",
            round(float(np.percentile(jumps, 90)), 2) if jumps else "n/a",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "candidate",
                "tip_ee_x_m",
                "tip_ee_y_m",
                "tip_ee_z_m",
                "shaft_ee_x",
                "shaft_ee_y",
                "shaft_ee_z",
                "tip_from_median_mm",
                "consecutive_tip_jump_mm",
                "consecutive_shaft_angle_deg",
            )
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
