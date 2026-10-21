"""Compare a candidate's contact-end motion with a rigid-grasp prediction.

This is a kinematic consistency check, not an assertion that no slip occurred.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-frame", required=True, type=int)
    parser.add_argument("--to-frame", required=True, type=int)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=POSE_TRACE.npz")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.to_frame <= args.from_frame:
        parser.error("to-frame must follow from-frame")
    with np.load(Path(__file__).with_name("pose_traces.npz")) as reference:
        tip = (
            reference["dowel_reference_center_camera_m"]
            + reference["dowel_reference_end_offsets_m"][1] * reference["dowel_reference_shaft_camera"]
        )
        base_from_camera = reference["base_from_camera"].copy()
        ee = reference["demo_ee_pose_base"].copy()
    if args.to_frame >= len(ee):
        parser.error("recorded end-effector poses do not cover these frames")
    camera_from_base = np.linalg.inv(base_from_camera)
    gripper_motion = (
        camera_from_base @ ee[args.to_frame] @ np.linalg.inv(ee[args.from_frame]) @ base_from_camera
    )
    rows = []
    for specification in args.candidate:
        name, separator, filename = specification.partition("=")
        if not separator or not name or not filename:
            parser.error("candidate must be NAME=POSE_TRACE.npz")
        with np.load(filename) as trace:
            frames = trace["frames"]
            indices = [np.flatnonzero(frames == frame) for frame in (args.from_frame, args.to_frame)]
            if any(len(index) != 1 for index in indices):
                continue
            i, j = (int(index[0]) for index in indices)
            if trace["lost"][i] or trace["lost"][j]:
                continue
            initial = trace["poses"][i].copy()
            current = trace["poses"][j].copy()
        predicted = gripper_motion @ initial
        first_tip = initial[:3, :3] @ tip + initial[:3, 3]
        current_tip = current[:3, :3] @ tip + current[:3, 3]
        predicted_tip = predicted[:3, :3] @ tip + predicted[:3, 3]
        rows.append(
            (
                name,
                args.from_frame,
                args.to_frame,
                float(np.linalg.norm(current_tip - first_tip) * 1000),
                float(np.linalg.norm(predicted_tip - first_tip) * 1000),
                float(np.linalg.norm(current_tip - predicted_tip) * 1000),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "candidate",
                "from_frame",
                "to_frame",
                "visual_pose_end_step_mm",
                "rigid_grasp_end_step_mm",
                "visual_vs_rigid_end_gap_mm",
            )
        )
        writer.writerows(rows)
    print(args.output, "valid candidates", len(rows))


if __name__ == "__main__":
    main()
