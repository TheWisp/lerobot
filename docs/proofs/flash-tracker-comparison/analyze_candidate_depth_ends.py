"""Compare causal dowel end predictions with a current-depth end hypothesis.

The visible depth lies on the front surface and the contact end is occluded.
The extrapolated contact point is therefore a diagnostic, not a ground-truth
label. A pregrasp Point2Pose pose transported by the recorded gripper selects
one end identity for all sources; it does not supply the insertion pose.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from analyze_hidden_tip_observability import depth_end_hypothesis
from analyze_multiview import frame_points, move


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = Path(__file__).parent
    with np.load(root / "pose_traces.npz") as reference:
        intrinsics = reference["intrinsics"].copy()
        center = reference["dowel_reference_center_camera_m"].copy()
        shaft = reference["dowel_reference_shaft_camera"].copy()
        offsets = reference["dowel_reference_end_offsets_m"].copy()
        gripper_poses = reference["demo_ee_pose_base"].copy()
        grasp_frame = int(reference["rigid_grasp_seed_frame"])
        reference_frame = int(reference["grasp_reference_frame"])
        base_from_camera = reference["base_from_camera"].copy()
    traces = {}
    for name, filename in (
        ("Point2Pose", "p2p_dowel_internal_control_100_550.npz"),
        ("TAPNext 256", "tapnext_fixed_dowel_100_550_pose.npz"),
        ("TAPNext 512", "tapnext512_fixed_dowel_100_550_pose.npz"),
    ):
        with np.load(root / filename) as trace:
            traces[name] = {
                int(frame): (pose, bool(lost))
                for frame, pose, lost in zip(trace["frames"], trace["poses"], trace["lost"], strict=True)
            }
    anchor = traces["Point2Pose"][reference_frame][0]
    camera_from_base = np.linalg.inv(base_from_camera)
    length_m = float(offsets[1] - offsets[0])
    endpoints = [center + offset * shaft for offset in offsets]
    rows = []
    for frame in range(400, 501):
        points, _ = frame_points(args.dataset, args.masks, "dowel", frame, intrinsics, edge_limit_m=0.01)
        if len(points) < 3:
            continue
        grasp_prediction = (
            camera_from_base
            @ gripper_poses[frame]
            @ np.linalg.inv(gripper_poses[grasp_frame])
            @ base_from_camera
            @ anchor
        )
        observed_end, inferred_tip, observed_axis, anisotropy = depth_end_hypothesis(
            points, length_m, grasp_prediction[:3, :3] @ shaft
        )
        for name, trace in traces.items():
            candidate = trace.get(frame)
            if candidate is None or candidate[1]:
                continue
            pose = candidate[0]
            near, tip = move(np.asarray(endpoints), pose)
            axis = pose[:3, :3] @ shaft
            rows.append(
                (
                    frame,
                    name,
                    len(points),
                    anisotropy,
                    float(np.linalg.norm(observed_end - near) * 1000),
                    float(np.linalg.norm(inferred_tip - tip) * 1000),
                    float(np.degrees(np.arccos(np.clip(observed_axis @ axis, -1, 1)))),
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "candidate",
                "current_depth_points",
                "depth_middle_axis_eigenratio",
                "visible_end_disagreement_mm",
                "extrapolated_hidden_end_disagreement_mm",
                "unoriented_shaft_axis_disagreement_deg",
            )
        )
        writer.writerows(rows)
    for name in traces:
        subset = [row for row in rows if row[1] == name]
        print(
            name,
            len(subset),
            "median visible/hidden/axis disagreements",
            np.median(np.asarray([row[4:] for row in subset]), axis=0),
        )


if __name__ == "__main__":
    main()
