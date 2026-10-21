"""Replay end-exchange canonicalization across the recorded dowel rotation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from lerobot.policies.flash.symmetry import AxialSymmetry


def turn_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def attachment_disagreement(
    first: np.ndarray, second: np.ndarray, first_ee: np.ndarray, second_ee: np.ndarray, center: np.ndarray
) -> tuple[float, float, float]:
    first_attachment = np.linalg.inv(first_ee) @ first
    second_attachment = np.linalg.inv(second_ee) @ second
    relative = np.linalg.inv(first_attachment) @ second_attachment
    origin_shift_mm = float(np.linalg.norm(relative[:3, 3]) * 1000)
    first_center = first_attachment[:3, :3] @ center + first_attachment[:3, 3]
    second_center = second_attachment[:3, :3] @ center + second_attachment[:3, 3]
    center_shift_mm = float(np.linalg.norm(first_center - second_center) * 1000)
    return origin_shift_mm, center_shift_mm, turn_deg(first_attachment, second_attachment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, default=Path(__file__).with_name("pose_traces.npz"))
    parser.add_argument(
        "--point-trace", type=Path, default=Path(__file__).with_name("p2p_dowel_internal_control_100_550.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("symmetry_branch_300_400.csv")
    )
    args = parser.parse_args()

    with np.load(args.recording) as recording, np.load(args.point_trace) as point_trace:
        base_from_camera = recording["base_from_camera"].copy()
        center = recording["dowel_reference_center_camera_m"].copy()
        axis = recording["dowel_reference_shaft_camera"].copy()
        ee_poses = recording["demo_ee_pose_base"].copy()
        frames = point_trace["frames"].copy()
        point_poses = point_trace["poses"].copy()
        lost = point_trace["lost"].copy()

    symmetry = AxialSymmetry("held", axis, center, base_from_camera, ends_interchangeable=True)
    rows = []
    poses: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    previous_raw = previous_fixed = previous_continuous = None
    start_axis = base_from_camera[:3, :3] @ symmetry.axis_object
    for frame, camera_motion, missing in zip(frames, point_poses, lost, strict=True):
        if frame < 300 or frame > 400 or missing:
            continue
        raw = base_from_camera @ camera_motion
        fixed = symmetry.canonicalize(raw)
        continuous = symmetry.canonicalize(raw, reference_pose=previous_continuous)
        observed_axis = raw[:3, :3] @ symmetry.axis_object
        rows.append(
            {
                "frame": int(frame),
                "baseline_axis_dot": float(np.dot(start_axis, observed_axis)),
                "raw_step_deg": turn_deg(previous_raw, raw) if previous_raw is not None else 0.0,
                "fixed_reference_step_deg": (
                    turn_deg(previous_fixed, fixed) if previous_fixed is not None else 0.0
                ),
                "continuous_step_deg": (
                    turn_deg(previous_continuous, continuous) if previous_continuous is not None else 0.0
                ),
                "fixed_correction_deg": turn_deg(raw, fixed),
                "continuous_correction_deg": turn_deg(raw, continuous),
            }
        )
        poses[int(frame)] = (raw, fixed, continuous)
        previous_raw, previous_fixed, previous_continuous = raw, fixed, continuous

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for frame in (360, 370, 380):
        print(next(row for row in rows if row["frame"] == frame))
    print(
        "largest fixed/continuous step",
        max(row["fixed_reference_step_deg"] for row in rows),
        max(row["continuous_step_deg"] for row in rows),
    )
    for name, index in (("raw", 0), ("fixed", 1), ("continuous", 2)):
        print(
            name,
            "340→380 attachment origin/center mm, rotation deg",
            attachment_disagreement(
                poses[340][index], poses[380][index], ee_poses[340], ee_poses[380], center
            ),
        )
    print(args.output)


if __name__ == "__main__":
    main()
