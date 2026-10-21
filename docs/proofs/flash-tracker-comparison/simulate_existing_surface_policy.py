"""Replay FLASH's current group/direct comparison on direct/point proposals.

This tests whether one existing source-neutral conflict rule can protect a
causal point track without an object-specific pose-jump threshold. It is not
yet wired into the live scene estimator and has no independent 6D labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from run_causal_points_demo import read_frame

from lerobot.policies.flash.pose_disagreement import PoseComparisonVerdict, compare_pose_proposals


def reference_points(dataset: Path, masks: Path, role: str, intrinsics: np.ndarray) -> np.ndarray:
    depth = cv2.imread(str(dataset / "depth" / "000100.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(masks / role / "000100.png"), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None:
        raise ValueError("missing first-view RGB-D")
    y, x = np.nonzero((mask > 0) & (depth > 0))
    z = depth[y, x] * 0.001
    return np.column_stack(
        ((x - intrinsics[0, 2]) * z / intrinsics[0, 0], (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    features = np.load(root / "pose_traces.npz")
    points = np.load(args.points)
    role = str(points["role"])
    if role == "dowel":
        direct = np.load(root / "p2p_dowel_internal_control_100_550.npz")
        poses, lost = direct["poses"], direct["lost"]
    elif role == "ring":
        poses, lost = features["bundlesdf_ring_delta"], features["bundlesdf_ring_failed"]
    else:
        raise ValueError(f"unsupported recorded role {role}")
    model = reference_points(args.dataset, args.masks, role, features["intrinsics"])
    center = model.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum((model - center) ** 2, axis=1))))
    fit_threshold_m = 0.005
    seed = int(points["seed_frame"])
    anchor_motion = np.eye(4)
    anchor_pose = poses[seed - 100]
    rows = []
    accepted_poses = []
    for frame, motion in zip(points["frames"], points["motions"], strict=True):
        frame = int(frame)
        if not np.isfinite(motion).all():
            break
        prediction = motion @ np.linalg.inv(anchor_motion) @ anchor_pose
        if lost[frame - 100]:
            verdict = "direct_lost"
            offset_m = float("nan")
            rotation_rad = float("nan")
            accepted = prediction
        else:
            source = read_frame(args.dataset, frame, features["intrinsics"])
            mask = cv2.imread(str(args.masks / role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"missing mask at frame {frame}")
            comparison = compare_pose_proposals(
                source,
                mask > 0,
                model,
                poses[frame - 100],
                prediction,
                max_agreement_offset_m=fit_threshold_m,
                max_agreement_angle_rad=float(np.arctan2(fit_threshold_m, radius)),
                min_masked_depth_points=12,
            )
            verdict = comparison.verdict.value
            offset_m = comparison.object_center_difference_m
            rotation_rad = comparison.rotation_difference_rad
            if comparison.verdict in (PoseComparisonVerdict.AGREE, PoseComparisonVerdict.CANDIDATE_FAVORED):
                anchor_motion = motion
                anchor_pose = poses[frame - 100]
                accepted = anchor_pose
            else:
                accepted = prediction
        accepted_poses.append(accepted)
        rows.append((frame, verdict, offset_m * 1000, np.degrees(rotation_rad)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("frame", "verdict", "object_center_difference_mm", "rotation_difference_deg"))
        writer.writerows(rows)
    np.savez_compressed(args.poses, frames=[row[0] for row in rows], poses=np.asarray(accepted_poses))
    for phase, subset in (
        ("before", [row for row in rows if row[0] <= 496]),
        ("contact", [row for row in rows if 497 <= row[0] <= 504]),
    ):
        labels = sorted({row[1] for row in subset})
        print(role, phase, {label: sum(row[1] == label for row in subset) for label in labels})
    for row in rows:
        if 496 <= row[0] <= 504:
            print(role, row[0], row[1], "center gap mm", round(row[2], 2))


if __name__ == "__main__":
    main()
