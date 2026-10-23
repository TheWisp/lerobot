"""Ablate source-neutral arbitration between direct and causal point poses.

Use the current RGB-D surface, shaft silhouette and measured free space. Only
when all three favor a direct pose does it replace the point-motion anchor.
This is an offline candidate policy, not a deployed acceptance rule or 6D GT.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from analyze_multiview import frame_points, move
from analyze_ray_consistency import reference_points
from analyze_temporal_pose_conflict import score, verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument(
        "--agreement-tip-mm",
        type=float,
        default=0.0,
        help="Ablation only: reanchor a direct pose within this task-point gap; zero disables it",
    )
    args = parser.parse_args()
    if args.agreement_tip_mm < 0:
        parser.error("agreement tip distance cannot be negative")
    root = Path(__file__).parent
    features = np.load(root / "pose_traces.npz")
    p2p = np.load(root / "p2p_dowel_internal_control_100_550.npz")
    points = np.load(args.points)
    intrinsics = features["intrinsics"]
    center = features["dowel_reference_center_camera_m"]
    axis = features["dowel_reference_shaft_camera"]
    tip = center + features["dowel_reference_end_offsets_m"][1] * axis
    model = reference_points(args.dataset, args.masks, "dowel", intrinsics)
    seed = int(points["seed_frame"])
    anchor_motion = np.eye(4)
    anchor_pose = p2p["poses"][seed - 100]
    rows = []
    accepted_poses = []
    for frame, motion in zip(points["frames"], points["motions"], strict=True):
        frame = int(frame)
        if not np.isfinite(motion).all():
            break
        mask = cv2.imread(str(args.masks / "dowel" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        raw_depth = cv2.imread(str(args.dataset / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if mask is None or raw_depth is None:
            raise ValueError(f"missing mask/depth at frame {frame}")
        mask = mask > 0
        depth = raw_depth.astype(np.float32) * 0.001
        neighborhood = np.ones((3, 3), dtype=np.uint8)
        lower = cv2.erode(depth, neighborhood)
        upper = cv2.dilate(depth, neighborhood)
        valid_depth = (depth > 0) & (lower > 0) & (upper - lower <= 0.01)
        live, _ = frame_points(args.dataset, args.masks, "dowel", frame, intrinsics, edge_limit_m=0.01)
        predicted = motion @ np.linalg.inv(anchor_motion) @ anchor_pose
        predicted_score = score(predicted, model, live, mask, depth, valid_depth, intrinsics, center, axis)
        if p2p["lost"][frame - 100]:
            decision = "direct_lost"
            direct_score = (float("nan"),) * 3
            gap_mm = float("nan")
            accepted = predicted
        else:
            direct = p2p["poses"][frame - 100]
            direct_score = score(direct, model, live, mask, depth, valid_depth, intrinsics, center, axis)
            decision = verdict(direct_score, predicted_score)
            gap_mm = float(np.linalg.norm(move(tip[None], direct) - move(tip[None], predicted)) * 1000)
            if args.agreement_tip_mm > 0 and gap_mm <= args.agreement_tip_mm:
                decision = "near_agreement"
            if decision in ("direct_dominates", "near_agreement"):
                anchor_motion = motion
                anchor_pose = direct
                accepted = direct
            else:
                accepted = predicted
        accepted_poses.append(accepted)
        rows.append((frame, decision, gap_mm, *direct_score, *predicted_score))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "decision",
                "direct_to_prediction_tip_gap_mm",
                "direct_surface_p90_mm",
                "direct_image_axis_deg",
                "direct_free_space_fraction",
                "prediction_surface_p90_mm",
                "prediction_image_axis_deg",
                "prediction_free_space_fraction",
            )
        )
        writer.writerows(rows)
    np.savez_compressed(args.poses, frames=[row[0] for row in rows], poses=np.asarray(accepted_poses))
    for phase, selected in (
        ("before", [row for row in rows if row[0] <= 496]),
        ("contact", [row for row in rows if 497 <= row[0] <= 504]),
    ):
        labels = sorted({row[1] for row in selected})
        print(phase, {label: sum(row[1] == label for row in selected) for label in labels})
    for row in rows:
        if 496 <= row[0] <= 504:
            print(row[0], row[1], "tip gap", round(row[2], 2))


if __name__ == "__main__":
    main()
