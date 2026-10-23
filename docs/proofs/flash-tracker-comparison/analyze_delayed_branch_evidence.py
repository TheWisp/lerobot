"""Carry both frame-498 dowel pose branches into later causal RGB-D frames.

The two candidate poses differ at the conflict frame. Both are propagated
using the *same* independently tracked relative point motion, so only their
different initial pose is compared. Each score uses observations no later
than that row's frame. The confirmed map uses only earlier source views.
This tests delayed disambiguation, not true hidden-tip accuracy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from analyze_multiview import build_models, frame_points, move, surface_p90_mm

from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.pose_disagreement import free_space_support, surface_support


def pareto(direct: tuple[float, float, float], point: tuple[float, float, float]) -> str:
    """Lower surface/contradiction and higher mask coverage are preferable."""
    if not np.isfinite((*direct, *point)).all():
        return "unscored"
    direct_better = direct[0] <= point[0] and direct[1] >= point[1] and direct[2] <= point[2]
    point_better = point[0] <= direct[0] and point[1] >= direct[1] and point[2] <= direct[2]
    if direct_better and not point_better:
        return "direct_favored"
    if point_better and not direct_better:
        return "point_favored"
    return "tradeoff"


def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--point-trace", type=Path, default=root / "points_dowel_430_510_scale1.npz")
    parser.add_argument("--direct-trace", type=Path, default=root / "p2p_dowel_internal_control_100_550.npz")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conflict-frame", type=int, default=498)
    parser.add_argument("--last-frame", type=int, default=504)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.005)
    parser.add_argument("--max-local-depth-span-m", type=float, default=0.01)
    parser.add_argument("--multiview-support-radius-m", type=float, default=0.003)
    args = parser.parse_args()
    if args.last_frame < args.conflict_frame:
        parser.error("last frame must follow conflict frame")
    with np.load(args.fixture) as fixture:
        if int(fixture["frame_id"]) != args.conflict_frame:
            raise ValueError("fixture does not describe the requested conflict frame")
        intrinsics = fixture["intrinsics"]
        service_model = fixture["dowel_points"]
        direct_pose = fixture["dowel_direct_pose"]
        point_pose = fixture["dowel_motion"] @ fixture["dowel_seed_pose"]
    with np.load(args.point_trace) as track:
        motion_by_frame = {
            int(frame): motion for frame, motion in zip(track["frames"], track["motions"], strict=True)
        }
    motion_at_conflict = motion_by_frame.get(args.conflict_frame)
    if motion_at_conflict is None or not np.isfinite(motion_at_conflict).all():
        raise ValueError("no valid point motion at conflict frame")
    with np.load(args.direct_trace) as direct:
        _, confirmed_model, _ = build_models(
            args.dataset,
            args.masks,
            "dowel",
            intrinsics,
            direct["poses"],
            direct["lost"],
            args.multiview_support_radius_m,
        )
    models = {"service_first_view": service_model, "confirmed_prior_views": confirmed_model}
    rows = []
    for frame_id in range(args.conflict_frame, args.last_frame + 1):
        motion = motion_by_frame.get(frame_id)
        if motion is None or not np.isfinite(motion).all():
            raise ValueError(f"no valid causal point motion at frame {frame_id}")
        relative = motion @ np.linalg.inv(motion_at_conflict)
        poses = {"direct": relative @ direct_pose, "point": relative @ point_pose}
        raw_depth = cv2.imread(str(args.dataset / "depth" / f"{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        raw_mask = cv2.imread(str(args.masks / "dowel" / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if raw_depth is None or raw_mask is None or raw_depth.shape != raw_mask.shape:
            raise ValueError(f"missing aligned depth or mask at frame {frame_id}")
        mask = raw_mask > 0
        observed, observed_count = frame_points(
            args.dataset,
            args.masks,
            "dowel",
            frame_id,
            intrinsics,
            edge_limit_m=args.max_local_depth_span_m,
        )
        frame = RGBDFrame(
            frame_id,
            np.zeros((*raw_depth.shape, 3), dtype=np.uint8),
            raw_depth,
            intrinsics,
            0.001,
            np.eye(4),
            frame_id / 30,
            frame_id / 30,
            "recorded-camera",
        )
        for model_name, model in models.items():
            scores = {}
            for candidate_name, pose in poses.items():
                support = surface_support(frame, mask, model, pose)
                free = free_space_support(
                    frame,
                    model,
                    pose,
                    depth_tolerance_m=args.depth_tolerance_m,
                    max_local_depth_span_m=args.max_local_depth_span_m,
                )
                scores[candidate_name] = (
                    surface_p90_mm(observed, move(model, pose)),
                    support.projected_point_mask_fraction,
                    free.contradiction_fraction if free.contradiction_fraction is not None else float("nan"),
                    free.valid_depth_pixels,
                )
            direct_score, point_score = scores["direct"], scores["point"]
            rows.append(
                (
                    frame_id,
                    model_name,
                    len(model),
                    observed_count,
                    *direct_score,
                    *point_score,
                    pareto(direct_score[:3], point_score[:3]),
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "model",
                "reference_points",
                "observed_depth_points",
                "direct_observed_surface_p90_mm",
                "direct_projected_mask_fraction",
                "direct_free_space_contradiction_fraction",
                "direct_valid_projected_depth_pixels",
                "point_observed_surface_p90_mm",
                "point_projected_mask_fraction",
                "point_free_space_contradiction_fraction",
                "point_valid_projected_depth_pixels",
                "pareto_verdict",
            )
        )
        writer.writerows(rows)
    for row in rows:
        print(row[0], row[1], row[-1], "surface_mm", round(row[4], 2), round(row[8], 2))


if __name__ == "__main__":
    main()
