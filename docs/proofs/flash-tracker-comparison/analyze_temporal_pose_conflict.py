"""Compare a causal motion hypothesis with late direct-pose updates.

All visible-surface tests reuse the candidate sources' RGB-D stream. They can
reject a contradicted branch but cannot certify the hidden insertion tip.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from analyze_multiview import frame_points, move, surface_p90_mm
from analyze_p2p_mask_ablation import projected_axis_error_deg
from analyze_ray_consistency import reference_points, score_rays


def score(
    pose: np.ndarray,
    reference: np.ndarray,
    live: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    valid_depth: np.ndarray,
    intrinsics: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
) -> tuple[float, float, float]:
    surface = surface_p90_mm(live, move(reference, pose))
    image_axis, _ = projected_axis_error_deg(mask, pose, intrinsics, center, axis)
    rays = score_rays(reference, pose, depth, valid_depth, mask, intrinsics, depth_tolerance_m=0.005)
    free_space = rays[3] / rays[1] if rays[1] else float("nan")
    return surface, image_axis, free_space


def verdict(direct: tuple[float, ...], motion: tuple[float, ...]) -> str:
    if not np.isfinite((*direct, *motion)).all():
        return "unscored"
    if all(a <= b for a, b in zip(direct, motion, strict=True)):
        return "direct_dominates"
    if all(a <= b for a, b in zip(motion, direct, strict=True)):
        return "motion_dominates"
    return "tradeoff"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    features = np.load(root / "pose_traces.npz")
    p2p = np.load(root / "p2p_dowel_internal_control_100_550.npz")
    points = np.load(args.points)
    intrinsics = features["intrinsics"]
    center = features["dowel_reference_center_camera_m"]
    axis = features["dowel_reference_shaft_camera"]
    reference = reference_points(args.dataset, args.masks, "dowel", intrinsics)
    tip = center + features["dowel_reference_end_offsets_m"][1] * axis
    seed = int(points["seed_frame"])
    seed_pose = p2p["poses"][seed - 100]
    rows = []
    for frame, motion in zip(points["frames"], points["motions"], strict=True):
        frame = int(frame)
        if not np.isfinite(motion).all():
            continue
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
        point_pose = motion @ seed_pose
        point_scores = score(point_pose, reference, live, mask, depth, valid_depth, intrinsics, center, axis)
        lost = bool(p2p["lost"][frame - 100])
        if lost:
            direct_scores = (float("nan"),) * 3
            gap_mm = float("nan")
            decision = "direct_lost"
        else:
            direct_pose = p2p["poses"][frame - 100]
            direct_scores = score(
                direct_pose, reference, live, mask, depth, valid_depth, intrinsics, center, axis
            )
            gap_mm = float(np.linalg.norm(move(tip[None], direct_pose) - move(tip[None], point_pose)) * 1000)
            decision = verdict(direct_scores, point_scores)
        rows.append((frame, decision, gap_mm, *direct_scores, *point_scores))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "pareto_verdict",
                "hidden_tip_hypothesis_gap_mm",
                "direct_surface_p90_mm",
                "direct_image_axis_deg",
                "direct_free_space_fraction",
                "motion_surface_p90_mm",
                "motion_image_axis_deg",
                "motion_free_space_fraction",
            )
        )
        writer.writerows(rows)
    for phase, selected in (
        ("before", [row for row in rows if 430 < row[0] <= 496]),
        ("contact", [row for row in rows if 497 <= row[0] <= 504]),
    ):
        outcomes = {label: sum(row[1] == label for row in selected) for label in {row[1] for row in selected}}
        print(phase, outcomes)
    for row in rows:
        if 496 <= row[0] <= 504:
            print(row[0], row[1], "tip gap", round(row[2], 2), "scores", tuple(round(x, 2) for x in row[3:]))


if __name__ == "__main__":
    main()
