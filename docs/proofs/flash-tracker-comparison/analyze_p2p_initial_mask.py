"""Compare Point2Pose traces that differ only in their first-frame mask.

Scores use the same current SAM3 mask and aligned RGB-D for both traces.
They check visible consistency, not independently labeled hidden-tip error.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from analyze_multiview import frame_points, move, surface_p90_mm
from analyze_ray_consistency import reference_points, score_rays
from analyze_temporal_pose_conflict import score

from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.pose_disagreement import compare_pose_proposals


def visible_verdict(baseline: tuple[float, ...], alternative: tuple[float, ...]) -> str:
    paired = [(a, b) for a, b in zip(baseline, alternative, strict=True) if np.isfinite((a, b)).all()]
    if not paired:
        return "unscored"
    if all(a <= b for a, b in paired):
        return "baseline_dominates"
    if all(b <= a for a, b in paired):
        return "alternative_dominates"
    return "tradeoff"


def ring_scores(
    pose: np.ndarray,
    reference: np.ndarray,
    live: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    valid_depth: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[float, float, float]:
    rays = score_rays(reference, pose, depth, valid_depth, mask, intrinsics, depth_tolerance_m=0.005)
    return (
        surface_p90_mm(live, move(reference, pose)),
        float("nan"),
        rays[3] / rays[1] if rays[1] else float("nan"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--alternative", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("dowel", "ring"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--first-frame", type=int, default=300)
    parser.add_argument("--last-frame", type=int, default=504)
    parser.add_argument("--fit-threshold-m", type=float, default=0.005)
    parser.add_argument("--min-depth-points", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).parent
    features = np.load(root / "pose_traces.npz")
    baseline = np.load(args.baseline)
    alternative = np.load(args.alternative)
    if not np.array_equal(baseline["frames"], alternative["frames"]):
        raise ValueError("pose traces must cover the same camera frames")
    frames = baseline["frames"]
    if args.first_frame < frames[0] or args.last_frame > frames[-1]:
        raise ValueError("requested interval exceeds the traces")
    intrinsics = features["intrinsics"]
    reference = reference_points(args.dataset, args.masks, args.role, intrinsics)
    radius = float(np.sqrt(np.mean(np.sum((reference - reference.mean(axis=0)) ** 2, axis=1))))
    center = features["dowel_reference_center_camera_m"] if args.role == "dowel" else reference.mean(axis=0)
    axis = features["dowel_reference_shaft_camera"] if args.role == "dowel" else None
    probe = center + features["dowel_reference_end_offsets_m"][1] * axis if axis is not None else center
    rows = []
    for frame in range(args.first_frame, args.last_frame + 1):
        index = frame - int(frames[0])
        if baseline["lost"][index] or alternative["lost"][index]:
            rows.append((frame, bool(baseline["lost"][index]), bool(alternative["lost"][index]), "lost"))
            continue
        mask = cv2.imread(str(args.masks / args.role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        raw_depth = cv2.imread(str(args.dataset / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if mask is None or raw_depth is None or mask.shape != raw_depth.shape:
            raise ValueError(f"missing aligned mask/depth for frame {frame}")
        mask = mask > 0
        depth = raw_depth.astype(np.float32) * 0.001
        neighborhood = np.ones((3, 3), dtype=np.uint8)
        lower = cv2.erode(depth, neighborhood)
        upper = cv2.dilate(depth, neighborhood)
        valid_depth = (depth > 0) & (lower > 0) & (upper - lower <= 0.01)
        live, _ = frame_points(args.dataset, args.masks, args.role, frame, intrinsics, edge_limit_m=0.01)
        baseline_pose = baseline["poses"][index]
        alternative_pose = alternative["poses"][index]
        if axis is not None:
            baseline_scores = score(
                baseline_pose, reference, live, mask, depth, valid_depth, intrinsics, center, axis
            )
            alternative_scores = score(
                alternative_pose, reference, live, mask, depth, valid_depth, intrinsics, center, axis
            )
        else:
            baseline_scores = ring_scores(
                baseline_pose, reference, live, mask, depth, valid_depth, intrinsics
            )
            alternative_scores = ring_scores(
                alternative_pose, reference, live, mask, depth, valid_depth, intrinsics
            )
        probe_gap_mm = float(
            np.linalg.norm(move(probe[None], baseline_pose) - move(probe[None], alternative_pose)) * 1000
        )
        decision = visible_verdict(baseline_scores, alternative_scores)
        source = RGBDFrame(
            frame,
            np.zeros((*depth.shape, 3), dtype=np.uint8),
            raw_depth,
            intrinsics,
            0.001,
            np.eye(4),
            frame / 30,
            frame / 30,
            "recorded-camera",
        )
        service_comparison = compare_pose_proposals(
            source,
            mask,
            reference,
            baseline_pose,
            alternative_pose,
            max_agreement_offset_m=args.fit_threshold_m,
            max_agreement_angle_rad=float(np.arctan2(args.fit_threshold_m, radius)),
            min_masked_depth_points=args.min_depth_points,
        )
        rows.append(
            (
                frame,
                False,
                False,
                decision,
                probe_gap_mm,
                *baseline_scores,
                *alternative_scores,
                service_comparison.verdict.value,
                service_comparison.candidate.projected_point_mask_fraction,
                service_comparison.candidate.depth_residual_p90_m * 1000,
                service_comparison.prior.projected_point_mask_fraction,
                service_comparison.prior.depth_residual_p90_m * 1000,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "baseline_lost",
                "alternative_lost",
                "visible_verdict",
                "probe_point_gap_mm",
                "baseline_surface_p90_mm",
                "baseline_image_axis_deg",
                "baseline_free_space_fraction",
                "alternative_surface_p90_mm",
                "alternative_image_axis_deg",
                "alternative_free_space_fraction",
                "service_surface_verdict",
                "baseline_projected_mask_fraction",
                "baseline_projected_depth_p90_mm",
                "alternative_projected_mask_fraction",
                "alternative_projected_depth_p90_mm",
            )
        )
        writer.writerows(rows)
    for phase, selected in (
        ("carry", [row for row in rows if 300 <= row[0] < 400 and len(row) == 16]),
        ("insertion", [row for row in rows if 400 <= row[0] <= 500 and len(row) == 16]),
    ):
        values = np.asarray([[row[index] for index in (4, 5, 8)] for row in selected], dtype=float)
        print(
            phase,
            "shared frames",
            len(selected),
            "verdicts",
            {
                label: sum(row[3] == label for row in selected)
                for label in sorted({row[3] for row in selected})
            },
            "service verdicts",
            {
                label: sum(row[11] == label for row in selected)
                for label in sorted({row[11] for row in selected})
            },
            "median probe gap / baseline surface / alternative surface mm",
            np.round(np.nanmedian(values, axis=0), 2),
        )


if __name__ == "__main__":
    main()
