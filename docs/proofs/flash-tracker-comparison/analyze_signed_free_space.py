"""Evaluate a one-sided RGB-D contradiction of complete pose proposals.

The camera may see an occluder in front of the projected object surface. If it
instead sees a farther surface on most valid projected rays, the proposed
object position is inconsistent with the current frame. This is a diagnostic
on a partial first-view model, not independently labeled 6D pose error.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from analyze_ray_consistency import reference_points

from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.pose_disagreement import FreeSpaceSupport, free_space_support


def verdict(support: FreeSpaceSupport, *, minimum_pixels: int, tolerated_gap_m: float) -> str:
    if support.valid_depth_pixels < minimum_pixels or support.median_signed_gap_m is None:
        return "insufficient_depth"
    return "contradicted" if support.median_signed_gap_m > tolerated_gap_m else "not_contradicted"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--alternative", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())["causal_tracking"]
    fit_tolerance = float(config["fit_threshold_m"])
    task_tolerance = float(config["max_pose_disagreement_translation_m"])
    minimum_pixels = int(config["min_inliers"])
    root = Path(__file__).parent
    intrinsics = np.load(root / "pose_traces.npz")["intrinsics"]
    points = reference_points(args.dataset, args.masks, args.role, intrinsics)
    baseline = np.load(args.baseline)
    alternative = np.load(args.alternative)
    if not np.array_equal(baseline["frames"], alternative["frames"]):
        raise ValueError("the two pose traces must cover the same frames")
    rows = []
    for index, frame_id in enumerate(baseline["frames"]):
        frame_id = int(frame_id)
        depth = cv2.imread(str(args.dataset / "depth" / f"{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"missing aligned depth for frame {frame_id}")
        frame = RGBDFrame(
            frame_id,
            np.zeros((*depth.shape, 3), dtype=np.uint8),
            depth,
            intrinsics,
            0.001,
            np.eye(4),
            frame_id / 30,
            frame_id / 30,
            "recorded-camera",
        )
        results = []
        for trace in (baseline, alternative):
            if trace["lost"][index]:
                results.extend(("lost", 0, 0, None, None))
                continue
            support = free_space_support(
                frame,
                points,
                trace["poses"][index],
                depth_tolerance_m=fit_tolerance,
                max_local_depth_span_m=task_tolerance,
            )
            results.extend(
                (
                    verdict(support, minimum_pixels=minimum_pixels, tolerated_gap_m=task_tolerance),
                    support.projected_pixels,
                    support.valid_depth_pixels,
                    support.median_signed_gap_m * 1000 if support.median_signed_gap_m is not None else None,
                    support.contradiction_fraction,
                )
            )
        rows.append((frame_id, *results))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "baseline_verdict",
                "baseline_projected_pixels",
                "baseline_valid_depth_pixels",
                "baseline_median_signed_gap_mm",
                "baseline_contradiction_fraction",
                "alternative_verdict",
                "alternative_projected_pixels",
                "alternative_valid_depth_pixels",
                "alternative_median_signed_gap_mm",
                "alternative_contradiction_fraction",
            )
        )
        writer.writerows(rows)
    for label, selected in (
        ("pregrasp", [row for row in rows if row[0] < 150]),
        ("carry", [row for row in rows if 300 <= row[0] < 400]),
        ("insertion", [row for row in rows if 400 <= row[0] <= 500]),
    ):
        print(
            label,
            "baseline",
            dict(Counter(row[1] for row in selected)),
            "alternative",
            dict(Counter(row[6] for row in selected)),
        )


if __name__ == "__main__":
    main()
