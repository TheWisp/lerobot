"""Measure whether the frame-498 pose preference survives valid model sampling.

The recorded service fixture contains the full first-view SAM3 depth cloud and
both contemporaneous pose candidates. This is a sensitivity check on visible
RGB-D agreement, not an independent label of the hidden insertion tip.

Example:
    PYTHONPATH=src python analyze_reference_sampling_sensitivity.py \
        --fixture ../../../tests/policies/flash/fixtures/direct_point_contact_20261021.npz \
        --dataset ~/.cache/huggingface/lerobot/demos/insertion/recording \
        --masks ~/.cache/flash-tracker-compare/sam3_masks \
        --output reference_sampling_sensitivity.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from analyze_multiview import frame_points, move, surface_p90_mm

from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.pose_disagreement import surface_support


def _eroded_first_view(dataset: Path, masks: Path, intrinsics: np.ndarray) -> np.ndarray:
    depth = cv2.imread(str(dataset / "depth" / "000100.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(masks / "dowel" / "000100.png"), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None or depth.shape != mask.shape:
        raise ValueError("missing aligned first-view depth or dowel mask")
    interior = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8)) > 0
    y, x = np.nonzero(interior & (depth > 0))
    z = depth[y, x] * 0.001
    return np.column_stack(
        (
            (x - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (y - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--max-local-depth-span-m", type=float, default=0.01)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    with np.load(args.fixture) as recorded:
        frame_id = int(recorded["frame_id"])
        depth = recorded["depth"]
        intrinsics = recorded["intrinsics"]
        full = recorded["dowel_points"]
        mask = recorded["dowel_mask"].astype(bool)
        direct = recorded["dowel_direct_pose"]
        motion = recorded["dowel_motion"] @ recorded["dowel_seed_pose"]
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
    observed, observed_count = frame_points(
        args.dataset,
        args.masks,
        "dowel",
        frame_id,
        intrinsics,
        edge_limit_m=args.max_local_depth_span_m,
    )
    if not len(observed):
        raise ValueError("no depth-backed dowel pixels in the recorded contact frame")
    eroded = _eroded_first_view(args.dataset, args.masks, intrinsics)
    selected = np.linspace(0, len(eroded) - 1, min(4000, len(eroded)), dtype=int)
    models = [("service_full", 0, full), ("eroded_cap4000", 0, eroded[selected])]
    rng = np.random.default_rng(20261021)
    for fraction in (0.9, 0.75, 0.5, 0.25):
        size = max(3, int(len(full) * fraction))
        for replicate in range(args.repeats):
            models.append(
                (f"random_{fraction:g}", replicate, full[rng.choice(len(full), size, replace=False)])
            )
    rows = []
    for name, replicate, model in models:
        direct_support = surface_support(frame, mask, model, direct)
        motion_support = surface_support(frame, mask, model, motion)
        direct_surface_mm = surface_p90_mm(observed, move(model, direct))
        motion_surface_mm = surface_p90_mm(observed, move(model, motion))
        rows.append(
            (
                name,
                replicate,
                len(model),
                observed_count,
                direct_surface_mm,
                motion_surface_mm,
                direct_surface_mm - motion_surface_mm,
                direct_support.projected_point_mask_fraction,
                motion_support.projected_point_mask_fraction,
                direct_support.depth_residual_p90_m * 1000,
                motion_support.depth_residual_p90_m * 1000,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "model",
                "replicate",
                "reference_points",
                "observed_depth_points",
                "direct_observed_surface_p90_mm",
                "motion_observed_surface_p90_mm",
                "direct_minus_motion_surface_p90_mm",
                "direct_projected_mask_fraction",
                "motion_projected_mask_fraction",
                "direct_conditional_depth_p90_mm",
                "motion_conditional_depth_p90_mm",
            )
        )
        writer.writerows(rows)
    for name in dict.fromkeys(row[0] for row in rows):
        differences = [row[6] for row in rows if row[0] == name]
        print(
            name,
            "n=",
            len(differences),
            "motion_better=",
            sum(value > 0 for value in differences),
            "direct_better=",
            sum(value < 0 for value in differences),
            "delta_median_mm=",
            round(float(np.median(differences)), 4),
            "delta_range_mm=",
            tuple(round(float(value), 4) for value in (min(differences), max(differences))),
        )


if __name__ == "__main__":
    main()
