"""Audit initial-mask and point ownership in late Point2Pose reseeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--sam3-masks", required=True, type=Path)
    parser.add_argument("--case", action="append", nargs=3, metavar=("SEED", "LABEL", "FRAME"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for seed_text, label, frame_text in args.case:
        seed, frame = int(seed_text), int(frame_text)
        stem = f"{frame:06d}.png"
        mask_path = args.runs / f"p2p_dowel_reseed{seed}_{label}_masks/dowel" / stem
        diagnostic_path = args.runs / f"p2p_dowel_reseed{seed}_{label}_diagnostics/{frame:06d}.npz"
        images = [
            cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            for path in (mask_path, args.sam3_masks / "dowel" / stem, args.sam3_masks / "ring" / stem)
        ]
        if any(image is None for image in images):
            raise FileNotFoundError(f"missing saved mask at seed {seed}, frame {frame}, case {label}")
        mask, dowel, ring = (image > 0 for image in images)
        with np.load(diagnostic_path) as diagnostic:
            uv = diagnostic["fit_uv"].copy()
            fit_inliers = int(diagnostic["fit_inlier"].sum())
            tracked = int(diagnostic["track_valid_0"].sum())
        px = np.rint(uv).astype(int)
        if len(px):
            px[:, 0] = np.clip(px[:, 0], 0, mask.shape[1] - 1)
            px[:, 1] = np.clip(px[:, 1], 0, mask.shape[0] - 1)
        rows.append(
            (
                seed,
                label,
                frame,
                int(mask.sum()),
                float((mask & dowel).sum() / (mask | dowel).sum()),
                int((mask & ring).sum()),
                tracked,
                len(px),
                fit_inliers,
                int(dowel[px[:, 1], px[:, 0]].sum()),
                int(ring[px[:, 1], px[:, 0]].sum()),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "seed_frame",
                "mask_policy",
                "frame",
                "p2p_mask_pixels",
                "sam3_dowel_iou",
                "p2p_mask_pixels_in_sam3_ring",
                "valid_point_tracks",
                "fit_points",
                "fit_inliers",
                "fit_points_in_sam3_dowel",
                "fit_points_in_sam3_ring",
            )
        )
        writer.writerows(rows)
    print(args.output, "rows", len(rows))


if __name__ == "__main__":
    main()
