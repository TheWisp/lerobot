"""Score causal dowel pose axes against the central observed SAM3 silhouette.

The projected shaft is a first-view depth-cloud proxy. The current mask gives
only a 2D centerline; agreement does not prove a hidden 3D tip or axial roll.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def project(point: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    return np.array(
        (
            intrinsics[0, 0] * point[0] / point[2] + intrinsics[0, 2],
            intrinsics[1, 1] * point[1] / point[2] + intrinsics[1, 2],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--first-frame", type=int, default=400)
    parser.add_argument("--last-frame", type=int, default=500)
    parser.add_argument("--central-trim", type=float, default=0.2)
    args = parser.parse_args()
    if args.last_frame < args.first_frame or not 0 <= args.central_trim < 0.5:
        parser.error("invalid frame interval or central trim")
    root = Path(__file__).parent
    with np.load(root / "pose_traces.npz") as reference:
        intrinsics = reference["intrinsics"].copy()
        center = reference["dowel_reference_center_camera_m"].copy()
        shaft = reference["dowel_reference_shaft_camera"].copy()
        offsets = reference["dowel_reference_end_offsets_m"].copy()
    candidates = {}
    for label, filename in (
        ("Point2Pose", "p2p_dowel_internal_control_100_550.npz"),
        ("TAPNext 256", "tapnext_fixed_dowel_100_550_pose.npz"),
        ("TAPNext 512", "tapnext512_fixed_dowel_100_550_pose.npz"),
    ):
        with np.load(root / filename) as trace:
            candidates[label] = {
                int(frame): (pose, bool(lost))
                for frame, pose, lost in zip(trace["frames"], trace["poses"], trace["lost"], strict=True)
            }
    records = []
    for frame in range(args.first_frame, args.last_frame + 1):
        mask = cv2.imread(str(args.masks / "dowel" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"missing current dowel mask at frame {frame}")
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        rows, columns = np.nonzero(mask)
        if len(rows) < 3:
            continue
        pixels = np.column_stack((columns, rows)).astype(np.float64)
        full_center = pixels.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(pixels.T))
        major_axis = eigenvectors[:, -1]
        axial = (pixels - full_center) @ major_axis
        low, high = np.quantile(axial, (args.central_trim, 1 - args.central_trim))
        central_pixels = pixels[(axial >= low) & (axial <= high)]
        observed_center = central_pixels.mean(axis=0)
        anisotropy = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else float("inf")
        for name, trace in candidates.items():
            candidate = trace.get(frame)
            if candidate is None or candidate[1]:
                continue
            pose = candidate[0]
            ends = [
                project(pose[:3, :3] @ (center + offset * shaft) + pose[:3, 3], intrinsics)
                for offset in offsets
            ]
            direction = ends[1] - ends[0]
            direction /= np.linalg.norm(direction)
            normal = np.array((-direction[1], direction[0]))
            offset_px = float(abs((observed_center - ends[0]) @ normal))
            angle_deg = float(np.degrees(np.arccos(np.clip(abs(direction @ major_axis), 0, 1))))
            records.append((frame, name, len(pixels), len(central_pixels), anisotropy, offset_px, angle_deg))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "candidate",
                "mask_pixels",
                "central_pixels",
                "mask_anisotropy",
                "projected_shaft_center_offset_px",
                "unoriented_axis_angle_deg",
            )
        )
        writer.writerows(records)
    by_candidate = defaultdict(list)
    for record in records:
        by_candidate[record[1]].append(record)
    for name, subset in by_candidate.items():
        print(
            name,
            len(subset),
            "median_offset_px",
            np.median([row[5] for row in subset]),
            "median_axis_deg",
            np.median([row[6] for row in subset]),
            "median_mask_anisotropy",
            np.median([row[4] for row in subset]),
        )


if __name__ == "__main__":
    main()
