"""Compare insertion ring center/plane hypotheses, modulo normal sign.

The reference center and plane normal come from aligned masked RGB-D in the
first view. They are geometric proxies, not labeled hole features or 6D truth.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--first-frame", type=int, default=400)
    parser.add_argument("--last-frame", type=int, default=500)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.last_frame < args.first_frame:
        parser.error("last frame precedes first frame")
    with np.load(root / "pose_traces.npz") as trace:
        intrinsics = trace["intrinsics"].copy()
        frames = trace["frames"].copy()
        candidates = {
            "tapir": (trace["p2p_ring_delta"].copy(), trace["p2p_ring_lost"].copy()),
            "bundle": (trace["bundlesdf_ring_delta"].copy(), trace["bundlesdf_ring_failed"].copy()),
        }
    with np.load(root / "tapnext_fixed_ring_100_550_pose.npz") as fixed:
        if not np.array_equal(frames, fixed["frames"]):
            raise ValueError("candidate camera frames are not aligned")
        candidates["tapnext_fixed"] = (fixed["poses"].copy(), fixed["lost"].copy())
    reference_frame = int(frames[0])
    depth = cv2.imread(str(args.dataset / "depth" / f"{reference_frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(args.masks / "ring" / f"{reference_frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None or depth.shape != mask.shape:
        raise ValueError("missing aligned reference depth and ring mask")
    rows, columns = np.nonzero((mask > 0) & (depth > 0))
    metres = depth[rows, columns].astype(np.float64) * 0.001
    points = np.column_stack(
        (
            (columns - intrinsics[0, 2]) / intrinsics[0, 0] * metres,
            (rows - intrinsics[1, 2]) / intrinsics[1, 1] * metres,
            metres,
        )
    )
    center = np.median(points, axis=0)
    _, axes = np.linalg.eigh(np.cov(points.T))
    normal = axes[:, 0]
    records = []
    for frame in range(args.first_frame, args.last_frame + 1):
        index = frame - reference_frame
        if index < 0 or index >= len(frames) or frames[index] != frame:
            raise ValueError(f"frame {frame} absent from reference trace")
        for (first_name, (first_poses, first_lost)), (
            second_name,
            (second_poses, second_lost),
        ) in combinations(candidates.items(), 2):
            if first_lost[index] or second_lost[index]:
                continue
            first_pose, second_pose = first_poses[index], second_poses[index]
            first_center = first_pose[:3, :3] @ center + first_pose[:3, 3]
            second_center = second_pose[:3, :3] @ center + second_pose[:3, 3]
            first_normal = first_pose[:3, :3] @ normal
            second_normal = second_pose[:3, :3] @ normal
            center_gap_mm = float(np.linalg.norm(first_center - second_center) * 1000)
            normal_gap_deg = float(np.degrees(np.arccos(np.clip(abs(first_normal @ second_normal), 0, 1))))
            records.append((frame, first_name, second_name, center_gap_mm, normal_gap_deg))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("frame", "first", "second", "center_gap_mm", "unoriented_normal_gap_deg"))
        writer.writerows(records)
    for first_name, second_name in combinations(candidates, 2):
        pair = [row for row in records if row[1:3] == (first_name, second_name)]
        print(
            first_name,
            second_name,
            len(pair),
            "center_mm",
            np.median([row[3] for row in pair]),
            "normal_deg",
            np.median([row[4] for row in pair]),
        )


if __name__ == "__main__":
    main()
