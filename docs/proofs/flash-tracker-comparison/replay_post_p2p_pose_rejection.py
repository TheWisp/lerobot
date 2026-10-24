"""Replay a pose-only rejection check after unchanged Point2Pose output.

The input pose trace and Point2Pose SDF are never modified. For each emitted
pose, project one earlier measured object surface into the current camera and
count reliable rays that agree with the current mask/depth or contradict it by
occupying measured free space. Spatial tile sizes are varied to expose whether
a rejection depends on treating neighboring depth pixels as independent.

This is an offline diagnostic, not a calibrated live acceptance policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from analyze_ray_consistency import reference_points


def ray_labels(
    reference: np.ndarray,
    pose: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    depth_tolerance_m: float,
    local_depth_span_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel coordinates and mutually exclusive supported/empty-space labels."""
    height, width = depth.shape
    camera_points = reference @ pose[:3, :3].T + pose[:3, 3]
    camera_points = camera_points[camera_points[:, 2] > 0]
    x = np.rint(camera_points[:, 0] / camera_points[:, 2] * intrinsics[0, 0] + intrinsics[0, 2]).astype(int)
    y = np.rint(camera_points[:, 1] / camera_points[:, 2] * intrinsics[1, 1] + intrinsics[1, 2]).astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    flat = y[inside] * width + x[inside]
    nearest = np.full(height * width, np.inf, dtype=float)
    np.minimum.at(nearest, flat, camera_points[inside, 2])
    pixels = np.flatnonzero(np.isfinite(nearest))
    projected_x, projected_y = pixels % width, pixels // width
    neighborhood = np.ones((3, 3), dtype=np.uint8)
    lower = cv2.erode(depth, neighborhood)
    upper = cv2.dilate(depth, neighborhood)
    valid = (depth.flat[pixels] > 0) & (lower.flat[pixels] > 0)
    valid &= upper.flat[pixels] - lower.flat[pixels] <= local_depth_span_m
    gaps = depth.flat[pixels] - nearest[pixels]
    supported = valid & mask.flat[pixels] & (np.abs(gaps) <= depth_tolerance_m)
    contradicted = valid & (gaps > depth_tolerance_m)
    return projected_x[valid], projected_y[valid], supported[valid], contradicted[valid]


def tile_interval(
    x: np.ndarray, y: np.ndarray, supported: np.ndarray, contradicted: np.ndarray, *, tile_px: int
) -> tuple[float, float, int]:
    """Cluster-robust interval for the contradiction-minus-support ray fraction."""
    if not len(x):
        return float("nan"), float("nan"), 0
    tile_xy = np.column_stack((x // tile_px, y // tile_px))
    _, groups = np.unique(tile_xy, axis=0, return_inverse=True)
    count = np.bincount(groups)
    difference = np.bincount(groups, weights=contradicted.astype(int) - supported.astype(int))
    total = int(count.sum())
    mean = float(difference.sum() / total)
    tiles = len(count)
    if tiles < 2:
        return float("nan"), float("nan"), tiles
    influence = difference - mean * count
    standard_error = float(np.sqrt(tiles / (tiles - 1) * np.square(influence).sum()) / total)
    # A conventional 95% two-sided interval; sensitivity to tile size is
    # reported because camera pixels are spatially correlated.
    return mean - 1.96 * standard_error, mean + 1.96 * standard_error, tiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--filtered-trace", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[3] / "src/lerobot/policies/flash/docs/trial_config.example.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())["causal_tracking"]
    depth_tolerance_m = float(config["fit_threshold_m"])
    local_depth_span_m = float(config["max_pose_disagreement_translation_m"])
    min_valid_rays = int(config["min_inliers"])
    intrinsics = np.load(Path(__file__).with_name("pose_traces.npz"))["intrinsics"]
    reference = reference_points(args.dataset, args.masks, args.role, intrinsics)
    with np.load(args.trace) as trace:
        frames = trace["frames"].copy()
        poses = trace["poses"].copy()
        source_lost = trace["lost"].copy()
    rejected = np.zeros(len(frames), dtype=bool)
    rows = []
    for index, frame in enumerate(frames):
        frame = int(frame)
        if source_lost[index]:
            rows.append((frame, "p2p_lost", 0, 0, 0, None, None, None, None, None, None))
            continue
        raw_depth = cv2.imread(str(args.dataset / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        raw_mask = cv2.imread(str(args.masks / args.role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if raw_depth is None or raw_mask is None or raw_depth.shape != raw_mask.shape:
            raise ValueError(f"missing aligned RGB-D/mask for frame {frame}")
        x, y, supported, contradicted = ray_labels(
            reference,
            poses[index],
            raw_depth.astype(np.float32) * 0.001,
            raw_mask > 0,
            intrinsics,
            depth_tolerance_m=depth_tolerance_m,
            local_depth_span_m=local_depth_span_m,
        )
        intervals = [tile_interval(x, y, supported, contradicted, tile_px=size) for size in (4, 8, 16)]
        if len(x) < min_valid_rays or any(tiles < 2 for _, _, tiles in intervals):
            verdict = "insufficient_depth"
        elif all(lower > 0 for lower, _, _ in intervals):
            verdict = "rejected"
            rejected[index] = True
        elif all(upper < 0 for _, upper, _ in intervals):
            verdict = "supported"
        else:
            verdict = "unresolved"
        rows.append(
            (
                frame,
                verdict,
                len(x),
                int(supported.sum()),
                int(contradicted.sum()),
                *(lower for lower, _, _ in intervals),
                *(tiles for _, _, tiles in intervals),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "verdict",
                "valid_rays",
                "supported_rays",
                "contradicted_rays",
                "lower_4px",
                "lower_8px",
                "lower_16px",
                "tiles_4px",
                "tiles_8px",
                "tiles_16px",
            )
        )
        writer.writerows(rows)
    args.filtered_trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.filtered_trace,
        frames=frames,
        poses=poses,
        lost=source_lost | rejected,
        source_lost=source_lost,
        rejected=rejected,
    )
    for label, first, last in (
        ("pregrasp", 100, 149),
        ("carry", 300, 399),
        ("insertion", 400, 500),
        ("after_release", 514, 550),
    ):
        selected = [row for row in rows if first <= row[0] <= last]
        print(
            label,
            {
                status: sum(row[1] == status for row in selected)
                for status in ("supported", "rejected", "unresolved", "insufficient_depth", "p2p_lost")
            },
        )


if __name__ == "__main__":
    main()
