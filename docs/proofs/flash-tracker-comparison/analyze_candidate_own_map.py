"""Score each causal pose trace with its own earlier-view RGB-D object map.

Every map uses the same source frames 100, 300, 340 and 380 and the same
two-view confirmation rule. Insertion frames 400-500 are withheld from map
building. Candidate-specific maps can be contaminated by their source poses,
so these scores complement rather than replace common-reference comparisons.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from analyze_multiview import build_models, frame_points, move, surface_p90_mm


def observed_axis(mask: np.ndarray, trim: float) -> tuple[np.ndarray, np.ndarray, float]:
    pixels_y, pixels_x = np.nonzero(cv2.erode(mask, np.ones((3, 3), np.uint8)))
    pixels = np.column_stack((pixels_x, pixels_y)).astype(np.float64)
    if len(pixels) < 3:
        raise ValueError("current mask has too few pixels for an axis")
    center = pixels.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(pixels.T))
    major = vectors[:, -1]
    along = (pixels - center) @ major
    lower, upper = np.quantile(along, (trim, 1 - trim))
    center = pixels[(along >= lower) & (along <= upper)].mean(axis=0)
    return center, major, float(values[-1] / values[0])


def projected_line(
    center: np.ndarray, shaft: np.ndarray, span: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ends = []
    for offset in span:
        point = move((center + offset * shaft)[None], pose)[0]
        ends.append(
            np.array(
                (
                    intrinsics[0, 0] * point[0] / point[2] + intrinsics[0, 2],
                    intrinsics[1, 1] * point[1] / point[2] + intrinsics[1, 2],
                )
            )
        )
    direction = ends[1] - ends[0]
    direction /= np.linalg.norm(direction)
    return ends[0], direction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--support-radius-mm", type=float, default=3)
    parser.add_argument("--depth-edge-limit-mm", type=float, default=10)
    parser.add_argument("--central-trim", type=float, default=0.2)
    args = parser.parse_args()
    if args.support_radius_mm <= 0 or args.depth_edge_limit_mm <= 0 or not 0 <= args.central_trim < 0.5:
        parser.error("invalid analysis parameters")
    root = Path(__file__).parent
    with np.load(root / "pose_traces.npz") as reference:
        intrinsics = reference["intrinsics"].copy()
        candidates = {
            "dowel": {
                "BundleSDF": (
                    reference["bundlesdf_dowel_delta"].copy(),
                    reference["bundlesdf_dowel_failed"].copy(),
                )
            },
            "ring": {
                "BundleSDF": (
                    reference["bundlesdf_ring_delta"].copy(),
                    reference["bundlesdf_ring_failed"].copy(),
                )
            },
        }
    for role, files in {
        "dowel": {
            "Point2Pose": "p2p_dowel_internal_control_100_550.npz",
            "TAPNext 256": "tapnext_fixed_dowel_100_550_pose.npz",
            "TAPNext 512": "tapnext512_fixed_dowel_100_550_pose.npz",
        },
        "ring": {
            "Point2Pose": "p2p_ring_sam2_100_550.npz",
            "TAPNext 256": "tapnext_fixed_ring_100_550_pose.npz",
            "TAPNext 512": "tapnext512_fixed_ring_100_550_pose.npz",
        },
    }.items():
        for name, filename in files.items():
            with np.load(root / filename) as trace:
                candidates[role][name] = (trace["poses"].copy(), trace["lost"].copy())
    records = []
    for role, role_candidates in candidates.items():
        observations = {
            frame: frame_points(
                args.dataset,
                args.masks,
                role,
                frame,
                intrinsics,
                edge_limit_m=args.depth_edge_limit_mm / 1000,
            )[0]
            for frame in range(400, 501)
        }
        mask_axes = {}
        if role == "dowel":
            for frame in observations:
                mask = cv2.imread(str(args.masks / role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"missing mask for {role} frame {frame}")
                mask_axes[frame] = observed_axis(mask, args.central_trim)
        for name, (poses, lost) in role_candidates.items():
            _, model, _ = build_models(
                args.dataset,
                args.masks,
                role,
                intrinsics,
                poses,
                lost,
                args.support_radius_mm / 1000,
            )
            model_center = np.median(model, axis=0)
            values, vectors = np.linalg.eigh(np.cov(model.T))
            model_shaft = vectors[:, -1]
            span = np.quantile((model - model_center) @ model_shaft, (0.05, 0.95))
            bbox_mm = np.ptp(model, axis=0) * 1000
            for frame, live in observations.items():
                index = frame - 100
                if lost[index]:
                    continue
                pose = poses[index]
                score = surface_p90_mm(live, move(model, pose))
                offset_px = angle_deg = anisotropy = float("nan")
                if role == "dowel":
                    observed_center, observed_direction, anisotropy = mask_axes[frame]
                    line_origin, line_direction = projected_line(
                        model_center, model_shaft, span, pose, intrinsics
                    )
                    normal = np.array((-line_direction[1], line_direction[0]))
                    offset_px = float(abs((observed_center - line_origin) @ normal))
                    angle_deg = float(
                        np.degrees(np.arccos(np.clip(abs(line_direction @ observed_direction), 0, 1)))
                    )
                records.append(
                    (
                        role,
                        name,
                        frame,
                        len(model),
                        *bbox_mm,
                        len(live),
                        score,
                        offset_px,
                        angle_deg,
                        anisotropy,
                    )
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "role",
                "candidate",
                "frame",
                "map_points",
                "map_bbox_x_mm",
                "map_bbox_y_mm",
                "map_bbox_z_mm",
                "observed_depth_points",
                "own_map_surface_p90_mm",
                "own_map_projected_shaft_center_offset_px",
                "own_map_unoriented_axis_angle_deg",
                "mask_anisotropy",
            )
        )
        writer.writerows(records)
    for role, role_candidates in candidates.items():
        for name in role_candidates:
            subset = [row for row in records if row[:2] == (role, name)]
            print(role, name, len(subset), "surface_p90_mm", np.median([row[8] for row in subset]))


if __name__ == "__main__":
    main()
