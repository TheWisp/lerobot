"""Ablate local depth ICP using the visible shaft, withholding its ends.

This is an experiment, not a FLASH production pose update. The objective is
nearest-surface distance on an incomplete reconstructed object, so a lower
fit residual cannot alone establish a better hidden contact-point estimate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from analyze_multiview import build_models, frame_points, move, surface_p90_mm
from scipy.spatial import cKDTree


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    left, _, right_transpose = np.linalg.svd((source - source_center).T @ (target - target_center))
    handedness = np.diag((1.0, 1.0, np.linalg.det(left @ right_transpose)))
    rotation = right_transpose.T @ handedness @ left.T
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = target_center - rotation @ source_center
    return pose


def refit(
    model: np.ndarray,
    live_middle: np.ndarray,
    initial: np.ndarray,
    *,
    max_iterations: int,
    gate_m: float,
    retain_quantile: float,
) -> np.ndarray:
    pose = initial.copy()
    for _ in range(max_iterations):
        surface = move(model, pose)
        distances, nearest = cKDTree(surface).query(live_middle)
        near = distances <= gate_m
        if near.sum() < 30:
            break
        accepted = near & (distances <= np.quantile(distances[near], retain_quantile))
        correction = rigid_fit(surface[nearest[accepted]], live_middle[accepted])
        pose = correction @ pose
        if (
            np.linalg.norm(correction[:3, 3]) < 0.0001
            and np.linalg.norm(correction[:3, :3] - np.eye(3)) < 0.001
        ):
            break
    return pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--support-radius-mm", type=float, default=3.0)
    parser.add_argument("--depth-edge-limit-mm", type=float, default=10.0)
    parser.add_argument("--correspondence-gate-mm", type=float, default=15.0)
    parser.add_argument("--retain-quantile", type=float, default=0.8)
    parser.add_argument("--max-iterations", type=int, default=12)
    args = parser.parse_args()
    if (
        min(args.support_radius_mm, args.depth_edge_limit_mm, args.correspondence_gate_mm) <= 0
        or not 0 < args.retain_quantile <= 1
        or args.max_iterations < 1
    ):
        parser.error("invalid refit parameters")
    args.output.mkdir(parents=True, exist_ok=True)
    traces = np.load(Path(__file__).with_name("pose_traces.npz"))
    intrinsics = traces["intrinsics"]
    poses = traces["p2p_dowel_delta"]
    lost = traces["p2p_dowel_lost"]
    _, model, _ = build_models(
        args.dataset,
        args.masks,
        "dowel",
        intrinsics,
        poses,
        lost,
        args.support_radius_mm / 1000,
    )
    center = traces["dowel_reference_center_camera_m"]
    axis = traces["dowel_reference_shaft_camera"]
    insertion_end = center + axis * traces["dowel_reference_end_offsets_m"][1]
    rows = []
    for frame in range(400, 501):
        index = frame - 100
        if lost[index]:
            continue
        live, _ = frame_points(
            args.dataset,
            args.masks,
            "dowel",
            frame,
            intrinsics,
            edge_limit_m=args.depth_edge_limit_mm / 1000,
        )
        initial = poses[index]
        projected_axis = initial[:3, :3] @ axis
        projected_center = move(center[None, :], initial)[0]
        axial_coordinate = (live - projected_center) @ projected_axis
        lower, upper = np.quantile(axial_coordinate, (0.2, 0.8))
        middle = live[(axial_coordinate >= lower) & (axial_coordinate <= upper)]
        withheld = live[(axial_coordinate < lower) | (axial_coordinate > upper)]
        corrected = refit(
            model,
            middle,
            initial,
            max_iterations=args.max_iterations,
            gate_m=args.correspondence_gate_mm / 1000,
            retain_quantile=args.retain_quantile,
        )
        rows.append(
            (
                frame,
                len(middle),
                len(withheld),
                surface_p90_mm(middle, move(model, initial)),
                surface_p90_mm(middle, move(model, corrected)),
                surface_p90_mm(withheld, move(model, initial)),
                surface_p90_mm(withheld, move(model, corrected)),
                1000
                * float(
                    np.linalg.norm(
                        move(insertion_end[None, :], corrected) - move(insertion_end[None, :], initial)
                    )
                ),
            )
        )
    with (args.output / "partial_refit_score.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "middle_points",
                "withheld_end_points",
                "middle_before_p90_mm",
                "middle_after_p90_mm",
                "withheld_before_p90_mm",
                "withheld_after_p90_mm",
                "predicted_insertion_end_shift_mm",
            )
        )
        writer.writerows(rows)
    data = np.asarray(rows)
    print("frames", len(rows))
    print("middle fit improves", int((data[:, 4] < data[:, 3]).sum()))
    print("withheld ends improve", int((data[:, 6] < data[:, 5]).sum()))
    print("median withheld change (mm)", float(np.median(data[:, 6] - data[:, 5])))
    print("maximum predicted tip shift (mm)", float(np.max(data[:, 7])))


if __name__ == "__main__":
    main()
