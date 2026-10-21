"""Causally refit one pose anchor to masked RGB-D, then propagate a relative trace.

This is an exploratory diagnostic. Nearest-surface agreement is not a full pose
label, and the first-view model can be incomplete. The output should be scored
against later frames and task features before interpreting the correction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from analyze_multiview import frame_points, move, surface_p90_mm
from scipy.spatial import cKDTree


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the proper rigid transform mapping paired source points to target."""
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    left, _, right = np.linalg.svd((source - source_center).T @ (target - target_center))
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(left @ right)
    rotation = (left @ sign @ right).T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def refit(
    reference: np.ndarray,
    observed: np.ndarray,
    initial: np.ndarray,
    *,
    iterations: int,
    mad_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    pose = initial.copy()
    scores = []
    for _ in range(iterations):
        transformed = move(reference, pose)
        distances, indices = cKDTree(transformed).query(observed)
        median = float(np.median(distances))
        scale = 1.4826 * float(np.median(np.abs(distances - median)))
        accepted = distances <= median + mad_scale * scale
        if accepted.sum() < 3:
            raise ValueError("fewer than three depth correspondences survive robust trimming")
        pose = rigid_fit(transformed[indices[accepted]], observed[accepted]) @ pose
        scores.append(surface_p90_mm(observed, move(reference, pose)))
    return pose, np.asarray(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--relative", required=True, type=Path)
    parser.add_argument("--anchor-trace", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("dowel", "ring"))
    parser.add_argument("--reference-frame", type=int, default=100)
    parser.add_argument("--seed-frame", required=True, type=int)
    parser.add_argument("--edge-limit-m", type=float, default=0.01)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mad-scale", type=float, default=2.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.seed_frame <= args.reference_frame or args.iterations < 1 or args.max_points < 3:
        parser.error("seed must follow reference; iterations and point cap must be positive")
    if args.mad_scale < 0 or args.edge_limit_m <= 0:
        parser.error("MAD scale must be nonnegative and depth-edge limit positive")

    with np.load(Path(__file__).parent / "pose_traces.npz") as features:
        intrinsics = features["intrinsics"].copy()
    reference, _ = frame_points(
        args.dataset,
        args.masks,
        args.role,
        args.reference_frame,
        intrinsics,
        edge_limit_m=args.edge_limit_m,
        max_points=args.max_points,
    )
    observed, _ = frame_points(
        args.dataset,
        args.masks,
        args.role,
        args.seed_frame,
        intrinsics,
        edge_limit_m=args.edge_limit_m,
        max_points=args.max_points,
    )
    with np.load(args.anchor_trace) as anchor:
        indices = np.flatnonzero(anchor["frames"] == args.seed_frame)
        if len(indices) != 1 or bool(anchor["lost"][indices[0]]):
            raise ValueError("the seed lacks one valid causal anchor pose")
        original = anchor["poses"][indices[0]].copy()
    refined, scores = refit(
        reference, observed, original, iterations=args.iterations, mad_scale=args.mad_scale
    )
    with np.load(args.relative) as relative:
        data = {key: relative[key].copy() for key in relative.files}
    if int(data["frames"][0]) != args.seed_frame:
        raise ValueError("relative trace does not begin at the seed frame")
    data["poses"] = data["poses"] @ refined
    data["seed_frame"] = np.asarray(args.seed_frame)
    data["anchor_source"] = np.asarray("causal_first_view_robust_icp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    np.savez_compressed(
        args.output.with_name(args.output.stem + "_fit.npz"),
        original=original,
        refined=refined,
        surface_p90_before_mm=surface_p90_mm(observed, move(reference, original)),
        surface_p90_by_iteration_mm=scores,
        reference_frame=args.reference_frame,
        seed_frame=args.seed_frame,
        edge_limit_m=args.edge_limit_m,
        max_points=args.max_points,
        mad_scale=args.mad_scale,
    )
    print(
        args.output,
        "seed surface p90",
        round(surface_p90_mm(observed, move(reference, original)), 3),
        "->",
        round(float(scores[-1]), 3),
    )


if __name__ == "__main__":
    main()
