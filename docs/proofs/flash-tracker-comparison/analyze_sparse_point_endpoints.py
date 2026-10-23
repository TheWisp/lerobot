"""Fit metric rigid poses from causally emitted sparse point-track endpoints.

Each endpoint uses only its first-frame RGB-D seed and its own observed depth.
Missing intermediary frames remain missing; no pose is backfilled into them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from run_causal_points_demo import read_frame

from lerobot.policies.flash.registration import (
    DegenerateRegistrationError,
    register_object,
    unproject_pixels,
    valid_depth_pixels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fit-threshold-m", type=float, default=0.005)
    parser.add_argument("--min-inliers", type=int, default=12)
    args = parser.parse_args()
    if args.fit_threshold_m <= 0 or args.min_inliers < 3:
        parser.error("fit threshold and inlier count must be positive")
    intrinsics = np.load(Path(__file__).with_name("pose_traces.npz"))["intrinsics"]
    with np.load(args.tracks) as track:
        role = str(track["role"])
        seed_frame = int(track["seed_frame"])
        seed_uv = track["seed_uv"].copy()
        frames = track["frames"].copy()
        uv = track["uv"].copy()
        visible = track["visible"].copy()
        step_ms = track["step_ms"].copy()
    if (
        len(frames) != len(uv)
        or len(frames) != len(visible)
        or not np.all(np.diff(frames) > 0)
        or frames[0] <= seed_frame
    ):
        raise ValueError("point endpoints need strictly increasing frames after the seed")
    seed = read_frame(args.dataset, seed_frame, intrinsics)
    seed_points = unproject_pixels(seed.depth, intrinsics, seed_uv, seed.depth_m_per_unit)
    poses = []
    inliers = []
    depth_counts = []
    errors = []
    for index, frame_id in enumerate(frames):
        frame = read_frame(args.dataset, int(frame_id), intrinsics)
        valid = visible[index] & valid_depth_pixels(frame.depth, uv[index])
        selected = np.flatnonzero(valid)
        depth_counts.append(len(selected))
        pose = np.full((4, 4), np.nan)
        count = 0
        error = float("nan")
        if len(selected) >= args.min_inliers:
            current_points = unproject_pixels(
                frame.depth, intrinsics, uv[index][selected], frame.depth_m_per_unit
            )
            try:
                fit = register_object(
                    seed_points[selected],
                    current_points,
                    threshold_m=args.fit_threshold_m,
                    min_inliers=args.min_inliers,
                    trials=100,
                )
            except DegenerateRegistrationError:
                pass
            else:
                pose = fit.base_from_demo_object
                count = len(fit.inlier_indices)
                error = fit.median_error_m
        poses.append(pose)
        inliers.append(count)
        errors.append(error)
    poses = np.stack(poses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        role=np.asarray(role),
        seed_frame=np.asarray(seed_frame),
        frames=frames,
        poses=poses,
        lost=~np.isfinite(poses).all(axis=(1, 2)),
        tracked_points=visible.sum(axis=1),
        depth_points=np.asarray(depth_counts),
        inliers=np.asarray(inliers),
        median_error_m=np.asarray(errors),
        step_ms=step_ms,
    )
    print(args.output, "valid endpoint fits", sum(np.isfinite(poses).all(axis=(1, 2))), "/", len(frames))


if __name__ == "__main__":
    main()
