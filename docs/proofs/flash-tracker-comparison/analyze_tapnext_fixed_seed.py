"""Fit rigid RGB-D motion from fixed TAPNext++ point tracks.

Registration and depth lifting use FLASH's same functions and settings as its
LK point-track replay. An optional current SAM3 mask gate tests whether point
identities that leave the object explain drift; the ungated run matches the
current LK tracker's depth-only admission rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
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
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pose-trace-output", type=Path)
    parser.add_argument("--gate-current-mask", action="store_true")
    parser.add_argument("--fit-threshold-m", type=float, default=0.005)
    parser.add_argument("--min-inliers", type=int, default=12)
    args = parser.parse_args()
    intrinsics = np.load(Path(__file__).with_name("pose_traces.npz"))["intrinsics"]
    with np.load(args.tracks) as track:
        role = str(track["role"])
        seed_frame = int(track["seed_frame"])
        seed_uv = track["seed_uv"]
        frames = track["frames"]
        uv = track["uv"]
        visible = track["visible"]
        step_ms = track["step_ms"]
    if not np.array_equal(frames, np.arange(seed_frame, int(frames[-1]) + 1)):
        raise ValueError("TAPNext++ trace skipped a causal camera frame")
    seed = read_frame(args.dataset, seed_frame, intrinsics)
    seed_points = unproject_pixels(seed.depth, intrinsics, seed_uv, seed.depth_m_per_unit)
    records = []
    for index, frame_id in enumerate(frames[1:], start=1):
        frame = read_frame(args.dataset, int(frame_id), intrinsics)
        valid = visible[index] & valid_depth_pixels(frame.depth, uv[index])
        if args.gate_current_mask:
            mask = cv2.imread(str(args.masks / role / f"{int(frame_id):06d}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != frame.depth.shape:
                raise ValueError(f"missing aligned current mask at frame {frame_id}")
            pixels = np.rint(uv[index][valid]).astype(int)
            current_indices = np.flatnonzero(valid)
            valid[current_indices] &= mask[pixels[:, 1], pixels[:, 0]] > 0
        selected = np.flatnonzero(valid)
        motion = np.full((4, 4), np.nan)
        inliers = 0
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
                motion = fit.base_from_demo_object
                inliers = len(fit.inlier_indices)
                error = fit.median_error_m
        records.append(
            (int(frame_id), motion, int(visible[index].sum()), len(selected), inliers, error, step_ms[index])
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        role=np.asarray(role),
        seed_frame=np.asarray(seed_frame),
        seed_points=np.asarray(len(seed_uv)),
        frames=np.asarray([row[0] for row in records]),
        motions=np.stack([row[1] for row in records]),
        tracked_points=np.asarray([row[2] for row in records]),
        depth_points=np.asarray([row[3] for row in records]),
        inliers=np.asarray([row[4] for row in records]),
        median_error_m=np.asarray([row[5] for row in records]),
        step_ms=np.asarray([row[6] for row in records]),
    )
    if args.pose_trace_output is not None:
        args.pose_trace_output.parent.mkdir(parents=True, exist_ok=True)
        poses = np.stack([np.eye(4), *(row[1] for row in records)])
        np.savez_compressed(
            args.pose_trace_output,
            frames=frames,
            poses=poses,
            lost=~np.isfinite(poses).all(axis=(1, 2)),
        )
    print(
        args.output,
        "seed",
        len(seed_uv),
        "valid fits",
        sum(np.isfinite(row[1]).all() for row in records),
        "/",
        len(records),
    )


if __name__ == "__main__":
    main()
