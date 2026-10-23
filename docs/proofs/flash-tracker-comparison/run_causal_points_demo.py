"""Replay FLASH's causal rigid point tracker from one saved mask.

Every update receives the next recorded RGB-D frame. This checks whether the
fast frontend can bridge slow pose updates; a successful rigid fit is not by
itself proof that it followed the right object.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.causal_points import CausalRigidPointTracker
from lerobot.policies.flash.perception import RGBDFrame


def read_frame(dataset: Path, frame_id: int, intrinsics: np.ndarray) -> RGBDFrame:
    stem = f"{frame_id:06d}"
    bgr = cv2.imread(str(dataset / "rgb" / f"{stem}.jpg"))
    depth = cv2.imread(str(dataset / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None or bgr.shape[:2] != depth.shape:
        raise ValueError(f"missing or misaligned RGB-D at frame {frame_id}")
    return RGBDFrame(
        frame_id,
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        depth,
        intrinsics,
        0.001,
        np.eye(4),
        frame_id / 30,
        frame_id / 30,
        "offline replay",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("dowel", "ring"))
    parser.add_argument("--seed-frame", required=True, type=int)
    parser.add_argument("--last-frame", type=int, default=500)
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 100 <= args.seed_frame < args.last_frame <= 550:
        parser.error("seed/last frames must lie in the recorded range")
    intrinsics = np.load(Path(__file__).with_name("pose_traces.npz"))["intrinsics"]
    mask = cv2.imread(str(args.masks / args.role / f"{args.seed_frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("missing seed mask")
    tracker = CausalRigidPointTracker(
        min_inliers=12, max_frame_gap_s=0.1, fit_threshold_m=0.005, image_scale=args.image_scale
    )
    seeded = tracker.seed(read_frame(args.dataset, args.seed_frame, intrinsics), mask > 0)
    records = []
    for frame_id in range(args.seed_frame + 1, args.last_frame + 1):
        frame = read_frame(args.dataset, frame_id, intrinsics)
        started = time.perf_counter()
        motion = tracker.update(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000
        records.append(
            (
                frame_id,
                motion.base_motion_from_seed
                if motion.base_motion_from_seed is not None
                else np.full((4, 4), np.nan),
                motion.tracked_points,
                motion.depth_points,
                motion.inliers,
                motion.median_error_m if motion.median_error_m is not None else np.nan,
                elapsed_ms,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        role=np.asarray(args.role),
        seed_frame=np.asarray(args.seed_frame),
        image_scale=np.asarray(args.image_scale),
        seed_points=np.asarray(seeded),
        frames=np.asarray([row[0] for row in records]),
        motions=np.stack([row[1] for row in records]),
        tracked_points=np.asarray([row[2] for row in records]),
        depth_points=np.asarray([row[3] for row in records]),
        inliers=np.asarray([row[4] for row in records]),
        median_error_m=np.asarray([row[5] for row in records]),
        step_ms=np.asarray([row[6] for row in records]),
    )
    print(
        args.role,
        args.seed_frame,
        "scale",
        args.image_scale,
        "seeded",
        seeded,
        "fit frames",
        sum(row[4] >= tracker.min_inliers for row in records),
        "of",
        len(records),
        "median tracker-step ms",
        round(float(np.median([row[6] for row in records])), 2),
    )


if __name__ == "__main__":
    main()
