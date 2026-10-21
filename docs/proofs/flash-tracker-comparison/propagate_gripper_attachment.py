"""Propagate one observed object pose through recorded gripper motion.

This is a causal rigid-attachment baseline. It assumes the grasp does not slip
after the seed frame and does not incorporate later visual observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-trace", required=True, type=Path)
    parser.add_argument("--recording-trace", type=Path, default=Path(__file__).with_name("pose_traces.npz"))
    parser.add_argument("--seed-frame", required=True, type=int)
    parser.add_argument("--first-frame", required=True, type=int)
    parser.add_argument("--last-frame", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.seed_frame > args.first_frame or args.first_frame > args.last_frame:
        parser.error("the seed must precede the output interval")

    with np.load(args.recording_trace) as recording:
        camera_from_base = np.linalg.inv(recording["base_from_camera"])
        gripper_poses = recording["demo_ee_pose_base"].copy()
    if args.last_frame >= len(gripper_poses):
        parser.error("the recording has no gripper pose for the last frame")
    with np.load(args.pose_trace) as source:
        indices = np.flatnonzero(source["frames"] == args.seed_frame)
        if len(indices) != 1 or bool(source["lost"][indices[0]]):
            parser.error("the seed has no valid object pose")
        seed_pose = source["poses"][indices[0]].copy()
    if not np.isfinite(seed_pose).all() or not np.allclose(seed_pose[3], [0, 0, 0, 1]):
        parser.error("the seed must be a finite homogeneous pose")

    camera_from_ee_at_seed = camera_from_base @ gripper_poses[args.seed_frame]
    ee_from_object = np.linalg.inv(camera_from_ee_at_seed) @ seed_pose
    frames = np.arange(args.first_frame, args.last_frame + 1, dtype=np.int32)
    poses = np.stack([camera_from_base @ gripper_poses[int(frame)] @ ee_from_object for frame in frames])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        frames=frames,
        poses=poses,
        lost=np.zeros(len(frames), dtype=bool),
        ee_from_object=ee_from_object,
        metadata=np.asarray(
            json.dumps(
                {
                    "method": "rigid_gripper_attachment",
                    "seed_frame": args.seed_frame,
                    "seed_source": str(args.pose_trace),
                }
            )
        ),
    )
    print(args.output)


if __name__ == "__main__":
    main()
