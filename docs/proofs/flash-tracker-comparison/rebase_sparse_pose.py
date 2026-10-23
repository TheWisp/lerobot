"""Express a later-seeded relative pose trace in the frame-100 reference.

The fitted transform maps points from the later seed image to each current image.
The supplied causal anchor maps the frame-100 reference into that seed image.
No future pose is used to rebase an endpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--anchor-pose-key", required=True)
    parser.add_argument("--anchor-lost-key", required=True)
    parser.add_argument(
        "--self-normalize",
        action="store_true",
        help="remove a relative estimator's nonidentity self-pose at its seed frame",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.relative) as relative:
        data = {key: relative[key].copy() for key in relative.files}
    seed_frame = int(data["seed_frame"]) if "seed_frame" in data else int(data["frames"][0])
    with np.load(args.anchor) as anchor:
        indices = np.flatnonzero(anchor["frames"] == seed_frame)
        if len(indices) != 1 or bool(anchor[args.anchor_lost_key][indices[0]]):
            raise ValueError("the seed has no valid causal anchor pose")
        seed_from_reference = anchor[args.anchor_pose_key][indices[0]].copy()
    if not np.isfinite(seed_from_reference).all() or not np.allclose(seed_from_reference[3], [0, 0, 0, 1]):
        raise ValueError("anchor must be a finite homogeneous pose")
    if args.self_normalize:
        if int(data["frames"][0]) != seed_frame or bool(data["lost"][0]):
            raise ValueError("self-normalization needs a valid first pose at the seed frame")
        seed_self_pose = data["poses"][0]
        if not np.isfinite(seed_self_pose).all() or not np.allclose(seed_self_pose[3], [0, 0, 0, 1]):
            raise ValueError("self-normalization needs a finite homogeneous seed pose")
        data["poses"] = data["poses"] @ np.linalg.inv(seed_self_pose)
    data["poses"] = data["poses"] @ seed_from_reference
    data["seed_frame"] = np.asarray(seed_frame)
    data["anchor_source"] = np.asarray(str(args.anchor))
    data["anchor_pose_key"] = np.asarray(args.anchor_pose_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(args.output, "valid", int((~data["lost"]).sum()), "/", len(data["frames"]))


if __name__ == "__main__":
    main()
