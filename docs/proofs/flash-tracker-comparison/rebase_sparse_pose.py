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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.relative) as relative:
        data = {key: relative[key].copy() for key in relative.files}
    with np.load(args.anchor) as anchor:
        indices = np.flatnonzero(anchor["frames"] == data["seed_frame"])
        if len(indices) != 1 or bool(anchor[args.anchor_lost_key][indices[0]]):
            raise ValueError("the seed has no valid causal anchor pose")
        seed_from_reference = anchor[args.anchor_pose_key][indices[0]].copy()
    if not np.isfinite(seed_from_reference).all() or not np.allclose(seed_from_reference[3], [0, 0, 0, 1]):
        raise ValueError("anchor must be a finite homogeneous pose")
    data["poses"] = data["poses"] @ seed_from_reference
    data["anchor_source"] = np.asarray(str(args.anchor))
    data["anchor_pose_key"] = np.asarray(args.anchor_pose_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(args.output, "valid", int((~data["lost"]).sum()), "/", len(data["frames"]))


if __name__ == "__main__":
    main()
