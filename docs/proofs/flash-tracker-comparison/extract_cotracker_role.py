"""Extract one role's strictly causal endpoint tracks from a shared run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", required=True, type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.shared) as track:
        roles = [str(role) for role in track["roles"]]
        counts = track["role_query_counts"].astype(int)
        if args.role not in roles:
            raise ValueError(f"role {args.role} absent from shared trace")
        index = roles.index(args.role)
        start = int(counts[:index].sum())
        stop = start + int(counts[index])
        if stop > len(track["all_seed_uv"]):
            raise ValueError("shared role point counts exceed query array")
        output = {
            "role": np.asarray(args.role),
            "seed_frame": track["seed_frame"].copy(),
            "seed_uv": track["all_seed_uv"][start:stop].copy(),
            "frames": track["frames"].copy(),
            "uv": track["all_uv"][:, start:stop].copy(),
            "visible": track["all_visible"][:, start:stop].copy(),
            "step_ms": track["step_ms"].copy(),
            "window_frames": track["window_frames"].copy(),
            "stride_frames": track["stride_frames"].copy(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(args.output, "queries", stop - start, "endpoints", len(output["frames"]))


if __name__ == "__main__":
    main()
