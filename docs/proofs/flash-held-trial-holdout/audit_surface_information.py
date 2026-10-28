"""Compare conditional surface information with later visible direction evidence.

The later-direction check is a counterfactual audit, not a pose ground truth or
a runtime threshold. Source frames are correlated and come from one attempt.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def audit(result: dict, future_axis: dict, *, horizon_s: float) -> dict:
    if horizon_s <= 0:
        raise ValueError("future-axis probe must have a positive horizon")
    horizon = next(
        (item for item in future_axis["horizons"] if np.isclose(item["requested_horizon_s"], horizon_s)),
        None,
    )
    if horizon is None:
        raise ValueError(f"future-axis audit has no {horizon_s:g} s probe")
    frames = {frame["row"]: frame for frame in result["frames"]}
    compared = [
        (sample, frames[sample["row"]]["conditional_surface_information"])
        for sample in horizon["samples"]
        if sample["row"] in frames
        and frames[sample["row"]].get("conditional_surface_information") is not None
        and sample.get("motion_only_gap_deg") is not None
    ]
    return {
        "trial": result["trial"],
        "future_horizon_s": horizon_s,
        "camera_frames": len(result["frames"]),
        "frames_with_surface_information": sum(
            frame.get("conditional_surface_information") is not None for frame in result["frames"]
        ),
        "future_direction_compared_frames": len(compared),
        "median_matched_reference_fraction": (
            float(
                np.median(
                    [info["matched_reference_points"] / info["reference_points"] for _, info in compared]
                )
            )
            if compared
            else None
        ),
        "vision_direction_closer_than_motion_only_frames": sum(
            sample["vision_gap_deg"] < sample["motion_only_gap_deg"] for sample, _ in compared
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--future-axis", required=True, type=Path)
    parser.add_argument("--horizon-s", type=float, default=0.6)
    args = parser.parse_args()
    print(json.dumps(audit(_read(args.result), _read(args.future_axis), horizon_s=args.horizon_s), indent=2))


if __name__ == "__main__":
    main()
