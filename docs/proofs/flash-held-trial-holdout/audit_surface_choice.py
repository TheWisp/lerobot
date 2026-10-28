"""Audit whether same-image mask/depth dominance selects a useful held direction.

This is an offline diagnostic on two recorded attempts, not a calibrated pose
confidence or an action rule. The later image checks direction only.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def dominance(visual: dict, gripper: dict, *, residual_key: str, min_supported_pixels: int) -> str:
    if min_supported_pixels < 1:
        raise ValueError("supported-pixel minimum must be positive")
    if any(item.get("supported_pixels", 0) < min_supported_pixels for item in (visual, gripper)):
        return "insufficient"
    v_residual, g_residual = visual.get(residual_key), gripper.get(residual_key)
    if v_residual is None or g_residual is None:
        return "insufficient"

    def dominates(first: dict, second: dict, first_residual: float, second_residual: float) -> bool:
        return (
            first["mask_fraction"] >= second["mask_fraction"]
            and first_residual <= second_residual
            and (first["mask_fraction"] > second["mask_fraction"] or first_residual < second_residual)
        )

    if dominates(visual, gripper, v_residual, g_residual):
        return "vision"
    if dominates(gripper, visual, g_residual, v_residual):
        return "gripper"
    return "ambiguous"


def audit(axis_audit: dict, future_axis: dict, *, horizon_s: float, min_supported_pixels: int) -> dict:
    if horizon_s <= 0:
        raise ValueError("future-axis probe must have a positive horizon")
    horizon = next(
        (item for item in future_axis["horizons"] if np.isclose(item["requested_horizon_s"], horizon_s)),
        None,
    )
    if horizon is None:
        raise ValueError(f"future-axis audit has no {horizon_s:g} s probe")
    measurements = {item["row"]: item for item in axis_audit["measurements"]}
    result = {}
    for residual_key in ("depth_residual_p90_mm", "clean_depth_residual_p90_mm"):
        counts = Counter()
        closer = Counter()
        for sample in horizon["samples"]:
            measurement = measurements.get(sample["row"])
            if measurement is None:
                counts["missing"] += 1
                continue
            proposals = measurement["proposals"]
            if "vision_candidate" not in proposals or "gripper_fallback" not in proposals:
                counts["missing"] += 1
                continue
            choice = dominance(
                proposals["vision_candidate"],
                proposals["gripper_fallback"],
                residual_key=residual_key,
                min_supported_pixels=min_supported_pixels,
            )
            counts[choice] += 1
            if (choice == "vision" and sample["vision_gap_deg"] < sample["gripper_gap_deg"]) or (
                choice == "gripper" and sample["gripper_gap_deg"] < sample["vision_gap_deg"]
            ):
                closer[choice] += 1
        result[residual_key] = {
            "choices": dict(counts),
            "chosen_direction_closer_to_later_visible_direction": dict(closer),
        }
    return {
        "trial": future_axis["trial"],
        "future_horizon_s": horizon_s,
        "min_supported_pixels": min_supported_pixels,
        "compared_frames": len(horizon["samples"]),
        "results": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-audit", required=True, type=Path)
    parser.add_argument("--future-axis", required=True, type=Path)
    parser.add_argument("--min-supported-pixels", required=True, type=int)
    parser.add_argument("--horizon-s", type=float, default=0.6)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                read(args.axis_audit),
                read(args.future_axis),
                horizon_s=args.horizon_s,
                min_supported_pixels=args.min_supported_pixels,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
