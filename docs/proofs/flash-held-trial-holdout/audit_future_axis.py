"""Check whether held-axis proposals predict later RGB-D directions.

The object is expressed in the measured gripper's local frame at each camera
time. A rigid held object should keep a stable local direction. This checks
future visible geometry, not the hidden tip or a ground-truth six-axis pose.
"""

import argparse
import gzip
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np


def load(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def gap_deg(first, second):
    return float(np.degrees(np.arccos(np.clip(abs(first @ second), -1, 1))))


def analyze(result_path: Path, audit_path: Path, horizons_s: list[float]) -> dict:
    result = load(result_path)
    held = [
        item
        for item in result["frames"]
        if result["grasp_time_s"] <= item["source_time_s"] < result["release_time_s"]
    ]
    direct_rows = [item["row"] for item in held if item["held_evidence"] == "observed"]
    if not direct_rows:
        raise ValueError("the held interval has no accepted direct pose")
    last_direct_row = max(direct_rows)
    measurements = [
        item
        for item in load(audit_path)["measurements"]
        if item["row"] > last_direct_row
        and item["tracked_surface_axis_gripper"] is not None
        and all(name in item["proposals"] for name in ("vision_candidate", "gripper_fallback"))
    ]
    times = [item["source_time_s"] for item in measurements]
    if any(right <= left for left, right in zip(times, times[1:], strict=False)):
        raise ValueError("camera measurements must increase in source time")
    summaries = []
    for horizon_s in horizons_s:
        if not np.isfinite(horizon_s) or horizon_s <= 0:
            raise ValueError("prediction horizon must be positive and finite")
        rows = []
        for index, item in enumerate(measurements):
            future_index = bisect_left(times, times[index] + horizon_s)
            if future_index == len(measurements):
                continue
            future = measurements[future_index]
            measured = np.asarray(future["tracked_surface_axis_gripper"])
            candidate = np.asarray(item["proposals"]["vision_candidate"]["axis_gripper"])
            gripper = np.asarray(item["proposals"]["gripper_fallback"]["axis_gripper"])
            rows.append(
                {
                    "row": item["row"],
                    "future_row": future["row"],
                    "actual_horizon_s": future["source_time_s"] - item["source_time_s"],
                    "vision_gap_deg": gap_deg(candidate, measured),
                    "gripper_gap_deg": gap_deg(gripper, measured),
                }
            )
        summaries.append(
            {
                "requested_horizon_s": horizon_s,
                "compared_frames": len(rows),
                "vision_closer_frames": sum(
                    item["vision_gap_deg"] < item["gripper_gap_deg"] for item in rows
                ),
                "median_vision_gap_deg": float(np.median([item["vision_gap_deg"] for item in rows]))
                if rows
                else None,
                "median_gripper_gap_deg": float(np.median([item["gripper_gap_deg"] for item in rows]))
                if rows
                else None,
                "samples": rows,
            }
        )
    return {
        "trial": result["trial"],
        "last_direct_row": last_direct_row,
        "measurement_frames_after_last_direct": len(measurements),
        "horizons": summaries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--axis-audit", type=Path, required=True)
    parser.add_argument("--horizon-s", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.result, args.axis_audit, args.horizon_s)
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wt", encoding="utf-8") as stream:
            json.dump(report, stream)
    else:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
