"""Compare dowel pose candidates at the shaft and candidate contact ends.

The center, shaft and two end offsets are frame-100 depth-cloud features, not
calibrated landmarks or independent ground truth. End-pair matching permits a
symmetry swap but does not decide which end is held or enters the ring.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-frame", type=int, default=400)
    parser.add_argument("--last-frame", type=int, default=500)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.last_frame < args.first_frame:
        parser.error("last frame precedes first frame")
    with np.load(root / "pose_traces.npz") as trace:
        frames = trace["frames"].copy()
        center = trace["dowel_reference_center_camera_m"].copy()
        shaft = trace["dowel_reference_shaft_camera"].copy()
        offsets = trace["dowel_reference_end_offsets_m"].copy()
        candidates = {
            "tapir": (trace["p2p_dowel_delta"].copy(), trace["p2p_dowel_lost"].copy()),
            "bundle": (trace["bundlesdf_dowel_delta"].copy(), trace["bundlesdf_dowel_failed"].copy()),
        }
    for specification in args.candidate:
        name, separator, source = specification.partition("=")
        if not separator or not name or not source or name in candidates:
            parser.error("candidate must be a distinct NAME=POSE_TRACE_PATH")
        with np.load(source) as candidate:
            if not np.array_equal(frames, candidate["frames"]):
                raise ValueError(f"candidate {name} camera frames are not aligned")
            candidates[name] = (candidate["poses"].copy(), candidate["lost"].copy())
    records = []
    for frame in range(args.first_frame, args.last_frame + 1):
        index = frame - int(frames[0])
        if index < 0 or index >= len(frames) or frames[index] != frame:
            raise ValueError(f"frame {frame} absent from reference trace")
        for (first_name, (first_poses, first_lost)), (
            second_name,
            (second_poses, second_lost),
        ) in combinations(candidates.items(), 2):
            if first_lost[index] or second_lost[index]:
                continue
            first_pose, second_pose = first_poses[index], second_poses[index]
            first_center = first_pose[:3, :3] @ center + first_pose[:3, 3]
            second_center = second_pose[:3, :3] @ center + second_pose[:3, 3]
            first_axis = first_pose[:3, :3] @ shaft
            second_axis = second_pose[:3, :3] @ shaft
            first_ends = np.stack([first_center + offset * first_axis for offset in offsets])
            second_ends = np.stack([second_center + offset * second_axis for offset in offsets])
            direct = np.linalg.norm(first_ends - second_ends, axis=1)
            swapped = np.linalg.norm(first_ends - second_ends[::-1], axis=1)
            best = direct if direct.sum() <= swapped.sum() else swapped
            records.append(
                (
                    frame,
                    first_name,
                    second_name,
                    float(np.linalg.norm(first_center - second_center) * 1000),
                    float(np.degrees(np.arccos(np.clip(abs(first_axis @ second_axis), 0, 1)))),
                    float(direct[1] * 1000),
                    float(np.mean(best) * 1000),
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "first",
                "second",
                "center_gap_mm",
                "unoriented_shaft_gap_deg",
                "same_end_1_gap_mm",
                "unordered_end_pair_mean_gap_mm",
            )
        )
        writer.writerows(records)
    for first_name, second_name in combinations(candidates, 2):
        pair = [row for row in records if row[1:3] == (first_name, second_name)]
        if not pair:
            continue
        print(
            first_name,
            second_name,
            len(pair),
            "center_mm",
            np.median([row[3] for row in pair]),
            "shaft_deg",
            np.median([row[4] for row in pair]),
            "same_end_mm",
            np.median([row[5] for row in pair]),
            "unordered_ends_mm",
            np.median([row[6] for row in pair]),
        )


if __name__ == "__main__":
    main()
