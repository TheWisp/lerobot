"""Compare causal pose histories at the dowel's inferred contact end.

The reference end is a first-view depth-cloud proxy, not an independently
measured insertion tip. All distances below are between predictions.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=POSE_TRACE.npz")
    parser.add_argument("--first-frame", type=int, default=400)
    parser.add_argument("--last-frame", type=int, default=500)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.candidate) < 2 or args.last_frame < args.first_frame:
        parser.error("provide at least two candidates and a valid frame interval")

    with np.load(Path(__file__).with_name("pose_traces.npz")) as reference:
        center = reference["dowel_reference_center_camera_m"].copy()
        axis = reference["dowel_reference_shaft_camera"].copy()
        offset = float(reference["dowel_reference_end_offsets_m"][1])
    end = center + offset * axis

    traces: dict[str, dict[int, np.ndarray]] = {}
    for specification in args.candidate:
        name, separator, filename = specification.partition("=")
        if not separator or not name or not filename or name in traces:
            parser.error("each candidate must be a unique NAME=POSE_TRACE.npz")
        with np.load(filename) as trace:
            traces[name] = {
                int(frame): pose.copy()
                for frame, pose, lost in zip(trace["frames"], trace["poses"], trace["lost"], strict=True)
                if not bool(lost)
            }

    rows = []
    for frame in range(args.first_frame, args.last_frame + 1):
        for (first_name, first), (second_name, second) in combinations(traces.items(), 2):
            if frame not in first or frame not in second:
                continue
            first_pose, second_pose = first[frame], second[frame]
            first_end = first_pose[:3, :3] @ end + first_pose[:3, 3]
            second_end = second_pose[:3, :3] @ end + second_pose[:3, 3]
            first_axis, second_axis = first_pose[:3, :3] @ axis, second_pose[:3, :3] @ axis
            unoriented_angle = np.degrees(np.arccos(np.clip(abs(first_axis @ second_axis), -1, 1)))
            rows.append(
                (
                    frame,
                    first_name,
                    second_name,
                    float(np.linalg.norm(first_end - second_end) * 1000),
                    float(unoriented_angle),
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ("frame", "first", "second", "predicted_contact_end_gap_mm", "unoriented_axis_gap_deg")
        )
        writer.writerows(rows)
    for first_name, second_name in combinations(traces, 2):
        paired = np.asarray(
            [(row[3], row[4]) for row in rows if row[1:3] == (first_name, second_name)], dtype=float
        )
        if len(paired):
            print(
                first_name, "vs", second_name, len(paired), "median end/axis gap", np.median(paired, axis=0)
            )


if __name__ == "__main__":
    main()
