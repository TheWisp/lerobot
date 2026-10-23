"""Compare two sparse pose traces at first-view dowel task features.

The chosen contact end is a geometric proxy; this compares two predictions,
not either prediction with independently measured hidden-tip ground truth.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(Path(__file__).with_name("pose_traces.npz")) as reference:
        center = reference["dowel_reference_center_camera_m"].copy()
        shaft = reference["dowel_reference_shaft_camera"].copy()
        offsets = reference["dowel_reference_end_offsets_m"].copy()
    with np.load(args.first) as source:
        first = {key: source[key].copy() for key in ("frames", "poses", "lost")}
    with np.load(args.second) as source:
        second = {key: source[key].copy() for key in ("frames", "poses", "lost")}
    if np.any(np.diff(first["frames"]) <= 0) or np.any(np.diff(second["frames"]) <= 0):
        raise ValueError("pose frames must be strictly increasing")
    points = np.stack((center, center + offsets[0] * shaft, center + offsets[1] * shaft))
    rows = []
    for frame in np.intersect1d(first["frames"], second["frames"]):
        i = int(np.searchsorted(first["frames"], frame))
        j = int(np.searchsorted(second["frames"], frame))
        if first["lost"][i] or second["lost"][j]:
            continue
        a, b = first["poses"][i], second["poses"][j]
        moved_a = points @ a[:3, :3].T + a[:3, 3]
        moved_b = points @ b[:3, :3].T + b[:3, 3]
        distances = np.linalg.norm(moved_a - moved_b, axis=1) * 1000
        shaft_cosine = abs(float((a[:3, :3] @ shaft) @ (b[:3, :3] @ shaft)))
        angle = float(np.degrees(np.arccos(np.clip(shaft_cosine, 0, 1))))
        rows.append((int(frame), *distances, angle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ("frame", "center_gap_mm", "visible_end_gap_mm", "contact_end_gap_mm", "shaft_axis_gap_deg")
        )
        writer.writerows(rows)
    print(args.output, "shared valid frames", len(rows))


if __name__ == "__main__":
    main()
