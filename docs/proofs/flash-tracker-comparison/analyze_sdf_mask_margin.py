"""Count where a mask-margin filter removes saved Point2Pose SDF inputs.

The generous shaft radius is an oracle for this recording only. It helps
diagnose whether bad keyframe pixels cluster at the mask edge; it is not a
proposed runtime object model or threshold.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--depth-factor", required=True, type=float)
    parser.add_argument("--shaft-radius-mm", required=True, type=float)
    parser.add_argument("--margins-px", nargs="+", type=float, default=[1, 2, 4])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = np.load(args.reference)
    center = reference["dowel_reference_center_camera_m"]
    axis = reference["dowel_reference_shaft_camera"]
    axis /= np.linalg.norm(axis)
    rows = []
    for path in sorted(args.snapshots.glob("object_0_frame_*.npz"))[1:]:
        snapshot = np.load(path)
        mask = snapshot["mask"].astype(bool)
        depth = snapshot["selected_depth"] / args.depth_factor
        y, x = np.nonzero(mask & (depth > 0))
        z = depth[y, x]
        k = snapshot["intrinsics"]
        pose = snapshot["pose"]
        camera = np.stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z), axis=1)
        object_points = (camera - pose[:3, 3]) @ pose[:3, :3]
        radial_mm = np.linalg.norm(np.cross(object_points - center, axis), axis=1) * 1000
        suspicious = radial_mm > args.shaft_radius_mm
        boundary_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for margin in args.margins_px:
            removed = boundary_distance[y, x] <= margin
            rows.append(
                (
                    path.stem,
                    margin,
                    len(z),
                    int(suspicious.sum()),
                    int((~suspicious).sum()),
                    int(np.count_nonzero(suspicious & removed)),
                    int(np.count_nonzero(~suspicious & removed)),
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "keyframe",
                "margin_px",
                "valid_mask_depth",
                "beyond_shaft_20mm",
                "within_shaft_20mm",
                "beyond_removed",
                "within_removed",
            )
        )
        writer.writerows(rows)
    for margin in args.margins_px:
        subset = [row for row in rows if row[1] == margin]
        far = sum(row[3] for row in subset)
        near = sum(row[4] for row in subset)
        far_removed = sum(row[5] for row in subset)
        near_removed = sum(row[6] for row in subset)
        print(
            f"{margin:g} px: {far_removed}/{far} beyond-shaft pixels removed; {near_removed}/{near} within-shaft removed"
        )


if __name__ == "__main__":
    main()
