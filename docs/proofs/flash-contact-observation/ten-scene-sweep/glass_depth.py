"""Compare jar-mask depth with a robust plane fit to the nearby tabletop.

This is a diagnostic for source episode 32, not a transparent-object depth model.
The source masks and RGB-D frames are in the imported local LeRobot dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def measure(root: Path, frame: int = 173) -> dict:
    flash = root / "flash" / "v1"
    source = json.loads((flash / "source.json").read_text())
    dataset = LeRobotDataset(source["repo_id"], root=root, video_backend="pyav")
    depth_mm = dataset[frame][source["depth_feature"]].numpy().squeeze()
    jar = cv2.imread(str(flash / f"imported-masks/object_b/{frame:06d}.png"), 0) > 0
    lid = cv2.imread(str(flash / f"imported-masks/object_a/{frame:06d}.png"), 0) > 0
    if not np.any(jar):
        raise ValueError("No source jar mask in the selected frame")

    # A five-pixel band around the mask samples the tabletop. Exclude both
    # annotated objects and invalid depth; the band may still contain other
    # surfaces, so iteratively reject values >10 mm from the fitted plane.
    neighborhood = cv2.dilate(jar.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    table = neighborhood & ~jar & ~lid & np.isfinite(depth_mm) & (depth_mm > 0)
    yy, xx = np.indices(depth_mm.shape)
    coordinates = np.stack((xx[table], yy[table], np.ones(np.count_nonzero(table))), axis=1)
    values = depth_mm[table]
    support = np.abs(values - np.median(values)) <= 10
    for _ in range(4):
        coefficients = np.linalg.lstsq(coordinates[support], values[support], rcond=None)[0]
        support = np.abs(values - coordinates @ coefficients) <= 10
    plane = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
    inside_residual = np.abs(depth_mm[jar] - plane[jar])
    outside_residual = np.abs(depth_mm[table] - plane[table])
    return {
        "dataset": str(root),
        "frame": frame,
        "source_jar_mask_pixels": int(np.count_nonzero(jar)),
        "nearby_table_pixels": int(np.count_nonzero(table)),
        "jar_nonzero_depth_fraction": float(np.mean(np.isfinite(depth_mm[jar]) & (depth_mm[jar] > 0))),
        "jar_depth_minus_table_median_absolute_mm": float(np.median(inside_residual)),
        "jar_depth_minus_table_p90_absolute_mm": float(np.percentile(inside_residual, 90)),
        "jar_pixels_within_5mm_of_table_plane_fraction": float(np.mean(inside_residual <= 5)),
        "outside_table_median_absolute_mm": float(np.median(outside_residual)),
        "plane_fit_support_fraction": float(np.mean(support)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--frame", type=int, default=173)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.dataset, args.frame)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
