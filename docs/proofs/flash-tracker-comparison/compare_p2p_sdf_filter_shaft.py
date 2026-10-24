"""Score replayed dowel poses against the current visible RGB-D shaft.

This is a cylinder-specific diagnostic, not a FLASH runtime pose check. The
radius comes from the first clear demo frame; the center is fit on one half of
the current shaft pixels and scored on the other half.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from analyze_dowel_release_geometry import (
    angle_degrees,
    cloud_and_pixels,
    fit_circle,
    held_out_circle_error,
    shaft_middle,
)

HERE = Path(__file__).parent
TRACES = {
    "baseline": "p2p_dowel_internal_control_100_550.npz",
    "window": "p2p_dowel_sdf_window_100_550.npz",
    "depth": "p2p_dowel_sdf_depth_component_100_550.npz",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "p2p_sdf_filter_shaft_evidence.csv")
    args = parser.parse_args()
    reference = np.load(HERE / "pose_traces.npz")
    k = reference["intrinsics"]
    reference_axis = reference["dowel_reference_shaft_camera"]
    initial_points, initial_pixels, _ = cloud_and_pixels(args.dataset, args.masks, 100, k)
    initial_middle, _, initial_axis, _ = shaft_middle(initial_points, initial_pixels)
    radius, _ = fit_circle(initial_middle, initial_axis, None)
    traces = {name: np.load(HERE / path)["poses"] for name, path in TRACES.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("frame", "candidate", "depth_axis_ratio", "axis_error_deg", "circle_p90_mm"))
        for frame in range(400, 551):
            points, pixels, _ = cloud_and_pixels(args.dataset, args.masks, frame, k)
            if len(points) < 20:
                continue
            middle, middle_pixels, _, ratio = shaft_middle(points, pixels)
            train = (middle_pixels[:, 0] + middle_pixels[:, 1]) % 2 == 0
            _, eigenvectors = np.linalg.eigh(np.cov(middle[train].T))
            depth_axis = eigenvectors[:, -1]
            for name, poses in traces.items():
                estimated_axis = poses[frame - 100, :3, :3] @ reference_axis
                circle_error = held_out_circle_error(middle, middle_pixels, estimated_axis, radius)
                writer.writerow(
                    (
                        frame,
                        name,
                        ratio,
                        angle_degrees(depth_axis, estimated_axis),
                        float(np.quantile(circle_error, 0.9) * 1000),
                    )
                )


if __name__ == "__main__":
    main()
