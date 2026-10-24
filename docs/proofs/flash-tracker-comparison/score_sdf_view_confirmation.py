"""Score a replay-only SDF trial against the same recorded RGB-D observations.

This measures visible-surface agreement, not the dowel's hidden tip or true 6D
pose. It intentionally does not import the FLASH runtime or control a robot.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from analyze_multiview import frame_points, move, surface_p90_mm
from analyze_ray_consistency import reference_points, score_rays
from analyze_temporal_pose_conflict import score as dowel_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("dowel", "ring"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    features = np.load(Path(__file__).parent / "pose_traces.npz")
    k = features["intrinsics"]
    reference = reference_points(args.dataset, args.masks, args.role, k)
    traces = {"baseline": np.load(args.baseline), "trial": np.load(args.trial)}
    if not np.array_equal(traces["baseline"]["frames"], traces["trial"]["frames"]):
        raise ValueError("the pose traces must have the same frame indices")
    first_frame = int(traces["baseline"]["frames"][0])
    rows = []
    for frame in range(400, 551):
        mask_image = cv2.imread(str(args.masks / args.role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        raw_depth = cv2.imread(str(args.dataset / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if mask_image is None or raw_depth is None or mask_image.shape != raw_depth.shape:
            raise ValueError(f"missing aligned RGB-D/mask frame {frame}")
        mask = mask_image > 0
        depth = raw_depth.astype(np.float32) * 0.001
        neighborhood = np.ones((3, 3), dtype=np.uint8)
        lower = cv2.erode(depth, neighborhood)
        upper = cv2.dilate(depth, neighborhood)
        valid_depth = (depth > 0) & (lower > 0) & (upper - lower <= 0.01)
        live, _ = frame_points(args.dataset, args.masks, args.role, frame, k, edge_limit_m=0.01)
        for name, trace in traces.items():
            index = frame - first_frame
            lost = bool(trace["lost"][index])
            if lost:
                rows.append((frame, name, True, np.nan, np.nan, np.nan))
                continue
            pose = trace["poses"][index]
            if args.role == "dowel":
                surface, axis_error, contradiction = dowel_score(
                    pose,
                    reference,
                    live,
                    mask,
                    depth,
                    valid_depth,
                    k,
                    features["dowel_reference_center_camera_m"],
                    features["dowel_reference_shaft_camera"],
                )
            else:
                surface = surface_p90_mm(live, move(reference, pose))
                axis_error = float("nan")
                rays = score_rays(reference, pose, depth, valid_depth, mask, k, depth_tolerance_m=0.005)
                contradiction = rays[3] / rays[1] if rays[1] else float("nan")
            rows.append((frame, name, False, surface, axis_error, contradiction))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ("frame", "candidate", "lost", "surface_p90_mm", "image_axis_deg", "ray_contradiction")
        )
        writer.writerows(rows)
    for start, stop in ((400, 500), (514, 550), (400, 550)):
        print(start, stop)
        for name in traces:
            values = np.asarray(
                [row[3:] for row in rows if start <= row[0] <= stop and row[1] == name and not row[2]],
                dtype=float,
            )
            lost_count = sum(row[2] for row in rows if start <= row[0] <= stop and row[1] == name)
            metric_columns = (0, 1, 2) if args.role == "dowel" else (0, 2)
            print(
                name,
                "valid",
                len(values),
                "lost",
                lost_count,
                "median",
                np.nanmedian(values[:, metric_columns], axis=0),
            )


if __name__ == "__main__":
    main()
