"""Score dowel reacquisition against the recorded RGB-D and gripper prior.

The single-view surface score is diagnostic only. In particular, a mask can
cover visible RGB while the depth camera returns zeros or foreground clutter.

Example:
    python analyze_reacquisition.py --dataset /path/to/insertion/recording \
        --masks /path/to/sam3_masks --output /tmp/flash-reacquisition
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from analyze_multiview import frame_points, move, surface_p90_mm

FIRST_FRAME = 100
GRASP_REFERENCE_FRAME = 130
ATTACH_FRAME = 228
SCORE_FRAMES = range(249, 301)
ILLUSTRATED_FRAMES = (249, 260, 262, 276)


def gripper_prediction(traces: np.lib.npyio.NpzFile, frame: int, shift: int = 0) -> np.ndarray:
    base_from_camera = traces["base_from_camera"]
    end_effector = traces["demo_ee_pose_base"]
    object_at_reference = traces["p2p_dowel_delta"][GRASP_REFERENCE_FRAME - FIRST_FRAME]
    attachment = np.linalg.inv(end_effector[ATTACH_FRAME + shift]) @ base_from_camera @ object_at_reference
    return np.linalg.inv(base_from_camera) @ end_effector[frame + shift] @ attachment


def plot_depth(dataset: Path, masks: Path, output: Path) -> None:
    figure, axes = plt.subplots(2, len(ILLUSTRATED_FRAMES), figsize=(12, 5.5), constrained_layout=True)
    roi = (slice(105, 335), slice(0, 320))
    for column, frame in enumerate(ILLUSTRATED_FRAMES):
        stem = f"{frame:06d}"
        rgb = cv2.cvtColor(cv2.imread(str(dataset / "rgb" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(dataset / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED) * 0.001
        mask = cv2.imread(str(masks / "dowel" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 0
        axes[0, column].imshow(rgb[roi])
        axes[0, column].contour(mask[roi], levels=[0.5], colors="lime", linewidths=1)
        axes[0, column].set_title(f"RGB {frame}")
        axes[1, column].imshow(
            np.ma.masked_where(depth[roi] == 0, depth[roi]), vmin=0.2, vmax=0.5, cmap="turbo"
        )
        axes[1, column].contour(mask[roi], levels=[0.5], colors="white", linewidths=1)
        axes[1, column].set_title(f"Depth {frame} (m); white is missing")
        for row in range(2):
            axes[row, column].axis("off")
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--edge-limit-mm", type=float, default=10.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    traces = np.load(Path(__file__).with_name("pose_traces.npz"))
    intrinsics = traces["intrinsics"]
    first, _ = frame_points(args.dataset, args.masks, "dowel", FIRST_FRAME, intrinsics, max_points=4000)
    centre = traces["dowel_reference_center_camera_m"][None, :]
    rows = []
    for frame in SCORE_FRAMES:
        mask = cv2.imread(str(args.masks / "dowel" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        depth = cv2.imread(str(args.dataset / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        mask_pixels = int(mask.sum())
        valid_depth_fraction = (
            float(((depth > 0) & mask).sum() / mask_pixels) if mask_pixels else float("nan")
        )
        interior, interior_count = frame_points(
            args.dataset, args.masks, "dowel", frame, intrinsics, edge_limit_m=args.edge_limit_mm / 1000
        )
        index = frame - FIRST_FRAME
        p2p_lost = bool(traces["p2p_dowel_lost"][index])
        p2p = traces["p2p_dowel_delta"][index]
        predicted = gripper_prediction(traces, frame)
        center_gap_mm = (
            float(1e3 * np.linalg.norm(move(centre, p2p) - move(centre, predicted)))
            if not p2p_lost
            else float("nan")
        )
        prediction_scores = [
            surface_p90_mm(interior, move(first, gripper_prediction(traces, frame, shift)))
            for shift in range(-3, 4)
        ]
        rows.append(
            (
                frame,
                mask_pixels,
                valid_depth_fraction,
                interior_count,
                p2p_lost,
                surface_p90_mm(interior, move(first, p2p)) if not p2p_lost else float("nan"),
                surface_p90_mm(interior, move(first, predicted)),
                min(prediction_scores),
                center_gap_mm,
            )
        )
    with (args.output / "reacquisition_score.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "sam3_mask_pixels",
                "valid_depth_fraction",
                "interior_depth_points",
                "p2p_lost",
                "p2p_surface_p90_mm",
                "gripper_surface_p90_mm",
                "best_gripper_shift_3_surface_p90_mm",
                "pose_center_gap_mm",
            )
        )
        writer.writerows(rows)
    plot_depth(args.dataset, args.masks, args.output / "reacquisition_depth.png")


if __name__ == "__main__":
    main()
