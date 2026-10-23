"""Show a late insertion pose disagreement on the recorded RGB-D frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from analyze_multiview import move

FRAMES = (496, 497, 498, 499, 500, 501, 502, 504)


def projected_ends(
    pose: np.ndarray, center: np.ndarray, axis: np.ndarray, offsets: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    ends = move(center[None] + offsets[:, None] * axis[None], pose)
    return np.column_stack(
        (
            ends[:, 0] / ends[:, 2] * intrinsics[0, 0] + intrinsics[0, 2],
            ends[:, 1] / ends[:, 2] * intrinsics[1, 1] + intrinsics[1, 2],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    traces = np.load(root / "pose_traces.npz")
    p2p = np.load(root / "p2p_dowel_internal_control_100_550.npz")
    points = np.load(args.points)
    seed_frame = int(points["seed_frame"])
    intrinsics = traces["intrinsics"]
    center = traces["dowel_reference_center_camera_m"]
    axis = traces["dowel_reference_shaft_camera"]
    offsets = traces["dowel_reference_end_offsets_m"]
    figure, axes = plt.subplots(2, 4, figsize=(12, 7.3), constrained_layout=True)
    x0, x1, y0, y1 = 155, 310, 70, 245
    for frame, panel in zip(FRAMES, axes.flat, strict=True):
        bgr = cv2.imread(str(args.dataset / "rgb" / f"{frame:06d}.jpg"))
        mask = cv2.imread(str(args.masks / "dowel" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise ValueError(f"missing frame {frame}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        panel.imshow(rgb[y0:y1, x0:x1])
        panel.contour(mask[y0:y1, x0:x1], levels=(0.5,), colors="lime", linewidths=1)
        if not p2p["lost"][frame - 100]:
            line = projected_ends(p2p["poses"][frame - 100], center, axis, offsets, intrinsics)
            panel.plot(line[:, 0] - x0, line[:, 1] - y0, color="#ff9f1a", linewidth=2.2)
        index = np.flatnonzero(points["frames"] == frame)
        if len(index) == 1 and np.isfinite(points["motions"][index[0]]).all():
            pose = points["motions"][index[0]] @ p2p["poses"][seed_frame - 100]
            line = projected_ends(pose, center, axis, offsets, intrinsics)
            panel.plot(line[:, 0] - x0, line[:, 1] - y0, color="#10c4e8", linewidth=2.2)
        title = f"{frame}: P2P {int(p2p['fit_inliers'][frame - 100])} inliers"
        if p2p["lost"][frame - 100]:
            title = f"{frame}: P2P lost"
        panel.set_title(title, fontsize=10)
        panel.axis("off")
    figure.text(
        0.48, 0.005, "Orange: Point2Pose   •   Cyan: causal points from frame 430   •   Green: SAM3 mask"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
