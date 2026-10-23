"""Show task-relevant shaft/end projections over recorded insertion RGB.

The line endpoints come from the frame-100 depth-cloud proxy transported by
each causal pose trace. They are hypotheses, not annotated physical tips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def project(point: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    if point[2] <= 0:
        raise ValueError("candidate point lies behind camera")
    return np.array(
        (
            intrinsics[0, 0] * point[0] / point[2] + intrinsics[0, 2],
            intrinsics[1, 1] * point[1] / point[2] + intrinsics[1, 2],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, nargs="+", default=(400, 430, 450, 475, 498, 500))
    args = parser.parse_args()
    if len(args.frames) != 6:
        parser.error("the two-by-three comparison requires exactly six frames")
    root = Path(__file__).parent
    with np.load(root / "pose_traces.npz") as reference:
        intrinsics = reference["intrinsics"].copy()
        center = reference["dowel_reference_center_camera_m"].copy()
        shaft = reference["dowel_reference_shaft_camera"].copy()
        offsets = reference["dowel_reference_end_offsets_m"].copy()
    candidates = {}
    for label, filename, color in (
        ("Point2Pose", "p2p_dowel_internal_control_100_550.npz", "#ffad20"),
        ("TAPNext 256", "tapnext_fixed_dowel_100_550_pose.npz", "#f24c93"),
        ("TAPNext 512", "tapnext512_fixed_dowel_100_550_pose.npz", "#a5dd47"),
    ):
        with np.load(root / filename) as trace:
            candidates[label] = (trace["frames"].copy(), trace["poses"].copy(), trace["lost"].copy(), color)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, frame in zip(axes.flat, args.frames, strict=True):
        image = cv2.imread(str(args.dataset / "rgb" / f"{frame:06d}.jpg"))
        mask = cv2.imread(str(args.masks / "dowel" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"missing recorded RGB/mask at frame {frame}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rows, columns = np.nonzero(mask)
        if not len(rows):
            raise ValueError(f"dowel mask empty at frame {frame}")
        pad = 70
        left, right = max(0, int(columns.min()) - pad), min(image.shape[1], int(columns.max()) + pad)
        top, bottom = max(0, int(rows.min()) - pad), min(image.shape[0], int(rows.max()) + pad)
        rgb = image[top:bottom, left:right]
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        edge_rows, edge_columns = np.nonzero(edges[top:bottom, left:right])
        rgb[edge_rows, edge_columns] = np.array((0, 240, 245), np.uint8)
        axis.imshow(rgb)
        for label, (frames, poses, lost, color) in candidates.items():
            index = np.flatnonzero(frames == frame)
            if len(index) != 1 or lost[index[0]]:
                continue
            pose = poses[index[0]]
            endpoints = [center + offset * shaft for offset in offsets]
            projected = [project(pose[:3, :3] @ point + pose[:3, 3], intrinsics) for point in endpoints]
            pixels = np.stack(projected) - np.array((left, top))
            axis.plot(pixels[:, 0], pixels[:, 1], "-o", color=color, linewidth=2, markersize=5, label=label)
        axis.set_title(f"frame {frame}")
        axis.set_axis_off()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    main()
