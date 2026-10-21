"""Project two causal Point2Pose branches from different first-frame masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from render_late_pose_jump import projected_ends

FRAMES = (300, 400, 470, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--alternative", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    features = np.load(Path(__file__).with_name("pose_traces.npz"))
    baseline = np.load(args.baseline)
    alternative = np.load(args.alternative)
    center = features["dowel_reference_center_camera_m"]
    axis = features["dowel_reference_shaft_camera"]
    offsets = features["dowel_reference_end_offsets_m"]
    intrinsics = features["intrinsics"]
    x0, x1, y0, y1 = 100, 330, 0, 270
    figure, axes = plt.subplots(2, 2, figsize=(9, 9.8))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.91, wspace=0.08, hspace=0.09)
    for frame, panel in zip(FRAMES, axes.flat, strict=True):
        bgr = cv2.imread(str(args.dataset / "rgb" / f"{frame:06d}.jpg"))
        mask = cv2.imread(str(args.masks / "dowel" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise ValueError(f"missing frame {frame}")
        panel.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1])
        panel.contour(mask[y0:y1, x0:x1], levels=(0.5,), colors="#46e377", linewidths=1.3)
        for trace, color in ((baseline, "#ff9f1a"), (alternative, "#d44af7")):
            if trace["lost"][frame - 100]:
                continue
            ends = projected_ends(trace["poses"][frame - 100], center, axis, offsets, intrinsics)
            panel.plot(ends[:, 0] - x0, ends[:, 1] - y0, color=color, linewidth=2.7)
            panel.scatter(ends[1, 0] - x0, ends[1, 1] - y0, s=35, color=color, marker="o")
        panel.set_title(f"Frame {frame}")
        panel.axis("off")
    figure.text(
        0.5,
        0.96,
        "Orange: sealed first mask  ·  Magenta: SAM3 first mask  ·  Green: current SAM3 outline",
        ha="center",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    main()
