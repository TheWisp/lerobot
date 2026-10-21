"""Show how the initial mask changes Point2Pose point ownership."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, seed in enumerate((340, 380)):
        frame = seed + 1
        stem = f"{frame:06d}"
        image = cv2.imread(str(args.dataset / "rgb" / f"{stem}.jpg"))
        masks = {
            role: cv2.imread(str(args.masks / role / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            for role in ("dowel", "ring")
        }
        if image is None or any(mask is None for mask in masks.values()):
            raise ValueError(f"missing recorded image/mask at frame {frame}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        union = masks["dowel"] | masks["ring"]
        yy, xx = np.nonzero(union)
        pad = 35
        left, right = max(0, int(xx.min()) - pad), min(image.shape[1], int(xx.max()) + pad)
        top, bottom = max(0, int(yy.min()) - pad), min(image.shape[0], int(yy.max()) + pad)
        for column, label in enumerate(("diagnostic", "exact")):
            axis = axes[row, column]
            p2p_mask = cv2.imread(
                str(args.runs / f"p2p_dowel_reseed{seed}_{label}_masks/dowel/{stem}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if p2p_mask is None:
                raise ValueError(f"missing Point2Pose mask for {seed} {label}")
            with np.load(args.runs / f"p2p_dowel_reseed{seed}_{label}_diagnostics/{stem}.npz") as diag:
                uv = diag["fit_uv"].copy()
                tracked = int(diag["track_valid_0"].sum())
            crop = image[top:bottom, left:right].copy()
            for mask, color in (
                (masks["dowel"], (0, 240, 245)),
                (masks["ring"], (245, 235, 0)),
                (p2p_mask, (250, 35, 220)),
            ):
                edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
                rows, columns = np.nonzero(edges[top:bottom, left:right])
                crop[rows, columns] = color
            axis.imshow(crop)
            if len(uv):
                px = np.rint(uv).astype(int)
                px[:, 0] = np.clip(px[:, 0], 0, image.shape[1] - 1)
                px[:, 1] = np.clip(px[:, 1], 0, image.shape[0] - 1)
                on_ring = masks["ring"][px[:, 1], px[:, 0]] > 0
                for selected, color in ((~on_ring, "white"), (on_ring, "red")):
                    axis.scatter(
                        uv[selected, 0] - left,
                        uv[selected, 1] - top,
                        s=27,
                        c=color,
                        edgecolors="black",
                        linewidths=0.3,
                    )
            policy = "SAM2 initial" if label == "diagnostic" else "exact SAM3 initial"
            axis.set_title(f"Seed {seed}: {policy}; {tracked} valid tracks, {len(uv)} fit points")
            axis.set_axis_off()
    figure.suptitle(
        "Cyan: SAM3 dowel  |  yellow: SAM3 ring  |  pink: Point2Pose mask  |  red dots: ring points"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
