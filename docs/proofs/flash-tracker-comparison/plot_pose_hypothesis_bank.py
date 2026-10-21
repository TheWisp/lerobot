"""Plot same-frame visible evidence from the saved candidate-bank replay."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_scores(path: Path) -> dict[str, dict[int, dict[str, str]]]:
    scores: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            scores[row["candidate"]][int(row["frame"])] = row
    return scores


def series(rows: dict[int, dict[str, str]], frames: np.ndarray, field: str) -> np.ndarray:
    return np.asarray(
        [float(rows[frame][field]) if frame in rows and rows[frame][field] else np.nan for frame in frames]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dowel", required=True, type=Path)
    parser.add_argument("--ring", required=True, type=Path)
    parser.add_argument("--contact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    colors = {
        "sealed": "#cf7414",
        "initial_sam3": "#aa518e",
        "perframe_sam3": "#6657bb",
        "bundlesdf": "#138a92",
        "point_430": "#238441",
    }
    names = {
        "sealed": "Point2Pose, sealed initializer",
        "initial_sam3": "Point2Pose, SAM3 initializer",
        "perframe_sam3": "Point2Pose, SAM3 each frame",
        "bundlesdf": "BundleSDF",
        "point_430": "Point motion from frame 430",
    }
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 9), constrained_layout=True)
    for axis, path, role, start, end in (
        (axes[0], args.dowel, "Dowel", 249, 500),
        (axes[1], args.ring, "Ring", 300, 500),
    ):
        scores = read_scores(path)
        frames = np.arange(start, end + 1)
        for candidate, rows in scores.items():
            axis.plot(
                frames,
                series(rows, frames, "observed_surface_p90_mm"),
                color=colors[candidate],
                linewidth=1.4,
                label=names[candidate],
            )
        axis.axvspan(400, 500, color="#dce9f7", alpha=0.45, zorder=-1)
        axis.set_title(f"{role}: visible RGB-D surface to the same first-view model")
        axis.set_ylabel("90th percentile distance (mm)")
        axis.set_xlabel("Demo camera frame")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2, loc="upper left")
    contact = read_scores(args.contact)
    frames = np.arange(494, 505)
    axis = axes[2]
    for candidate in ("sealed", "point_430"):
        rows = contact[candidate]
        axis.plot(
            frames,
            series(rows, frames, "ray_support_fraction"),
            color=colors[candidate],
            marker="o",
            linewidth=1.5,
            label=names[candidate],
        )
    axis.axvline(498, color="black", linewidth=1, linestyle="--", label="Direct-pose jump")
    axis.set_title("Late dowel contact: fraction of projected rays supported by mask and depth")
    axis.set_xlabel("Demo camera frame")
    axis.set_ylabel("Supported rays / valid rays")
    axis.set_xticks(frames)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=3)
    figure.suptitle(
        "Causal pose candidates; current RGB-D consistency, not hidden-tip ground truth", fontsize=12
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
