"""Render the dowel-rotation shaft-agreement timeline from audit JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ORDER = (
    ("point2pose", "Point2Pose full"),
    ("bundlesdf", "BundleSDF full"),
    ("trackeverything_native", "TrackEverything native 3D"),
    ("trackeverything_depth_lift", "TrackEverything depth-lift"),
    ("litetracker", "LiteTracker + fit"),
    ("bootstapir", "BootsTAPIR + fit"),
    ("cotracker3", "CoTracker3 + fit"),
    ("lk", "Optical flow + fit"),
    ("tapnext", "TAPNext + fit"),
)


def render(audit: Path, output: Path) -> None:
    data = json.loads(audit.read_text())
    first, last = data["frames"]
    values = np.ma.masked_all((len(ORDER), last - first + 1))
    for index, (name, _) in enumerate(ORDER):
        for frame_index, row in enumerate(data["methods"][name]["rows"]):
            if row is not None and data["mask"][frame_index]["elongation"] >= data["min_mask_elongation"]:
                values[index, frame_index] = row["shaft_to_mask_deg"]
    figure, axes = plt.subplots(figsize=(12, 5.6), dpi=160)
    palette = plt.get_cmap("RdYlGn_r").copy()
    palette.set_bad("#d9d9d9")
    scale = Normalize(0, 90)
    axes.imshow(
        values,
        cmap=palette,
        norm=scale,
        aspect="auto",
        extent=(first - 0.5, last + 0.5, len(ORDER) - 0.5, -0.5),
        interpolation="nearest",
    )
    axes.set_yticks(np.arange(len(ORDER)), [label for _, label in ORDER], fontsize=9)
    axes.set_xticks(np.linspace(first, last, 7, dtype=int))
    axes.set_xlabel("Demo frame (30 Hz) — dowel rotates from horizontal to upright")
    for frame in np.linspace(first, last, 7, dtype=int)[1:-1]:
        axes.axvline(frame, color="black", alpha=0.18, linewidth=0.7)
    axes.set_title(
        "Visible dowel shaft: projected pose direction vs SAM3 mask\n"
        "Grey = no pose or mask shape ambiguous; greener = closer alignment",
        fontsize=11,
    )
    for index, (name, _) in enumerate(ORDER):
        poses = data["methods"][name]["pose_frames"]
        axes.text(last + 2, index, f"{poses}/{last - first + 1}", va="center", fontsize=8, clip_on=False)
    axes.set_xlim(first - 0.5, last + 18)
    bar = figure.colorbar(ScalarMappable(norm=scale, cmap=palette), ax=axes, pad=0.05, shrink=0.85)
    bar.set_label("Unoriented shaft angle error (degrees)")
    figure.text(
        0.15,
        0.02,
        f"Gripper heading vs mask: {data['gripper_heading_to_mask_deg_median_when_elongated']:.1f}° "
        "median. Mask and gripper are checks, not independent 6D ground truth.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.audit, args.output)


if __name__ == "__main__":
    main()
