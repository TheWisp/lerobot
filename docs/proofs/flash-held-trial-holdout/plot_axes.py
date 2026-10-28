"""Render the RGB-D holdout's two late candidate/fallback axis checks."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root = Path(__file__).parent
measurements = {
    item["row"]: item for item in json.loads((root / "axis_audit.json").read_text())["measurements"]
}
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
for ax, row in zip(axes, (417, 511), strict=True):
    image = np.asarray(Image.open(root / f"rgb-{row}.png"))
    mask = np.asarray(Image.open(root / f"mask-{row}.png")) > 0
    ax.imshow(image)
    ax.contour(mask.astype(float), levels=[0.5], colors=["white"], linewidths=1.5)
    result = measurements[row]
    for key, color, label in (
        ("vision_candidate", "#00ffff", "rejected vision"),
        ("gripper_fallback", "#ff8a00", "gripper fallback"),
    ):
        proposal = result["proposals"][key]
        endpoints = np.asarray(proposal["projected_endpoints_xy"])
        ax.plot(endpoints[:, 0], endpoints[:, 1], color=color, linewidth=3, marker="o", markersize=4)
        ax.plot(
            [],
            [],
            color=color,
            linewidth=3,
            label=f"{label}: {proposal['axis_gap_surface_3d_deg']:.0f}°",
        )
    ax.set_xlim(180, 375)
    ax.set_ylim(340, 110)
    ax.set_title(f"Recorded camera frame {result['camera_frame_id']}", fontsize=16)
    ax.legend(loc="upper right", fontsize=12, facecolor="#111111", labelcolor="white")
    ax.axis("off")
fig.savefig(root / "axes_comparison.png", dpi=150, facecolor="white")
