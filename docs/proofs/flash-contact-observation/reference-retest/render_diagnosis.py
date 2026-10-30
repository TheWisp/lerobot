"""Evidence figures from actual recorded pixels; no simulated tracking results."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

out = Path("/home/feit/.codex/visualizations/2026/10/12/01a12ff6-a1d5-7af1-a00f-80c5f356f546")
data = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17, "text.color": "#172334"})
green = "#19e2a0"
orange = "#ff8b28"
blue = "#64a8ff"


def photo(ax, rgb, box, title):
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[3], box[2])
    ax.set_axis_off()
    ax.set_title(title, fontsize=20, pad=12, weight="bold")


fig = plt.figure(figsize=(10, 7), facecolor="white")
fig.text(
    0.5, 0.95, "Same rule: useful on the dowel, harmful on the mug", ha="center", fontsize=22, weight="bold"
)
for index, (name, box, title) in enumerate(
    [
        ("dowel", (155, 295, 180, 265), "Dowel: wrong depth at the edge"),
        ("mug", (60, 155, 60, 180), "Mug: part of the same cup"),
    ]
):
    d = np.load(data / f"{name}_stray_audit.npz")
    a = fig.add_axes([0.05 + index * 0.5, 0.28, 0.40, 0.51])
    if name == "mug":
        yy, xx = np.nonzero(d["mask"])
        box = (int(xx.min()) - 8, int(xx.max()) + 8, int(yy.min()) - 8, int(yy.max()) + 8)
    photo(a, d["rgb"], box, title)
    p = d["pixels"]
    keep = d["keep"]
    a.scatter(p[keep, 0], p[keep, 1], s=9 if name == "dowel" else 4, c=green, linewidths=0, alpha=0.8)
    a.scatter(
        p[~keep, 0],
        p[~keep, 1],
        s=38 if name == "dowel" else 7,
        c=orange,
        linewidths=0.5,
        edgecolors="black",
        alpha=1,
    )
    fig.text(
        0.25 + index * 0.5,
        0.19,
        "5 unwanted points removed" if name == "dowel" else "239 real surface points removed",
        ha="center",
        fontsize=19,
        weight="bold",
    )
fig.text(
    0.5, 0.10, "Green: kept     Orange: discarded by the largest-3D-patch rule", ha="center", fontsize=17
)
fig.text(
    0.5, 0.04, "Both sets of orange points are inside their object’s main 2D mask.", ha="center", fontsize=17
)
fig.savefig(out / "flash_stray_tradeoff_20261029.png", dpi=160)
plt.close(fig)

fig = plt.figure(figsize=(10, 7), facecolor="white")
d = np.load(data / "clock_outline_149.npz")
fig.text(
    0.5,
    0.95,
    "Clock: a partial surface is asked to cover a whole mask",
    ha="center",
    fontsize=22,
    weight="bold",
)
a = fig.add_axes([0.06, 0.28, 0.39, 0.52])
photo(a, d["source_rgb"], (273, 338, 43, 112), "Previous frame")
y, x = np.nonzero(d["source_valid"])
a.scatter(x, y, s=9, c=blue, linewidths=0)
a = fig.add_axes([0.55, 0.28, 0.39, 0.52])
photo(a, d["rgb"], (273, 338, 43, 112), "Mapped into the next frame")
a.contour(d["mask"], [0.5], colors=[orange], linewidths=2)
a.add_patch(Polygon(d["polygon"], facecolor=blue, edgecolor=blue, alpha=0.45, lw=2))
fig.text(0.5, 0.20, "Blue is incomplete because depth points were filtered out.", ha="center", fontsize=18)
fig.text(
    0.5,
    0.13,
    "The score penalizes the orange outline left uncovered.",
    ha="center",
    fontsize=19,
    weight="bold",
)
fig.text(
    0.5,
    0.05,
    "A larger, wrong-side match wins. This good match has only 1.5° motion error.",
    ha="center",
    fontsize=17,
)
fig.savefig(out / "flash_clock_ranking_cause_20261029.png", dpi=160)
plt.close(fig)

fig = plt.figure(figsize=(10, 9), facecolor="white")
fig.text(
    0.5,
    0.96,
    "Dish: the depth shape changes while the RGB looks similar",
    ha="center",
    fontsize=21,
    weight="bold",
)
for row, i in enumerate((0, 10)):
    d = np.load(data / f"dish_{i:03}.npz")
    src = np.load(data / f"dish_source_{i:03}.npz")
    for col in (0, 1):
        a = fig.add_axes([0.07 + col * 0.43, 0.53 - row * 0.37, 0.34, 0.29])
        if col == 0:
            a.imshow(d["rgb"], interpolation="nearest")
        else:
            im = a.imshow(src["depth"] * 1000, cmap="turbo", vmin=350, vmax=650, interpolation="nearest")
        a.contour(d["mask"], [0.5], colors=["white"], linewidths=1.5)
        a.set_xlim(60, 125)
        a.set_ylim(192, 90)
        a.set_axis_off()
        a.set_title(f"Frame {i}: " + ("RGB" if col == 0 else "depth"), fontsize=19, weight="bold")
cax = fig.add_axes([0.90, 0.23, 0.018, 0.50])
bar = fig.colorbar(im, cax=cax)
bar.set_label("Distance from camera (mm)", fontsize=15)
fig.text(
    0.5,
    0.09,
    "White outline = SAM-selected dish. Depth colours use the same scale.",
    ha="center",
    fontsize=16,
)
fig.text(
    0.5,
    0.035,
    "Our import adds <1.5 mm here; it does not explain the larger disagreement.",
    ha="center",
    fontsize=16,
)
fig.savefig(out / "flash_dish_depth_cause_20261029.png", dpi=160)
plt.close(fig)
for p in (
    "flash_stray_tradeoff_20261029.png",
    "flash_clock_ranking_cause_20261029.png",
    "flash_dish_depth_cause_20261029.png",
):
    print(out / p)
