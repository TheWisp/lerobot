"""Plain-language figures from recorded pixels and measured selection/matching results."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("/home/feit/.codex/visualizations/2026/10/12/01a12ff6-a1d5-7af1-a00f-80c5f356f546")
data = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "text.color": "#172334"})


def photo(ax, rgb, box, title):
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[3], box[2])
    ax.set_axis_off()
    ax.set_title(title, fontsize=19, pad=10, weight="bold")


d = np.load(data / "dish_000.npz")
fig = plt.figure(figsize=(10, 7), facecolor="white")
fig.text(
    0.5, 0.95, "Dish: most of the measured surface gets discarded", ha="center", fontsize=23, weight="bold"
)
box = (60, 121, 88, 192)
a = fig.add_axes([0.07, 0.24, 0.38, 0.59])
b = fig.add_axes([0.55, 0.24, 0.38, 0.59])
photo(a, d["rgb"], box, "Dish selected correctly")
a.contour(d["mask"], [0.5], colors=["#00bd8a"], linewidths=1.7)
photo(b, d["rgb"], box, "Points kept for tracking")
y, x = np.nonzero(d["selected"])
b.scatter(x, y, s=16, c="#00dc8c", linewidths=0)
fig.text(0.5, 0.15, "Only the small green strip survives.", ha="center", fontsize=21, weight="bold")
fig.text(
    0.5,
    0.065,
    "The depth-change rule rejects smooth slopes;\nthen the largest-patch rule discards more of the dish.",
    ha="center",
    fontsize=18,
    linespacing=1.4,
)
fig.savefig(out / "flash_dish_rejection_plain_20261029.png", dpi=160)
plt.close(fig)

frames = np.load(data / "clock_inputs.npz")["rgb"]
fig = plt.figure(figsize=(10, 9), facecolor="white")
fig.text(
    0.5, 0.955, "Clock: a good match exists, but loses the ranking", ha="center", fontsize=23, weight="bold"
)
a = fig.add_axes([0.30, 0.55, 0.40, 0.32])
photo(a, frames[150], (265, 343, 40, 125), "Current frame 150: back visible")
a = fig.add_axes([0.06, 0.14, 0.39, 0.30])
b = fig.add_axes([0.55, 0.14, 0.39, 0.30])
photo(a, frames[0], (257, 335, 49, 141), "Match from frame 0")
photo(b, frames[149], (265, 343, 40, 125), "Match from frame 149")
fig.text(0.255, 0.09, "180° wrong", ha="center", fontsize=23, weight="bold", color="#bb3232")
fig.text(0.745, 0.09, "1.5° error", ha="center", fontsize=23, weight="bold", color="#087a52")
fig.text(0.255, 0.045, "Our surface score prefers this", ha="center", fontsize=15)
fig.text(0.745, 0.045, "Found by image + depth matching", ha="center", fontsize=15)
fig.text(0.5, 0.505, "Pairwise motion errors, checked against independent labels", ha="center", fontsize=16)
fig.savefig(out / "flash_clock_best_match_plain_20261029.png", dpi=160)
plt.close(fig)
print(out / "flash_dish_rejection_plain_20261029.png")
print(out / "flash_clock_best_match_plain_20261029.png")
