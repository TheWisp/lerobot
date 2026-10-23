"""Render rigidity violations from the saved tracks, with original capture crops."""

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "config.json").read_text())
FIG, AX = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [1, 1.4]})
COLORS = ["#27b9cd", "#ee8056"]
for row, (name, clip_name, title) in enumerate(
    [("dowel", "dowel_rotation", "Rotating dowel"), ("yellow", "yellow_full", "Moving yellow object")]
):
    s = np.load(ROOT / (name + "_original.npz"))
    p, support, frames = s["points"], s["support"], s["frames"]
    ids = np.flatnonzero(support.all(axis=0))
    pairs = np.array([(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]])
    distances = np.linalg.norm(p[:, pairs[:, 0]] - p[:, pairs[:, 1]], axis=-1) * 1000
    change = distances - distances[0]
    score = np.median(np.abs(change), axis=0)
    chosen = [int(np.argmin(score)), int(np.argmax(score))]
    clip = next(c for c in CONFIG["clips"] if c["name"] == clip_name)
    recording = Path(clip["recording"])
    rgb = cv2.cvtColor(cv2.imread(str(recording / "rgb" / f"{frames[0]:06d}.jpg")), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(Path(clip["mask_dir"]) / f"{frames[0]:06d}.png"), 0) > 0
    k = np.loadtxt(recording / "cam_K.txt").reshape(3, 3)
    uv = p[0] @ k.T
    uv = np.divide(uv[:, :2], uv[:, 2:], out=np.zeros_like(uv[:, :2]), where=uv[:, 2:] > 0)
    AX[row, 0].imshow(rgb)
    for n, pair_index in enumerate(chosen):
        pair = pairs[pair_index]
        AX[row, 0].plot(uv[pair, 0], uv[pair, 1], "-o", color=COLORS[n], linewidth=2.5, markersize=7)
        AX[row, 1].plot(
            frames,
            change[:, pair_index],
            color=COLORS[n],
            linewidth=2.3,
            label=["More stable pair", "Less stable pair"][n],
        )
    y, x = np.nonzero(mask)
    padding = 30
    AX[row, 0].set_xlim(x.min() - padding, x.max() + padding)
    AX[row, 0].set_ylim(y.max() + padding, y.min() - padding)
    AX[row, 0].axis("off")
    AX[row, 0].set_title(title, fontsize=17, loc="left")
    AX[row, 1].axhline(0, color="#777777", linewidth=1, linestyle="--")
    AX[row, 1].set_ylabel("Change in separation (mm)", fontsize=12)
    AX[row, 1].set_xlabel("Recorded frame", fontsize=12)
    AX[row, 1].grid(alpha=0.2)
    AX[row, 1].spines[["top", "right"]].set_visible(False)
AX[0, 1].legend(frameon=False, fontsize=12, loc="upper left")
FIG.suptitle("Rigid points should keep the same separation", fontsize=23, x=0.08, ha="left")
FIG.text(
    0.08,
    0.025,
    "Both pairs pass mask + depth checks in every frame. Best and worst pairs shown among continuously supported IDs.",
    fontsize=11,
)
FIG.tight_layout(rect=[0.02, 0.06, 0.99, 0.94])
FIG.savefig(ROOT / "drift_patterns.png", dpi=170)
