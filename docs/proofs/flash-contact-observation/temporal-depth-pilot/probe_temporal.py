"""Audit recurrence of the already inspected dowel background-depth contamination.

The 400 mm split is an explicit diagnostic for this recorded window, not a
generic outlier rule or a change to the tracker. Recurrence is not point identity.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot.policies.flash.surface_observation import observation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BASE = Path.home() / ".cache/huggingface/lerobot/demos/insertion/recording"
MASKS = Path.home() / ".cache/flash-mask-depth-experiment/sam3-full"
K = np.loadtxt(BASE / "cam_K.txt")
CFG = json.loads((REPO / "docs/proofs/flash-mask-depth-tracker/config.json").read_text())
CFG["component_gap_m"] = None


def read(i):
    rgb = cv2.cvtColor(cv2.imread(str(BASE / f"rgb/{i:06}.jpg")), cv2.COLOR_BGR2RGB)
    d = cv2.imread(str(BASE / f"depth/{i:06}.png"), -1).astype(np.float32) * 0.001
    m = cv2.imread(str(MASKS / f"{i:06}.png"), 0) > 0
    return rgb, m, observation(d, m, K, CFG)


rows = []
for i in range(250, 311):
    _, m, o = read(i)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype("uint8"), 8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    p, px = o["points"], o["pixels"]
    far = p[:, 2] > 0.4
    rows.append(
        {
            "frame": i,
            "mask_pixels": int(m.sum()),
            "mask_component_areas": sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True),
            "points": len(p),
            "far_points": int(far.sum()),
            "far_in_main_mask": int(np.sum(far & (lab[px[:, 1], px[:, 0]] == largest))),
            "median_depth_mm": float(np.median(p[:, 2]) * 1000),
        }
    )
(HERE / "temporal.json").write_text(json.dumps(rows, indent=2))


def font(n):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", n)


canvas = Image.new("RGB", (1320, 620), "#101827")
draw = ImageDraw.Draw(canvas)
draw.text((28, 20), "Bad depth can recur inside a tracked mask", font=font(39), fill="white")
draw.text((28, 76), "Cyan: SAM boundary. Red: samples deeper than 40 cm.", font=font(25), fill="#c5d2e6")
for j, i in enumerate([278, 280, 286]):
    rgb, m, o = read(i)
    x, y = o["pixels"].T
    image = rgb.copy()
    cv2.drawContours(
        image,
        cv2.findContours(m.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        -1,
        (30, 235, 250),
        1,
    )
    for (u, v), z in zip(o["pixels"], o["points"][:, 2], strict=False):
        if z > 0.4:
            cv2.circle(image, (int(u), int(v)), 2, (255, 55, 65), -1)
    yy, xx = np.where(m)
    crop = image[
        max(0, yy.min() - 20) : min(m.shape[0], yy.max() + 21),
        max(0, xx.min() - 25) : min(m.shape[1], xx.max() + 26),
    ]
    tile = Image.fromarray(crop)
    scale = min(410 / tile.width, 265 / tile.height)
    tile = tile.resize((round(tile.width * scale), round(tile.height * scale)), Image.Resampling.NEAREST)
    left = 28 + j * 435
    draw.text((left, 127), f"Frame {i}", font=font(29), fill="white")
    canvas.paste(tile, (left + (410 - tile.width) // 2, 178 + (265 - tile.height) // 2))
draw.text((28, 461), "The dowel is about 34 cm from the camera here.", font=font(29), fill="white")
draw.text((28, 508), "These deeper readings survive the depth cleaning.", font=font(27), fill="#ffbd8c")
draw.text(
    (28, 553), "Presence over time alone cannot certify a surface patch.", font=font(27), fill="#c5d2e6"
)
canvas.save(HERE / "recurring_depth.jpg", quality=95)
print(HERE / "recurring_depth.jpg")
