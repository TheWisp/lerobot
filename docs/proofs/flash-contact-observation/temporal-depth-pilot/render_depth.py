"""Actual RGB/depth/model outputs, with identical depth colours in every panel."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache/flash-depth-completion-pilot-20261029"
a = np.load(CACHE / "box_holes_dense_prompt.npz")
b = np.load(CACHE / "box_holes_lingbot_r9.npz")
assert np.array_equal(a["depth"], b["depth"]) and np.array_equal(a["rgb"], b["rgb"])
mask, rgb = a["mask"], a["rgb"]
ys, xs = np.where(mask)
y0, y1 = max(0, ys.min() - 18), min(mask.shape[0], ys.max() + 19)
x0, x1 = max(0, xs.min() - 18), min(mask.shape[1], xs.max() + 19)
contours = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
views = [rgb.copy()]
for z in [a["depth"], a["predicted"], b["predicted"]]:
    value = np.nan_to_num((z - 0.4) / 0.25, nan=0, posinf=1, neginf=0)
    image = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(np.clip(value, 0, 1) * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    image[(z <= 0) | ~np.isfinite(z)] = 0
    image[~mask] //= 4
    views.append(image)
for v in views:
    cv2.drawContours(v, contours, -1, (235, 245, 250), 1)


def font(n):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", n)


out = Image.new("RGB", (1220, 1060), "#101827")
draw = ImageDraw.Draw(out)
draw.text((30, 22), "Can a depth model fill the missing box?", font=font(40), fill="white")
draw.text((30, 83), "Same frame. Same colours: blue = 40 cm, red = 65 cm.", font=font(27), fill="#c5d2e6")
titles = [
    "RGB + object boundary",
    "Sensor depth: 71% missing",
    "PromptDA Small + filled sensor input",
    "LingBot-Depth + raw sensor input",
]
for i, (v, title) in enumerate(zip(views, titles, strict=False)):
    left, top = 30 + (i % 2) * 600, 145 + (i // 2) * 405
    draw.text((left, top), title, font=font(24), fill="white")
    tile = Image.fromarray(v[y0:y1, x0:x1])
    scale = min(560 / tile.width, 335 / tile.height)
    tile = tile.resize((round(tile.width * scale), round(tile.height * scale)), Image.Resampling.NEAREST)
    out.paste(tile, (left + (560 - tile.width) // 2, top + 48 + (335 - tile.height) // 2))
draw.text(
    (30, 975), "Filled pixels are estimates. We have no truth for these holes.", font=font(26), fill="#ffbd8c"
)
out.save(HERE / "depth_completion.jpg", quality=95)
print(HERE / "depth_completion.jpg")
