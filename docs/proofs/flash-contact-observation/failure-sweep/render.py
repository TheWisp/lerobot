"""Real recorded RGB-D and retained measurements; no generated imagery."""

import gzip
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot.policies.flash.surface_observation import observation

base = Path.home() / ".cache/flash-failure-sweep-20261029/cracker_box_yalehand0-sam3"
p = json.loads((base / "provenance.json").read_text())
e = json.loads(gzip.decompress((base / "evidence/000000.json.gz").read_bytes()))
source = Path(p["source"])
rgb = cv2.cvtColor(cv2.imread(str(source / "rgb" / p["names"][0])), cv2.COLOR_BGR2RGB)
depth = cv2.imread(str(source / "depth" / p["names"][0]), -1).astype(np.float32) * 0.001
mask = np.zeros(depth.shape, bool)
for start, count in e["masks_rle"]["object"]:
    mask.flat[start : start + count] = True
obs = observation(depth, mask, np.array(p["intrinsics"]), p["config"]["vision_pipeline"]["observation"])
outline, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
y, x = np.where(mask)
padding = 18
x0, x1 = max(0, x.min() - padding), min(rgb.shape[1], x.max() + padding)
y0, y1 = max(0, y.min() - padding), min(rgb.shape[0], y.max() + padding)
views = [rgb.copy()]
cv2.drawContours(views[0], outline, -1, (70, 235, 240), 1)
z = cv2.applyColorMap(np.uint8(np.clip((depth - 0.35) / 0.45, 0, 1) * 255), cv2.COLORMAP_TURBO)
z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
z[depth == 0] = 0
z[~mask] //= 5
cv2.drawContours(z, outline, -1, (70, 235, 240), 1)
views.append(z)
for pixels in [np.column_stack(np.where(obs["valid_image"])[::-1]), obs["pixels"]]:
    image = (rgb * 0.35).astype("uint8")
    for u, v in pixels:
        cv2.circle(image, (int(u), int(v)), 1, (120, 255, 95), -1)
    cv2.drawContours(image, outline, -1, (70, 235, 240), 1)
    views.append(image)
W, H = 1240, 1180
out = Image.new("RGB", (W, H), "#101827")
draw = ImageDraw.Draw(out)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(n):
    return ImageFont.truetype(font_path, n)


draw.text((34, 24), "Visible object. Incomplete depth.", font=font(45), fill="white")
draw.text((34, 86), "Same cracker-box frame in all four panels", font=font(27), fill="#b8c6d9")
titles = [
    "SAM3 finds the box",
    "Recorded depth: black means missing",
    f"Depth checks retain {obs['clean_count']} pixels",
    f"Saved shape: only {len(obs['points'])} points",
]
for idx, (view, title) in enumerate(zip(views, titles, strict=False)):
    left, top = 34 + (idx % 2) * 605, 150 + (idx // 2) * 440
    draw.text((left, top), title, font=font(25), fill="white")
    crop = Image.fromarray(view[y0:y1, x0:x1])
    scale = min(570 / crop.width, 375 / crop.height)
    crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.NEAREST)
    out.paste(crop, (left + (570 - crop.width) // 2, top + 49))
missing = 100 * np.mean(depth[mask] == 0)
draw.text((34, 1043), f"{missing:.0f}% of the mask has no depth reading.", font=font(31), fill="#ffcf86")
draw.text((34, 1090), "Better segmentation alone did not restore pose tracking.", font=font(28), fill="white")
target = Path(__file__).with_name("box_depth_failure.jpg")
out.save(target, quality=95)
print(target)
