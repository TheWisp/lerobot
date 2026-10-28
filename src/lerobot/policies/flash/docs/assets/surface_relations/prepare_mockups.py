"""Prepare a design mockup from insertion RGB-D frame 100; does not run a policy.

uv run --no-sync python <this file> --recording <episode>/recording \
    --mask <saved SAM mask> --output /tmp/flash-design-review

The surface selections and rule handles are illustrative user annotations.
Depth discontinuities are left open. No geometry completion or tracking claim.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def prepare(recording: Path, mask_path: Path, output: Path) -> None:
    rgb = cv2.cvtColor(cv2.imread(str(recording / "rgb/000100.jpg")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(recording / "depth/000100.png"), -1).astype(float) / 1000
    mask = cv2.imread(str(mask_path), 0) > 0
    k = np.loadtxt(recording / "cam_K.txt").reshape(3, 3)
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    xyz = np.stack(((xx - k[0, 2]) * depth / k[0, 0], -(yy - k[1, 2]) * depth / k[1, 1], -depth), -1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ring = (
        (xx > 210)
        & (xx < 295)
        & (yy > 160)
        & (yy < 244)
        & (hsv[..., 0] > 35)
        & (hsv[..., 0] < 95)
        & (hsv[..., 1] > 65)
    )

    # Two-pixel sampling is a rendering choice, independent of tracker settings.
    pos = xyz[::2, ::2].reshape(-1, 3)
    colors = rgb[::2, ::2].reshape(-1, 3) / 255
    z = depth[::2, ::2]
    ids = np.arange(z.size).reshape(z.shape)
    a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
    triangles = np.concatenate((np.stack((a, c, b), -1), np.stack((b, c, d), -1)))
    zs = z.ravel()[triangles]
    triangles = triangles[(zs.min(1) > 0.15) & (zs.max(1) < 0.85) & (np.ptp(zs, axis=1) < 0.012)]

    def at(u: float, v: float, selection: np.ndarray | None = None) -> np.ndarray:
        x, y = int(round(u)), int(round(v))
        cut = depth[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
        valid = cut > 0
        if selection is not None:
            valid &= selection[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
        if not valid.any():
            raise ValueError(f"No measured depth at selected pixel {x}, {y}")
        distance = float(np.median(cut[valid]))
        return np.array([(u - k[0, 2]) * distance / k[0, 0], -(v - k[1, 2]) * distance / k[1, 1], -distance])

    def contour(selection: np.ndarray) -> list:
        contours, _ = cv2.findContours(selection.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pixels = max(contours, key=cv2.contourArea).reshape(-1, 2)
        points = [at(float(u), float(v), selection).tolist() for u, v in pixels]
        return points + points[:1]

    # A brush stroke across the visible curved surface, selected in image space.
    ribbon, curve, normals = [], [], []
    for offset in np.linspace(-10, 10, 15):
        u, v = 157 - 0.5 * offset, 269 + offset
        point = at(u, v, mask)
        patch = xyz[(np.abs(xx - u) < 5) & (np.abs(yy - v) < 5) & mask & (depth > 0)]
        _, _, vt = np.linalg.svd(patch - patch.mean(0), full_matrices=False)
        normal = vt[-1]
        if normal @ -point < 0:
            normal *= -1
        curve.append(point.tolist())
        normals.append(normal.tolist())
        ribbon.extend([at(u - 16, v - 8, mask).tolist(), at(u + 16, v + 8, mask).tolist()])
    ring_points = xyz[ring & (depth > 0)]
    _, _, vt = np.linalg.svd(ring_points - ring_points.mean(0), full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal *= -1
    distance = float(np.median(depth[ring & (depth > 0)]))
    pivot = [(253 - k[0, 2]) * distance / k[0, 0], -(199 - k[1, 2]) * distance / k[1, 1], -distance]
    data = {
        "positions": pos.round(6).ravel().tolist(),
        "colors": colors.round(4).ravel().tolist(),
        "indices": triangles.ravel().tolist(),
        "dowel_indices": triangles[mask[::2, ::2].ravel()[triangles].all(1)].ravel().tolist(),
        "ring_indices": triangles[ring[::2, ::2].ravel()[triangles].all(1)].ravel().tolist(),
        "dowel": contour(mask),
        "ring": contour(ring),
        "curve": curve,
        "normals": normals,
        "ribbon": ribbon,
        "pivot": pivot,
        "normal": normal.tolist(),
        "source": "insertion / recording / frame 100",
        "time_s": float(np.loadtxt(recording / "times.txt")[100] - np.loadtxt(recording / "times.txt")[0]),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "scene.json").write_text(json.dumps(data, separators=(",", ":")))
    source = Path(__file__).resolve().parent
    shutil.copy(source / "scene_mockup.html", output / "index.html")
    vendor = source.parents[4] / "gui/static/vendor"
    for name in ("three.module.js", "OrbitControls.js"):
        shutil.copy(vendor / name, output / name)
    print(f"Prepared measured RGB-D scene in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.recording, args.mask, args.output)
