"""Replay an act from its record outside the GUI: every recorded frame drawn as the Approach tab's replay draws it.

    python benchmarks/act_overlay.py ACT_DIR [--fps 5]

Writes ``ACT_DIR/overlay/NNNNNN.jpg`` (one per recorded frame) and ``ACT_DIR/overlay.mp4``. Every value drawn is as the
run recorded it (lerobot.gui.api.pregrasp._replay_draw); what the act did not record is said on the frame, not rebuilt.
"""

import argparse
import json
import pathlib
import subprocess

import cv2
import numpy as np

from lerobot.gui.api import pregrasp


def demo_surface(demo_root: pathlib.Path, name: str) -> np.ndarray | None:
    """The place object's surface in the demo's view of it (camera frame, metres), from the frame it was designated on:
    what the replay carries by the act's held pose to draw where the act held it."""
    f = demo_root / "objects.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    rows = json.loads(str(z["names"]))
    n = next((i for i, r in enumerate(rows) if r[0] == name), None)
    if n is None:
        return None
    rec = demo_root / "recording"
    depth = cv2.imread(str(rec / "depth" / f"{int(rows[n][1]):06d}.png"), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    k = np.loadtxt(rec / "cam_K.txt")
    mask = cv2.erode(np.asarray(z[f"o{n}_mask"], np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ys, xs = np.nonzero(mask & (depth > 100))
    zz = depth[ys, xs] / 1000.0
    return np.stack([(xs - k[0, 2]) / k[0, 0] * zz, (ys - k[1, 2]) / k[1, 1] * zz, zz], 1)


def render(act: pathlib.Path, fps: float) -> pathlib.Path:
    meta = json.loads((act / "act.json").read_text())
    onto = (meta.get("target") or {}).get("object")
    points = demo_surface(pathlib.Path(meta["demo_root"]), onto) if onto and meta.get("demo_root") else None
    out = act / "overlay"
    out.mkdir(exist_ok=True)
    for i in range(len(meta.get("frames") or [])):
        cv2.imwrite(str(out / f"{i:06d}.jpg"), pregrasp._replay_draw(act, meta, i, points))
    video = act / "overlay.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(out / "%06d.jpg"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    return video


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("act", type=pathlib.Path)
    ap.add_argument("--fps", type=float, default=5.0)
    args = ap.parse_args()
    print(render(args.act, args.fps))
