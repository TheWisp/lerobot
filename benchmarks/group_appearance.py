"""Offline appearance cue for a designated object in a recorded act.

This is a measurement experiment, not a live pose source. It tests whether the
object's original SAM mask has enough distinctive color to locate a partly
visible object after its point tracks stop placing it. It estimates translation
only; a colored distractor or a changed viewpoint can make it wrong.

Examples (use the Point2Pose environment when --click runs SAM):

  python benchmarks/group_appearance.py --act ACT --click 368,340 --out /tmp/cue.json
  python benchmarks/group_appearance.py --act ACT --mask /tmp/mask.npy \
      --group-timeline /tmp/groups/timeline.json --object ring --out /tmp/cue.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def _source(path: Path, duration: float | None) -> tuple[list[Path], list[Path], np.ndarray, np.ndarray]:
    if (path / "camera" / "timeline.json").exists():
        timeline = json.loads((path / "camera" / "timeline.json").read_text())
        rows = [r for r in timeline["rows"] if r["status"] == "captured" and r.get("image") is not None]
        if duration is not None:
            rows = [r for r in rows if r["t_wall"] - timeline["origin_wall"] <= duration]
        rgb = [path / "camera" / "rgb" / f"{r['image']:06d}.jpg" for r in rows]
        depth = [path / "camera" / "depth" / f"{r['image']:06d}.png" for r in rows]
        times = np.array([r["t_wall"] for r in rows])
        intr = json.loads((path / "act.json").read_text())["intr"]
        k = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    else:
        rgb = sorted((path / "rgb").glob("*.jpg"))
        depth = [path / "depth" / f"{p.stem}.png" for p in rgb]
        k = np.loadtxt(path / "cam_K.txt")
        times = np.atleast_1d(np.loadtxt(path / "times.txt"))
        if duration is not None:
            n = int(np.searchsorted(times - times[0], duration, side="right"))
            rgb, depth, times = rgb[:n], depth[:n], times[:n]
    if not rgb:
        raise ValueError(f"no captured images in {path}")
    return rgb, depth, k, times


def _model(bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    ys, xs = np.where(mask)
    if len(xs) < 20:
        raise ValueError("initial mask is empty or too small")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    local = mask[y0:y1, x0:x1]
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hue, sat, val = (hsv[:, :, i][local] for i in range(3))
    # Hue is undefined on grey/dark pixels. The saturation/brightness floors
    # remove those pixels; the hue margin tolerates shading and JPEG changes.
    mid_hue = float(np.median(hue))
    lo = np.array(
        [max(0, mid_hue - 11), max(80, np.percentile(sat, 10)), max(32, np.percentile(val, 10))], np.uint8
    )
    hi = np.array([min(179, mid_hue + 11), 255, 255], np.uint8)
    template = (local & (cv2.inRange(hsv, lo, hi) > 0)).astype(np.float32)
    if template.sum() < 20:
        raise ValueError("mask has too little chromatic support for this cue")
    return template, lo, hi, (x0, y0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--act", type=Path, required=True, help="act directory or sequential RGB/depth recording")
    seed = ap.add_mutually_exclusive_group(required=True)
    seed.add_argument("--mask", type=Path, help="saved initial boolean mask (.npy)")
    seed.add_argument("--click", help="initial object click x,y; runs SAM 2.1 once")
    ap.add_argument("--duration", type=float, help="seconds from the first captured image")
    ap.add_argument("--max-shift-px", type=int, default=16, help="local search radius per camera frame")
    ap.add_argument(
        "--track-support", type=float, default=0.28, help="minimum match to advance the local search"
    )
    ap.add_argument("--group-timeline", type=Path, help="optional group_live.py timeline to compare")
    ap.add_argument("--object", help="object name to compare against live act groups, or --group-timeline")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.group_timeline and not args.object:
        ap.error("--object is required with --group-timeline")
    rgb_files, depth_files, k, times = _source(args.act, args.duration)
    first = cv2.imread(str(rgb_files[0]))
    if first is None:
        raise FileNotFoundError(rgb_files[0])
    if args.mask:
        mask = np.load(args.mask).astype(bool)
    else:
        from group_live import segment_clicks

        click = tuple(map(int, args.click.split(",")))
        mask = segment_clicks(cv2.cvtColor(first, cv2.COLOR_BGR2RGB), [click])[0]
    if mask.shape != first.shape[:2]:
        raise ValueError("mask and first RGB image must have the same size")
    template, lo, hi, (x0, y0) = _model(first, mask)
    first_chroma = cv2.inRange(cv2.cvtColor(first, cv2.COLOR_BGR2HSV), lo, hi) > 0
    surround = (cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~mask
    object_color_share = float(first_chroma[mask].mean())
    surround_color_share = float(first_chroma[surround].mean()) if surround.any() else 0.0
    centre = np.array([x0 + template.shape[1] / 2, y0 + template.shape[0] / 2], dtype=float)
    by_frame = {}
    if args.group_timeline:
        by_frame = {r["frame"]: r for r in json.loads(args.group_timeline.read_text())}
    live_groups = None
    if args.object and not args.group_timeline and (args.act / "act.json").exists():
        live_groups = (
            json.loads((args.act / "act.json").read_text())
            .get("groups", {})
            .get("frames", {})
            .get(args.object)
        )
        if live_groups:
            live_times = np.array([r[0] for r in live_groups])
    rows = []
    for frame, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files, strict=True)):
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            raise FileNotFoundError(rgb_path)
        t0 = time.perf_counter()  # RGB capture/decode is shared with the existing camera path.
        px = round(centre[0] - template.shape[1] / 2)
        py = round(centre[1] - template.shape[0] / 2)
        rad = args.max_shift_px
        xa, ya = max(0, px - rad), max(0, py - rad)
        xb, yb = (
            min(bgr.shape[1], px + rad + template.shape[1]),
            min(bgr.shape[0], py + rad + template.shape[0]),
        )
        hsv = cv2.cvtColor(bgr[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
        chroma = (cv2.inRange(hsv, lo, hi) > 0).astype(np.float32)
        scores = cv2.matchTemplate(chroma, template, cv2.TM_CCORR)
        _, score, _, at = cv2.minMaxLoc(scores)
        support = float(score / template.sum())
        candidate = np.array([xa + at[0] + template.shape[1] / 2, ya + at[1] + template.shape[0] / 2])
        if support >= args.track_support:
            centre = candidate
        match_ms = (time.perf_counter() - t0) * 1000
        row = {"frame": frame, "pixel": centre.tolist(), "support": support, "match_ms": match_ms}
        if depth_path.exists():
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            cx, cy = round(centre[0] - template.shape[1] / 2), round(centre[1] - template.shape[0] / 2)
            patch = depth[cy : cy + template.shape[0], cx : cx + template.shape[1]]
            view = cv2.cvtColor(
                bgr[cy : cy + template.shape[0], cx : cx + template.shape[1]], cv2.COLOR_BGR2HSV
            )
            visible = (cv2.inRange(view, lo, hi) > 0) & template.astype(bool) & (patch > 0)
            if visible.any():
                z = float(np.median(patch[visible])) / 1000
                row["depth_m"] = z
                row["xyz_m"] = [(centre[0] - k[0, 2]) * z / k[0, 0], (centre[1] - k[1, 2]) * z / k[1, 1], z]
        if frame in by_frame:
            obj = by_frame[frame]["objects"][args.object]
            x, y, z = obj["pose"]
            group_uv = [x * k[0, 0] / z + k[0, 2], y * k[1, 1] / z + k[1, 2]]
            row["group_pixel"] = group_uv
            row["group_gap_px"] = float(np.linalg.norm(centre - group_uv))
            row["group_own_ok"] = obj["own_ok"]
        elif live_groups:
            at = int(np.searchsorted(live_times, times[frame]))
            near = min(
                (i for i in (at - 1, at) if 0 <= i < len(live_times)),
                key=lambda i: abs(live_times[i] - times[frame]),
            )
            lag = float(live_times[near] - times[frame])
            if abs(lag) <= 0.15:
                pose = np.asarray(live_groups[near][1:]).reshape(4, 4)
                x, y, z = pose[:3, 3]
                group_uv = [x * k[0, 0] / z + k[0, 2], y * k[1, 1] / z + k[1, 2]]
                row["group_pixel"] = group_uv
                row["group_gap_px"] = float(np.linalg.norm(centre - group_uv))
                row["group_age_ms"] = lag * 1000
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": str(args.act),
                "color_lower_hsv": lo.tolist(),
                "color_upper_hsv": hi.tolist(),
                "template_pixels": int(template.sum()),
                "object_color_share": object_color_share,
                "surround_color_share": surround_color_share,
                "rows": rows,
            },
            indent=2,
        )
    )
    ms = np.array([r["match_ms"] for r in rows])
    print(f"{len(rows)} frames; ROI match median {np.median(ms):.3f} ms, p95 {np.percentile(ms, 95):.3f} ms")
    print(f"initial color support {object_color_share:.1%} inside mask, {surround_color_share:.1%} around it")
    for threshold in (0.9, 0.4):
        gaps = [r["group_gap_px"] for r in rows if r["support"] >= threshold and "group_gap_px" in r]
        if gaps:
            print(
                f"support >= {threshold}: {len(gaps)} group comparisons, "
                f"gap median {np.median(gaps):.1f} px, p95 {np.percentile(gaps, 95):.1f} px"
            )


if __name__ == "__main__":
    main()
