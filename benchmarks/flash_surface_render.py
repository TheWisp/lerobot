"""Render source-time, uncensored point-tracking comparisons as H.264 MP4.

All panels show the same recording/crop. Cyan is each model's raw visible
points; magenta is an independently labelled visible point where available.
No rejected or failed frame is omitted from the video.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_surface_benchmark import _load, validate_prediction


def _panel(
    rgb: np.ndarray,
    mask: np.ndarray,
    label: str,
    pred_xy: np.ndarray | None,
    pred_visible: np.ndarray | None,
    truth_xy: np.ndarray | None,
    truth_visible: np.ndarray | None,
    roi: tuple[int, int, int, int],
    size: tuple[int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = roi
    width, height = size
    crop = rgb[y0:y1, x0:x1].copy()
    contours, _ = cv2.findContours(mask[y0:y1, x0:x1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, (235, 235, 235), 1)
    if truth_xy is not None:
        for xy, visible in zip(truth_xy, truth_visible, strict=True):
            if visible and np.isfinite(xy).all():
                cv2.circle(crop, (round(float(xy[0] - x0)), round(float(xy[1] - y0))), 2, (230, 30, 230), -1)
    if pred_xy is not None:
        for xy, visible in zip(pred_xy, pred_visible, strict=True):
            if visible and np.isfinite(xy).all():
                cv2.circle(crop, (round(float(xy[0] - x0)), round(float(xy[1] - y0))), 2, (255, 255, 0), -1)
    crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    panel = np.full((height + 48, width, 3), 18, np.uint8)
    panel[48:] = crop
    cv2.putText(panel, label, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def render(experiment: Path, models: list[str], output: Path, roi: tuple[int, int, int, int] | None) -> dict:
    manifest = json.loads((experiment / "manifest.json").read_text())
    reference = _load(experiment / "queries.npz")
    predictions = {}
    for model in models:
        predictions[model] = _load(experiment / f"{model}.npz")
        validate_prediction(reference, predictions[model])
    frames, capture_s = reference["frames"], reference["capture_s"]
    recording = Path(manifest["recording"])
    first = cv2.imread(str(recording / "rgb" / f"{frames[0]:06d}.jpg"), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError("first RGB frame unavailable")
    h, w = first.shape[:2]
    if roi is None:
        roi = (0, 0, w, h)
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("crop must be inside the source frame")
    count = len(models) + 1
    cols = min(3, count)
    rows = int(np.ceil(count / cols))
    panel_width = 420
    panel_height = round(panel_width * (y1 - y0) / (x1 - x0))
    # yuv420p requires even frame dimensions.
    panel_height += panel_height % 2
    frame_w, frame_h = cols * panel_width, rows * (panel_height + 48)
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = 30
    steps = np.arange(int(np.ceil(capture_s[-1] * fps)) + 1) / fps
    indices = np.clip(np.searchsorted(capture_s, steps, side="right") - 1, 0, len(frames) - 1)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{frame_w}x{frame_h}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "27",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for index in indices:
            frame_id = frames[index]
            rgb = cv2.imread(str(recording / "rgb" / f"{frame_id:06d}.jpg"), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(Path(manifest["mask_dir"]) / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if rgb is None or mask is None or mask.shape != rgb.shape[:2]:
                raise ValueError(f"bad RGB/mask frame {frame_id}")
            truth_xy = reference["truth_xy"][index] if "truth_xy" in reference else None
            truth_visible = reference["truth_visible"][index] if "truth_visible" in reference else None
            panels = [
                _panel(
                    rgb,
                    mask,
                    "RGB-D SOURCE",
                    None,
                    None,
                    truth_xy,
                    truth_visible,
                    roi,
                    (panel_width, panel_height),
                )
            ]
            for model in models:
                prediction = predictions[model]
                panels.append(
                    _panel(
                        rgb,
                        mask,
                        model.upper(),
                        prediction["xy"][index],
                        prediction["visible"][index],
                        truth_xy,
                        truth_visible,
                        roi,
                        (panel_width, panel_height),
                    )
                )
            canvas = np.full((frame_h, frame_w, 3), 18, np.uint8)
            for cell, panel in enumerate(panels):
                row, col = divmod(cell, cols)
                sy, sx = row * (panel_height + 48), col * panel_width
                canvas[sy : sy + panel.shape[0], sx : sx + panel.shape[1]] = panel
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error[-1000:]}")
    return {
        "output": str(output),
        "frames": len(steps),
        "source_frames": len(frames),
        "duration_s": len(steps) / fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--roi", type=int, nargs=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(render(args.experiment, args.models, args.output, tuple(args.roi) if args.roi else None))
    )


if __name__ == "__main__":
    main()
