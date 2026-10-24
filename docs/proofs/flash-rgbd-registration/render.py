"""Render synchronized fixed-landmark transport for the benchmark interval."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
from compare import MASKS, RECORDING, move, reference

WIDTH, HEIGHT = 720, 2150
CROP = (0, 15, 550, 365)
PANEL_WIDTH, PANEL_HEIGHT = 704, 448
YELLOW = (45, 245, 255)
WHITE = (245, 245, 245)
CYAN = (240, 210, 55)
RED = (60, 70, 255)


def write_text(image, value, x, y, scale=0.7, color=WHITE):
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (5, 5, 5), 5, cv2.LINE_AA)
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def pixels(points, k):
    uv = points[:, :2] / points[:, 2, None] @ k[:2, :2].T + k[:2, 2]
    left, top, right, bottom = CROP
    return np.rint((uv - (left, top)) * (PANEL_WIDTH / (right - left), PANEL_HEIGHT / (bottom - top))).astype(
        int
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    methods = ["colored-icp-968", "roma-turbo", "roma-fast", "ufm"]
    traces = {m: np.load(args.results / m / "trace.npz") for m in methods}
    status = {
        m: {
            int(r["frame"]): bool(int(r["valid"]))
            for r in csv.DictReader((args.results / m / "trace.csv").open())
        }
        for m in methods
    }
    frames = traces[methods[0]]["frames"]
    assert all(np.array_equal(frames, traces[m]["frames"]) for m in methods)
    start = int(frames[0])
    ee, anchors = reference()
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    times = np.loadtxt(RECORDING / "times.txt")
    fps = (len(times) - 1) / (times[-1] - times[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        for j, frame in enumerate(frames):
            rgb = cv2.imread(str(RECORDING / "rgb" / f"{frame:06d}.jpg"))
            mask = cv2.imread(str(MASKS / f"{frame:06d}.png"), 0)
            if rgb is None or mask is None:
                raise ValueError(f"missing image or mask at {frame}")
            left, top, right, bottom = CROP
            base = cv2.resize(rgb[top:bottom, left:right], (PANEL_WIDTH, PANEL_HEIGHT))
            mask_crop = cv2.resize(
                mask[top:bottom, left:right], (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_NEAREST
            )
            contours, _ = cv2.findContours(
                (mask_crop > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            expected = ee[frame] @ np.linalg.inv(ee[start])
            expected_uv = pixels(move(anchors, expected), k)
            canvas = np.full((HEIGHT, WIDTH, 3), (20, 24, 29), np.uint8)
            write_text(canvas, "DOWEL: FIXED LANDMARKS | 1x", 15, 38, 0.78)
            for slot, method in enumerate(methods):
                y = 62 + slot * 505
                write_text(canvas, method.upper(), 17, y + 25, 0.8)
                image = base.copy()
                cv2.drawContours(image, contours, -1, CYAN, 2, cv2.LINE_AA)
                if frame > start and not status[method][int(frame)]:
                    write_text(image, "NO MATCH - LAST POSE SHOWN", 14, PANEL_HEIGHT - 20, 0.65, RED)
                for p in expected_uv:
                    cv2.circle(image, tuple(p), 8, WHITE, 2, cv2.LINE_AA)
                actual = pixels(move(anchors, traces[method]["poses"][j]), k)
                cv2.line(image, tuple(actual[0]), tuple(actual[1]), YELLOW, 3, cv2.LINE_AA)
                cv2.line(image, tuple(actual[0]), tuple(actual[2]), YELLOW, 2, cv2.LINE_AA)
                for name, p in zip("ABC", actual, strict=True):
                    cv2.circle(image, tuple(p), 5, YELLOW, -1, cv2.LINE_AA)
                    write_text(image, name, int(p[0] + 8), int(p[1] - 8), 0.57, YELLOW)
                canvas[y + 37 : y + 37 + PANEL_HEIGHT, 8 : 8 + PANEL_WIDTH] = image
            write_text(canvas, "Cyan: visible mask   Yellow: fixed A/B/C, even if hidden", 15, 2094, 0.57)
            write_text(canvas, "White rings: gripper-motion reference (approx)", 15, 2129, 0.60)
            if frame in (280, 310, 340, 399):
                cv2.imwrite(str(args.output.with_name(f"frame_{frame}.jpg")), canvas)
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("video encoding failed")


if __name__ == "__main__":
    main()
