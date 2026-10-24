"""Show the actual frozen measured surface before and after shape correction."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from compare import HERE, RECORDING, frame_data, move
from shape_grounding import project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = np.load(args.result / "trace.npz")
    summary = json.loads((args.result / "summary.json").read_text())
    method = summary["method"]
    baseline = np.load(HERE / "results" / method / "trace.npz")
    source, initial = result["reference_points"], result["initial_pose"]
    center = source.mean(0)
    first = np.argmax(np.linalg.norm(source - center, axis=1))
    second = np.argmax(np.linalg.norm(source - source[first], axis=1))
    axis = source[second] - source[first]
    axis /= np.linalg.norm(axis)
    delta = source - source[first]
    third = np.argmax(np.linalg.norm(delta - (delta @ axis)[:, None] * axis, axis=1))
    anchors = source[[first, second, third]]
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    times = np.loadtxt(RECORDING / "times.txt")
    fps = (len(times) - 1) / (times[-1] - times[0])
    width, height = 720, 1510
    crop = (100, 35, 360, 295)
    size = 700

    def text(image, value, xy, scale=0.85, color=(245, 245, 245)):
        cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 5, cv2.LINE_AA)
        cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def pix(points):
        return np.rint((project(points, k) - crop[:2]) * size / (crop[2] - crop[0])).astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
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
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        "scale=576:1208",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        for index, frame in enumerate(result["frames"]):
            data = frame_data(int(frame), k)
            image = cv2.cvtColor(data["rgb"], cv2.COLOR_RGB2BGR)
            left, top, right, bottom = crop
            base = cv2.resize(image[top:bottom, left:right], (size, size))
            mask = cv2.resize(
                data["mask"][top:bottom, left:right].astype(np.uint8),
                (size, size),
                interpolation=cv2.INTER_NEAREST,
            )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            canvas = np.full((height, width, 3), 23, np.uint8)
            for slot, (title, pose) in enumerate(
                (
                    (method.upper(), baseline["poses"][index] @ initial),
                    (method.upper() + " + whole-shape correction", result["poses"][index]),
                )
            ):
                y = slot * 750
                text(canvas, title, (12, y + 32), 0.88)
                panel = base.copy()
                cv2.drawContours(panel, contours, -1, (240, 210, 55), 2, cv2.LINE_AA)
                points = move(source, pose)
                uv = np.rint(project(points, k)).astype(int)
                inside = (
                    (uv[:, 0] >= 0)
                    & (uv[:, 0] < data["depth"].shape[1])
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < data["depth"].shape[0])
                )
                visible = inside.copy()
                visible[inside] = (
                    data["depth"][uv[inside, 1], uv[inside, 0]]
                    >= points[inside, 2] - summary["config"]["depth_tolerance_m"]
                )
                layer = panel.copy()
                for pixel, shown in zip(pix(points), visible, strict=True):
                    if shown:
                        cv2.circle(layer, tuple(pixel), 2, (30, 225, 255), -1, cv2.LINE_AA)
                panel = cv2.addWeighted(layer, 0.7, panel, 0.3, 0)
                anchor_pixels = pix(move(anchors, pose))
                cv2.line(
                    panel, tuple(anchor_pixels[0]), tuple(anchor_pixels[1]), (255, 255, 255), 3, cv2.LINE_AA
                )
                for name, pixel in zip("ABC", anchor_pixels, strict=True):
                    cv2.circle(panel, tuple(pixel), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    text(panel, name, tuple(pixel + [8, -8]), 0.7)
                canvas[y + 44 : y + 744, 10:710] = panel
            text(canvas, "Same measured reference surface | 1x", (12, 1500), 0.7)
            if frame in (280, 310, 340, 360, 399):
                cv2.imwrite(str(args.output.with_name(f"shape_frame_{frame}.jpg")), canvas)
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("video encoding failed")


if __name__ == "__main__":
    main()
