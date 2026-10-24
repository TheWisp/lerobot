"""Render every demo frame at recorded speed with fixed reference landmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from compare import MASKS, RECORDING, move
from shape_grounding import project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--title", default="Full demo - whole-shape correction")
    args = parser.parse_args()
    trace = np.load(args.result / "trace.npz")
    summary = json.loads((args.result / "summary.json").read_text())
    frames, poses = trace["frames"], trace["poses"]
    source, anchors = trace["reference_points"], trace["landmarks_reference"]
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    times = np.loadtxt(RECORDING / "times.txt")
    fps = (len(times) - 1) / (times[-1] - times[0])
    # One fixed crop covers all observed motion. This presentation choice does
    # not feed later-frame information into the causal tracker.
    union = None
    for frame in frames:
        mask = cv2.imread(str(MASKS / f"{frame:06d}.png"), 0) > 0
        union = mask.copy() if union is None else union | mask
    y, x = np.nonzero(union)
    center = np.array([(x.min() + x.max()) / 2, (y.min() + y.max()) / 2])
    extent = int(np.ceil(max(np.ptp(x), np.ptp(y)) * 1.3))
    origin = center - extent / 2
    size, header, footer = 640, 56, 44
    height = size + header + footer
    matrix = np.array(
        [[size / extent, 0, -origin[0] * size / extent], [0, size / extent, -origin[1] * size / extent]],
        np.float32,
    )

    def text(image, value, xy, scale=0.75, color=(245, 245, 245)):
        cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 5, cv2.LINE_AA)
        cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def pixels(points):
        return np.rint((project(points, k) - origin) * size / extent).astype(int)

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
        f"{size}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        for index, frame in enumerate(frames):
            rgb = cv2.imread(str(RECORDING / "rgb" / f"{frame:06d}.jpg"))
            mask = cv2.imread(str(MASKS / f"{frame:06d}.png"), 0)
            panel = cv2.warpAffine(rgb, matrix, (size, size))
            scaled_mask = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(scaled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, (240, 210, 55), 2, cv2.LINE_AA)
            status = str(trace["statuses"][index])
            if status in ("observed", "reference"):
                points = move(source, poses[index])
                depth = cv2.imread(str(RECORDING / "depth" / f"{frame:06d}.png"), -1) * 0.001
                uv = np.rint(project(points, k)).astype(int)
                inside = (
                    (uv[:, 0] >= 0)
                    & (uv[:, 0] < depth.shape[1])
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < depth.shape[0])
                )
                visible = inside.copy()
                visible[inside] = depth[uv[inside, 1], uv[inside, 0]] >= (
                    points[inside, 2] - summary["config"]["depth_tolerance_m"]
                )
                layer = panel.copy()
                for pixel in pixels(points)[visible]:
                    cv2.circle(layer, tuple(pixel), 2, (30, 225, 255), -1, cv2.LINE_AA)
                panel = cv2.addWeighted(layer, 0.7, panel, 0.3, 0)
                anchor_pixels = pixels(move(anchors, poses[index]))
                cv2.line(
                    panel, tuple(anchor_pixels[0]), tuple(anchor_pixels[1]), (255, 255, 255), 2, cv2.LINE_AA
                )
                for name, pixel in zip("ABC", anchor_pixels, strict=True):
                    cv2.circle(panel, tuple(pixel), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    text(panel, name, tuple(pixel + [9, -9]), 0.7)
            else:
                text(panel, "No observation - estimate unavailable", (12, size - 24), 0.72, (60, 120, 255))
            canvas = np.full((height, size, 3), 23, np.uint8)
            text(canvas, args.title, (12, 36), 0.83)
            canvas[header : header + size] = panel
            text(
                canvas,
                f"{times[frame] - times[0]:.1f} s  |  1x  |  Fixed reference A, B, C",
                (12, height - 13),
                0.67,
            )
            if frame in (0, 150, 180, 210, 250, 280, 310, 340, 399, 480, 540, 600, 712):
                cv2.imwrite(str(args.output.with_name(f"full_frame_{frame}.jpg")), canvas)
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")


if __name__ == "__main__":
    main()
