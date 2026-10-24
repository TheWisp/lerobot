"""Full-duration, synchronized comparison; failures and the one restart remain visible."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from run_surface_tracker import move

WIDTH, HEIGHT = 720, 1650
CROP = (0, 15, 550, 365)
PANEL_HEIGHT = 448
PANEL_WIDTH = 704
WHITE = (245, 245, 245)
CYAN = (245, 220, 40)
YELLOW = (30, 245, 255)
RED = (85, 85, 255)


def text(image, value, x, y, scale=0.8, color=WHITE):
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (8, 8, 8), 5, cv2.LINE_AA)
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def pixels(points, intrinsics):
    uv = points[:, :2] / points[:, 2, None] @ intrinsics[:2, :2].T + intrinsics[:2, 2]
    left, top, right, bottom = CROP
    return (uv - (left, top)) * (PANEL_WIDTH / (right - left), PANEL_HEIGHT / (bottom - top))


def panel(rgb, trace, frame):
    left, top, right, bottom = CROP
    image = cv2.resize(rgb[top:bottom, left:right], (PANEL_WIDTH, PANEL_HEIGHT))
    seed = int(trace["seed_frame"])
    if frame < seed:
        text(image, f"New landmarks begin at frame {seed}", 20, 40, 0.72)
        return image
    index = frame - int(trace["frames"][0])
    accepted = trace["state"][index] in ("initialized", "surface_supported")
    color = CYAN if accepted else RED
    points = trace["display_points_camera"][index]
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]
    overlay = image.copy()
    for x, y in np.rint(pixels(points, trace["intrinsics"])).astype(int):
        if 0 <= x < PANEL_WIDTH and 0 <= y < PANEL_HEIGHT:
            cv2.circle(overlay, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    pose = trace["poses"][index]
    landmarks = trace["landmarks_object"]
    axis_points = landmarks[0] + trace["basis_object"].T * 0.025
    uv = pixels(move(np.concatenate((landmarks, axis_points)), pose), trace["intrinsics"])
    uv = np.rint(uv).astype(int)
    for endpoint, axis_color in zip(uv[3:], [(75, 75, 255), (80, 240, 80), (255, 140, 80)], strict=True):
        cv2.line(image, tuple(uv[0]), tuple(endpoint), axis_color, 3, cv2.LINE_AA)
    missing = []
    for label, (x, y) in zip("ABC", uv[:3], strict=True):
        if 0 <= x < PANEL_WIDTH and 0 <= y < PANEL_HEIGHT:
            cv2.circle(image, (int(x), int(y)), 6, YELLOW, -1, cv2.LINE_AA)
            text(image, label, int(x + 8), int(y - 8), 0.65, YELLOW)
        else:
            missing.append(label)
    if not accepted:
        text(image, "WITHHELD - showing last pose", 18, PANEL_HEIGHT - 20, 0.7, RED)
    elif missing:
        text(image, f"Landmarks off screen: {', '.join(missing)}", 18, PANEL_HEIGHT - 20, 0.7, RED)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    traces = [
        np.load(args.results / name / "trace.npz") for name in ("connected_full", "previous", "connected280")
    ]
    labels = [
        "Keep surface + depth cleanup",
        "Replace surface every frame",
        "Depth cleanup; restart at frame 280",
    ]
    times = np.loadtxt(args.recording / "times.txt")
    fps = (len(times) - 1) / (times[-1] - times[0])
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
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        for frame in range(len(times)):
            rgb = cv2.imread(str(args.recording / "rgb" / f"{frame:06d}.jpg"))
            if rgb is None:
                raise ValueError(f"missing RGB at {frame}")
            canvas = np.full((HEIGHT, WIDTH, 3), (20, 24, 29), np.uint8)
            text(canvas, "MASK + DEPTH ONLY | FULL DEMO | 1x", 16, 40, 0.8)
            for j, (trace, label) in enumerate(zip(traces, labels, strict=True)):
                y = 90 + j * 510
                text(canvas, label, 16, y, 0.79)
                canvas[y + 12 : y + 12 + PANEL_HEIGHT, 8 : 8 + PANEL_WIDTH] = panel(rgb, trace, frame)
            text(canvas, "Cyan: surface   Yellow: fixed landmarks", 16, 1600, 0.72)
            text(
                canvas,
                f"Frame {frame:03d} | {times[frame] - times[0]:.1f}s   No P2P / gripper input",
                16,
                1635,
                0.66,
            )
            if frame in (100, 300, 340, 540):
                cv2.imwrite(str(args.output.with_name(f"frame_{frame}.jpg")), canvas)
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("video encoding failed")


if __name__ == "__main__":
    main()
