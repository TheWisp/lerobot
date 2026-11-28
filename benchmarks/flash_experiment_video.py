"""Render identical source frames with persistent axes for both candidate traces."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_experiment_inputs import samples
from lerobot.policies.flash.surface_grounding import project
from lerobot.policies.flash.surface_observation import initial_landmarks, move


def render(run: Path, left: str = "ufm_chain"):
    case = json.loads((run / "case.json").read_text())
    report = json.loads((run / "summary.json").read_text())  # complete runs only
    rows = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    with np.load(run / "reference_points.npz", allow_pickle=False) as saved:
        refs = {role: saved[role].copy() for role in saved.files}
    glyphs = {}
    for role, ref in refs.items():
        anchors, basis = initial_landmarks(ref)
        length = np.linalg.norm(np.ptp(ref, axis=0)) / 3
        glyphs[role] = np.concatenate((anchors, anchors[0] + basis.T * length))
    interval = np.diff([row["time_s"] for row in rows])
    fps = 1 / float(np.median(interval))
    encoder = None
    output = run / ("comparison.mp4" if left == "ufm_chain" else "comparison-retained-reference.mp4")
    if output.exists():
        raise FileExistsError(output)
    try:
        for (frame, masks), row in zip(samples(case), rows, strict=True):
            panels = []
            left_title = "UFM motion" if left == "ufm_chain" else "UFM: retain matched image"
            for method, title in ((left, left_title), ("surface_corrected", "UFM + surface fit")):
                panel = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                for index, role in enumerate(case["roles"]):
                    contours, _ = cv2.findContours(
                        masks[role].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(panel, contours, -1, (180, 180, 180), 1)
                    pose = row["objects"][role][method]
                    if pose is None or role not in glyphs:
                        cv2.putText(
                            panel,
                            f"{role}: no pose",
                            (12, 28 + 28 * index),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 180, 255),
                            2,
                        )
                        continue
                    positions = move(glyphs[role], np.asarray(pose))
                    pixels = project(positions, frame.intrinsics)
                    if not np.isfinite(pixels).all() or np.any(positions[:, 2] <= 0):
                        continue
                    xy = np.clip(np.rint(pixels), -10000, 10000).astype(int)
                    for endpoint, color in zip(xy[3:], ((0, 0, 255), (0, 255, 0), (255, 0, 0)), strict=True):
                        cv2.line(panel, tuple(xy[0]), tuple(endpoint), color, 3, cv2.LINE_AA)
                    for p in xy[:3]:
                        cv2.circle(panel, tuple(p), 4, (255, 255, 255), -1, cv2.LINE_AA)
                panel = cv2.copyMakeBorder(panel, 48, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(panel, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                panels.append(panel)
            canvas = np.concatenate(panels, axis=1)
            if encoder is None:
                h, w = canvas.shape[:2]
                encoder = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "bgr24",
                        "-s",
                        f"{w}x{h}",
                        "-r",
                        str(fps),
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(output),
                    ],
                    stdin=subprocess.PIPE,
                )
            encoder.stdin.write(canvas.tobytes())
    finally:
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("video encoding failed")
    report["video"] = {
        "path": str(output),
        "fps": fps,
        "timing": "one frame per recorded frame; median interval playback; gaps not expanded",
        "glyph": "same initial three landmarks and axes, transformed by each candidate",
    }
    (run / "video.json").write_text(json.dumps(report["video"], indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--left", choices=("ufm_chain", "ufm_retained_reference"), default="ufm_chain")
    args = parser.parse_args()
    render(args.run, args.left)
