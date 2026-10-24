"""Show the full scene and full demo, retaining source-specific evidence.

The current experiment updates the dowel. The archived ring candidates are
explicitly labelled as previous tracking, not a new multi-object benchmark.
The renderer never fills missing poses or repeats the last available frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from compare import MASKS as DOWEL_MASKS, RECORDING, move
from shape_grounding import project as camera_project

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flash-tracker-comparison"))
from visualize_full_demo_fusion import (  # noqa: E402
    CAUSAL,
    MASKS,
    MODELS,
    P2P,
    PROOF,
    bars,
    draw_mask,
    draw_ring,
    label,
)

FIT = (245, 245, 245)
MASK = (240, 210, 55)
GRIPPER = (30, 155, 255)
GROUP = (90, 235, 90)
OLD_RING = (255, 175, 90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crf", type=int, default=29)
    args = parser.parse_args()
    trace = np.load(args.result / "trace.npz")
    summary = json.loads((args.result / "summary.json").read_text())
    rows = list(csv.DictReader((args.result / "trace.csv").open()))
    times = np.loadtxt(RECORDING / "times.txt")
    total = len(times)
    if not np.array_equal(trace["frames"], np.arange(total)):
        raise ValueError("The scene video requires the complete demo, starting at frame zero")
    if [int(r["frame"]) for r in rows] != list(range(total)):
        raise ValueError("Frame evidence must cover the same complete sequence")
    source, anchors = trace["reference_points"], trace["landmarks_reference"]
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    calibration = np.load(PROOF / "pose_traces.npz")
    camera_from_base = np.linalg.inv(calibration["base_from_camera"])
    causal = np.load(CAUSAL)
    models = np.load(MODELS)
    ring = np.load(P2P["ring"])
    ring_index = {int(frame): i for i, frame in enumerate(ring["frames"])}
    visual = np.array([r["status"] == "observed" for r in rows])
    gripper = np.array([r["prior_source"].startswith("gripper_") for r in rows])
    group = np.array([r["prior_source"] == "environment_group_motion" for r in rows])
    ring_p2p = np.zeros(total, bool)
    ring_group = np.zeros(total, bool)
    for frame, index in ring_index.items():
        if 0 <= frame < total:
            ring_p2p[frame] = not bool(ring["lost"][index])
    count = min(total, len(causal["ring_estimator"]))
    ring_group[:count] = causal["ring_estimator"][:count] == "environment_group_motion"
    available = visual | gripper | group
    held_flags = np.array([bool(int(r["held"])) for r in rows])
    previously_held_flags = np.r_[False, np.maximum.accumulate(held_flags[:-1])]
    timeline = [
        ("MASK + DEPTH", visual, MASK),
        ("CURRENT DOWEL", available, FIT),
        ("GRIPPER PRED.", gripper, GRIPPER),
        ("SCENE GROUP", group, GROUP),
        ("RING P2P (PREVIOUS)", ring_p2p, OLD_RING),
        ("RING GROUP (PREVIOUS)", ring_group, GROUP),
    ]
    fps = (total - 1) / (times[-1] - times[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
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
            "1280x720",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
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
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None

    def draw_anchors(image, pose, color, width, names=False):
        xyz = move(anchors, pose)
        if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
            return
        pixels = np.rint(camera_project(xyz, k)).astype(int)
        cv2.line(image, tuple(pixels[0]), tuple(pixels[1]), color, width, cv2.LINE_AA)
        if names:
            for name, pixel in zip("ABC", pixels, strict=True):
                cv2.circle(image, tuple(pixel), 3, color, -1, cv2.LINE_AA)
                label(image, name, tuple(pixel + [5, -5]), 0.45, (15, 15, 15), 3)
                label(image, name, tuple(pixel + [5, -5]), 0.45, color, 1)

    try:
        for frame in range(total):
            image = cv2.imread(str(RECORDING / "rgb" / f"{frame:06d}.jpg"))
            if image is None or image.shape != (480, 848, 3):
                raise ValueError(f"Unexpected recording image at frame {frame}")
            dowel_mask = cv2.imread(str(DOWEL_MASKS / f"{frame:06d}.png"), 0)
            if dowel_mask is None:
                raise ValueError(f"Missing recorded dowel mask at frame {frame}")
            draw_mask(image, dowel_mask, MASK)
            ring_mask_path = MASKS / "ring" / f"{frame:06d}.png"
            if ring_mask_path.exists():
                draw_mask(image, cv2.imread(str(ring_mask_path), 0), (240, 135, 220))
            if ring_p2p[frame]:
                draw_ring(
                    image, models["points_1"], ring["poses"][ring_index[frame]], camera_from_base, k, OLD_RING
                )
            if ring_group[frame]:
                draw_ring(image, models["points_1"], causal["ring_poses"][frame], camera_from_base, k, GROUP)
            if group[frame] or gripper[frame]:
                draw_anchors(image, trace["prior_poses"][frame], GROUP if group[frame] else GRIPPER, 5)
            if available[frame]:
                pose = trace["poses"][frame]
                # Show only model samples that are not behind measured foreground.
                depth = cv2.imread(str(RECORDING / "depth" / f"{frame:06d}.png"), -1) * 0.001
                xyz = move(source, pose)
                pixels = np.rint(camera_project(xyz, k)).astype(int)
                inside = (
                    (xyz[:, 2] > 0)
                    & (pixels[:, 0] >= 0)
                    & (pixels[:, 0] < 848)
                    & (pixels[:, 1] >= 0)
                    & (pixels[:, 1] < 480)
                )
                seen = inside.copy()
                seen[inside] = (
                    depth[pixels[inside, 1], pixels[inside, 0]]
                    >= xyz[inside, 2] - summary["config"]["depth_tolerance_m"]
                )
                for pixel in pixels[seen]:
                    cv2.circle(image, tuple(pixel), 1, (200, 200, 200), -1, cv2.LINE_AA)
                draw_anchors(image, pose, FIT, 2, names=True)
            screen = np.full((720, 1280, 3), (25, 27, 31), np.uint8)
            screen[:480, :848] = image
            screen[:52, :848] = (screen[:52, :848].astype(float) * 0.28).astype(np.uint8)
            label(screen, "FULL DEMO / WHOLE SCENE", (24, 35), 0.82)
            label(screen, f"{times[frame] - times[0]:.1f}s / 1x", (663, 35), 0.65)
            panel = screen[:480, 848:]
            label(panel, "CURRENT DOWEL TEST", (22, 39), 0.76)
            label(panel, "SAM outline", (22, 83), 0.66, MASK)
            label(panel, "Estimated A / B / C", (22, 121), 0.66, FIT)
            label(panel, "Gripper prediction", (22, 159), 0.66, GRIPPER)
            label(panel, "Scene-group prediction", (22, 197), 0.65, GROUP)
            cv2.line(panel, (20, 218), (412, 218), (80, 80, 80), 1)
            label(panel, "RING: PREVIOUS TRACKING", (22, 254), 0.65, OLD_RING)
            label(panel, "Point2Pose + scene group", (22, 285), 0.58, (200, 200, 200), 1)
            cv2.line(panel, (20, 307), (412, 307), (80, 80, 80), 1)
            held = held_flags[frame]
            previously_held = previously_held_flags[frame]
            phase = (
                "HELD / INSERTION"
                if held
                else ("RELEASE / RETREAT" if previously_held else "APPROACH / PICKUP")
            )
            label(panel, phase, (22, 344), 0.71)
            label(
                panel, "Mask + depth: " + ("available" if visual[frame] else "unavailable"), (22, 382), 0.59
            )
            motion = "gripper" if gripper[frame] else ("scene group" if group[frame] else "unavailable")
            label(panel, "Motion bridge: " + motion, (22, 414), 0.59)
            if not available[frame]:
                label(panel, "No current dowel estimate", (22, 454), 0.59, (70, 150, 255))
            cv2.line(screen, (0, 480), (1279, 480), (100, 100, 100), 2)
            label(screen, "SOURCE AVAILABILITY / COMPLETE RECORDING", (24, 512), 0.73)
            bars(screen, 538, timeline, frame, total)
            if frame in (0, 150, 180, 240, 310, 399, 540, 712):
                cv2.imwrite(str(args.output.with_name(f"scene_frame_{frame}.jpg")), screen)
            encoder.stdin.write(screen.tobytes())
    finally:
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
    manifest = {
        "frames": total,
        "fps": fps,
        "source": str(args.result),
        "new_tracking_role": "dowel",
        "previous_ring_candidates": str(P2P["ring"]),
        "previous_group_recording": str(CAUSAL),
        "note": "Every original frame, full camera field of view; no tracking result is extended beyond its recorded availability.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
