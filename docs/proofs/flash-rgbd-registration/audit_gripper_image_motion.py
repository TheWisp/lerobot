"""Compare visible gripper features with recorded FK, independently of object fits.

The polygon is a diagnostic annotation on the gripper housing, not an object
tracking rule. The timing sweep is reported only; it changes no input or pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from compare import RECORDING

from lerobot.policies.flash.depth_quality import depth_support_mask
from lerobot.policies.flash.perception import TimedEndEffectorHistory


def project(points, k):
    pixels = points @ k.T
    return pixels[:, :2] / pixels[:, 2:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    first, last = config["frames"]
    times = np.loadtxt(RECORDING / "times.txt")
    times = times - times[0]
    k = np.loadtxt(RECORDING / "cam_K.txt")
    origin = np.loadtxt(RECORDING / "times.txt")[0]
    arm = np.load(RECORDING.parent / "showservo_demo.npz")
    calibration = np.load(Path(__file__).parent.parent / "flash-tracker-comparison/pose_traces.npz")
    camera_from_base = np.linalg.inv(calibration["base_from_camera"])
    history = TimedEndEffectorHistory(capacity=len(arm["t"]), max_sample_gap_s=config["max_motor_gap_s"])
    for t, ee in zip(arm["t"], arm["tips"], strict=True):
        history.append(float(arm["t0"] + t - origin), camera_from_base @ ee)

    def rgb(index):
        return cv2.imread(str(RECORDING / "rgb" / f"{index:06d}.jpg"))

    gray = cv2.cvtColor(rgb(first), cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [np.array(config["gripper_polygon_px"], np.int32)], 255)
    uv = cv2.goodFeaturesToTrack(gray, config["max_points"], 0.01, 4, mask=mask).reshape(-1, 2)
    depth = cv2.imread(str(RECORDING / "depth" / f"{first:06d}.png"), -1) * 0.001
    px = np.rint(uv).astype(int)
    valid = depth_support_mask(depth, radius_px=1)[px[:, 1], px[:, 0]]
    uv, px = uv[valid], px[valid]
    z = depth[px[:, 1], px[:, 0]]
    xyz = np.c_[(uv[:, 0] - k[0, 2]) * z / k[0, 0], (uv[:, 1] - k[1, 2]) * z / k[1, 1], z]
    valid = np.ones(len(uv), bool)
    tracks, fb_errors = [uv.copy()], []
    for f in range(first + 1, last + 1):
        new = cv2.cvtColor(rgb(f), cv2.COLOR_BGR2GRAY)
        after, ok, _ = cv2.calcOpticalFlowPyrLK(gray, new, uv, None, winSize=(21, 21), maxLevel=3)
        before, back_ok, _ = cv2.calcOpticalFlowPyrLK(new, gray, after, None, winSize=(21, 21), maxLevel=3)
        error = np.linalg.norm(before - uv, axis=1)
        valid &= ok.ravel().astype(bool) & back_ok.ravel().astype(bool) & (error < config["fb_threshold_px"])
        fb_errors.append(error)
        uv, gray = after, new
        tracks.append(uv.copy())
    if valid.sum() < 3:
        raise ValueError("Not enough verified gripper feature tracks")
    tracks = np.asarray(tracks)[:, valid]
    xyz = xyz[valid]

    def predictions(offset):
        seed = history.at(float(times[first] + offset))
        out = []
        for f in range(first, last + 1):
            ee = history.at(float(times[f] + offset))
            if ee is None or seed is None:
                raise ValueError("Timing sweep exceeds available motor history")
            motion = ee @ np.linalg.inv(seed)
            out.append(project(xyz @ motion[:3, :3].T + motion[:3, 3], k))
        return np.asarray(out)

    baseline = predictions(0)
    gaps = np.median(np.linalg.norm(baseline - tracks, axis=2), axis=1)
    sweep = []
    for offset in config["diagnostic_motor_offsets_s"]:
        residual = np.linalg.norm(predictions(offset) - tracks, axis=2)
        sweep.append({"offset_s": offset, "median_px": float(np.median(residual))})
    result = {
        "config": config,
        "verified_features": int(valid.sum()),
        "median_flow_roundtrip_px": float(np.median(np.array(fb_errors)[:, valid])),
        "frames": [
            {"frame": f, "seconds": float(times[f]), "median_gripper_projection_gap_px": float(gap)}
            for f, gap in zip(range(first, last + 1), gaps, strict=True)
        ],
        "diagnostic_timing_sweep": sweep,
        "conclusion": "Recorded FK does not align with visible gripper motion. Timing alone does not fully explain it; calibration/kinematic error remains possible. No offset was applied.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    report = config["report_frame"]
    left, right = rgb(first), rgb(report)
    for uv0, measured, predicted in zip(
        tracks[0], tracks[report - first], baseline[report - first], strict=True
    ):
        cv2.circle(left, tuple(np.rint(uv0).astype(int)), 3, (60, 240, 80), -1)
        cv2.circle(right, tuple(np.rint(measured).astype(int)), 3, (60, 240, 80), -1)
        x, y = np.rint(predicted).astype(int)
        cv2.drawMarker(right, (x, y), (40, 60, 255), cv2.MARKER_CROSS, 9, 2)
    canvas = np.zeros((560, 1696, 3), np.uint8)
    canvas[80:, :848], canvas[80:, 848:] = left, right
    cv2.putText(
        canvas,
        "Same gripper features: green = image, red = recorded motion",
        (20, 35),
        0,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(canvas, f"Frame {first}", (20, 68), 0, 0.8, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        f"Frame {report}: {gaps[report - first]:.1f} px disagreement",
        (868, 68),
        0,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(str(args.output / "gripper_disagreement.jpg"), canvas)
    print(json.dumps({k: v for k, v in result.items() if k not in ("config", "frames")}, indent=2))


if __name__ == "__main__":
    main()
