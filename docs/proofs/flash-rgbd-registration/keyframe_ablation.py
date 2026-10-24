"""Test direct clear-frame relocalization instead of adjacent-frame chaining."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from compare import (
    RECORDING,
    crop_pair,
    frame_data,
    move,
    reference,
    robust_fit,
    roma_pairs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from romav2 import RoMaV2

    cv2.setNumThreads(1)
    model = RoMaV2()
    model.apply_setting("fast")
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    frames = {n: frame_data(n, k) for n in range(280, 400)}
    ee, landmarks = reference()
    axis = landmarks[1] - landmarks[0]
    axis /= np.linalg.norm(axis)
    args.output.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for stride in (10, 20, 40, 119):
        source_frame = 280
        source_pose = np.eye(4)
        last_pose = np.eye(4)
        rows = []
        poses = [last_pose.copy()]
        for target_frame in range(281, 400):
            if target_frame - source_frame > stride:
                source_frame = target_frame - 1
                source_pose = last_pose.copy()
            a, b = frames[source_frame], frames[target_frame]
            rgb_a, rgb_b, crop = crop_pair(a, b)
            start = time.perf_counter()
            source, target = roma_pairs(model, rgb_a, rgb_b, crop, a, b, k)
            relative, inliers, residual = robust_fit(source, target)
            inference_ms = (time.perf_counter() - start) * 1000
            if relative is not None:
                last_pose = relative @ source_pose
            poses.append(last_pose.copy())
            expected = ee[target_frame] @ np.linalg.inv(ee[280])
            err = (
                np.mean(np.linalg.norm(move(landmarks, last_pose) - move(landmarks, expected), axis=1)) * 1000
            )
            cosine = np.clip((last_pose[:3, :3] @ axis) @ (expected[:3, :3] @ axis), -1, 1)
            rows.append(
                {
                    "frame": target_frame,
                    "source_frame": source_frame,
                    "valid": int(relative is not None),
                    "candidate_pairs": len(source),
                    "inliers": inliers,
                    "fit_residual_mm": residual * 1000,
                    "landmark_error_vs_gripper_mm": err,
                    "axis_error_vs_gripper_deg": np.degrees(np.arccos(cosine)),
                    "inference_ms": inference_ms,
                }
            )
        with (args.output / f"stride_{stride}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(
            args.output / f"stride_{stride}.npz",
            frames=np.arange(280, 400),
            poses=np.array(poses),
            landmarks_camera_at_start=landmarks,
        )
        summary = {
            "stride": stride,
            "valid": sum(row["valid"] for row in rows),
            "pairs": len(rows),
            "median_landmark_error_vs_gripper_mm": float(
                np.median([row["landmark_error_vs_gripper_mm"] for row in rows])
            ),
            "last_landmark_error_vs_gripper_mm": float(rows[-1]["landmark_error_vs_gripper_mm"]),
            "median_axis_error_vs_gripper_deg": float(
                np.median([row["axis_error_vs_gripper_deg"] for row in rows])
            ),
            "median_inference_ms": float(np.median([row["inference_ms"] for row in rows])),
        }
        all_summaries[str(stride)] = summary
        print(json.dumps(summary), flush=True)
    (args.output / "summary.json").write_text(json.dumps(all_summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
