"""Test causal RoMa matches against several earlier dowel views.

This is an offline comparison, not a runtime estimator. All source poses come
from earlier estimated frames; the gripper is used only to score the result.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from compare import RECORDING, crop_pair, frame_data, move, reference, robust_fit, roma_pairs


def sources_for(target: int, frames: dict[int, dict], clear_window: int) -> list[int]:
    sources = [target - gap for gap in (1, 2, 4) if target - gap >= 280]
    eligible = range(max(280, target - clear_window), target - 4)
    if eligible:
        clearest = max(eligible, key=lambda frame: len(frames[frame]["obs"]["points"]))
        sources.append(clearest)
    return list(dict.fromkeys(sources))


def candidate_medoid(candidates: list[tuple[int, np.ndarray]], landmarks: np.ndarray) -> int:
    points = np.array([move(landmarks, pose) for _, pose in candidates])
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1).mean(axis=-1)
    return int(np.argmin(np.median(distances, axis=1)))


def evaluate(poses, ee, landmarks, axis, frame):
    expected = ee[frame] @ np.linalg.inv(ee[280])
    error_mm = np.mean(np.linalg.norm(move(landmarks, poses[-1]) - move(landmarks, expected), axis=1)) * 1000
    dot = np.clip((poses[-1][:3, :3] @ axis) @ (expected[:3, :3] @ axis), -1, 1)
    return float(error_mm), float(np.degrees(np.arccos(dot)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clear-window", type=int, default=20)
    args = parser.parse_args()
    if args.clear_window < 5:
        parser.error("clear-window must be at least 5 frames")

    from romav2 import RoMaV2

    cv2.setNumThreads(1)
    model = RoMaV2()
    model.apply_setting("fast")
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    frames = {frame: frame_data(frame, k) for frame in range(280, 400)}
    ee, landmarks = reference()
    axis = landmarks[1] - landmarks[0]
    axis /= np.linalg.norm(axis)
    estimated = {"medoid": [np.eye(4)], "joint": [np.eye(4)]}
    rows = []
    for target_frame in range(281, 400):
        history = sources_for(target_frame, frames, args.clear_window)
        start = time.perf_counter()
        pairs = []
        for source_frame in history:
            a, b = frames[source_frame], frames[target_frame]
            image_a, image_b, crop = crop_pair(a, b)
            source, target = roma_pairs(model, image_a, image_b, crop, a, b, k)
            relative, inliers, residual = robust_fit(source, target)
            pairs.append((source_frame, source, target, relative, inliers, residual))
        inference_ms = (time.perf_counter() - start) * 1000
        row = {"frame": target_frame, "sources": ",".join(map(str, history)), "inference_ms": inference_ms}
        for mode in estimated:
            poses = estimated[mode]
            candidates = [
                (f, relative @ poses[f - 280]) for f, _, _, relative, _, _ in pairs if relative is not None
            ]
            if candidates:
                medoid_index = candidate_medoid(candidates, landmarks)
                chosen_frame, chosen_pose = candidates[medoid_index]
                if mode == "joint":
                    # Equalize frame contributions before a joint 3D rigid fit.
                    # Only include sources that agree with the medoid's three
                    # transported landmarks at least as well as the median source.
                    predicted = move(landmarks, chosen_pose)
                    disagreements = {
                        frame: np.linalg.norm(move(landmarks, pose) - predicted, axis=1).mean()
                        for frame, pose in candidates
                    }
                    limit = np.median(list(disagreements.values()))
                    selected = [
                        item for item in pairs if item[0] in disagreements and disagreements[item[0]] <= limit
                    ]
                    source_clouds, target_clouds = [], []
                    for frame, source, target, _, _, _ in selected:
                        if len(source) == 0:
                            continue
                        sampled = np.linspace(0, len(source) - 1, min(256, len(source)), dtype=int)
                        object_from_camera = np.linalg.inv(poses[frame - 280])
                        source_clouds.append(move(source[sampled], object_from_camera))
                        target_clouds.append(target[sampled])
                    if source_clouds:
                        joint_pose, _, _ = robust_fit(
                            np.concatenate(source_clouds), np.concatenate(target_clouds)
                        )
                        if joint_pose is not None:
                            chosen_pose = joint_pose
                poses.append(chosen_pose)
                row[f"{mode}_source"] = chosen_frame
                row[f"{mode}_valid"] = 1
            else:
                poses.append(poses[-1].copy())
                row[f"{mode}_source"] = -1
                row[f"{mode}_valid"] = 0
            error_mm, axis_deg = evaluate(poses, ee, landmarks, axis, target_frame)
            row[f"{mode}_landmark_error_mm"] = error_mm
            row[f"{mode}_axis_error_deg"] = axis_deg
        rows.append(row)
        if target_frame % 20 == 0:
            print(
                target_frame,
                {mode: round(row[f"{mode}_axis_error_deg"], 1) for mode in estimated},
                flush=True,
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output / "trace.npz",
        frames=np.arange(280, 400),
        **{k: np.array(v) for k, v in estimated.items()},
    )
    summary = {
        mode: {
            "valid": sum(row[f"{mode}_valid"] for row in rows),
            "median_landmark_error_mm": float(np.median([row[f"{mode}_landmark_error_mm"] for row in rows])),
            "last_landmark_error_mm": rows[-1][f"{mode}_landmark_error_mm"],
            "median_axis_error_deg": float(np.median([row[f"{mode}_axis_error_deg"] for row in rows])),
            "last_axis_error_deg": rows[-1][f"{mode}_axis_error_deg"],
        }
        for mode in estimated
    }
    summary["median_inference_ms"] = float(np.median([row["inference_ms"] for row in rows]))
    summary["clear_window"] = args.clear_window
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
