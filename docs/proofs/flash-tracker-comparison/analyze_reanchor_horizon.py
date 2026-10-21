"""Measure how long causal point motion agrees with a withheld direct pose.

Every fifth frame in the insertion interval seeds an independent point track.
The track consumes only consecutive RGB-D frames, never the later direct poses
used for evaluation. Agreement with those poses is not ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from analyze_multiview import frame_points, move
from run_causal_points_demo import read_frame

from lerobot.policies.flash.causal_points import CausalRigidPointTracker


def normal_or_axis(points: np.ndarray, role: str) -> np.ndarray:
    values, basis = np.linalg.eigh(np.cov(points.T))
    return basis[:, -1] if role == "dowel" else basis[:, 0]


def unoriented_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(left @ right), -1, 1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed-step", type=int, default=5)
    parser.add_argument("--last-frame", type=int, default=500)
    args = parser.parse_args()
    if args.seed_step < 1 or not 401 <= args.last_frame <= 550:
        parser.error("seed step must be positive and last frame must be in the insertion recording")
    root = Path(__file__).parent
    traces = np.load(root / "pose_traces.npz")
    p2p = np.load(root / "p2p_dowel_internal_control_100_550.npz")
    direct = {
        "dowel": (p2p["poses"], p2p["lost"]),
        "ring": (traces["bundlesdf_ring_delta"], traces["bundlesdf_ring_failed"]),
    }
    features = {}
    for role in direct:
        points, _ = frame_points(args.dataset, args.masks, role, 100, traces["intrinsics"])
        features[role] = (points.mean(axis=0), normal_or_axis(points, role))
    tip = traces["dowel_reference_center_camera_m"] + (
        traces["dowel_reference_end_offsets_m"][1] * traces["dowel_reference_shaft_camera"]
    )
    active: dict[str, list[tuple[int, CausalRigidPointTracker]]] = {role: [] for role in direct}
    rows = []
    for frame_id in range(400, args.last_frame + 1):
        frame = read_frame(args.dataset, frame_id, traces["intrinsics"])
        for role, (poses, lost) in direct.items():
            index = frame_id - 100
            for seed_frame, tracker in active[role]:
                motion = tracker.update(frame)
                if motion.base_motion_from_seed is None:
                    rows.append((role, seed_frame, frame_id, frame_id - seed_frame, 0, 0, 0, np.nan, np.nan))
                    continue
                if lost[index]:
                    rows.append(
                        (
                            role,
                            seed_frame,
                            frame_id,
                            frame_id - seed_frame,
                            1,
                            motion.inliers,
                            0,
                            np.nan,
                            np.nan,
                        )
                    )
                    continue
                point_pose = motion.base_motion_from_seed @ poses[seed_frame - 100]
                direct_pose = poses[index]
                point = tip if role == "dowel" else features[role][0]
                _, direction = features[role]
                position_gap_mm = float(
                    np.linalg.norm(move(point[None], point_pose) - move(point[None], direct_pose)) * 1000
                )
                direction_gap_deg = unoriented_angle_deg(
                    point_pose[:3, :3] @ direction, direct_pose[:3, :3] @ direction
                )
                rows.append(
                    (
                        role,
                        seed_frame,
                        frame_id,
                        frame_id - seed_frame,
                        1,
                        motion.inliers,
                        1,
                        position_gap_mm,
                        direction_gap_deg,
                    )
                )
            if frame_id < args.last_frame and (frame_id - 400) % args.seed_step == 0 and not lost[index]:
                mask = cv2.imread(str(args.masks / role / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"missing {role} mask at frame {frame_id}")
                tracker = CausalRigidPointTracker(
                    min_inliers=12, max_frame_gap_s=0.1, fit_threshold_m=0.005, image_scale=1.0
                )
                if tracker.seed(frame, mask > 0) >= tracker.min_inliers:
                    active[role].append((frame_id, tracker))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "role",
                "seed_frame",
                "frame",
                "age_frames",
                "point_fit",
                "inliers",
                "direct_available",
                "task_point_gap_mm",
                "axis_or_normal_gap_deg",
            )
        )
        writer.writerows(rows)
    for role in direct:
        print(role, "tracks", len(active[role]))
        for age in (5, 10, 20, 40, 60):
            selected = [row for row in rows if row[0] == role and row[3] == age]
            evaluated = [row for row in selected if row[4] and row[6]]
            gaps = [row[7] for row in evaluated]
            print(
                "  age",
                age,
                "fits",
                sum(row[4] for row in selected),
                "/",
                len(selected),
                "task-point gap median mm",
                round(float(np.median(gaps)), 2) if gaps else None,
                "p90",
                round(float(np.quantile(gaps, 0.9)), 2) if gaps else None,
            )


if __name__ == "__main__":
    main()
