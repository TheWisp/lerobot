"""Measure the saved vision-only trajectory without feeding it motor poses.

Intervals are review annotations of this recording, never runtime criteria.
Depth agreement uses the connected cleaned patch, not other disconnected
surfaces inside the mask. Neither that agreement nor motor motion is ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from compare import RECORDING, frame_data, move
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from shape_grounding import project


def angle(a, b, unoriented=False):
    value = float(a @ b)
    return float(np.degrees(np.arccos(np.clip(abs(value) if unoriented else value, -1, 1))))


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(zip(("median", "p95", "max"), np.percentile(values, [50, 95, 100]).tolist(), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vision = np.load(args.vision / "trace.npz")
    fusion = np.load(args.fusion / "trace.npz")
    with (args.vision / "trace.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    with (args.fusion / "trace.csv").open() as stream:
        fused_rows = list(csv.DictReader(stream))
    frames, poses, source = vision["frames"], vision["poses"], vision["reference_points"]
    np.testing.assert_array_equal(frames, fusion["frames"])
    np.testing.assert_array_equal(frames, [int(r["frame"]) for r in saved])
    ee = fusion["end_effector_poses"]
    held = np.array([bool(int(r["held"])) for r in fused_rows])
    selected = np.flatnonzero(held)
    times = np.loadtxt(RECORDING / "times.txt")
    recording = np.load(RECORDING.parent / "showservo_demo.npz")
    motor_times = float(recording["t0"]) + recording["t"]
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    center = source.mean(0)
    basis = np.linalg.svd(source - center, full_matrices=False)[2]
    centers = poses[:, :3, :3] @ center + poses[:, :3, 3]
    axes = poses[:, :3, :3] @ basis[0]
    args.output.mkdir(parents=True, exist_ok=True)
    fps = (len(times) - 1) / (times[-1] - times[0])
    video = subprocess.Popen(
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
            "29",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output / "held_vision.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    tiles, rows = [], []
    review_frames = {260, 279, 300, 340, 399, 435}
    try:
        for i in selected:
            f = int(frames[i])
            observed = frame_data(f, k)
            points = observed["obs"]["points"]
            xyz = move(source, poses[i])
            row = {
                "frame": f,
                "seconds": float(times[f] - times[0]),
                "status": saved[i]["status"],
                "mask_pixels": int(observed["mask"].sum()),
                "connected_surface_samples": len(points),
                "shape_cost": float(saved[i]["shape_cost_after"]),
                "coarse_reinitialization": int(saved[i]["used_coarse"]),
                "accepted_by_fusion": int(fused_rows[i]["visual_changed"]),
            }
            motor_index = int(np.argmin(abs(motor_times - times[f])))
            row.update(
                gripper_measured=float(recording["grippers"][motor_index]),
                gripper_command=float(recording["q_cmd"][motor_index, -1]),
                nearest_motor_offset_ms=float(motor_times[motor_index] - times[f]) * 1000,
            )
            if len(points) >= 3:
                current_basis = np.linalg.svd(points - points.mean(0), full_matrices=False)[2]
                row["axis_vs_connected_depth_deg"] = angle(axes[i], current_basis[0], True)
                row["visible_surface_gap_mm"] = float(np.median(cKDTree(xyz).query(points)[0]) * 1000)
            else:
                row.update(axis_vs_connected_depth_deg=np.nan, visible_surface_gap_mm=np.nan)
            for key in (
                "axis_step_deg",
                "axis_line_step_deg",
                "center_step_mm",
                "full_rotation_step_deg",
                "axis_step_vs_gripper_deg",
                "center_step_vs_gripper_mm",
                "rotation_step_vs_gripper_deg",
            ):
                row[key] = np.nan
            if i > 0 and held[i - 1]:
                row["axis_step_deg"] = angle(axes[i], axes[i - 1])
                row["axis_line_step_deg"] = angle(axes[i], axes[i - 1], True)
                row["center_step_mm"] = float(np.linalg.norm(centers[i] - centers[i - 1]) * 1000)
                row["full_rotation_step_deg"] = float(
                    np.degrees(Rotation.from_matrix(poses[i, :3, :3] @ poses[i - 1, :3, :3].T).magnitude())
                )
                if np.isfinite(ee[i - 1 : i + 1]).all():
                    delta = ee[i] @ np.linalg.inv(ee[i - 1])
                    predicted = delta @ poses[i - 1]
                    row["axis_step_vs_gripper_deg"] = angle(axes[i], predicted[:3, :3] @ basis[0])
                    row["center_step_vs_gripper_mm"] = float(
                        np.linalg.norm(centers[i] - (predicted[:3, :3] @ center + predicted[:3, 3])) * 1000
                    )
                    row["rotation_step_vs_gripper_deg"] = float(
                        np.degrees(Rotation.from_matrix(poses[i, :3, :3] @ predicted[:3, :3].T).magnitude())
                    )
            rows.append(row)
            image = cv2.cvtColor(observed["rgb"], cv2.COLOR_RGB2BGR)
            contours = cv2.findContours(
                observed["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            cv2.drawContours(image, contours, -1, (255, 255, 0), 1, cv2.LINE_AA)
            uv = np.rint(project(xyz, k)).astype(np.int32)
            cv2.polylines(image, [cv2.convexHull(uv)], True, (255, 255, 255), 1, cv2.LINE_AA)
            anchors = np.rint(project(move(vision["landmarks_reference"], poses[i]), k)).astype(int)
            for name, pixel in zip("ABC", anchors, strict=True):
                cv2.circle(image, tuple(pixel), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    image,
                    name,
                    tuple(pixel + [5, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            screen = cv2.resize(image, (1280, 720))
            cv2.rectangle(screen, (0, 0), (1280, 49), (20, 20, 20), -1)
            cv2.putText(
                screen,
                "VISION ONLY | white: estimate | cyan: SAM",
                (18, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                screen,
                f"{row['seconds']:.2f}s",
                (1135, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            video.stdin.write(screen.tobytes())
            if f in review_frames:
                tile = cv2.resize(image, (636, 360))
                cv2.rectangle(tile, (0, 0), (636, 36), (20, 20, 20), -1)
                cv2.putText(
                    tile,
                    f"Vision only - {row['seconds']:.2f}s / frame {f}",
                    (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
            if f % 60 == 0:
                print("audited", f, flush=True)
    finally:
        video.stdin.close()
        if video.wait() != 0:
            raise RuntimeError("Video encoding failed")
    cv2.imwrite(
        str(args.output / "held_vision.jpg"),
        np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]),
    )
    with (args.output / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    intervals = []
    for start, stop, name in [
        (230, 233, "fingers_still_closing"),
        (234, 279, "closed_fingers_pickup_and_early_rotation"),
        (280, 359, "rotation_and_carry"),
        (360, 399, "previously_reviewed_clear_view"),
        (400, 487, "insertion_before_opening_command"),
        (488, 514, "opening_but_annotation_still_says_held"),
    ]:
        part = [r for r in rows if start <= r["frame"] <= stop]
        block = {"name": name, "frames": [start, stop], "seconds": [part[0]["seconds"], part[-1]["seconds"]]}
        for key in (
            "axis_vs_connected_depth_deg",
            "visible_surface_gap_mm",
            "axis_line_step_deg",
            "center_step_mm",
            "full_rotation_step_deg",
            "axis_step_vs_gripper_deg",
            "center_step_vs_gripper_mm",
            "rotation_step_vs_gripper_deg",
        ):
            block[key] = stats([r[key] for r in part if r["frame"] > start or "step" not in key])
        # This is drift relative to the motor pose, not independent accuracy.
        idx = np.arange(start, stop + 1)
        attachment = np.linalg.inv(ee[idx]) @ poses[idx]
        object_centers = attachment[:, :3, :3] @ center + attachment[:, :3, 3]
        block["center_change_in_gripper_frame_mm"] = stats(
            np.linalg.norm(object_centers - object_centers[0], axis=1) * 1000
        )
        object_axes = attachment[:, :3, :3] @ basis[0]
        block["axis_change_in_gripper_frame_deg"] = stats([angle(a, object_axes[0]) for a in object_axes])
        intervals.append(block)
    finite = [r for r in rows if np.isfinite(r["full_rotation_step_deg"])]
    summary = {
        "vision_input": str(args.vision),
        "vision_sha256": hashlib.sha256((args.vision / "trace.npz").read_bytes()).hexdigest(),
        "fusion_input": str(args.fusion),
        "frames": [int(selected[0]), int(selected[-1])],
        "pose_frames": len(rows),
        "missing_visual_geometry": sum(r["status"] != "observed" for r in rows),
        "intervals": intervals,
        "largest_rotation_steps": sorted(finite, key=lambda r: r["full_rotation_step_deg"], reverse=True)[:8],
        "fusion_visual_updates_during_hold": [r["frame"] for r in rows if r["accepted_by_fusion"]],
        "reviewed_motor_events": {
            "closure_settled_camera_frame": 234,
            "opening_command_camera_frame": 488,
            "measured_opening_camera_frame": 490,
            "note": "Manually reviewed motor trace, not a new contact detector; these events do not prove physical attachment/separation times.",
        },
        "limits": [
            "Cached UFM plus mask/depth shape fit; no gripper input. This is not a new tracker run.",
            "Held interval comes from existing annotations; actual contact timing is not established.",
            "Connected depth PCA under partial visibility is not the full object's ground-truth axis.",
            "Surface gap reuses the fitting observations and is not an independent accuracy metric.",
            "Gripper agreement assumes rigid attachment, synchronized recordings and accurate calibration; these are imperfect here.",
            "Low step jitter can coexist with a consistently wrong pose. Full rotation includes axial twist.",
            "Rendered hull is the projected fixed reference surface, including occluded parts, not a reconstructed complete mesh.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {"intervals": intervals, "fusion_updates": summary["fusion_visual_updates_during_hold"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
