"""Check dowel pose hypotheses against an independently extracted ring opening.

The opening is estimated from the frame-100 SAM3 mask hole and a robust plane
fit to that frame's aligned ring depth. Its motion then comes from each ring
pose trace. These are task-geometry consistency checks, not hidden-tip labels
or permission to impose insertion contact on every frame.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def ring_hole_measurement(
    dataset: Path,
    masks: Path,
    frame: int,
    intrinsics: np.ndarray,
    plane_tolerance_m: float,
    trials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mask = cv2.imread(str(masks / "ring" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(dataset / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if mask is None or depth is None or mask.shape != depth.shape:
        raise ValueError(f"missing aligned ring mask/depth at frame {frame}")
    mask = mask > 0
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError(f"no ring contours at frame {frame}")
    holes = [
        index for index, contour in enumerate(contours) if hierarchy[0, index, 3] >= 0 and len(contour) >= 5
    ]
    if not holes:
        raise ValueError(f"no visible ring opening at frame {frame}")
    hole_contour = contours[max(holes, key=lambda index: cv2.contourArea(contours[index]))]
    hole_uv = np.asarray(cv2.fitEllipse(hole_contour)[0], dtype=np.float64)

    rows, columns = np.nonzero(mask & (depth > 0))
    metres = depth[rows, columns].astype(np.float64) * 0.001
    points = np.column_stack(
        (
            (columns - intrinsics[0, 2]) * metres / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * metres / intrinsics[1, 1],
            metres,
        )
    )
    if len(points) < 3:
        raise ValueError(f"no ring depth at frame {frame}")
    rng = np.random.default_rng(frame)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(trials):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        magnitude = np.linalg.norm(normal)
        if magnitude < np.finfo(float).eps:
            continue
        residuals = np.abs((points - a) @ (normal / magnitude))
        inliers = residuals <= plane_tolerance_m
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 3:
        raise ValueError(f"no supported ring plane at frame {frame}")
    plane_points = points[best]
    plane_center = plane_points.mean(axis=0)
    normal = np.linalg.eigh(np.cov(plane_points.T))[1][:, 0]
    if normal[2] < 0:
        normal = -normal
    ray = np.linalg.inv(intrinsics) @ np.array([*hole_uv, 1.0])
    hole_xyz = ray * (normal @ plane_center) / (normal @ ray)
    return hole_xyz, normal, hole_uv, float(best.mean())


def candidate_trace(specification: str) -> tuple[str, dict[int, np.ndarray]]:
    name, separator, filename = specification.partition("=")
    if not separator or not name or not filename:
        raise ValueError("candidate must be NAME=POSE_TRACE.npz")
    with np.load(filename) as trace:
        poses = {
            int(frame): pose.copy()
            for frame, pose, lost in zip(trace["frames"], trace["poses"], trace["lost"], strict=True)
            if not bool(lost)
        }
    return name, poses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=POSE_TRACE.npz")
    parser.add_argument("--ring-candidate", action="append", default=[], help="NAME=POSE_TRACE.npz")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--first-frame", type=int, default=400)
    parser.add_argument("--last-frame", type=int, default=500)
    parser.add_argument("--plane-tolerance-m", type=float, default=0.0025)
    parser.add_argument("--ransac-trials", type=int, default=1000)
    parser.add_argument("--hole-validation-frames", type=int, nargs="*", default=[])
    parser.add_argument("--hole-validation-output", type=Path)
    args = parser.parse_args()
    if args.first_frame > args.last_frame or args.plane_tolerance_m <= 0 or args.ransac_trials <= 0:
        parser.error("invalid frame interval or plane-fit parameters")
    if args.hole_validation_frames and args.hole_validation_output is None:
        parser.error("--hole-validation-output is required with --hole-validation-frames")

    with np.load(Path(__file__).with_name("pose_traces.npz")) as reference:
        intrinsics = reference["intrinsics"].copy()
        ring_poses = {
            "Point2Pose ring": (reference["p2p_ring_delta"].copy(), reference["p2p_ring_lost"].copy()),
            "BundleSDF ring": (
                reference["bundlesdf_ring_delta"].copy(),
                reference["bundlesdf_ring_failed"].copy(),
            ),
        }
        center = reference["dowel_reference_center_camera_m"].copy()
        axis = reference["dowel_reference_shaft_camera"].copy()
        tip = center + float(reference["dowel_reference_end_offsets_m"][1]) * axis
        first_reference_frame = int(reference["frames"][0])
        reference_frame_count = len(reference["frames"])
    for specification in args.ring_candidate:
        name, sampled_poses = candidate_trace(specification)
        if name in ring_poses:
            parser.error(f"duplicate ring candidate: {name}")
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], reference_frame_count, axis=0)
        lost = np.ones(reference_frame_count, dtype=bool)
        for frame, pose in sampled_poses.items():
            index = frame - first_reference_frame
            if 0 <= index < reference_frame_count:
                poses[index] = pose
                lost[index] = False
        ring_poses[name] = poses, lost
    opening, normal, _, support = ring_hole_measurement(
        args.dataset,
        args.masks,
        first_reference_frame,
        intrinsics,
        args.plane_tolerance_m,
        args.ransac_trials,
    )
    validation, _, _, _ = ring_hole_measurement(
        args.dataset, args.masks, 340, intrinsics, args.plane_tolerance_m, args.ransac_trials
    )
    print("first-view plane support", round(support, 3))
    for ring_name, (poses, lost) in ring_poses.items():
        if not lost[340 - first_reference_frame]:
            ring_pose = poses[340 - first_reference_frame]
            predicted = ring_pose[:3, :3] @ opening + ring_pose[:3, 3]
            print(
                ring_name, "frame-340 hole validation gap mm", np.linalg.norm(predicted - validation) * 1000
            )

    if args.hole_validation_output is not None:
        validation_rows = []
        for frame in args.hole_validation_frames:
            observed, _, _, plane_support = ring_hole_measurement(
                args.dataset, args.masks, frame, intrinsics, args.plane_tolerance_m, args.ransac_trials
            )
            index = frame - first_reference_frame
            for ring_name, (poses, lost) in ring_poses.items():
                if 0 <= index < len(poses) and not lost[index]:
                    pose = poses[index]
                    predicted = pose[:3, :3] @ opening + pose[:3, 3]
                    validation_rows.append(
                        (frame, ring_name, float(np.linalg.norm(predicted - observed) * 1000), plane_support)
                    )
        args.hole_validation_output.parent.mkdir(parents=True, exist_ok=True)
        with args.hole_validation_output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("frame", "ring_source", "observed_hole_center_gap_mm", "plane_support_fraction"))
            writer.writerows(validation_rows)

    candidates = dict(candidate_trace(specification) for specification in args.candidate)
    if len(candidates) != len(args.candidate):
        parser.error("candidate names must be unique")
    rows = []
    for frame in range(args.first_frame, args.last_frame + 1):
        index = frame - first_reference_frame
        for ring_name, (poses, lost) in ring_poses.items():
            if index >= len(poses) or lost[index]:
                continue
            ring_pose = poses[index]
            hole = ring_pose[:3, :3] @ opening + ring_pose[:3, 3]
            ring_normal = ring_pose[:3, :3] @ normal
            for dowel_name, trace in candidates.items():
                if frame not in trace:
                    continue
                dowel_pose = trace[frame]
                predicted_tip = dowel_pose[:3, :3] @ tip + dowel_pose[:3, 3]
                dowel_center = dowel_pose[:3, :3] @ center + dowel_pose[:3, 3]
                dowel_axis = dowel_pose[:3, :3] @ axis
                cosine = float(ring_normal @ dowel_axis)
                if abs(cosine) < np.finfo(float).eps:
                    continue
                intersection = dowel_center + (ring_normal @ (hole - dowel_center)) / cosine * dowel_axis
                difference = predicted_tip - hole
                tip_lateral = difference - (difference @ ring_normal) * ring_normal
                rows.append(
                    (
                        frame,
                        ring_name,
                        dowel_name,
                        float(np.linalg.norm(tip_lateral) * 1000),
                        float(difference @ ring_normal * 1000),
                        float(np.linalg.norm(intersection - hole) * 1000),
                        float(np.degrees(np.arccos(np.clip(abs(cosine), 0, 1)))),
                    )
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "ring_source",
                "dowel_source",
                "predicted_tip_lateral_mm",
                "predicted_tip_axial_mm",
                "shaft_at_hole_plane_lateral_mm",
                "unoriented_shaft_to_ring_normal_deg",
            )
        )
        writer.writerows(rows)
    for phase, minimum_frame in (
        ("insertion", args.first_frame),
        ("last_10", max(args.first_frame, args.last_frame - 9)),
    ):
        for ring_name in ring_poses:
            for dowel_name in candidates:
                paired = np.asarray(
                    [
                        row[3:]
                        for row in rows
                        if row[0] >= minimum_frame and row[1:3] == (ring_name, dowel_name)
                    ]
                )
                if len(paired):
                    print(phase, ring_name, dowel_name, len(paired), np.round(np.median(paired, axis=0), 2))


if __name__ == "__main__":
    main()
