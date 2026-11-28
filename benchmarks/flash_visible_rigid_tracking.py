"""Measure visible rigid tracking on a recorded RGB-D interval.

This reads an already computed TrackEverything trace. It fits supported point
IDs causally, then measures a local RGB-D surface refinement separately. The
refinement is a proposal, not an automatically accepted FLASH pose. Run one
object and visible interval at a time; this does not assess occlusion recovery.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import cv2
import numpy as np
from flash_rgbd_registration import cloud, projection_evidence

from lerobot.policies.flash.registration import DegenerateRegistrationError, transform_points
from lerobot.policies.flash.visible_points import read_mask, surface_support
from lerobot.policies.flash.visible_rigid_tracking import VisibleRigidTracker, refine_visible_surface


def evaluate(
    scene_path: Path,
    recording: Path,
    mask_dir: Path,
    *,
    role: str,
    start: int,
    end: int,
    track_tolerance_m: float,
    surface_tolerance_m: float,
    min_inliers: int,
    fit_trials: int,
    edge_pixels: int,
    icp_iterations: int,
    trim_fraction: float,
) -> dict:
    with np.load(scene_path, allow_pickle=False) as saved:
        frames = np.asarray(saved["frames"])
        coords = np.asarray(saved["query_coords_m"])
        visibility = np.asarray(saved["query_visibility_logits"])
        roles = np.asarray(saved["query_roles"])
        births = np.asarray(saved["query_points_2d"][:, 0])
    if start >= end or start not in frames or end not in frames or not np.all(np.diff(frames) == 1):
        raise ValueError("select an ordered, contiguous interval inside the saved trace")
    columns = np.flatnonzero(roles == role)
    if len(columns) < min_inliers:
        raise ValueError(f"role {role!r} has too few tracked points")
    k = np.loadtxt(recording / "cam_K.txt").reshape(3, 3)
    first = int(frames[0])
    tracker = VisibleRigidTracker(
        len(columns), tolerance_m=track_tolerance_m, min_inliers=min_inliers, trials=fit_trials
    )
    hybrid = VisibleRigidTracker(
        len(columns), tolerance_m=track_tolerance_m, min_inliers=min_inliers, trials=fit_trials
    )
    reference_surface = None
    reference_points = None
    rows = []
    for frame in range(start, end + 1):
        index = frame - first
        depth_mm = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise ValueError(f"missing aligned depth frame {frame}")
        mask = read_mask(mask_dir, frame, depth_mm.shape)
        if mask is None:
            missing = tracker.update(
                points=coords[index, columns], supported=np.zeros(len(columns), dtype=bool)
            )
            hybrid.update(points=coords[index, columns], supported=np.zeros(len(columns), dtype=bool))
            rows.append(
                {
                    "frame": frame,
                    "supported": 0,
                    "matched": missing.matched_points,
                    "inliers": 0,
                    "new_points": 0,
                    "inlier_median_mm": None,
                    "pose_available": False,
                    "reason": "instance mask unavailable",
                }
            )
            continue
        depth_m = depth_mm.astype(float) * 0.001
        points = coords[index, columns]
        supported, _, _ = surface_support(
            points,
            np.where(births[columns] <= index, visibility[index, columns], -np.inf),
            mask,
            depth_m,
            k,
            surface_tolerance_m,
        )
        measured = tracker.update(points, supported)
        row = {
            "frame": frame,
            "supported": measured.supported_points,
            "matched": measured.matched_points,
            "inliers": measured.rigid_inliers,
            "new_points": measured.new_points,
            "inlier_median_mm": None
            if measured.median_inlier_error_m is None
            else measured.median_inlier_error_m * 1000,
            "pose_available": measured.pose is not None,
        }
        if measured.pose is None:
            hybrid.update(points, supported)
            rows.append(row)
            continue
        if reference_surface is None:
            reference_surface = cloud(depth_mm, mask, k, edge_pixels)
            reference_points = points.copy()
            reference_supported = supported.copy()
        target_surface = cloud(depth_mm, mask, k, edge_pixels)
        hybrid_observation, decision = hybrid.update_with_surface(
            points,
            supported,
            source_surface=reference_surface,
            target_surface=target_surface,
            target_mask=mask,
            target_depth=depth_mm,
            intrinsics=k,
            depth_m_per_unit=0.001,
            surface_tolerance_m=surface_tolerance_m,
            iterations=icp_iterations,
            trim_fraction=trim_fraction,
        )
        row["surface_accepted"] = bool(decision is not None and decision.accepted)
        if decision is not None:
            row["surface_evidence"] = {
                "baseline_supported": decision.baseline.surface_supported,
                "baseline_front": decision.baseline.in_front_of_depth,
                "proposal_supported": None
                if decision.proposal is None
                else decision.proposal.surface_supported,
                "proposal_front": None if decision.proposal is None else decision.proposal.in_front_of_depth,
            }
        poses = {"point_fit": measured.pose}
        if hybrid_observation.pose is not None:
            poses["surface_selected"] = hybrid_observation.pose
        with contextlib.suppress(DegenerateRegistrationError):
            poses["surface_proposal"] = refine_visible_surface(
                reference_surface,
                target_surface,
                measured.pose,
                max_correspondence_m=surface_tolerance_m,
                iterations=icp_iterations,
                trim_fraction=trim_fraction,
            )
        shared = reference_supported & supported
        for name, pose in poses.items():
            evidence, _, _ = projection_evidence(
                transform_points(pose, reference_surface), mask, depth_mm, k, surface_tolerance_m
            )
            residual = (
                np.linalg.norm(transform_points(pose, reference_points[shared]) - points[shared], axis=1)
                * 1000
            )
            row[name] = {
                **evidence,
                "pose_camera_from_local": pose.tolist(),
                "shared_reference_ids": int(shared.sum()),
                "shared_track_median_mm": float(np.median(residual)) if len(residual) else None,
                "shared_track_p90_mm": float(np.quantile(residual, 0.9)) if len(residual) else None,
            }
        rows.append(row)
    return {
        "source": str(scene_path),
        "recording": str(recording),
        "mask_dir": str(mask_dir),
        "role": role,
        "interval": [start, end],
        "settings": {
            "track_tolerance_m": track_tolerance_m,
            "surface_tolerance_m": surface_tolerance_m,
            "min_inliers": min_inliers,
            "fit_trials": fit_trials,
            "edge_pixels": edge_pixels,
            "icp_iterations": icp_iterations,
            "trim_fraction": trim_fraction,
        },
        "frames_with_pose": sum(row["pose_available"] for row in rows),
        "frame_count": len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--track-tolerance-m", type=float, required=True)
    parser.add_argument("--surface-tolerance-m", type=float, required=True)
    parser.add_argument("--min-inliers", type=int, required=True)
    parser.add_argument("--fit-trials", type=int, required=True)
    parser.add_argument("--edge-pixels", type=int, default=2)
    parser.add_argument("--icp-iterations", type=int, default=20)
    parser.add_argument("--trim-fraction", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.scene,
        args.recording,
        args.mask_dir,
        role=args.role,
        start=args.start,
        end=args.end,
        track_tolerance_m=args.track_tolerance_m,
        surface_tolerance_m=args.surface_tolerance_m,
        min_inliers=args.min_inliers,
        fit_trials=args.fit_trials,
        edge_pixels=args.edge_pixels,
        icp_iterations=args.icp_iterations,
        trim_fraction=args.trim_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("role", "interval", "frames_with_pose", "frame_count")}))


if __name__ == "__main__":
    main()
