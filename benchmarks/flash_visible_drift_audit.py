"""Separate raw point deformation from visible-rigid pose correction on a saved clip.

No tracker inference or robot motion. Masks and RGB-D must be frame aligned.
Pairwise distances test rigidity without assuming a known object pose. Pose
step sizes measure discontinuity, not ground-truth pose error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.registration import register_object, transform_points
from lerobot.policies.flash.visible_points import surface_support
from lerobot.policies.flash.visible_rigid_tracking import (
    VisibleRGBDFrame,
    VisibleRigidTracker,
    _surface_evidence,
    fit_visible_trace,
    visible_surface_cloud,
)


def distribution(values: np.ndarray) -> dict | None:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return dict(zip(("median", "p90", "max"), np.quantile(values, [0.5, 0.9, 1]).tolist(), strict=True))


def pose_steps(poses: np.ndarray, lost: np.ndarray, probes: np.ndarray) -> dict:
    valid = ~lost[:-1] & ~lost[1:]
    if not valid.any():
        return {"rotation_step_deg": None, "median_cloud_step_mm": None}
    first, second = poses[:-1][valid], poses[1:][valid]
    angles = np.degrees(
        Rotation.from_matrix(first[:, :3, :3].transpose(0, 2, 1) @ second[:, :3, :3]).magnitude()
    )
    moved = np.asarray([transform_points(pose, probes) for pose in poses])
    steps = np.median(np.linalg.norm(np.diff(moved, axis=0), axis=2), axis=1)[valid] * 1000
    return {"rotation_step_deg": distribution(angles), "median_cloud_step_mm": distribution(steps)}


def frozen_consensus(
    points: np.ndarray, support: np.ndarray, *, tolerance: float, min_inliers: int, trials: int, seeds: int
) -> dict:
    """Hold measurements/map fixed; vary only RANSAC's seed at the final frame."""
    tracker = VisibleRigidTracker(
        points.shape[1], tolerance_m=tolerance, min_inliers=min_inliers, trials=trials
    )
    for xyz, supported in zip(points, support, strict=True):
        tracker.update(xyz, supported)
    known = support[-1] & np.isfinite(tracker.canonical).all(axis=1)
    poses, counts = [], []
    for seed in range(1, seeds + 1):
        result = register_object(
            tracker.canonical[known],
            points[-1, known],
            threshold_m=tolerance,
            min_inliers=min_inliers,
            trials=trials,
            seed=seed,
        )
        poses.append(result.base_from_demo_object)
        counts.append(len(result.inlier_indices))
    rotations = np.asarray(poses)[:, :3, :3]
    angles = [np.degrees(Rotation.from_matrix(a.T @ b).magnitude()) for a in rotations for b in rotations]
    return {
        "seeds": seeds,
        "inlier_count": distribution(counts),
        "max_orientation_disagreement_deg": max(angles),
        "scope": "Point bank only; identical measurements, no surface correction or new motion.",
    }


def run(args: argparse.Namespace) -> dict:
    with np.load(args.scene, allow_pickle=False) as saved:
        scene = {key: saved[key] for key in saved.files}
    metadata = json.loads(str(scene["metadata"]))
    selected = (scene["frames"] >= args.first_frame) & (scene["frames"] <= args.last_frame)
    role_ids = scene["query_roles"] == args.role
    frames = scene["frames"][selected]
    if len(frames) < 2 or not role_ids.any():
        raise ValueError("need at least two saved frames and a known role")
    points = scene["query_coords_m"][selected][:, role_ids].astype(float)
    visibility = scene["query_visibility_logits"][selected][:, role_ids]
    births = scene["query_points_2d"][role_ids, 0].astype(int) + int(scene["frames"][0])
    intrinsics = np.loadtxt(args.recording / "cam_K.txt").reshape(3, 3)
    depth_tolerance = metadata["support_tolerance_m"]
    fit_tolerance = metadata["fit_threshold_m"]
    support = np.zeros(points.shape[:2], dtype=bool)
    lifted = points.copy()
    depth_errors = []
    rgbd = []
    surface_settings = metadata.get("surface_refinement")
    for index, frame in enumerate(frames):
        depth = cv2.imread(str(args.recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(args.mask_dir / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if depth is None or mask is None or depth.shape != mask.shape:
            raise ValueError(f"missing aligned mask/depth at frame {frame}")
        depth = depth.astype(np.float32) * 0.001
        mask = mask > 0
        # Unborn tracks may contain zero coordinates. Do not project them.
        valid = np.isfinite(points[index]).all(axis=1) & (points[index, :, 2] > 0)
        safe = np.where(valid[:, None], points[index], [0, 0, 1])
        visible = np.where(valid & (births <= frame), visibility[index], -np.inf)
        supported, x, y = surface_support(safe, visible, mask, depth, intrinsics, depth_tolerance)
        support[index] = supported
        measured_z = depth[y[supported], x[supported]]
        depth_errors.extend((safe[supported, 2] - measured_z) * 1000)
        lifted[index, supported] = safe[supported] * (measured_z / safe[supported, 2])[:, None]
        if args.compare_surface:
            if surface_settings is None:
                raise ValueError("saved scene has no surface-refinement settings")
            rgbd.append(
                VisibleRGBDFrame(
                    visible_surface_cloud(
                        depth,
                        mask,
                        intrinsics,
                        edge_pixels=surface_settings["edge_pixels"],
                        max_points=surface_settings["max_points"],
                    ),
                    mask,
                    depth,
                    intrinsics,
                )
            )
    report = {
        "scene": str(args.scene),
        "recording": str(args.recording),
        "mask_dir": str(args.mask_dir),
        "role": args.role,
        "frames": [int(frames[0]), int(frames[-1])],
        "fit_threshold_m": fit_tolerance,
        "support_tolerance_m": depth_tolerance,
        "birth_frames_and_counts": dict(
            zip(*[v.tolist() for v in np.unique(births, return_counts=True)], strict=True)
        ),
        "supported_count": distribution(support.sum(axis=1)),
        "accepted_depth_error_mm": distribution(np.abs(depth_errors)),
    }
    for name, ids in (
        ("endpoint_shared", support[0] & support[-1]),
        ("continuously_supported", support.all(axis=0)),
    ):
        report[name] = {"count": int(ids.sum())}
        for label, xyz in (("raw", points), ("measured_depth", lifted)):
            report[name][label + "_pair_distance_change_mm"] = distribution(
                np.abs(pdist(xyz[-1, ids]) - pdist(xyz[0, ids])) * 1000
            )
    if args.consensus_frame is not None:
        index = int(np.flatnonzero(frames == args.consensus_frame)[0])
        report["frozen_consensus"] = frozen_consensus(
            points[: index + 1],
            support[: index + 1],
            tolerance=fit_tolerance,
            min_inliers=args.min_inliers,
            trials=args.trials,
            seeds=args.seeds,
        )
        report["frozen_consensus"]["frame"] = args.consensus_frame
    if args.compare_surface:
        # Compare only the original reference span: slicing it would change the
        # canonical point frame and confound the correction-policy comparison.
        if not selected.all():
            raise ValueError("surface comparison requires the whole saved scene")
        source_index = next(i for i, supported in enumerate(support) if supported.sum() >= args.min_inliers)
        trace = fit_visible_trace(
            points,
            support,
            tolerance_m=fit_tolerance,
            min_inliers=args.min_inliers,
            trials=args.trials,
            reference_surface=rgbd[source_index].surface_m,
            rgbd_frames=rgbd,
            depth_m_per_unit=1.0,
            surface_tolerance_m=depth_tolerance,
            icp_iterations=surface_settings["icp_iterations"],
            trim_fraction=surface_settings["trim_fraction"],
        )
        role_index = list(scene["roles"]).index(args.role)
        probes = points[source_index, support[source_index]]
        report["saved_pose"] = pose_steps(scene["poses"][role_index], scene["lost"][role_index], probes)
        report["current_pose"] = pose_steps(trace["poses"], trace["lost"], probes)
        for label, poses, lost in (
            ("saved_pose", scene["poses"][role_index], scene["lost"][role_index]),
            ("current_pose", trace["poses"], trace["lost"]),
        ):
            evidence = [
                _surface_evidence(
                    rgbd[source_index].surface_m,
                    pose,
                    frame.mask,
                    frame.depth,
                    frame.intrinsics,
                    depth_m_per_unit=1.0,
                    tolerance_m=depth_tolerance,
                )
                for pose, frame, is_lost in zip(poses, rgbd, lost, strict=True)
                if not is_lost
            ]
            report[label]["surface_agreement_percent"] = distribution(
                np.array([e.surface_supported / e.projected_points * 100 for e in evidence])
            )
            report[label]["foreground_conflict_percent"] = distribution(
                np.array([e.in_front_of_depth / e.projected_points * 100 for e in evidence])
            )
        report["saved_pose"]["surface_corrections"] = int(scene["surface_refined"][role_index].sum())
        report["current_pose"]["surface_corrections"] = int(trace["surface_refined"].sum())
        report["current_pose"]["lost_frames"] = int(trace["lost"].sum())
        np.savez_compressed(args.output.with_suffix(".npz"), frames=frames, **trace)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("scene", "recording", "mask-dir", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--first-frame", type=int, required=True)
    parser.add_argument("--last-frame", type=int, required=True)
    parser.add_argument("--compare-surface", action="store_true")
    parser.add_argument("--min-inliers", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--consensus-frame", type=int)
    parser.add_argument(
        "--seeds", type=int, default=75, help="Repeatability audit trials, not a tracker setting"
    )
    print(json.dumps(run(parser.parse_args()), indent=2))
