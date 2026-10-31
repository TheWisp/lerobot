"""Read-only reference-view review for recordings made by the sealed rig importer."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import QhullError

from .geometry import inverse
from .held_pose_diagnostics import _arm_pose_at
from .reference_view_candidates import MeasuredReferenceView, review_reference_view_candidates
from .reference_view_registration import fit_reference_view_candidates
from .surface_grounding import project
from .surface_observation import move, observation, spatial_sample
from .surface_reference import PreparedSurfaceAgreement, RegisteredSurfaceModel, registered_surface_agreement
from .trajectory import Evidence, PoseEstimate
from .vision_pipeline import VisionPipelineConfig


def _gripper_motion_prior(
    root: Path,
    bundle_path: Path,
    annotation: dict,
    report: dict,
    role: str,
    frame_index: int,
    grasp_reference_pose: np.ndarray | None,
    base_from_camera: np.ndarray,
) -> np.ndarray | None:
    """Predict a carried view from one grasp seed and recorded motor motion.

    This deliberately ignores later visual attachment corrections, which
    would make the selected view part of its own registration prior.
    """
    if role != annotation.get("held_role") or grasp_reference_pose is None:
        return None
    trace_name = report.get("held_attachment_trace")
    if not isinstance(trace_name, str):
        return None
    trace_path = (bundle_path.parent / trace_name).resolve()
    if not trace_path.is_relative_to(bundle_path.parent.resolve()) or not trace_path.is_file():
        return None
    arm_path = root / "showservo_demo.npz"
    if not arm_path.is_file():
        return None
    with np.load(trace_path, allow_pickle=False) as trace:
        times = trace["camera_times_s"]
        if frame_index >= len(times):
            return None
        capture_time_s = float(times[frame_index])
    grasp_time_s = float(annotation["grasp_time_s"])
    release_time_s = float(annotation["release_time_s"])
    if not grasp_time_s <= capture_time_s < release_time_s:
        return None
    with np.load(arm_path, allow_pickle=False) as arm:
        motor_times = arm["t"]
        motor_poses = arm["tips"]
        grasp_index = int(np.argmin(np.abs(motor_times - grasp_time_s)))
        capture_pose = _arm_pose_at(motor_times, motor_poses, capture_time_s)
        if capture_pose is None:
            return None
        return (
            inverse(base_from_camera)
            @ capture_pose
            @ inverse(motor_poses[grasp_index])
            @ grasp_reference_pose
        )


def review_sealed_reference_view(
    root: Path, bundle_path: Path, episode: int, role: str, frame_index: int
) -> dict:
    """Compare a selected view without changing the saved demo or active reference."""
    from .prepared import PreparedFlashSkill

    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("choose a nonnegative demo camera frame")
    prepared = PreparedFlashSkill.load(bundle_path)
    if prepared.skill.source_episode != episode:
        raise ValueError("selected view belongs to a different demo episode")
    reference = next((item for item in prepared.references if item.role == role), None)
    if reference is None:
        raise ValueError("selected object is not in this prepared demo")
    annotation = json.loads((bundle_path.parent / "annotation.json").read_text(encoding="utf-8"))
    report = json.loads((bundle_path.parent / "preparation-report.json").read_text(encoding="utf-8"))
    if annotation["episode"] != prepared.skill.source_episode:
        raise ValueError("prepared annotation belongs to another demo episode")
    cache_path = (root / report["pose_trace_cache"]).resolve()
    if not cache_path.is_relative_to((root / "flash" / "v1" / "preparation-cache").resolve()):
        raise ValueError("prepared pose trace is outside this dataset")
    evidence_root = bundle_path.parent / "scene-evidence"
    if not cache_path.is_file() or not evidence_root.is_dir():
        raise ValueError("this prepared version lacks the pose or mask evidence needed for review")
    intrinsics = np.loadtxt(root / "recording" / "cam_K.txt").reshape(3, 3)
    calibration = json.loads(Path(annotation["calibration_path"]).expanduser().read_text(encoding="utf-8"))
    base_from_camera = np.asarray(calibration["camera"]["T_base_cam"], dtype=float)
    config = VisionPipelineConfig.from_settings(annotation, backend="appearance")
    selected_mask = None

    def measured(name: str, index: int) -> MeasuredReferenceView | None:
        nonlocal selected_mask
        path = evidence_root / f"{index:06d}.json.gz"
        if not path.is_file():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            evidence = json.load(stream)
        if evidence["frame_index"] != index:
            raise ValueError("prepared mask evidence has a different source frame")
        runs = evidence["masks_rle"].get(name)
        if runs is None:
            return None
        depth = cv2.imread(str(root / "recording" / "depth" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.shape != tuple(evidence["image_size"]):
            raise ValueError(f"frame {index} has no aligned recorded depth")
        mask = np.zeros(depth.size, dtype=bool)
        for start, length in runs:
            if (
                type(start) is not int
                or type(length) is not int
                or start < 0
                or length < 0
                or start + length > mask.size
            ):
                raise ValueError(f"frame {index} has invalid object mask runs")
            mask[start : start + length] = True
        mask = mask.reshape(depth.shape)
        if name == role and index == frame_index:
            selected_mask = mask
        depth_m = depth.astype(np.float32) * 0.001
        observed = observation(depth_m, mask, intrinsics, asdict(config.observation))
        return MeasuredReferenceView(
            index, depth_m, observed["valid_image"], observed["points"], int(mask.sum())
        )

    snippet = reference.snippets[0]
    first_mask_path = (root / snippet.mask_artifact).resolve()
    if not first_mask_path.is_relative_to((root / "flash" / "v1" / "masks").resolve()):
        raise ValueError("prepared reference mask is outside this dataset")
    first_mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
    first_depth = cv2.imread(
        str(root / "recording" / "depth" / f"{snippet.frame_index:06d}.png"), cv2.IMREAD_UNCHANGED
    )
    if first_mask is None or first_depth is None or first_mask.shape != first_depth.shape:
        raise ValueError("prepared reference has no aligned mask and depth")
    model = next((item for item in prepared.surface_models if item.role == role), None)
    if model is None:
        if len(reference.snippets) != 1:
            raise ValueError("prepared multi-view reference needs its saved surface model")
        # Older prepared bundles predate saved surface models. Rebuild only the
        # first view from their immutable reference mask and recorded depth.
        depth_m = first_depth.astype(np.float32) * 0.001
        observed = observation(depth_m, first_mask.astype(bool), intrinsics, asdict(config.observation))
        if len(observed["points"]) < 3:
            raise ValueError("prepared first view has insufficient reliable depth")
        points = move(observed["points"], snippet.object_from_camera)
        limit = config.surface_reference_max_points_per_view
        model = RegisteredSurfaceModel(role, reference.version, (points[spatial_sample(points, limit)],))
    existing_view_index = next(
        (
            index
            for index, item in enumerate(reference.snippets[1:], start=1)
            if item.frame_index == frame_index
        ),
        None,
    )
    if existing_view_index is not None:
        # Rechecking an already registered frame must compare against the
        # remaining views, rather than scoring the candidate against itself.
        model = RegisteredSurfaceModel(
            role,
            reference.version,
            tuple(
                points
                for index, points in enumerate(model.view_points_object_m)
                if index != existing_view_index
            ),
        )

    with np.load(cache_path, allow_pickle=False) as cache:
        count = int(cache["frames"])
        if frame_index >= count:
            raise ValueError("selected frame is outside this prepared pose trace")
        estimates = {}
        for index in range(count):
            evidence = Evidence(str(cache[f"{role}_evidence"][index]))
            estimates[index] = {
                role: PoseEstimate(
                    role,
                    cache[f"{role}_poses"][index] if evidence != Evidence.LOST else None,
                    float(index),
                    float(index),
                    evidence,
                    reference.version,
                    tuple(bool(x) for x in cache[f"{role}_translation_supported"][index]),
                    tuple(bool(x) for x in cache[f"{role}_rotation_supported"][index]),
                    estimator=str(cache[f"{role}_estimator"][index]),
                    source_frame_id=index,
                )
            }
    summaries, details = review_reference_view_candidates(
        estimates,
        (model,),
        {role: [frame_index]},
        measured,
        intrinsics,
        base_from_camera,
        config.surface_reference_max_points_per_view,
    )
    comparisons = details[0]["comparisons"]
    depth_rows = [
        row
        for row in comparisons
        if row["saved_views"]["reference_to_depth_p90_m"] is not None
        and row["with_candidate"]["reference_to_depth_p90_m"] is not None
    ]
    front_worsened = sorted(
        (
            row
            for row in depth_rows
            if row["with_candidate"]["reference_in_front_p90_m"]
            > row["saved_views"]["reference_in_front_p90_m"]
        ),
        key=lambda row: row["with_candidate"]["reference_in_front_p90_m"]
        - row["saved_views"]["reference_in_front_p90_m"],
        reverse=True,
    )
    behind_worsened = sorted(
        (
            row
            for row in depth_rows
            if row["with_candidate"]["reference_behind_p90_m"] > row["saved_views"]["reference_behind_p90_m"]
        ),
        key=lambda row: row["with_candidate"]["reference_behind_p90_m"]
        - row["saved_views"]["reference_behind_p90_m"],
        reverse=True,
    )
    independent_fits = []
    gripper_motion_fits = []
    independent_fit_error = None
    candidate_view = measured(role, frame_index)
    if candidate_view is not None and selected_mask is not None and len(candidate_view.points_camera_m) >= 3:
        try:
            grasp_frame = annotation.get("grasp_reference_frame")
            grasp_estimate = estimates.get(grasp_frame, {}).get(role)
            grasp_pose = (
                grasp_estimate.pose
                if grasp_estimate is not None and grasp_estimate.evidence == Evidence.OBSERVED
                else None
            )
            gripper_prior = _gripper_motion_prior(
                root, bundle_path, annotation, report, role, frame_index, grasp_pose, base_from_camera
            )
            saved_points = np.concatenate(model.view_points_object_m)
            fits = fit_reference_view_candidates(
                saved_points,
                candidate_view,
                selected_mask,
                intrinsics,
                config.shape,
                prior_poses_camera_from_object={"gripper_motion": gripper_prior}
                if gripper_prior is not None
                else None,
            )

            def preview(pose: np.ndarray) -> list[list[int]]:
                camera_points = move(saved_points, pose)
                camera_points = camera_points[camera_points[:, 2] > 0]
                xy = project(camera_points, intrinsics)
                inside = (
                    (xy[:, 0] >= 0)
                    & (xy[:, 0] < candidate_view.depth_m.shape[1])
                    & (xy[:, 1] >= 0)
                    & (xy[:, 1] < candidate_view.depth_m.shape[0])
                )
                return np.rint(xy[inside]).astype(int).tolist()

            for fit in fits:
                result = {
                    "seed": fit.seed,
                    "converged": fit.converged,
                    "fit_cost": fit.fit_cost,
                    "measured_to_reference_p90_m": fit.measured_to_reference_p90_m,
                    "reference_depth_pixels": fit.reference_depth_pixels,
                    "reference_to_depth_p90_m": fit.reference_to_depth_p90_m,
                    "reference_in_front_p90_m": fit.reference_in_front_p90_m,
                    "reference_behind_p90_m": fit.reference_behind_p90_m,
                    "surface_center_camera_m": fit.surface_center_camera_m,
                    "object_from_camera": inverse(fit.camera_from_object).tolist(),
                    "projected_points_xy": preview(fit.camera_from_object),
                }
                (gripper_motion_fits if fit.seed == "gripper_motion" else independent_fits).append(result)
            if gripper_prior is not None:
                prepared_agreement = PreparedSurfaceAgreement.build(model)
                score, _ = registered_surface_agreement(
                    model,
                    gripper_prior,
                    candidate_view.points_camera_m,
                    candidate_view.clean_depth_pixels,
                    candidate_view.depth_m,
                    intrinsics,
                    prepared=prepared_agreement,
                )
                gripper_motion_fits.insert(
                    0,
                    {
                        "seed": "gripper_prediction",
                        "converged": None,
                        "fit_cost": None,
                        "measured_to_reference_p90_m": score.measured_to_reference_p90_m,
                        "reference_depth_pixels": score.reference_depth_pixels,
                        "reference_to_depth_p90_m": score.reference_to_depth_p90_m,
                        "reference_in_front_p90_m": score.reference_in_front_p90_m,
                        "reference_behind_p90_m": score.reference_behind_p90_m,
                        "surface_center_camera_m": tuple(
                            float(x) for x in move(saved_points, gripper_prior).mean(axis=0)
                        ),
                        "object_from_camera": inverse(gripper_prior).tolist(),
                        "projected_points_xy": preview(gripper_prior),
                    },
                )
        except (ValueError, RuntimeError, QhullError) as exc:
            independent_fit_error = str(exc)
    return {
        "reference_frame_index": snippet.frame_index,
        "first_view_points": len(model.view_points_object_m[0]),
        "saved_view_count": len(model.view_points_object_m),
        "replacing_saved_view": existing_view_index is not None,
        "candidate": summaries[0],
        "largest_front_conflicts": front_worsened[:5],
        "largest_behind_gaps": behind_worsened[:5],
        "independent_fits": independent_fits,
        "gripper_motion_fits": gripper_motion_fits,
        "independent_fit_error": independent_fit_error,
        "source_bundle": str(bundle_path.resolve().relative_to(root.resolve())),
        "source_reference_version": reference.version,
        "base_reference_frame_index": snippet.frame_index,
        "base_calibration_version": snippet.calibration_version,
        "mask_sha256": (
            hashlib.sha256(selected_mask.astype(np.uint8).tobytes()).hexdigest()
            if selected_mask is not None
            else None
        ),
        "disclaimer": "Same pose trace registered and checked this view; review cannot establish independent accuracy.",
    }
