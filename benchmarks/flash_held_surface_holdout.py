"""Check a held-object surface/attachment hypothesis on different demo frames.

This audits the historical sealed-demo importer, whose aligned RGB-D frames,
scene evidence and motor states are local files. It does not estimate a pose:
the candidate view uses the prepared demo's accepted pose, and each fixed
gripper/object relation uses the saved attachment at its chosen anchor frame.
Validation frames must exclude the candidate view. A low surface distance is
evidence of consistency with those frames, not ground-truth pose accuracy.
Optional local fitting reports both full rotation and the direction change of
the first view's longest principal axis; that axis is meaningful only when its
first singular value is well separated from the others.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.held_pose_diagnostics import _arm_pose_at
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.surface_grounding import PreparedReferenceSurface, ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import move, observation
from lerobot.policies.flash.surface_reference import (
    PreparedSurfaceAgreement,
    RegisteredSurfaceModel,
    registered_surface_agreement,
)


def _mask(scene_evidence: Path, index: int, role: str) -> np.ndarray:
    with gzip.open(scene_evidence / f"{index:06d}.json.gz", "rt", encoding="utf-8") as file:
        row = json.load(file)
    height, width = row["image_size"]
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in row["masks_rle"][role]:
        mask[start : start + length] = 1
    return mask.reshape((height, width)).astype(bool)


def _observation(
    dataset_root: Path, scene_evidence: Path, index: int, role: str, intrinsics: np.ndarray, config: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    path = dataset_root / "recording" / "depth" / f"{index:06d}.png"
    raw_depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw_depth is None:
        raise FileNotFoundError(path)
    depth = raw_depth.astype(np.float32) * 0.001
    mask = _mask(scene_evidence, index, role)
    return depth, mask, observation(depth, mask, intrinsics, config)


def _score(
    model: RegisteredSurfaceModel,
    prepared: PreparedSurfaceAgreement,
    camera_from_object: np.ndarray,
    measured: dict,
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> dict:
    first, combined = registered_surface_agreement(
        model,
        camera_from_object,
        measured["points"],
        measured["valid_image"],
        depth,
        intrinsics,
        prepared=prepared,
    )

    def values(agreement) -> dict:
        return {
            "measured_to_reference_p90_mm": agreement.measured_to_reference_p90_m * 1000,
            "reference_to_depth_p90_mm": (
                None
                if agreement.reference_to_depth_p90_m is None
                else agreement.reference_to_depth_p90_m * 1000
            ),
            "reference_depth_pixels": agreement.reference_depth_pixels,
        }

    return {"first_view": values(first), "two_views": values(combined)}


def run(
    dataset_root: Path,
    bundle_path: Path,
    role: str,
    *,
    baseline_anchor_frame: int,
    candidate_frame: int,
    validation_frames: range,
    fit: bool = False,
) -> dict:
    bundle_dir = bundle_path.parent
    report = json.loads((bundle_dir / "preparation-report.json").read_text(encoding="utf-8"))
    annotation = json.loads((bundle_dir / "annotation.json").read_text(encoding="utf-8"))
    calibration_path = Path(annotation["calibration_path"])
    if not calibration_path.is_absolute():
        calibration_path = dataset_root / calibration_path
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    camera_from_base = np.linalg.inv(np.asarray(calibration["camera"]["T_base_cam"], dtype=float))
    base_from_camera = np.linalg.inv(camera_from_base)
    intrinsics = np.loadtxt(dataset_root / "recording" / "cam_K.txt").reshape((3, 3))
    scene_evidence = bundle_dir / "scene-evidence"
    observation_config = report["vision_pipeline"]["observation"]
    skill = PreparedFlashSkill.load(bundle_path)
    model = next((item for item in skill.surface_models if item.role == role), None)
    if model is None:
        raise ValueError(f"{role}: no prepared measured surface")
    if len(model.view_points_object_m) != 1:
        raise ValueError("this audit needs the current single-view prepared model")
    with np.load(dataset_root / report["pose_trace_cache"]) as data:
        accepted_pose = data[f"{role}_poses"][candidate_frame].copy()
        evidence = str(data[f"{role}_evidence"][candidate_frame])
    if evidence != "observed" or not np.isfinite(accepted_pose).all():
        raise ValueError(f"{role}: candidate frame {candidate_frame} has no accepted direct pose")
    with np.load(bundle_dir / report["held_attachment_trace"]) as data:
        if str(data["role"]) != role:
            raise ValueError(f"{role}: held attachment trace belongs to {data['role']}")
        camera_times = data["camera_times_s"].copy()
        attachments = data["gripper_from_object"].copy()
    with np.load(dataset_root / "showservo_demo.npz") as data:
        arm_times = data["t"].copy()
        arm_poses = data["tips"].copy()
    if candidate_frame in validation_frames:
        raise ValueError("validation frames must exclude the candidate view")
    if not (0 <= baseline_anchor_frame < len(camera_times) and 0 <= candidate_frame < len(camera_times)):
        raise ValueError("anchor or candidate frame is outside the recording")

    _, _, candidate_observed = _observation(
        dataset_root, scene_evidence, candidate_frame, role, intrinsics, observation_config
    )
    candidate_points = candidate_observed["points"]
    if len(candidate_points) < 3:
        raise ValueError(f"{role}: candidate frame {candidate_frame} has insufficient clean depth")
    candidate_object_from_camera = np.linalg.inv(accepted_pose) @ base_from_camera
    second_view = move(candidate_points, candidate_object_from_camera)
    combined = RegisteredSurfaceModel(
        role, model.reference_version, (model.view_points_object_m[0], second_view)
    )
    prepared = PreparedSurfaceAgreement.build(combined)
    reference_points = model.view_points_object_m[0]
    _, singular_values, basis = np.linalg.svd(
        reference_points - reference_points.mean(axis=0), full_matrices=False
    )
    dominant_axis_object = basis[0]
    shape_config = ShapeConfig(**report["vision_pipeline"]["shape"]) if fit else None
    fit_surfaces = (
        (
            (
                "first_view",
                model.view_points_object_m[0],
                PreparedReferenceSurface.build(model.view_points_object_m[0], shape_config),
            ),
            (
                "two_views",
                prepared.all_points,
                PreparedReferenceSurface.build(prepared.all_points, shape_config),
            ),
        )
        if shape_config is not None
        else ()
    )
    baseline_attachment = attachments[baseline_anchor_frame]
    candidate_attachment = attachments[candidate_frame]
    rows = []
    for index in validation_frames:
        if not 0 <= index < len(camera_times):
            raise ValueError(f"validation frame {index} is outside the recording")
        depth, mask, measured = _observation(
            dataset_root, scene_evidence, index, role, intrinsics, observation_config
        )
        if len(measured["points"]) < 3:
            rows.append({"frame": index, "measured_points": len(measured["points"]), "score": None})
            continue
        ee = _arm_pose_at(arm_times, arm_poses, float(camera_times[index]))
        if ee is None:
            raise ValueError(f"validation frame {index} has no timestamp-aligned motor pose")
        baseline_pose = camera_from_base @ ee @ baseline_attachment
        candidate_pose = camera_from_base @ ee @ candidate_attachment
        baseline = _score(combined, prepared, baseline_pose, measured, depth, intrinsics)["first_view"]
        current = _score(combined, prepared, candidate_pose, measured, depth, intrinsics)
        row = {
            "frame": index,
            "measured_points": len(measured["points"]),
            "score": {
                "baseline_attachment_first_view": baseline,
                "candidate_attachment_first_view": current["first_view"],
                "candidate_attachment_two_views": current["two_views"],
            },
        }
        if shape_config is not None:
            observed = {"obs": measured, "depth": depth, "mask": mask}
            center = reference_points.mean(axis=0)
            fitted = {}
            for name, points, prepared_reference in fit_surfaces:
                fitter = WholeSurfaceFit(
                    points,
                    observed,
                    intrinsics,
                    shape_config,
                    prepared_reference=prepared_reference,
                )
                pose, detail = fitter.refine(candidate_pose)
                before_center = move(center[None], candidate_pose)[0]
                after_center = move(center[None], pose)[0]
                displacement = np.linalg.norm(
                    move(reference_points, pose) - move(reference_points, candidate_pose), axis=1
                )
                before_axis = candidate_pose[:3, :3] @ dominant_axis_object
                after_axis = pose[:3, :3] @ dominant_axis_object
                fitted[name] = {
                    "initial_cost": fitter.score(candidate_pose),
                    "fitted_cost": fitter.score(pose),
                    "center_shift_mm": float(np.linalg.norm(after_center - before_center) * 1000),
                    "reference_point_displacement_p90_mm": float(np.percentile(displacement, 90) * 1000),
                    "rotation_deg": float(
                        np.degrees(Rotation.from_matrix(pose[:3, :3] @ candidate_pose[:3, :3].T).magnitude())
                    ),
                    "dominant_axis_tilt_deg": float(
                        np.degrees(np.arccos(np.clip(abs(np.dot(before_axis, after_axis)), 0, 1)))
                    ),
                    "solver": detail,
                }
            row["fit_from_candidate_attachment"] = fitted
        rows.append(row)
    return {
        "dataset_root": str(dataset_root),
        "bundle": str(bundle_path),
        "role": role,
        "baseline_anchor_frame": baseline_anchor_frame,
        "candidate_frame": candidate_frame,
        "reference_points": len(model.view_points_object_m[0]),
        "candidate_points": len(candidate_points),
        "first_view_singular_values": singular_values.tolist(),
        "fit_from_candidate_attachment": fit,
        "validation": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("bundle", type=Path, help="prepared bundle.json under this dataset")
    parser.add_argument("role")
    parser.add_argument("--baseline-anchor-frame", type=int, required=True)
    parser.add_argument("--candidate-frame", type=int, required=True)
    parser.add_argument("--validation-first", type=int, required=True)
    parser.add_argument("--validation-last", type=int, required=True, help="inclusive")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--fit", action="store_true", help="also test how the local shape fitter moves the pose"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step < 1 or args.validation_first > args.validation_last:
        parser.error("validation range must be increasing with positive step")
    result = run(
        args.dataset_root,
        args.bundle,
        args.role,
        baseline_anchor_frame=args.baseline_anchor_frame,
        candidate_frame=args.candidate_frame,
        validation_frames=range(args.validation_first, args.validation_last + 1, args.step),
        fit=args.fit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
