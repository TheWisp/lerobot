"""Migrate a sealed RGB-D demonstration into a reviewable FLASH dataset extension.

This is an explicit adapter for the old ``showservo_demo.npz`` recording
layout. It does not infer task intent: a JSON annotation names the held and
target roles, grasp/release times, prompts, and registration tolerances. Both
the demonstration and a subsequent rollout use the same FLASH RGB-D matcher.
Held poses after grasp are deliberately labelled predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.appearance import (
    AppearancePixelMatcher,
    ReferenceAppearanceFrame,
    Sam3MaskSource,
    build_appearance_model,
)
from lerobot.policies.flash.dino_features import DinoPatchExtractor
from lerobot.policies.flash.geometry import inverse
from lerobot.policies.flash.matched_rgbd import MatchedRGBDPoseBackend
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.skill import (
    AttachmentAction,
    AttachmentTransition,
    FlashSkill,
    Interval,
    MotionMode,
    Relation,
    SpatialSample,
)
from lerobot.policies.flash.symmetry import AxialSymmetry, SymmetryCanonicalizingBackend
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate
from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics


def _frame(
    root: Path, index: int, intrinsics: np.ndarray, base_from_camera: np.ndarray, version: str
) -> RGBDFrame:
    rgb = cv2.imread(str(root / "recording" / "rgb" / f"{index:06d}.jpg"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / "recording" / "depth" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise FileNotFoundError(f"missing aligned RGB-D demo frame {index}")
    return RGBDFrame(
        index,
        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
        depth,
        intrinsics,
        0.001,  # sealed recording depth PNGs are millimetres
        base_from_camera,
        index / 30,
        index / 30,
        version,
    )


def _nearest_index(times: np.ndarray, time_s: float) -> int:
    return int(np.argmin(np.abs(times - time_s)))


def _interval_layout(
    times: np.ndarray,
    keypoints: list[dict],
    grasp: int,
    release: int,
    start_anchor_end: int,
) -> list[tuple[int, int, dict, bool, bool, bool]]:
    """Resolve annotated event times once onto the arm clock before assigning relations."""
    bounds = sorted(
        {
            0,
            grasp,
            start_anchor_end,
            release,
            *(_nearest_index(times, float(item["t"])) for item in keypoints),
        }
    )
    layout = []
    for first, last in zip(bounds, bounds[1:], strict=False):
        if last <= first:
            continue
        end_key = next(
            (item for item in keypoints if _nearest_index(times, float(item["t"])) >= last), keypoints[-1]
        )
        demonstrated = end_key["move"] == "follow" or last == grasp
        start_anchored = first >= grasp and last <= start_anchor_end
        carrying = first >= grasp and last <= release and not start_anchored
        layout.append((first, last, end_key, demonstrated, start_anchored, carrying))
    return layout


def _interval_reference(
    *,
    first: int,
    grasp: int,
    end_key: dict,
    start_anchored: bool,
    carrying: bool,
    held: str,
    target: str,
) -> tuple[str, Relation, bool]:
    """Preserve each keypoint's declared reference rather than inferring it from time alone."""
    relative_to_entry = start_anchored or end_key.get("frame") == "world"
    if relative_to_entry:
        return "__start__", Relation.EE_TO_START, True
    if carrying:
        return target, Relation.HELD_TO_TARGET, False
    return (target if first >= grasp else held), Relation.EE_TO_OBJECT, False


def _metric_pose_at(
    estimates: list[dict], role: str, camera_index: int, *, max_previous: int
) -> tuple[np.ndarray, str]:
    for index in range(camera_index, max(-1, camera_index - max_previous - 1), -1):
        estimate = estimates[index].get(role)
        if estimate is not None and estimate.evidence != Evidence.LOST:
            return (
                estimate.pose,
                "observed"
                if index == camera_index and estimate.evidence == Evidence.OBSERVED
                else "predicted",
            )
    raise ValueError(f"{role} has no recent metric demo pose at camera frame {camera_index}")


def prepare(root: Path, annotation_path: Path, *, output_name: str) -> tuple[Path, dict]:
    annotation = json.loads(annotation_path.read_text())
    required = (
        "repo_id",
        "episode",
        "robot_arm",
        "calibration_path",
        "reference_frame",
        "prompts_by_role",
        "held_role",
        "target_role",
        "grasp_time_s",
        "release_time_s",
        "max_demo_pose_gap_frames",
        "connection_speed_m_s",
        "min_similarity",
        "registration_threshold_m",
        "registration_min_inliers",
        "registration_trials",
        "sam_resolution",
        "descriptor_model_id",
        "start_anchored_until_s",
        "grasp_reference_frame",
        "axial_symmetry_roles",
    )
    missing = [key for key in required if key not in annotation]
    if missing:
        raise ValueError(f"annotation is missing: {', '.join(missing)}")
    held, target = annotation["held_role"], annotation["target_role"]
    prompts = annotation["prompts_by_role"]
    if held == target or set(prompts) != {held, target}:
        raise ValueError("the annotation must give distinct prompts for its held and target roles")
    if annotation["episode"] != 0:
        raise ValueError("this sealed-format adapter supports its one recorded episode only")
    calibration = json.loads(Path(annotation["calibration_path"]).expanduser().read_text())
    base_from_camera = np.asarray(calibration["camera"]["T_base_cam"], dtype=float)
    kinematics = make_calibrated_so107_kinematics(calibration, arm=annotation["robot_arm"])
    with np.load(root / "showservo_demo.npz", allow_pickle=False) as arm:
        times = np.asarray(arm["t"], dtype=float)
        joints = np.asarray(arm["q_obs"], dtype=float)
        actions = np.asarray(arm["q_cmd"], dtype=float)
        recorded_tips = np.asarray(arm["tips"], dtype=float)
        origin_s = float(arm["t0"])
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("demo arm samples must have an increasing clock")
    # A tool-frame mismatch is a safety failure, not a trajectory offset to hide.
    for index in np.unique(np.linspace(0, len(times) - 1, min(len(times), 25), dtype=int)):
        calculated = kinematics.forward_kinematics(joints[index])
        if not np.allclose(calculated, recorded_tips[index], atol=1e-5):
            raise ValueError(f"calibrated fingertip differs from recorded demo at sample {index}")
    camera_times = np.atleast_1d(np.loadtxt(root / "recording" / "times.txt")) - origin_s
    intrinsics = np.loadtxt(root / "recording" / "cam_K.txt").reshape(3, 3)
    keypoints = json.loads((root / "keypoints.json").read_text())["keypoints"]
    if not keypoints or any(b["t"] <= a["t"] for a, b in zip(keypoints, keypoints[1:], strict=False)):
        raise ValueError("keypoints must increase in demo time")
    grasp = _nearest_index(times, annotation["grasp_time_s"])
    release = _nearest_index(times, annotation["release_time_s"])
    if not 0 < grasp < release < len(times) - 1:
        raise ValueError("grasp and release must lie inside the recorded demo")
    start_anchor_end = _nearest_index(times, float(annotation["start_anchored_until_s"]))
    if not grasp <= start_anchor_end <= release:
        raise ValueError("the start-anchored phase must end between grasp and release")

    version = hashlib.sha256(
        intrinsics.tobytes() + base_from_camera.tobytes() + str(annotation["reference_frame"]).encode()
    ).hexdigest()[:12]
    reference_index = int(annotation["reference_frame"])
    reference_frame = _frame(root, reference_index, intrinsics, base_from_camera, version)
    mask_source = Sam3MaskSource(
        prompts, camera_id="flash-demo-preparation", resolution=annotation["sam_resolution"]
    )
    reference_masks = mask_source.masks(reference_frame)
    if set(reference_masks) != set(prompts):
        raise ValueError(f"reference frame is missing masks for {set(prompts) - set(reference_masks)}")
    extractor = DinoPatchExtractor(model_id=annotation["descriptor_model_id"])
    references, models = [], []
    mask_paths = {}
    for role in prompts:
        mask_rel = f"flash/v1/masks/{output_name}-{role}-{reference_index:06d}.png"
        snippet = ReferenceSnippet(
            annotation["repo_id"],
            annotation["episode"],
            reference_index,
            role,
            np.eye(4),
            mask_rel,
            version,
        )
        reference = ObjectReference(role, f"{output_name}:{version}", (snippet,))
        model = build_appearance_model(
            reference, (ReferenceAppearanceFrame(snippet, reference_frame, reference_masks[role]),), extractor
        )
        references.append(reference)
        models.append(model)
        mask_paths[role] = mask_rel

    axial_symmetries = []
    for role, spec in annotation["axial_symmetry_roles"].items():
        axis_kind = spec["axis_kind"]
        if role not in prompts or axis_kind not in ("surface_normal", "long_axis"):
            raise ValueError("axial symmetry needs a prepared role and a supported PCA axis")
        points = next(model.points.points_object_m for model in models if model.points.role == role)
        center = points.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh((points - center).T @ (points - center))
        axis = eigenvectors[
            :, int(np.argmin(eigenvalues) if axis_kind == "surface_normal" else np.argmax(eigenvalues))
        ]
        if axis[2] < 0:
            axis = -axis
        axial_symmetries.append(
            AxialSymmetry(
                role,
                axis,
                center,
                base_from_camera,
                lock_to_baseline=bool(spec.get("lock_to_baseline", False)),
                max_observed_tilt_rad=(
                    np.deg2rad(float(spec["max_observed_tilt_deg"]))
                    if "max_observed_tilt_deg" in spec
                    else None
                ),
            )
        )

    through_frame = _nearest_index(camera_times, times[_nearest_index(times, keypoints[-1]["t"])])
    estimates: list[dict] = []
    observed_counts = dict.fromkeys(prompts, 0)
    cache_identity = {
        key: annotation[key]
        for key in (
            "reference_frame",
            "prompts_by_role",
            "min_similarity",
            "registration_threshold_m",
            "registration_min_inliers",
            "registration_trials",
            "sam_resolution",
            "descriptor_model_id",
            "axial_symmetry_roles",
        )
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True).encode()
        + intrinsics.tobytes()
        + base_from_camera.tobytes()
    ).hexdigest()[:12]
    cache_path = root / "flash" / "v1" / "preparation-cache" / f"{output_name}-{cache_key}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if int(cache["frames"]) != through_frame + 1:
                raise ValueError("cached pose trace has a different demo length")
            for index in range(through_frame + 1):
                row = {}
                for role in prompts:
                    evidence = (
                        Evidence(str(cache[f"{role}_evidence"][index]))
                        if f"{role}_evidence" in cache
                        else Evidence.OBSERVED
                        if bool(cache[f"{role}_observed"][index])
                        else Evidence.LOST
                    )
                    available = evidence != Evidence.LOST
                    row[role] = PoseEstimate(
                        role,
                        cache[f"{role}_poses"][index] if available else None,
                        index / 30,
                        index / 30,
                        evidence,
                        f"{output_name}:{version}",
                        (available,) * 3,
                        (available,) * 3,
                    )
                    observed_counts[role] += evidence == Evidence.OBSERVED
                estimates.append(row)
    else:
        matcher = AppearancePixelMatcher(
            tuple(models), extractor, mask_source, min_similarity=float(annotation["min_similarity"])
        )
        backend = MatchedRGBDPoseBackend(
            tuple(model.points for model in models),
            matcher,
            threshold_m=float(annotation["registration_threshold_m"]),
            min_inliers=int(annotation["registration_min_inliers"]),
            trials=int(annotation["registration_trials"]),
            clock=lambda: 0.0,
        )
        if axial_symmetries:
            backend = SymmetryCanonicalizingBackend(backend, tuple(axial_symmetries))
        backend.initialize(tuple(references))
        for index in range(through_frame + 1):
            found = backend.update(_frame(root, index, intrinsics, base_from_camera, version))
            estimates.append({estimate.role: estimate for estimate in found})
            for estimate in found:
                observed_counts[estimate.role] += estimate.evidence == Evidence.OBSERVED
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"frames": through_frame + 1}
        for role in prompts:
            arrays[f"{role}_observed"] = np.array(
                [row[role].evidence == Evidence.OBSERVED for row in estimates], dtype=bool
            )
            arrays[f"{role}_evidence"] = np.array([row[role].evidence.value for row in estimates])
            arrays[f"{role}_poses"] = np.stack(
                [row[role].pose if row[role].pose is not None else np.eye(4) for row in estimates]
            )
        np.savez_compressed(cache_path, **arrays)
    camera_for_arm = np.array([_nearest_index(camera_times, time_s) for time_s in times])
    max_gap = int(annotation["max_demo_pose_gap_frames"])
    selected_grasp_frame = int(annotation["grasp_reference_frame"])
    if not 0 <= selected_grasp_frame <= int(camera_for_arm[grasp]):
        raise ValueError("the selected grasp reference must precede the demonstrated grasp")
    selected = estimates[selected_grasp_frame][held]
    if selected.evidence != Evidence.OBSERVED:
        raise ValueError("the selected grasp reference has no observed metric pose")
    grasp_held, grasp_evidence = selected.pose, "predicted from selected pre-occlusion frame"
    grip_from_held = inverse(recorded_tips[grasp]) @ grasp_held
    roles_version = {reference.role: reference.version for reference in references}
    intervals = []
    for number, (first, last, end_key, demonstrated, start_anchored, carrying) in enumerate(
        _interval_layout(times, keypoints, grasp, release, start_anchor_end)
    ):
        mode = MotionMode.DEMONSTRATED if demonstrated else MotionMode.CONNECTION
        reference_role, relation, relative_to_entry = _interval_reference(
            first=first,
            grasp=grasp,
            end_key=end_key,
            start_anchored=start_anchored,
            carrying=carrying,
            held=held,
            target=target,
        )
        carrying = relation == Relation.HELD_TO_TARGET
        indices = range(first, last + 1) if demonstrated else (first, last)
        samples = []
        for index in indices:
            camera_index = int(camera_for_arm[index])
            if relative_to_entry:
                reference_pose, reference_evidence = recorded_tips[first], "recorded_start"
            elif reference_role == held and index <= grasp:
                reference_pose = grasp_held
                reference_evidence = "observed" if camera_index == selected_grasp_frame else "predicted"
            else:
                reference_pose, reference_evidence = _metric_pose_at(
                    estimates, reference_role, camera_index, max_previous=max_gap
                )
            demo_held = recorded_tips[index] @ grip_from_held if carrying else None
            event_id = (
                "attach-held"
                if index == grasp and last == grasp
                else ("detach-held" if index == release and last == release else None)
            )
            samples.append(
                SpatialSample(
                    time_s=float(times[index]),
                    demo_ee=recorded_tips[index],
                    demo_reference=reference_pose,
                    demo_held=demo_held,
                    gripper=float(actions[index, -1]),
                    source_frame_index=index,
                    reference_evidence=reference_evidence,
                    held_evidence="predicted" if carrying else None,
                    event_id=event_id,
                )
            )
        intervals.append(
            Interval(
                name=f"segment-{number:02d}-{end_key['move']}",
                mode=mode,
                relation=relation,
                reference_role=reference_role,
                samples=tuple(samples),
                speed_m_s=float(annotation["connection_speed_m_s"])
                if mode == MotionMode.CONNECTION
                else None,
                reference_version=(
                    f"{output_name}:start-anchor" if relative_to_entry else roles_version[reference_role]
                ),
                held_reference_version=roles_version[held] if carrying else None,
                held_role=held if carrying else None,
            )
        )
    skill = FlashSkill(
        annotation["repo_id"],
        annotation["episode"],
        tuple(intervals),
        attachment_transitions=(
            AttachmentTransition("attach-held", AttachmentAction.ATTACH, held, roles_version[held]),
            AttachmentTransition("detach-held", AttachmentAction.DETACH, held, roles_version[held]),
        ),
    )
    prepared = PreparedFlashSkill(
        skill,
        tuple(references),
        tuple(models),
        prompts,
        annotation["descriptor_model_id"],
        tuple(axial_symmetries),
    )
    destination = root / "flash" / "v1" / "prepared" / output_name
    if destination.exists():
        raise FileExistsError(f"prepared bundle already exists: {destination}")
    for role, rel in mask_paths.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), reference_masks[role].astype(np.uint8) * 255):
            raise OSError(f"cannot write reference mask {path}")
    manifest = prepared.save(destination)
    report = {
        "source_episode": annotation["episode"],
        "demo_frames_evaluated": len(estimates),
        "observed_pose_frames": observed_counts,
        "grasp_sample": grasp,
        "grasp_pose_evidence": grasp_evidence,
        "release_sample": release,
        "start_anchored_until_sample": start_anchor_end,
        "held_contact_pose": "predicted from recorded calibrated fingertip and grasp pose",
        "intervals": len(intervals),
        "calibration_version": version,
    }
    (destination / "preparation-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    manifest, report = prepare(
        args.dataset_root.expanduser(), args.annotation.expanduser(), output_name=args.output_name
    )
    print(json.dumps({"bundle": str(manifest), **report}, indent=2))


if __name__ == "__main__":
    main()
