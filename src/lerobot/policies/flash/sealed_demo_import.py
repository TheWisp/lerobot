"""Migrate a sealed RGB-D demonstration into a reviewable FLASH dataset extension.

This is an explicit adapter for the old ``showservo_demo.npz`` recording
layout. It does not infer task intent: a JSON annotation names the held and
target roles, grasp/release times, prompts, and registration tolerances. Both
the demonstration and a subsequent rollout use the same FLASH RGB-D matcher.
Held poses after grasp are predictions from the latest supported gripper-to-object
relation, optionally corrected by source-time visual evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.appearance import (
    ReferenceAppearanceFrame,
    Sam3MaskSource,
    build_appearance_model,
)
from lerobot.policies.flash.causal_service import CausalTrackingConfig
from lerobot.policies.flash.dino_features import DinoPatchExtractor
from lerobot.policies.flash.geometry import inverse
from lerobot.policies.flash.held_pose_diagnostics import _arm_pose_at, compare_demo_held_pose
from lerobot.policies.flash.offline_perception import POSE_TRACE_SELECTION_VERSION, extract_pose_trace
from lerobot.policies.flash.perception import (
    HeldAttachmentUpdater,
    ObjectReference,
    ReferenceSnippet,
    RGBDFrame,
)
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.runtime import AttachmentRuntimeLimits
from lerobot.policies.flash.scene_evidence_writer import (
    SCENE_EVIDENCE_CAPTURE_VERSION,
    SceneEvidenceWriter,
)
from lerobot.policies.flash.skill import (
    AttachmentAction,
    AttachmentTransition,
    EndpointRequirement,
    FlashSkill,
    Interval,
    MotionMode,
    Relation,
    SpatialSample,
    TurnAtPoint,
)
from lerobot.policies.flash.surface_reference import build_registered_surface_model
from lerobot.policies.flash.symmetry import AxialSymmetry
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend
from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics


def _frame(
    root: Path,
    index: int,
    intrinsics: np.ndarray,
    base_from_camera: np.ndarray,
    version: str,
    source_time_s: float,
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
        source_time_s,
        source_time_s,
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
    *,
    implicit_grasp_follow: bool = False,
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
        # Version-one sealed recipes replayed the grasp boundary regardless of
        # its label. New recipes encode that choice in the key frame itself.
        demonstrated = end_key["move"] == "follow" or (implicit_grasp_follow and last == grasp)
        start_anchored = first >= grasp and last <= start_anchor_end
        carrying = first >= grasp and last <= release and not start_anchored
        layout.append((first, last, end_key, demonstrated, start_anchored, carrying))
    return layout


def explicit_paths_from_prepared_skill(
    keypoints_document: dict, skill_document: dict, arm_times: np.ndarray
) -> tuple[dict, bool]:
    """Show a legacy compiled path as editable key frames without changing its motion.

    The prepared skill, rather than an inference from grasp timing, is the
    authority for what the old editor actually compiled. An inserted key frame
    is needed when grasp split a longer interval between two authored keys.
    """
    version = keypoints_document.get("path_selection_version", 1)
    if version != 1:
        if version != 2:
            raise ValueError("unsupported keypoint path-selection version")
        return copy.deepcopy(keypoints_document), False
    document = copy.deepcopy(keypoints_document)
    points = document["keypoints"]
    changed = False
    for interval in skill_document["intervals"]:
        if interval["mode"] != MotionMode.DEMONSTRATED.value:
            continue
        last_index = interval["samples"][-1].get("source_frame_index")
        if last_index is None or not 0 <= last_index < len(arm_times):
            raise ValueError("prepared interval lacks a valid source arm frame")
        endpoint = next(
            (point for point in points if _nearest_index(arm_times, float(point["t"])) >= last_index),
            points[-1],
        )
        if endpoint["move"] == "follow":
            continue
        if _nearest_index(arm_times, float(endpoint["t"])) == last_index:
            endpoint["move"] = "follow"
        else:
            split = {**endpoint, "t": float(arm_times[last_index]), "move": "follow"}
            points.append(split)
            points.sort(key=lambda point: float(point["t"]))
        changed = True
    document["path_selection_version"] = 2
    return document, changed


def _interval_reference(
    *,
    first: int,
    grasp: int,
    end_key: dict,
    start_anchored: bool,
    carrying: bool,
    held: str,
    target: str,
    available_roles: set[str] | None = None,
) -> tuple[str, Relation, bool]:
    """Use the authored participant, except during an explicit start-anchored interval."""
    selected = end_key.get("frame")
    relative_to_entry = start_anchored or selected == "world"
    if relative_to_entry:
        return "__start__", Relation.EE_TO_START, True
    if selected is None:
        # Old keypoint files sometimes omitted frame; retain their previous
        # timing-based interpretation rather than changing archived skills.
        selected = target if carrying or first >= grasp else held
    if selected not in (available_roles if available_roles is not None else {held, target}):
        raise ValueError(f"keypoint reference {selected!r} is not a prepared object")
    relation = Relation.HELD_TO_TARGET if carrying and selected != held else Relation.EE_TO_OBJECT
    return selected, relation, False


def _endpoint_requirement(end_key: dict) -> EndpointRequirement:
    constraint = end_key.get("constraint", "pose")
    if constraint == "best_effort":
        return EndpointRequirement.PREFERRED
    if constraint == "pose":
        return EndpointRequirement.REQUIRED
    raise ValueError(f"unsupported keypoint constraint: {constraint!r}")


def _turn_for_interval(
    rules: list[dict], times: np.ndarray, first: int, last: int, reference_role: str
) -> TurnAtPoint | None:
    """Apply one authored turn across complete, adjacent timeline intervals."""
    selected = None
    for item in rules:
        start = _nearest_index(times, float(item["start_s"]))
        end = _nearest_index(times, float(item["end_s"]))
        if start >= end:
            raise ValueError("turn rule needs an increasing key-frame span")
        intersects = first < end and last > start
        within = first >= start and last <= end
        if intersects and not within:
            raise ValueError("turn rule must begin and end at interval boundaries")
        if not within:
            continue
        if reference_role != item["reference_role"]:
            raise ValueError("turn rule span crosses a different reference object")
        if selected is not None:
            raise ValueError("overlapping turn rules are not supported")
        selected = TurnAtPoint(
            item["rule_id"],
            tuple(item["pivot_reference_m"]),
            tuple(item["axis_reference"]),
            float(item["min_angle_rad"]),
            float(item["max_angle_rad"]),
            int(item["source_frame_index"]) if item.get("source_frame_index") is not None else None,
            item.get("source_evidence", "unverified"),
        )
    return selected


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


def _symmetry_from_annotation(
    role: str, spec: dict, points: np.ndarray, base_from_camera: np.ndarray
) -> AxialSymmetry:
    axis_kind = spec["axis_kind"]
    if axis_kind not in ("surface_normal", "long_axis"):
        raise ValueError("axial symmetry needs a supported PCA axis")
    center = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh((points - center).T @ (points - center))
    axis = eigenvectors[
        :, int(np.argmin(eigenvalues) if axis_kind == "surface_normal" else np.argmax(eigenvalues))
    ]
    if axis[2] < 0:
        axis = -axis
    return AxialSymmetry(
        role,
        axis,
        center,
        base_from_camera,
        lock_to_baseline=bool(spec.get("lock_to_baseline", False)),
        max_observed_tilt_rad=(
            np.deg2rad(float(spec["max_observed_tilt_deg"])) if "max_observed_tilt_deg" in spec else None
        ),
        ends_interchangeable=bool(spec.get("ends_interchangeable", False)),
    )


def _apply_reference_polygons(
    masks: dict[str, np.ndarray], polygons: dict, roles: set[str], image_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Replace selected SAM3 reference masks with explicitly annotated pixels."""
    height, width = image_size
    result = dict(masks)
    for role, vertices in polygons.items():
        if role not in roles:
            raise ValueError(f"reference mask polygon names unknown role {role!r}")
        points = np.asarray(vertices, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1:] != (2,)
            or len(points) < 3
            or not np.isfinite(points).all()
            or np.any(points < 0)
            or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] >= height)
        ):
            raise ValueError(f"{role}: reference polygon needs three or more in-frame pixel vertices")
        mask = np.zeros(image_size, dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        result[role] = mask.astype(bool)
    return result


def _reference_observations(
    prompts: dict[str, str],
    frame_indices: dict[str, int],
    frame_loader,
    mask_source: Sam3MaskSource,
    polygons: dict,
) -> dict[str, tuple[RGBDFrame, np.ndarray]]:
    """Select each object's clear view without requiring all objects in one image."""
    if set(frame_indices) != set(prompts) or any(index < 0 for index in frame_indices.values()):
        raise ValueError("each prepared object needs a nonnegative reference frame")
    if set(polygons) - set(prompts):
        raise ValueError(f"reference mask polygons name unknown roles: {set(polygons) - set(prompts)}")
    observations = {}
    for view_number, frame_index in enumerate(sorted(set(frame_indices.values()))):
        if view_number:
            mask_source.reset_tracking()
        frame = frame_loader(frame_index)
        masks = mask_source.masks(frame)
        roles = {role for role, index in frame_indices.items() if index == frame_index}
        selected = _apply_reference_polygons(
            {role: masks[role] for role in roles if role in masks},
            {role: polygons[role] for role in roles if role in polygons},
            roles,
            frame.depth.shape,
        )
        if set(selected) != roles:
            raise ValueError(f"reference frame {frame_index} is missing masks for {roles - set(selected)}")
        observations.update({role: (frame, selected[role]) for role in roles})
    return observations


def prepare(
    root: Path, annotation_path: Path, *, output_name: str, keypoints_path: Path | None = None
) -> tuple[Path, dict]:
    destination = root / "flash" / "v1" / "prepared" / output_name
    if destination.exists():
        raise FileExistsError(f"prepared bundle already exists: {destination}")
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
    if held == target or not {held, target} <= set(prompts) or len(prompts) < 2:
        raise ValueError("the annotation must name distinct held and target objects")
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
    if len(camera_times) < 2 or not np.all(np.diff(camera_times) > 0):
        raise ValueError("demo RGB-D frames must have an increasing recorded clock")
    intrinsics = np.loadtxt(root / "recording" / "cam_K.txt").reshape(3, 3)
    keypoints_document = json.loads((keypoints_path or root / "keypoints.json").read_text())
    keypoints = keypoints_document["keypoints"]
    path_version = keypoints_document.get("path_selection_version", 1)
    if path_version not in (1, 2):
        raise ValueError("unsupported keypoint path-selection version")
    turn_rules = keypoints_document.get("turn_rules", [])
    if not keypoints or any(b["t"] <= a["t"] for a, b in zip(keypoints, keypoints[1:], strict=False)):
        raise ValueError("keypoints must increase in demo time")
    if not isinstance(turn_rules, list):
        raise ValueError("turn rules must be a list")
    grasp = _nearest_index(times, annotation["grasp_time_s"])
    release = _nearest_index(times, annotation["release_time_s"])
    if not 0 < grasp < release < len(times) - 1:
        raise ValueError("grasp and release must lie inside the recorded demo")
    if not np.allclose(kinematics.forward_kinematics(joints[grasp]), recorded_tips[grasp], atol=1e-5):
        raise ValueError("calibrated fingertip differs from the grasp-time demo pose")
    start_anchor_end = _nearest_index(times, float(annotation["start_anchored_until_s"]))
    if not grasp <= start_anchor_end <= release:
        raise ValueError("the start-anchored phase must end between grasp and release")

    reference_index = int(annotation["reference_frame"])
    custom_reference_indices = annotation.get("reference_frames_by_role") or {}
    if (
        not isinstance(custom_reference_indices, dict)
        or set(custom_reference_indices) - set(prompts)
        or any(type(index) is not int or index < 0 for index in custom_reference_indices.values())
    ):
        raise ValueError("object reference frames must name prepared roles and nonnegative frame indices")
    reference_indices = {role: custom_reference_indices.get(role, reference_index) for role in prompts}
    reference_identity = (
        json.dumps(reference_indices, sort_keys=True) if custom_reference_indices else str(reference_index)
    )
    version = hashlib.sha256(
        intrinsics.tobytes() + base_from_camera.tobytes() + reference_identity.encode()
    ).hexdigest()[:12]
    vision_config = VisionPipelineConfig.from_settings(annotation, backend="appearance")
    mask_source = Sam3MaskSource(
        prompts, camera_id="flash-demo-preparation", resolution=vision_config.sam_resolution
    )
    reference_observations = _reference_observations(
        prompts,
        reference_indices,
        lambda index: _frame(root, index, intrinsics, base_from_camera, version, float(camera_times[index])),
        mask_source,
        annotation.get("reference_polygons_by_role", {}),
    )
    extractor = DinoPatchExtractor(model_id=annotation["descriptor_model_id"])
    references, models, surface_models = [], [], []
    mask_paths = {}
    for role in prompts:
        role_frame_index = reference_indices[role]
        reference_frame, reference_mask = reference_observations[role]
        mask_rel = f"flash/v1/masks/{output_name}-{role}-{role_frame_index:06d}.png"
        snippet = ReferenceSnippet(
            annotation["repo_id"],
            annotation["episode"],
            role_frame_index,
            role,
            np.eye(4),
            mask_rel,
            version,
        )
        reference = ObjectReference(role, f"{output_name}:{version}", (snippet,))
        views = (ReferenceAppearanceFrame(snippet, reference_frame, reference_mask),)
        model = build_appearance_model(reference, views, extractor)
        surface_model = build_registered_surface_model(
            reference,
            views,
            observation_config=vision_config.observation,
            max_points_per_view=vision_config.surface_reference_max_points_per_view,
        )
        references.append(reference)
        models.append(model)
        surface_models.append(surface_model)
        mask_paths[role] = mask_rel

    axial_symmetries = []
    for role, spec in annotation["axial_symmetry_roles"].items():
        if role not in prompts:
            raise ValueError("axial symmetry needs a prepared role")
        points = next(model.points.points_object_m for model in models if model.points.role == role)
        axial_symmetries.append(_symmetry_from_annotation(role, spec, points, base_from_camera))

    through_frame = _nearest_index(camera_times, times[_nearest_index(times, keypoints[-1]["t"])])
    estimates: list[dict] = []
    held_observed_masks: dict[int, np.ndarray] = {}
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
    cache_identity["reference_polygons_by_role"] = annotation.get("reference_polygons_by_role", {})
    if custom_reference_indices:
        cache_identity["reference_frames_by_role"] = reference_indices
    causal_config = (
        CausalTrackingConfig.from_config(annotation["causal_tracking"])
        if annotation.get("causal_tracking") is not None
        else None
    )
    cache_identity["causal_tracking"] = annotation.get("causal_tracking")
    cache_identity["vision_pipeline"] = vision_config.to_dict()
    cache_identity["pose_trace_selection_version"] = POSE_TRACE_SELECTION_VERSION
    cache_identity["scene_evidence_capture_version"] = SCENE_EVIDENCE_CAPTURE_VERSION
    cache_key = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True).encode()
        + intrinsics.tobytes()
        + base_from_camera.tobytes()
        + camera_times.tobytes()
    ).hexdigest()[:12]
    cache_path = root / "flash" / "v1" / "preparation-cache" / f"{output_name}-{cache_key}.npz"
    evidence_cache_path = cache_path.with_suffix(".scene-evidence")
    cache_hit = cache_path.exists()
    if cache_hit:
        evidence_manifest = evidence_cache_path / "manifest.json"
        if not evidence_manifest.is_file():
            cache_hit = False
        else:
            evidence_status = json.loads(evidence_manifest.read_text(encoding="utf-8"))
            cache_hit = (
                evidence_status.get("complete") is True
                and evidence_status.get("total_frames") == through_frame + 1
                and evidence_status.get("frames_completed") == through_frame + 1
                and evidence_status.get("roles") == list(prompts)
                and all(
                    (evidence_cache_path / f"{index:06d}.json.gz").is_file()
                    for index in range(through_frame + 1)
                )
            )
    if cache_hit and annotation.get("held_attachment") is not None:
        with np.load(cache_path, allow_pickle=False) as cache:
            cache_hit = "held_mask_indices" in cache and "held_masks" in cache
    if cache_hit:
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
                        float(camera_times[index]),
                        float(camera_times[index]),
                        evidence,
                        f"{output_name}:{version}",
                        tuple(bool(value) for value in cache[f"{role}_translation_supported"][index])
                        if f"{role}_translation_supported" in cache
                        else (available,) * 3,
                        tuple(bool(value) for value in cache[f"{role}_rotation_supported"][index])
                        if f"{role}_rotation_supported" in cache
                        else (available,) * 3,
                        direct_translation_observed=(
                            bool(cache[f"{role}_direct_translation"][index])
                            if f"{role}_direct_translation" in cache
                            else False
                        ),
                        estimator=(
                            str(cache[f"{role}_estimator"][index])
                            if f"{role}_estimator" in cache
                            else "legacy_demo_cache"
                        ),
                        last_direct_observation_s=(
                            float(cache[f"{role}_last_direct_s"][index])
                            if f"{role}_last_direct_s" in cache
                            and np.isfinite(cache[f"{role}_last_direct_s"][index])
                            else None
                        ),
                        symmetry_canonicalized=(
                            bool(cache[f"{role}_symmetry_canonicalized"][index])
                            if f"{role}_symmetry_canonicalized" in cache
                            else False
                        ),
                        source_frame_id=index,
                    )
                    observed_counts[role] += evidence == Evidence.OBSERVED
                estimates.append(row)
            if "held_mask_indices" in cache:
                held_observed_masks = {
                    int(index): mask.astype(bool)
                    for index, mask in zip(cache["held_mask_indices"], cache["held_masks"], strict=True)
                }
    else:
        writer = SceneEvidenceWriter(
            evidence_cache_path, total_frames=through_frame + 1, roles=tuple(prompts)
        )
        backend = build_vision_backend(
            vision_config,
            mask_source,
            models=tuple(models),
            extractor=extractor,
            axial_symmetries=tuple(axial_symmetries),
            clock=lambda: 0.0,
        )
        grasp_camera = _nearest_index(camera_times, times[grasp])
        release_camera = _nearest_index(camera_times, times[release])

        def capture_held_mask(frame, result, _groups, by_role) -> None:
            writer.record(frame, result, _groups, by_role)
            if (
                times[grasp] <= frame.source_time_s < times[release]
                and by_role[held].evidence == Evidence.OBSERVED
                and held in result.masks
            ):
                held_observed_masks[frame.frame_id] = result.masks[held].copy()

        estimates = extract_pose_trace(
            (
                _frame(root, index, intrinsics, base_from_camera, version, float(camera_times[index]))
                for index in range(through_frame + 1)
            ),
            backend,
            tuple(references),
            causal_tracking=causal_config,
            object_points_by_role={model.points.role: model.points.points_object_m for model in models},
            axial_symmetries=tuple(axial_symmetries),
            attached_role_at_frame=(
                lambda frame_id: held if grasp_camera <= frame_id < release_camera else None
            ),
            on_frame_evidence=capture_held_mask,
        )
        writer.finish()
        for row in estimates:
            for role, estimate in row.items():
                observed_counts[role] += estimate.evidence == Evidence.OBSERVED
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"frames": through_frame + 1}
        arrays["held_mask_indices"] = np.asarray(sorted(held_observed_masks), dtype=np.int32)
        image_shape = next(iter(reference_observations.values()))[0].depth.shape
        arrays["held_masks"] = (
            np.stack([held_observed_masks[index] for index in sorted(held_observed_masks)])
            if held_observed_masks
            else np.empty((0, *image_shape), dtype=bool)
        )
        for role in prompts:
            arrays[f"{role}_observed"] = np.array(
                [row[role].evidence == Evidence.OBSERVED for row in estimates], dtype=bool
            )
            arrays[f"{role}_evidence"] = np.array([row[role].evidence.value for row in estimates])
            arrays[f"{role}_estimator"] = np.array([row[role].estimator for row in estimates])
            arrays[f"{role}_translation_supported"] = np.array(
                [row[role].translation_supported for row in estimates], dtype=bool
            )
            arrays[f"{role}_rotation_supported"] = np.array(
                [row[role].rotation_supported for row in estimates], dtype=bool
            )
            arrays[f"{role}_direct_translation"] = np.array(
                [row[role].direct_translation_observed for row in estimates], dtype=bool
            )
            arrays[f"{role}_symmetry_canonicalized"] = np.array(
                [row[role].symmetry_canonicalized for row in estimates], dtype=bool
            )
            arrays[f"{role}_last_direct_s"] = np.array(
                [
                    row[role].last_direct_observation_s
                    if row[role].last_direct_observation_s is not None
                    else np.nan
                    for row in estimates
                ]
            )
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
    held_model = next(model.points.points_object_m for model in models if model.points.role == held)
    held_pose_comparison = compare_demo_held_pose(
        role=held,
        estimates=estimates,
        arm_times_s=times,
        arm_poses=recorded_tips,
        grasp_time_s=float(times[grasp]),
        release_time_s=float(times[release]),
        grasp_attachment=grip_from_held,
        object_center_m=held_model.mean(axis=0),
    )
    # Replay the same source-time attachment updater as a live attempt. A
    # captured visual pose can revise the grasp assumption without changing
    # the saved object frame. When the image cannot arbitrate a large change,
    # retain the last accepted relation and mark the bundle for review.
    held_attachment_by_camera = [grip_from_held.copy() for _ in estimates]
    held_conflicted_by_camera = [False for _ in estimates]
    held_ambiguous_by_camera = [False for _ in estimates]
    held_corrected_by_camera = [False for _ in estimates]
    held_visual_evidence_by_camera: list[dict | None] = [None for _ in estimates]
    held_correction_frames: list[int] = []
    held_conflict_frames: list[int] = []
    held_attachment_policy = annotation.get("held_attachment")
    if held_attachment_policy is not None:
        missing_masks = [
            frame_index
            for frame_index, row in enumerate(estimates)
            if times[grasp] <= row[held].source_time_s < times[release]
            and row[held].evidence == Evidence.OBSERVED
            and row[held].supports()
            and frame_index not in held_observed_masks
        ]
        if missing_masks:
            raise ValueError(
                f"held correction needs recorded aligned masks for {len(missing_masks)} observed frames; "
                "regenerate the pose-trace cache"
            )
        limits = AttachmentRuntimeLimits.from_config(held_attachment_policy)
        updater = HeldAttachmentUpdater(
            held,
            next(reference.version for reference in references if reference.role == held),
            kinematics,
            tuple(f"joint-{index}" for index in range(joints.shape[1])),
            history_capacity=limits.history_capacity,
            max_motor_sample_gap_s=limits.max_motor_sample_gap_s,
            max_attachment_age_s=limits.max_attachment_age_s,
            max_update_shift_m=limits.max_update_shift_m,
            max_update_turn_rad=limits.max_update_turn_rad,
            object_center_object_m=held_model.mean(axis=0),
            object_points_object_m=held_model,
            min_visual_support_points=(
                causal_config.min_inliers
                if causal_config is not None
                else vision_config.registration_min_inliers
            ),
            depth_occlusion_tolerance_m=(
                causal_config.fit_threshold_m
                if causal_config is not None
                else vision_config.registration_threshold_m
            ),
            observation_quality=vision_config.observation,
        )
        if not updater.activate(float(times[grasp]), selected, joints[grasp]):
            raise ValueError("selected grasp pose cannot seed the held attachment")
        last_attachment = grip_from_held
        has_correction = False
        for frame_index, row in enumerate(estimates):
            estimate = row[held]
            if not float(times[grasp]) <= estimate.source_time_s < float(times[release]):
                held_attachment_by_camera[frame_index] = last_attachment.copy()
                continue
            if estimate.evidence == Evidence.OBSERVED and estimate.supports():
                mask = held_observed_masks.get(frame_index)
                ee_at_capture = _arm_pose_at(times, recorded_tips, estimate.source_time_s)
                if mask is not None and ee_at_capture is not None:
                    frame = _frame(
                        root,
                        frame_index,
                        intrinsics,
                        base_from_camera,
                        version,
                        estimate.source_time_s,
                    )
                    accepted_update = updater.on_pose_update(
                        frame_index,
                        {held: estimate},
                        frame=frame,
                        mask=mask,
                        ee_at_capture=ee_at_capture,
                    )
                    visual_evidence = updater.latest_visual_evidence()
                    if visual_evidence is not None and visual_evidence["frame_id"] == frame_index:
                        held_visual_evidence_by_camera[frame_index] = visual_evidence
                    if accepted_update:
                        last_attachment = updater.last_accepted_gripper_from_object
                        held_correction_frames.append(frame_index)
                        has_correction = True
                    elif updater.state(estimate.source_time_s) is None:
                        held_conflict_frames.append(frame_index)
            held_attachment_by_camera[frame_index] = last_attachment.copy()
            state, visual_ambiguous = updater.state_with_visual_status(estimate.source_time_s)
            held_conflicted_by_camera[frame_index] = state is None
            held_ambiguous_by_camera[frame_index] = visual_ambiguous
            held_corrected_by_camera[frame_index] = has_correction
    roles_version = {reference.role: reference.version for reference in references}
    intervals = []
    for number, (first, last, end_key, demonstrated, start_anchored, carrying) in enumerate(
        _interval_layout(
            times,
            keypoints,
            grasp,
            release,
            start_anchor_end,
            implicit_grasp_follow=path_version == 1,
        )
    ):
        mode = MotionMode.DEMONSTRATED if demonstrated else MotionMode.CONNECTION
        endpoint_requirement = _endpoint_requirement(end_key)
        reference_role, relation, relative_to_entry = _interval_reference(
            first=first,
            grasp=grasp,
            end_key=end_key,
            start_anchored=start_anchored,
            carrying=carrying,
            held=held,
            target=target,
            available_roles=set(prompts),
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
            causal_camera_index = max(
                0, min(len(estimates) - 1, int(np.searchsorted(camera_times, times[index], side="right") - 1))
            )
            demo_held = (
                recorded_tips[index] @ held_attachment_by_camera[causal_camera_index] if carrying else None
            )
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
                    held_evidence=(
                        "conflicted_prediction"
                        if carrying and held_conflicted_by_camera[causal_camera_index]
                        else (
                            "ambiguous_visual_prediction"
                            if carrying and held_ambiguous_by_camera[causal_camera_index]
                            else (
                                "vision_corrected_prediction"
                                if carrying and held_corrected_by_camera[causal_camera_index]
                                else ("predicted" if carrying else None)
                            )
                        )
                    ),
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
                endpoint_requirement=endpoint_requirement,
                turn_at_point=_turn_for_interval(turn_rules, times, first, last, reference_role),
            )
        )
    used_turns = {item.turn_at_point.rule_id for item in intervals if item.turn_at_point is not None}
    declared_turns = {str(item["rule_id"]) for item in turn_rules}
    if used_turns != declared_turns:
        raise ValueError(f"turn rules did not cover an interval: {sorted(declared_turns - used_turns)}")
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
        vision_config,
        causal_config,
        tuple(surface_models),
    )
    for role, rel in mask_paths.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), reference_observations[role][1].astype(np.uint8) * 255):
            raise OSError(f"cannot write reference mask {path}")
    manifest = prepared.save(destination)
    if held_attachment_policy is not None:
        np.savez_compressed(
            destination / "held-attachment-trace.npz",
            role=np.asarray(held),
            camera_times_s=camera_times[: len(estimates)],
            gripper_from_object=np.stack(held_attachment_by_camera),
            conflicted=np.asarray(held_conflicted_by_camera),
            visual_ambiguous=np.asarray(held_ambiguous_by_camera),
            corrected=np.asarray(held_corrected_by_camera),
            visual_evidence=np.asarray(
                [json.dumps(value) if value is not None else "" for value in held_visual_evidence_by_camera]
            ),
        )
    (destination / "annotation.json").write_text(json.dumps(annotation, indent=2) + "\n")
    (destination / "keypoints.json").write_text(json.dumps(keypoints_document, indent=2) + "\n")
    report = {
        "review_status": "candidate",
        "source_episode": annotation["episode"],
        "demo_frames_evaluated": len(estimates),
        "observed_pose_frames": observed_counts,
        "pose_evidence_frames": {
            role: {
                evidence.value: sum(row[role].evidence == evidence for row in estimates)
                for evidence in Evidence
            }
            for role in prompts
        },
        "vision_pipeline": vision_config.to_dict(),
        "pose_pipeline": "direct_plus_causal_scene" if causal_config is not None else "direct_only",
        "pose_estimator_frames": {
            role: {
                estimator: sum(row[role].estimator == estimator for row in estimates)
                for estimator in sorted({row[role].estimator for row in estimates})
            }
            for role in prompts
        },
        "grasp_sample": grasp,
        "grasp_pose_evidence": grasp_evidence,
        "release_sample": release,
        "start_anchored_until_sample": start_anchor_end,
        "held_contact_pose": (
            "source-time visual corrections with gripper prediction between observations"
            if held_correction_frames
            else "predicted from recorded calibrated fingertip and grasp pose"
        ),
        "held_attachment_policy": held_attachment_policy,
        "held_attachment_trace": "held-attachment-trace.npz" if held_attachment_policy else None,
        "held_correction_frames": held_correction_frames,
        "held_conflict_frames": held_conflict_frames,
        "held_ambiguous_frames": [
            index for index, ambiguous in enumerate(held_ambiguous_by_camera) if ambiguous
        ],
        "held_pose_comparison": held_pose_comparison,
        "intervals": len(intervals),
        "calibration_version": version,
        "reference_masks": mask_paths,
        "pose_trace_cache": str(cache_path.relative_to(root)),
        "pose_trace_cache_hit": cache_hit,
        "pose_trace_selection_version": POSE_TRACE_SELECTION_VERSION,
        "annotation_artifact": "annotation.json",
        "keypoints_artifact": "keypoints.json",
    }
    (destination / "preparation-report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copytree(evidence_cache_path, destination / "scene-evidence")
    sidecar_manifest = destination / "scene-evidence" / "manifest.json"
    sidecar_status = json.loads(sidecar_manifest.read_text(encoding="utf-8"))
    sidecar_status["source_sha256"] = hashlib.sha256(
        manifest.read_bytes() + (destination / "annotation.json").read_bytes()
    ).hexdigest()
    temporary_manifest = sidecar_manifest.with_suffix(".tmp")
    temporary_manifest.write_text(json.dumps(sidecar_status, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(sidecar_manifest)
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument(
        "--keypoints", type=Path, help="versioned keypoint annotation (defaults to dataset keypoints)"
    )
    args = parser.parse_args()
    manifest, report = prepare(
        args.dataset_root.expanduser(),
        args.annotation.expanduser(),
        output_name=args.output_name,
        keypoints_path=args.keypoints.expanduser() if args.keypoints is not None else None,
    )
    print(json.dumps({"bundle": str(manifest), **report}, indent=2))


if __name__ == "__main__":
    main()
