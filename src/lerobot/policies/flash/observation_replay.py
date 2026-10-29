"""Replay a LeRobot RGB-D episode in the FLASH review surface.

An RGB-D episode can carry masks, scene groups and pose evidence whether or
not robot actions were recorded. Vision evidence alone is not an executable skill.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .geometry import transform
from .observation_annotations import load_annotations
from .observation_dataset import (
    camera_feature_keys,
    depth_scale_m,
    episode_camera_samples,
    robot_actions_available,
    scene_from_camera,
)


def load_observation_replay(root: Path, episode: int = 0) -> dict:
    root = root.expanduser().resolve()
    source = json.loads((root / "flash" / "v1" / "source.json").read_text(encoding="utf-8"))
    if source.get("status") not in {"perception_only", "robot_recording"}:
        raise ValueError("this dataset has no registered RGB-D source")
    dataset = LeRobotDataset(source["repo_id"], root=root)
    has_robot_actions = robot_actions_available(dataset.meta.features)
    if source["status"] == "robot_recording" and not has_robot_actions:
        raise ValueError("robot recording needs both measured state and action features")
    rgb_key, depth_key = camera_feature_keys(dataset.meta.features, source)
    scale_m = depth_scale_m(source)
    samples, source_trace_sha256 = episode_camera_samples(root, dataset, source, episode)
    sidecar = root / "flash" / "v1" / "observation-evidence" / f"episode_{episode:06d}"
    manifest_path = sidecar / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    observation_path = sidecar / "observation.json"
    observation = (
        json.loads(observation_path.read_text(encoding="utf-8")) if observation_path.is_file() else None
    )
    evidence_available = bool(
        manifest
        and observation
        and manifest.get("complete")
        and manifest.get("total_frames") == len(samples)
        and observation.get("evidence_identity")
        and manifest.get("source_sha256") == observation["evidence_identity"]
        and isinstance(observation.get("surface_reference_max_points_per_view"), int)
        and observation.get("source_metadata_sha256")
        == hashlib.sha256((root / "flash" / "v1" / "source.json").read_bytes()).hexdigest()
        and observation.get("source_trace_sha256") == source_trace_sha256
        and observation.get("source_row_indices", [sample.row_index for sample in samples])
        == [sample.row_index for sample in samples]
        and observation.get("source_camera_frame_ids", [sample.camera_frame_id for sample in samples])
        == [sample.camera_frame_id for sample in samples]
    )
    height, width = dataset.meta.features[depth_key]["shape"][:2]
    intrinsics = source["camera_intrinsics"]
    frames = []
    geometry = {"points_object_m": {}, "centers_object_m": {}}
    point_limit = None
    if evidence_available:
        point_limit = int(observation["surface_reference_max_points_per_view"])
    processing_config = None
    if evidence_available and observation.get("config_bundle"):
        config_path = Path(observation["config_bundle"])
        if not config_path.is_absolute():
            config_path = root / config_path
        try:
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            if "vision_pipeline" in saved_config and "causal_tracking" in saved_config:
                processing_config = {
                    "vision_pipeline": saved_config["vision_pipeline"],
                    "causal_tracking": saved_config["causal_tracking"],
                }
        except (OSError, ValueError, TypeError):
            pass
    for frame_id, sample in enumerate(samples):
        # The saved evidence already contains each mask and pose. Loading an
        # image-backed LeRobot row here decodes both RGB and depth; fixed-camera
        # replay only needs depth once per role to build display geometry.
        row = dataset[sample.row_index] if source.get("camera_pose_feature") else None
        time_s = sample.source_time_s
        scene_pose, calibration_version = scene_from_camera(row or {}, dataset.meta.features, source)
        frame_evidence = None
        if evidence_available:
            with gzip.open(sidecar / f"{frame_id:06d}.json.gz", "rt", encoding="utf-8") as stream:
                frame_evidence = json.load(stream)
            if frame_evidence["frame_index"] != frame_id:
                raise ValueError("observation evidence and RGB-D episode are out of sequence")
            if frame_evidence.get("base_from_camera") is not None:
                # The perception pass may use a reviewed camera-pose revision.
                # Replay must project its saved object/scene poses through that
                # same transform, while the dataset's original pose stays intact.
                scene_pose = transform(frame_evidence["base_from_camera"])
                calibration_version = frame_evidence["calibration_version"]
        accepted = {}
        pose_status = {}
        masks = {}
        groups = {}
        proposals = {}
        raw_proposals = {}
        if frame_evidence:
            accepted = {
                role: {**estimate, "pose": estimate["pose"]}
                for role, estimate in frame_evidence["estimates"].items()
                if estimate["pose"] is not None
            }
            pose_status = {
                role: {
                    "evidence": estimate["evidence"],
                    "estimator": estimate["estimator"],
                    "source_time_s": estimate.get("source_time_s", time_s),
                    "source_frame_id": estimate.get("source_frame_id"),
                    "available_time_s": estimate.get("available_time_s"),
                    "reference_version": estimate.get("reference_version"),
                    "last_direct_observation_s": estimate.get("last_direct_observation_s"),
                }
                for role, estimate in frame_evidence["estimates"].items()
            }
            for role, runs in frame_evidence["masks_rle"].items():
                binary = np.zeros(height * width, dtype=np.uint8)
                for start, length in runs:
                    binary[start : start + length] = 1
                binary = binary.reshape(height, width)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boundary = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist() if contours else []
                masks[role] = {"frame_id": frame_id, "runs": runs, "boundary_xy": boundary}
                if role not in geometry["points_object_m"] and role in accepted and point_limit:
                    if row is None:
                        row = dataset[sample.row_index]
                    depth = row[depth_key].numpy().squeeze().astype(np.float64) * scale_m
                    valid = np.flatnonzero((binary != 0) & np.isfinite(depth) & (depth > 0))
                    if len(valid):
                        valid = valid[np.linspace(0, len(valid) - 1, min(len(valid), point_limit), dtype=int)]
                        yy, xx = np.divmod(valid, width)
                        zz = depth[yy, xx]
                        points_camera = np.stack(
                            (
                                (xx - intrinsics[0][2]) * zz / intrinsics[0][0],
                                (yy - intrinsics[1][2]) * zz / intrinsics[1][1],
                                zz,
                            ),
                            axis=1,
                        )
                        pose = np.asarray(accepted[role]["pose"], dtype=float)
                        points_scene = points_camera @ scene_pose[:3, :3].T + scene_pose[:3, 3]
                        points_object = (points_scene - pose[:3, 3]) @ pose[:3, :3]
                        geometry["points_object_m"][role] = points_object.round(6).tolist()
                        geometry["centers_object_m"][role] = points_object.mean(axis=0).round(6).tolist()
            groups = frame_evidence["groups"]
            proposals = {
                role: {
                    "evidence": proposal["evidence"],
                    "source_time_s": proposal.get("source_time_s", time_s),
                    "source_frame_id": proposal.get("source_frame_id"),
                    "available_time_s": proposal.get("available_time_s"),
                    "reference_version": proposal.get("reference_version"),
                }
                for role, proposal in frame_evidence["proposals"].items()
            }
            raw_proposals = {
                role: {
                    "pose": proposal["raw_pose"],
                    "frame_id": proposal.get("frame_id", frame_id),
                    "calibration_version": proposal.get(
                        "calibration_version", frame_evidence.get("calibration_version", "camera-frame")
                    ),
                    "failure": proposal["failure"],
                }
                for role, proposal in frame_evidence["proposals"].items()
                if proposal["raw_pose"] is not None and proposal["failure"]
            }
        frames.append(
            {
                "frame_index": frame_id,
                "source_row_index": sample.row_index,
                "source_camera_frame_id": sample.camera_frame_id,
                "evidence": {"available": True, **frame_evidence} if frame_evidence else None,
                "snapshot": {
                    "recorded_at_s": time_s,
                    "camera": {
                        "frame_id": frame_id,
                        "source_time_s": time_s,
                        "width": width,
                        "height": height,
                        "intrinsics": intrinsics,
                        "base_from_camera": scene_pose.tolist(),
                        "calibration_version": calibration_version,
                    },
                    "accepted": accepted,
                    "pose_status": pose_status,
                    "observed_masks": masks,
                    "scene_support": {"frame_id": frame_id, "groups": groups},
                    "proposals": proposals,
                    "raw_proposals": raw_proposals,
                    "demo": {"phase": "observation", "time_s": time_s},
                },
            }
        )
    annotation_warning = None
    try:
        annotations = load_annotations(root, episode)
    except ValueError as exc:
        annotations = None
        annotation_warning = str(exc)
    relationship_evidence = []
    for keyframe in (annotations or {}).get("keyframes", []):
        index = keyframe["frame_index"]
        if not 0 <= index < len(frames):
            continue
        estimates = frames[index]["snapshot"]["accepted"]
        moving = estimates.get(keyframe["moving_role"])
        reference = estimates.get(keyframe["reference_role"])
        relation = {
            "frame_index": index,
            "moving_role": keyframe["moving_role"],
            "reference_role": keyframe["reference_role"],
        }
        if moving and reference:
            relative = np.linalg.inv(np.asarray(reference["pose"])) @ np.asarray(moving["pose"])
            relation.update(
                {
                    "relative_pose": relative.tolist(),
                    "moving_evidence": moving["evidence"],
                    "reference_evidence": reference["evidence"],
                    "both_observed": moving["evidence"] == reference["evidence"] == "observed",
                }
            )
        else:
            relation.update(
                {
                    "relative_pose": None,
                    "both_observed": False,
                    "reason": "one or both object poses are unavailable",
                }
            )
        relationship_evidence.append(relation)
    return {
        "available": True,
        "kind": "observation_demo",
        "fps": (
            float(1.0 / np.median(np.diff([sample.source_time_s for sample in samples])))
            if len(samples) > 1
            else float(dataset.fps)
        ),
        "frames": frames,
        "source_episode": episode,
        "source": source.get("source"),
        "camera_features": {"rgb": rgb_key, "depth": depth_key},
        "geometry": geometry,
        "evidence_available": evidence_available,
        "processing": observation if evidence_available else None,
        "processing_config": processing_config,
        "robot_actions_available": has_robot_actions,
        "gripper_prediction_available": False,
        "annotations": annotations,
        "annotation_warning": annotation_warning,
        "relationship_evidence": relationship_evidence,
    }
