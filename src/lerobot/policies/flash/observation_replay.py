"""Replay an RGB-D-only LeRobot episode in the FLASH review surface.

An observation episode can carry masks, scene groups and pose evidence without
claiming robot motion or an executable spatial skill.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .observation_annotations import load_annotations
from .observation_dataset import camera_feature_keys


def load_observation_replay(root: Path, episode: int = 0) -> dict:
    root = root.expanduser().resolve()
    source = json.loads((root / "flash" / "v1" / "source.json").read_text(encoding="utf-8"))
    if source.get("status") != "perception_only":
        raise ValueError("this dataset is not an observation-only FLASH import")
    dataset = LeRobotDataset(source["repo_id"], root=root)
    rgb_key, depth_key = camera_feature_keys(dataset.meta.features, source)
    indices = [
        index for index in range(dataset.num_frames) if int(dataset[index]["episode_index"]) == episode
    ]
    if not indices:
        raise ValueError(f"observation episode {episode} is empty")
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
        and manifest.get("total_frames") == len(indices)
        and observation.get("evidence_identity")
        and manifest.get("source_sha256") == observation["evidence_identity"]
        and isinstance(observation.get("surface_reference_max_points_per_view"), int)
        and observation.get("source_metadata_sha256")
        == hashlib.sha256((root / "flash" / "v1" / "source.json").read_bytes()).hexdigest()
    )
    first = dataset[indices[0]]
    height, width = first[depth_key].numpy().squeeze().shape
    intrinsics = source["camera_intrinsics"]
    identity = np.eye(4).tolist()
    frames = []
    geometry = {"points_object_m": {}, "centers_object_m": {}}
    point_limit = None
    if evidence_available:
        point_limit = int(observation["surface_reference_max_points_per_view"])
    for frame_id, row_index in enumerate(indices):
        row = dataset[row_index]
        time_s = float(row["flash.source_time_s"].item())
        frame_evidence = None
        if evidence_available:
            with gzip.open(sidecar / f"{frame_id:06d}.json.gz", "rt", encoding="utf-8") as stream:
                frame_evidence = json.load(stream)
            if frame_evidence["frame_index"] != frame_id:
                raise ValueError("observation evidence and RGB-D episode are out of sequence")
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
                role: {"evidence": estimate["evidence"], "estimator": estimate["estimator"]}
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
                    depth = row[depth_key].numpy().squeeze().astype(np.float64) / 1000
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
                        points_object = (points_camera - pose[:3, 3]) @ pose[:3, :3]
                        geometry["points_object_m"][role] = points_object.round(6).tolist()
                        geometry["centers_object_m"][role] = points_object.mean(axis=0).round(6).tolist()
            groups = frame_evidence["groups"]
            proposals = {
                role: {"evidence": proposal["evidence"], "source_time_s": time_s}
                for role, proposal in frame_evidence["proposals"].items()
            }
            raw_proposals = {
                role: {
                    "pose": proposal["raw_pose"],
                    "frame_id": frame_id,
                    "calibration_version": "camera-frame",
                    "failure": proposal["failure"],
                }
                for role, proposal in frame_evidence["proposals"].items()
                if proposal["raw_pose"] is not None and proposal["failure"]
            }
        frames.append(
            {
                "frame_index": frame_id,
                "evidence": {"available": True, **frame_evidence} if frame_evidence else None,
                "snapshot": {
                    "recorded_at_s": time_s,
                    "camera": {
                        "frame_id": frame_id,
                        "source_time_s": time_s,
                        "width": width,
                        "height": height,
                        "intrinsics": intrinsics,
                        "base_from_camera": identity,
                        "calibration_version": "camera-frame",
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
        "fps": float(dataset.fps),
        "frames": frames,
        "source_episode": episode,
        "source": source.get("source"),
        "camera_features": {"rgb": rgb_key, "depth": depth_key},
        "geometry": geometry,
        "evidence_available": evidence_available,
        "robot_actions_available": False,
        "gripper_prediction_available": False,
        "annotations": annotations,
        "annotation_warning": annotation_warning,
        "relationship_evidence": relationship_evidence,
    }
