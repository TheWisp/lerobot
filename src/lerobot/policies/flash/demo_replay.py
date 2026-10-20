"""Review the exact spatial samples retained from a migrated RGB-D demonstration.

This is a read-only adapter for the sealed recording layout. It does not rerun
perception or invent per-frame tracker observations absent from the prepared
skill. The camera images are the same numbered RGB frames used by preparation.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from pathlib import Path

import cv2
import numpy as np

from .prepared import PreparedFlashSkill
from .skill import MotionMode, Relation


def _nearest_index(values: np.ndarray, target: float) -> int:
    right = min(bisect_left(values, target), len(values) - 1)
    if right and abs(values[right - 1] - target) < abs(values[right] - target):
        return right - 1
    return right


def _pose(pose: np.ndarray, *, evidence: str, estimator: str, source_time_s: float) -> dict:
    return {
        "pose": pose.tolist(),
        "evidence": evidence,
        "estimator": estimator,
        "source_time_s": source_time_s,
        "available_time_s": source_time_s,
    }


def build_demo_replay(
    prepared: PreparedFlashSkill,
    *,
    arm_times_s: np.ndarray,
    arm_tips: np.ndarray,
    camera_times_s: np.ndarray,
    intrinsics: np.ndarray,
    base_from_camera: np.ndarray,
    image_size: tuple[int, int],
    fps: float,
) -> dict:
    """Align saved skill samples to capture times without altering their evidence."""
    skill = prepared.skill
    arm_times_s = np.asarray(arm_times_s, dtype=float)
    camera_times_s = np.asarray(camera_times_s, dtype=float)
    arm_tips = np.asarray(arm_tips, dtype=float)
    if (
        not len(arm_times_s)
        or not len(camera_times_s)
        or arm_tips.shape != (len(arm_times_s), 4, 4)
        or not np.all(np.diff(arm_times_s) > 0)
        or not np.all(np.diff(camera_times_s) > 0)
    ):
        raise ValueError("demo camera and recorded arm need monotonic, aligned samples")
    width, height = image_size
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("demo replay needs image dimensions and positive FPS")

    intervals = skill.intervals
    starts = [interval.samples[0].time_s for interval in intervals]
    ends = [interval.samples[-1].time_s for interval in intervals]
    geometry = {
        "points_object_m": {
            model.points.role: model.points.points_object_m.tolist() for model in prepared.appearance_models
        },
        "centers_object_m": {
            model.points.role: model.points.points_object_m.mean(axis=0).tolist()
            for model in prepared.appearance_models
        },
    }
    camera_geometry = {
        "width": width,
        "height": height,
        "intrinsics": np.asarray(intrinsics, dtype=float).tolist(),
        "base_from_camera": np.asarray(base_from_camera, dtype=float).tolist(),
    }
    frames = []
    for frame_index, camera_time_s in enumerate(camera_times_s):
        arm_index = _nearest_index(arm_times_s, float(camera_time_s))
        interval_index = bisect_right(starts, camera_time_s) - 1
        interval = (
            intervals[interval_index]
            if 0 <= interval_index < len(intervals) and camera_time_s <= ends[interval_index]
            else None
        )
        accepted = {
            "__gripper__": _pose(
                arm_tips[arm_index],
                evidence="recorded",
                estimator="demo_ee",
                source_time_s=float(arm_times_s[arm_index]),
            )
        }
        held_prediction = None
        if interval is None:
            detail = {"phase": "outside_skill", "time_s": float(camera_time_s)}
        else:
            if interval.mode == MotionMode.CONNECTION:
                sample = interval.samples[-1]
                sample_description = "connection_endpoint"
            else:
                sample_index = _nearest_index(
                    np.asarray([item.time_s for item in interval.samples]), float(camera_time_s)
                )
                sample = interval.samples[sample_index]
                sample_description = "demonstrated_sample"
            if interval.relation != Relation.EE_TO_START:
                accepted[interval.reference_role] = _pose(
                    sample.demo_reference,
                    evidence=sample.reference_evidence,
                    estimator="demo_reference",
                    source_time_s=sample.time_s,
                )
            if interval.held_role is not None and sample.demo_held is not None:
                held_prediction = {
                    "role": interval.held_role,
                    **_pose(
                        sample.demo_held,
                        evidence=sample.held_evidence or "unverified",
                        estimator="demo_held",
                        source_time_s=sample.time_s,
                    ),
                }
            detail = {
                "phase": sample_description,
                "time_s": float(camera_time_s),
                "interval_index": interval_index,
                "interval_name": interval.name,
                "mode": interval.mode.value,
                "relation": interval.relation.value,
                "reference_role": interval.reference_role,
                "held_role": interval.held_role,
                "source_arm_frame_index": sample.source_frame_index,
                "sample_time_s": sample.time_s,
                "event_id": sample.event_id,
                "endpoint_requirement": interval.endpoint_requirement.value,
            }
        detail["nearest_recorded_arm_frame_index"] = arm_index
        detail["camera_arm_delta_s"] = float(camera_time_s - arm_times_s[arm_index])
        frames.append(
            {
                "frame_index": frame_index,
                "snapshot": {
                    "recorded_at_s": float(camera_time_s),
                    "camera": {
                        "frame_id": frame_index,
                        "source_time_s": float(camera_time_s),
                        **camera_geometry,
                    },
                    "accepted": accepted,
                    "held_prediction": held_prediction,
                    "demo": detail,
                },
            }
        )
    return {
        "available": True,
        "kind": "demo",
        "fps": fps,
        "geometry": geometry,
        "pose_constraints": {
            item.role: {
                "orientation_locked_to_baseline": item.lock_to_baseline,
                "ends_interchangeable": item.ends_interchangeable,
            }
            for item in prepared.axial_symmetries
        },
        "frames": frames,
        "source_episode": skill.source_episode,
        "skill_sample_count": sum(len(interval.samples) for interval in intervals),
        "raw_tracker_history": False,
    }


def _demo_annotation_path(root: Path, bundle: Path) -> Path:
    versioned = bundle.parent / "annotation.json"
    return versioned if versioned.is_file() else root / "flash" / "v1" / "import-annotations.json"


def load_legacy_demo_replay(dataset_root: Path, episode: int, bundle_path: Path) -> dict:
    """Load a prepared skill and its original aligned camera/arm clocks."""
    root = dataset_root.expanduser().resolve()
    bundle = bundle_path.expanduser().resolve()
    bundle.relative_to(root / "flash" / "v1" / "prepared")
    if bundle.name != "bundle.json" or not (root / "meta" / "info.json").is_file():
        raise ValueError("select a prepared FLASH skill beside a local dataset")
    prepared = PreparedFlashSkill.load(bundle)
    if prepared.skill.source_episode != episode:
        raise ValueError("prepared skill belongs to a different demo episode")
    annotation_path = _demo_annotation_path(root, bundle)
    if not annotation_path.is_file():
        raise ValueError("this demo has no saved capture calibration for aligned pose replay")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    calibration = json.loads(Path(annotation["calibration_path"]).expanduser().read_text(encoding="utf-8"))
    base_from_camera = np.asarray(calibration["camera"]["T_base_cam"], dtype=float)
    with np.load(root / "showservo_demo.npz", allow_pickle=False) as arm:
        arm_times_s = np.asarray(arm["t"], dtype=float)
        arm_tips = np.asarray(arm["tips"], dtype=float)
        origin_s = float(arm["t0"])
    camera_times_s = np.atleast_1d(np.loadtxt(root / "recording" / "times.txt")) - origin_s
    intrinsics = np.loadtxt(root / "recording" / "cam_K.txt").reshape(3, 3)
    image_dir = root / "recording" / "rgb"
    if any(not (image_dir / f"{index:06d}.jpg").is_file() for index in range(len(camera_times_s))):
        raise ValueError("demo RGB frames do not match its camera timestamps")
    first = cv2.imread(str(image_dir / "000000.jpg"), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError("first demo RGB frame cannot be decoded")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    result = build_demo_replay(
        prepared,
        arm_times_s=arm_times_s,
        arm_tips=arm_tips,
        camera_times_s=camera_times_s,
        intrinsics=intrinsics,
        base_from_camera=base_from_camera,
        image_size=(first.shape[1], first.shape[0]),
        fps=float(info["fps"]),
    )
    result["bundle_path"] = str(bundle)
    result["grasp_reference"] = {
        "role": annotation["held_role"],
        "camera_frame_index": annotation["grasp_reference_frame"],
        "grasp_time_s": annotation["grasp_time_s"],
    }
    return result
