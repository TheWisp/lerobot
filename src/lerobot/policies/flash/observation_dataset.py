"""Read synchronized RGB-D from an ordinary LeRobot episode, with or without actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def source_time_s(row: dict, features: dict, source: dict) -> float:
    """Use the recorded camera clock when supplied, otherwise the episode clock."""
    key = source.get(
        "source_time_feature", "flash.source_time_s" if "flash.source_time_s" in features else "timestamp"
    )
    if key not in features:
        raise ValueError(f"recorded camera time feature {key!r} is missing")
    value = np.asarray(row[key], dtype=np.float64).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError("recorded camera time must be one finite value per frame")
    return float(value[0])


def depth_scale_m(source: dict) -> float:
    """Convert the saved depth samples to metres without guessing their unit."""
    unit = source.get("depth_unit")
    if unit == "millimetres":
        return 0.001
    if unit == "metres":
        return 1.0
    raise ValueError("RGB-D source metadata must declare depth_unit as metres or millimetres")


def robot_actions_available(features: dict) -> bool:
    """Presence of action and measured state is data, not a pose-fallback claim."""
    return "action" in features and "observation.state" in features


def register_rgbd_dataset(
    root: Path,
    *,
    repo_id: str,
    camera_intrinsics: np.ndarray,
    depth_unit: str,
    rgb_feature: str | None = None,
    depth_feature: str | None = None,
    source_time_feature: str | None = None,
    camera_pose_feature: str | None = None,
    fixed_scene_from_camera: np.ndarray | None = None,
    stationary_camera: bool = False,
    source_name: str = "local RGB-D recording",
) -> Path:
    """Register existing LeRobot data for the shared RGB-D scene pipeline.

    No action, robot state, or image data is rewritten. Camera geometry and
    timing remain explicit rather than inferred from a dataset's robot type.
    """
    root = root.expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info["features"]
    source = {"rgb_feature": rgb_feature, "depth_feature": depth_feature}
    rgb_key, depth_key = camera_feature_keys(features, source)
    depth_scale_m({"depth_unit": depth_unit})
    declared_depth_unit = features[depth_key].get("info", {}).get("depth_unit")
    known_units = {"mm": "millimetres", "m": "metres"}
    if declared_depth_unit in known_units and known_units[declared_depth_unit] != depth_unit:
        raise ValueError("depth unit conflicts with the dataset's depth feature")
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    if (
        intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0, 0, 1])
    ):
        raise ValueError("camera intrinsics must be a finite 3x3 pixel calibration")
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("dataset identity is required")
    if source_time_feature is not None and source_time_feature not in features:
        raise ValueError("source time feature is missing from the dataset")
    if source_time_feature is None and "flash.source_time_s" not in features and "timestamp" not in features:
        raise ValueError("dataset needs a recorded camera time or episode timestamp")
    if camera_pose_feature is not None and camera_pose_feature not in features:
        raise ValueError("camera pose feature is missing from the dataset")
    if sum((camera_pose_feature is not None, fixed_scene_from_camera is not None, stationary_camera)) != 1:
        raise ValueError("declare a camera trajectory, a fixed scene transform, or a stationary camera")
    fixed_pose = None
    if fixed_scene_from_camera is not None:
        fixed_pose, _ = scene_from_camera(
            {}, {}, {"fixed_scene_from_camera": np.asarray(fixed_scene_from_camera).tolist()}
        )
    extension = root / "flash" / "v1"
    extension.mkdir(parents=True, exist_ok=True)
    destination = extension / "source.json"
    if destination.exists():
        raise FileExistsError(f"RGB-D source already registered at {destination}")
    payload = {
        "schema": "registered-rgbd-source-v1",
        "status": "robot_recording" if robot_actions_available(features) else "perception_only",
        "repo_id": repo_id,
        "source": source_name,
        "camera_intrinsics": intrinsics.tolist(),
        "rgb_feature": rgb_key,
        "depth_feature": depth_key,
        "depth_unit": depth_unit,
        "source_time_feature": source_time_feature
        or ("flash.source_time_s" if "flash.source_time_s" in features else "timestamp"),
        "camera_pose_feature": camera_pose_feature,
        "fixed_scene_from_camera": fixed_pose.tolist() if fixed_pose is not None else None,
        "camera_motion_assumption": "stationary" if camera_pose_feature is None else None,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def camera_feature_keys(features: dict, source: dict) -> tuple[str, str]:
    images = {
        key: value
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") in {"video", "image"}
    }
    depth = [key for key, feature in images.items() if feature.get("info", {}).get("is_depth_map")]
    rgb = [key for key in images if key not in depth]
    requested_rgb = source.get("rgb_feature")
    requested_depth = source.get("depth_feature")
    if requested_rgb is not None and requested_rgb not in rgb:
        raise ValueError("recorded RGB feature does not match dataset metadata")
    if requested_depth is not None and requested_depth not in depth:
        raise ValueError("recorded depth feature does not match dataset metadata")
    if requested_rgb is None and len(rgb) != 1:
        raise ValueError("choose one RGB image feature in the observation source metadata")
    if requested_depth is None and len(depth) != 1:
        raise ValueError("choose one aligned depth image feature in the observation source metadata")
    rgb_key, depth_key = requested_rgb or rgb[0], requested_depth or depth[0]
    if tuple(features[rgb_key]["shape"][:2]) != tuple(features[depth_key]["shape"][:2]):
        raise ValueError("RGB and depth features have different image sizes")
    return rgb_key, depth_key


def scene_from_camera(row: dict, features: dict, source: dict) -> tuple[np.ndarray, str]:
    """Return the camera pose on the recorded scene clock, or a fixed-camera frame."""
    pose_feature = source.get("camera_pose_feature")
    fixed_pose = source.get("fixed_scene_from_camera")
    if pose_feature is not None and fixed_pose is not None:
        raise ValueError("choose either a recorded camera trajectory or a fixed camera transform")
    if pose_feature is None:
        if source.get("camera_motion_assumption", "stationary") != "stationary":
            raise ValueError("moving-camera RGB-D needs a recorded camera pose for scene tracking")
        if fixed_pose is None:
            return np.eye(4), "camera-frame"
        pose = np.asarray(fixed_pose, dtype=np.float64)
        version = "fixed-scene-camera"
    else:
        if pose_feature not in features:
            raise ValueError("recorded camera pose feature is missing from the dataset")
        pose = np.asarray(row[pose_feature], dtype=np.float64)
        version = "source-world"
    if pose.size != 16:
        raise ValueError("recorded camera pose must contain one 4x4 transform")
    pose = pose.reshape(4, 4)
    rotation = pose[:3, :3]
    if (
        not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)
    ):
        raise ValueError("recorded camera pose is not a finite rigid transform")
    return pose, version


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register an existing LeRobot RGB-D dataset for FLASH vision"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--camera-intrinsics", required=True, type=Path, help="3x3 pixel calibration text file"
    )
    parser.add_argument("--depth-unit", required=True, choices=("metres", "millimetres"))
    parser.add_argument("--rgb-feature")
    parser.add_argument("--depth-feature")
    parser.add_argument("--source-time-feature")
    camera = parser.add_mutually_exclusive_group(required=True)
    camera.add_argument("--camera-pose-feature", help="Per-frame 4x4 scene-from-camera feature")
    camera.add_argument("--fixed-scene-from-camera", type=Path, help="Fixed 4x4 transform text file")
    camera.add_argument("--stationary-camera", action="store_true", help="Use the camera as the scene frame")
    parser.add_argument("--source-name", default="local RGB-D recording")
    args = parser.parse_args()
    print(
        register_rgbd_dataset(
            args.dataset_root,
            repo_id=args.repo_id,
            camera_intrinsics=np.loadtxt(args.camera_intrinsics),
            depth_unit=args.depth_unit,
            rgb_feature=args.rgb_feature,
            depth_feature=args.depth_feature,
            source_time_feature=args.source_time_feature,
            camera_pose_feature=args.camera_pose_feature,
            fixed_scene_from_camera=(
                np.loadtxt(args.fixed_scene_from_camera) if args.fixed_scene_from_camera is not None else None
            ),
            stationary_camera=args.stationary_camera,
            source_name=args.source_name,
        )
    )


if __name__ == "__main__":
    main()
