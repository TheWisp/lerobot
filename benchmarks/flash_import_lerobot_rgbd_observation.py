"""Copy one LeRobot RGB-D episode into an actionless, depth-aligned observation.

This adapter is for external recordings whose RGB and depth grids share rays
but have different resolutions. The source must explicitly declare that
relationship with ``--same-rays``; the source camera calibration must be for
the depth grid. Robot or human action fields are intentionally not copied.

Example::

    python benchmarks/flash_import_lerobot_rgbd_observation.py \
      OWNER/SOURCE 0 /path/to/observation --repo-id local/observation \
      --rgb-feature observation.images.ego \
      --depth-feature observation.images.depth \
      --camera-intrinsics /path/to/depth_K.txt \
      --camera-pose-feature observation.camera_pose --same-rays \
      --mask-feature observation.images.object_mask --mask object=0 --mask target=2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.geometry import transform


def _image(row: dict, feature: str, *, channels: int) -> np.ndarray:
    value = row[feature]
    image = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    if image.ndim != 3 or image.shape[0] != channels:
        raise ValueError(f"{feature} must decode as a {channels}-channel image")
    return np.moveaxis(image, 0, -1)


def _depth_mm(row: dict, feature: str) -> np.ndarray:
    depth = _image(row, feature, channels=1)[:, :, 0]
    if np.any(np.isfinite(depth) & (depth < 0)):
        raise ValueError("depth contains negative values")
    result = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0)
    if np.any(depth[valid] > np.iinfo(np.uint16).max / 1000):
        raise ValueError("depth exceeds the millimetre video range")
    result[valid] = np.rint(depth[valid] * 1000).astype(np.uint16)
    return result


def _aligned_image(image: np.ndarray, depth_shape: tuple[int, int], same_rays: bool) -> np.ndarray:
    height, width = depth_shape
    if image.shape[:2] == depth_shape:
        return image
    source_height, source_width = image.shape[:2]
    if not same_rays or source_height % height or source_width % width:
        raise ValueError("RGB and depth need equal grids or an explicit same-rays integer downscale")
    scale_y, scale_x = source_height // height, source_width // width
    if scale_y != scale_x or scale_x <= 1:
        raise ValueError("RGB and depth must have the same pixel scale in both axes")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def import_episode(
    source_repo_id: str,
    source_episode: int,
    destination: Path,
    *,
    repo_id: str,
    rgb_feature: str,
    depth_feature: str,
    camera_intrinsics: np.ndarray,
    camera_pose_feature: str | None = None,
    pose_camera_from_rgbd_camera: np.ndarray | None = None,
    mask_feature: str | None = None,
    mask_channels: dict[str, int] | None = None,
    mask_threshold: float = 0.5,
    same_rays: bool = False,
    source_root: Path | None = None,
) -> Path:
    """Preserve RGB-D and camera motion, dropping every action-related feature."""
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    if (
        intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0, 0, 1])
    ):
        raise ValueError("camera intrinsics must describe the depth grid in pixels")
    pose_basis = transform(pose_camera_from_rgbd_camera) if pose_camera_from_rgbd_camera is not None else None
    if pose_basis is not None and camera_pose_feature is None:
        raise ValueError("camera-frame alignment needs a source camera pose feature")
    masks = dict(mask_channels or {})
    if bool(mask_feature) != bool(masks) or len(set(masks.values())) != len(masks):
        raise ValueError("choose a mask feature and distinct channels for each named object")
    if not np.isfinite(mask_threshold) or not 0 < mask_threshold < 1:
        raise ValueError("mask threshold must be between zero and one")
    if any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", role)
        or isinstance(channel, bool)
        or not isinstance(channel, int)
        or channel not in range(3)
        for role, channel in masks.items()
    ):
        raise ValueError("each mask must name an object and an RGB channel 0, 1 or 2")
    cameras = [rgb_feature, depth_feature]
    if mask_feature:
        cameras.append(mask_feature)
    source = LeRobotDataset(
        source_repo_id,
        root=source_root,
        episodes=[source_episode],
        cameras=cameras,
        depth_output_unit="m",
        video_backend="pyav",
    )
    features = source.meta.features
    for feature in (rgb_feature, depth_feature, camera_pose_feature, mask_feature):
        if feature and feature not in features:
            raise ValueError(f"source feature {feature!r} is missing")
    if not features[depth_feature].get("info", {}).get("is_depth_map"):
        raise ValueError("source depth feature is not declared as a depth map")
    first = source[0]
    depth = _depth_mm(first, depth_feature)
    height, width = depth.shape
    _aligned_image(_image(first, rgb_feature, channels=3), depth.shape, same_rays)
    if mask_feature:
        mask_shape = _image(first, mask_feature, channels=3).shape[:2]
        if mask_shape not in {depth.shape, _image(first, rgb_feature, channels=3).shape[:2]}:
            raise ValueError("object masks must share the RGB or depth pixel grid")
    output_features = {
        "observation.images.rgb": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.depth": {
            "dtype": "video",
            "shape": (height, width, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        },
        "flash.source_time_s": {"dtype": "float64", "shape": (1,), "names": None},
        "flash.source_frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    if camera_pose_feature:
        output_features["flash.world_from_camera"] = {"dtype": "float64", "shape": (16,), "names": None}
        if pose_basis is not None:
            output_features["flash.source_world_from_pose_camera"] = {
                "dtype": "float64",
                "shape": (16,),
                "names": None,
            }
    depth_encoder = DepthEncoderConfig(depth_min=0.0)
    result = LeRobotDataset.create(
        repo_id=repo_id,
        fps=source.fps,
        features=output_features,
        root=destination,
        robot_type="external_rgbd_observation",
        depth_encoder=depth_encoder,
    )
    mask_root = destination / "flash" / "v1" / "imported-masks"
    for role in masks:
        (mask_root / role).mkdir(parents=True, exist_ok=True)
    source_shapes = {}
    previous_time = -np.inf
    for index in range(len(source)):
        row = first if index == 0 else source[index]
        depth = _depth_mm(row, depth_feature)
        if depth.shape != (height, width):
            raise ValueError(f"frame {index}: depth size changed")
        if np.any(depth > depth_encoder.depth_max * 1000):
            raise ValueError(f"frame {index}: depth exceeds output video range")
        rgb_original = _image(row, rgb_feature, channels=3)
        rgb = _aligned_image(rgb_original, depth.shape, same_rays)
        source_shapes = {"rgb_hw": list(rgb_original.shape[:2]), "depth_hw": [height, width]}
        time_s = float(row["timestamp"])
        if not np.isfinite(time_s) or time_s <= previous_time:
            raise ValueError(f"frame {index}: camera time did not advance")
        previous_time = time_s
        source_frame = int(row["frame_index"])
        output = {
            "task": f"RGB-D observation from {source_repo_id} episode {source_episode}",
            "observation.images.rgb": np.rint(rgb * 255).clip(0, 255).astype(np.uint8),
            "observation.images.depth": depth[:, :, None],
            "flash.source_time_s": np.asarray([time_s], dtype=np.float64),
            "flash.source_frame_index": np.asarray([source_frame], dtype=np.int64),
        }
        if camera_pose_feature:
            try:
                raw_pose = transform(np.asarray(row[camera_pose_feature], dtype=np.float64).reshape(4, 4))
            except ValueError as exc:
                raise ValueError(f"frame {index}: invalid world-from-camera transform") from exc
            if pose_basis is not None:
                output["flash.source_world_from_pose_camera"] = raw_pose.reshape(16)
            output["flash.world_from_camera"] = (
                raw_pose @ pose_basis if pose_basis is not None else raw_pose
            ).reshape(16)
        result.add_frame(output)
        if mask_feature:
            channels = _image(row, mask_feature, channels=3)
            for role, channel in masks.items():
                selected = channels[:, :, channel]
                if selected.shape != depth.shape:
                    selected = cv2.resize(selected, (width, height), interpolation=cv2.INTER_NEAREST)
                path = mask_root / role / f"{index:06d}.png"
                if not cv2.imwrite(str(path), (selected > mask_threshold).astype(np.uint8) * 255):
                    raise OSError(f"cannot save {path}")
    result.save_episode()
    result.finalize()
    metadata = {
        "schema": "external-rgbd-observation-v1",
        "status": "perception_only",
        "repo_id": repo_id,
        "source": f"{source_repo_id} episode {source_episode}",
        "source_repo_id": source_repo_id,
        "source_episode": source_episode,
        "camera_intrinsics": intrinsics.tolist(),
        "rgb_feature": "observation.images.rgb",
        "depth_feature": "observation.images.depth",
        "depth_unit": "millimetres",
        "source_time_feature": "flash.source_time_s",
        "camera_pose_feature": "flash.world_from_camera" if camera_pose_feature else None,
        "source_camera_pose_feature": (
            "flash.source_world_from_pose_camera" if pose_basis is not None else None
        ),
        "pose_camera_from_rgbd_camera": pose_basis.tolist() if pose_basis is not None else None,
        "camera_motion_assumption": None if camera_pose_feature else "stationary",
        "coordinate_frame": "world" if camera_pose_feature else "camera",
        "rgb_depth_alignment": "same_rays_area_downsample"
        if source_shapes["rgb_hw"] != source_shapes["depth_hw"]
        else "same_grid",
        "source_image_sizes": source_shapes,
        "source_mask_feature": mask_feature,
        "source_mask_channels": masks,
        "source_mask_threshold": mask_threshold,
        "output_depth_max_m": depth_encoder.depth_max,
        "missing_for_spatial_skill": [
            "measured_end_effector_trajectory",
            "robot_actions",
            "camera_to_robot_transform",
        ],
    }
    extension = destination / "flash" / "v1"
    extension.mkdir(parents=True, exist_ok=True)
    (extension / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo_id")
    parser.add_argument("source_episode", type=int)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--rgb-feature", required=True)
    parser.add_argument("--depth-feature", required=True)
    parser.add_argument("--camera-intrinsics", required=True, type=Path)
    parser.add_argument("--camera-pose-feature")
    parser.add_argument(
        "--pose-camera-from-rgbd-camera",
        type=Path,
        help="Explicit 4x4 transform from RGB-D pixel/depth axes to the source pose's camera axes",
    )
    parser.add_argument("--mask-feature")
    parser.add_argument("--mask", action="append", default=[], metavar="ROLE=CHANNEL")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--same-rays", action="store_true")
    args = parser.parse_args()
    masks = {}
    for entry in args.mask:
        role, separator, channel = entry.partition("=")
        if not separator or role in masks:
            parser.error("each mask must be a unique ROLE=CHANNEL")
        try:
            masks[role] = int(channel)
        except ValueError:
            parser.error("mask channel must be an integer 0, 1 or 2")
    print(
        import_episode(
            args.source_repo_id,
            args.source_episode,
            args.destination,
            repo_id=args.repo_id,
            source_root=args.source_root,
            rgb_feature=args.rgb_feature,
            depth_feature=args.depth_feature,
            camera_intrinsics=np.loadtxt(args.camera_intrinsics),
            camera_pose_feature=args.camera_pose_feature,
            pose_camera_from_rgbd_camera=(
                np.loadtxt(args.pose_camera_from_rgbd_camera)
                if args.pose_camera_from_rgbd_camera is not None
                else None
            ),
            mask_feature=args.mask_feature,
            mask_channels=masks,
            mask_threshold=args.mask_threshold,
            same_rays=args.same_rays,
        )
    )


if __name__ == "__main__":
    main()
