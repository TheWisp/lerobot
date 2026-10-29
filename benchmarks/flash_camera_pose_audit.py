"""Audit a recorded moving camera against persistent background RGB-D tracks.

The source camera transform is never changed. Background points from the first
frame are tracked in RGB; their measured pixels are compared with the pixels
predicted by the recorded camera trajectory. A held-out PnP fit is a diagnostic
alternative, not a replacement calibration or a claim that the scene is static.

Example::

    python benchmarks/flash_camera_pose_audit.py /path/to/episode \
      --mask object=/path/to/object-masks --mask target=/path/to/target-masks
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.observation_dataset import (
    camera_feature_keys,
    episode_camera_samples,
    scene_from_camera,
)
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.scene_groups import ScenePointTracker


def _reprojection_error(world: np.ndarray, xy: np.ndarray, camera_from_world: np.ndarray, k: np.ndarray):
    camera = world @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    positive = camera[:, 2] > 0
    error = np.full(len(world), np.inf)
    projected = np.column_stack(
        (
            camera[positive, 0] * k[0, 0] / camera[positive, 2] + k[0, 2],
            camera[positive, 1] * k[1, 1] / camera[positive, 2] + k[1, 2],
        )
    )
    error[positive] = np.linalg.norm(projected - xy[positive], axis=1)
    return error


def _summary(error: np.ndarray) -> dict[str, float] | None:
    error = error[np.isfinite(error)]
    if not len(error):
        return None
    return {"median_px": float(np.median(error)), "p90_px": float(np.percentile(error, 90))}


def compare_pose_to_background(
    world: np.ndarray,
    xy: np.ndarray,
    ids: np.ndarray,
    world_from_camera: np.ndarray,
    intrinsics: np.ndarray,
    *,
    pnp_reprojection_px: float = 2.0,
) -> dict:
    """Compare a supplied pose with a candidate fitted from other tracked points.

    Every fifth point is held out of PnP. Its error is reported separately so
    an attractive fit on the points it just used cannot masquerade as evidence.
    """
    if world.shape != (len(xy), 3) or xy.shape != (len(ids), 2):
        raise ValueError("tracked background needs aligned world points, pixels and IDs")
    if not np.isfinite(world).all() or not np.isfinite(xy).all() or pnp_reprojection_px <= 0:
        raise ValueError("tracked background and PnP resolution must be finite")
    if len(ids) == 0:
        return {"tracks": 0, "source": None, "pnp": None}
    camera_from_world = np.linalg.inv(world_from_camera)
    source_error = _reprojection_error(world, xy, camera_from_world, intrinsics)
    result = {"tracks": len(ids), "source": _summary(source_error), "pnp": None}
    holdout = np.asarray(ids, dtype=np.int64) % 5 == 0
    training = ~holdout
    if training.sum() < 6 or holdout.sum() < 3:
        return result
    found, rotation, translation, inliers = cv2.solvePnPRansac(
        world[training].astype(np.float64),
        xy[training].astype(np.float64),
        intrinsics.astype(np.float64),
        None,
        iterationsCount=200,
        reprojectionError=pnp_reprojection_px,
        confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not found or inliers is None or len(inliers) < 6:
        return result
    matrix, _ = cv2.Rodrigues(rotation)
    fitted_camera_from_world = np.eye(4)
    fitted_camera_from_world[:3, :3] = matrix
    fitted_camera_from_world[:3, 3] = translation[:, 0]
    correction = fitted_camera_from_world @ world_from_camera
    angle = np.arccos(np.clip((np.trace(correction[:3, :3]) - 1) / 2, -1, 1))
    result["pnp"] = {
        "training_inliers": len(inliers),
        "heldout_tracks": int(holdout.sum()),
        "heldout_source": _summary(source_error[holdout]),
        "heldout_fit": _summary(
            _reprojection_error(world[holdout], xy[holdout], fitted_camera_from_world, intrinsics)
        ),
        "source_delta_translation_m": float(np.linalg.norm(correction[:3, 3])),
        "source_delta_rotation_deg": float(np.degrees(angle)),
        "candidate_world_from_camera": np.linalg.inv(fitted_camera_from_world).tolist(),
    }
    return result


def audit(root: Path, mask_directories: dict[str, Path], *, episode: int = 0) -> dict:
    """Measure source camera-pose consistency on an ordinary RGB-D episode."""
    root = root.expanduser().resolve()
    source = json.loads((root / "flash/v1/source.json").read_text(encoding="utf-8"))
    source_metadata_sha256 = hashlib.sha256((root / "flash/v1/source.json").read_bytes()).hexdigest()
    if source.get("camera_pose_feature") is None:
        raise ValueError("camera-pose audit needs a measured moving-camera trajectory")
    if not mask_directories:
        raise ValueError("camera-pose audit needs object exclusion masks")
    dataset = LeRobotDataset(source["repo_id"], root=root, depth_output_unit="m", video_backend="pyav")
    samples, _ = episode_camera_samples(root, dataset, source, episode)
    recorded_pose_trace = np.asarray(
        dataset.episode_column(source["camera_pose_feature"], episode), dtype=np.float64
    )
    rgb_key, depth_key = camera_feature_keys(dataset.meta.features, source)
    intrinsics = np.asarray(source["camera_intrinsics"], dtype=np.float64)
    tracker = ScenePointTracker(image_scale=1.0)
    anchor: dict[int, np.ndarray] = {}
    masks_digest = hashlib.sha256()
    entries = []
    for frame_id, sample in enumerate(samples):
        row = dataset[sample.row_index]
        pose, version = scene_from_camera(row, dataset.meta.features, source)
        rgb = np.rint(np.moveaxis(row[rgb_key].numpy(), 0, -1) * 255).clip(0, 255).astype(np.uint8)
        depth = row[depth_key].numpy().squeeze().astype(np.float32)
        frame = RGBDFrame(
            frame_id, rgb, depth, intrinsics, 1.0, pose, sample.source_time_s, sample.source_time_s, version
        )
        tracks = tracker.update(frame)
        excluded = np.zeros(depth.shape, dtype=bool)
        for role, directory in sorted(mask_directories.items()):
            path = directory / f"{frame_id:06d}.png"
            masks_digest.update(role.encode() + b"\0" + path.read_bytes())
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != depth.shape:
                raise ValueError(f"{role}: mask {frame_id} is missing or not depth-aligned")
            excluded |= mask != 0
        outside = ~excluded[tracks.xy[:, 1], tracks.xy[:, 0]]
        if frame_id == 0:
            anchor = {
                int(point_id): xyz.copy()
                for point_id, xyz in zip(tracks.ids[outside], tracks.base_xyz[outside], strict=True)
            }
        indices = [i for i, point_id in enumerate(tracks.ids) if outside[i] and int(point_id) in anchor]
        ids = tracks.ids[indices]
        world = np.asarray([anchor[int(point_id)] for point_id in ids], dtype=np.float64).reshape(-1, 3)
        comparison = compare_pose_to_background(world, tracks.xy[indices], ids, pose, intrinsics)
        entries.append({"frame": frame_id, "time_s": sample.source_time_s, **comparison})
    return {
        "schema": "flash-camera-pose-audit-v1",
        "source": str(root),
        "source_metadata_sha256": source_metadata_sha256,
        "recorded_pose_trace_sha256": hashlib.sha256(recorded_pose_trace.tobytes()).hexdigest(),
        "mask_sha256": masks_digest.hexdigest(),
        "episode": episode,
        "reference_frame": 0,
        "initial_background_tracks": len(anchor),
        "assumption": "the first frame's unmasked tracked scene points remain stationary",
        "frames": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--mask", action="append", required=True, metavar="ROLE=DIR")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    masks = {}
    for item in args.mask:
        role, separator, directory = item.partition("=")
        if not separator or not role or not directory or role in masks:
            parser.error("each mask must have a unique ROLE=DIR")
        masks[role] = Path(directory).expanduser().resolve()
    result = audit(args.dataset_root, masks, episode=args.episode)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
