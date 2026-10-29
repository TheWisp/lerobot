"""Run an observation-only LeRobot RGB-D episode through FLASH perception.

This uses the same surface backend, causal scene estimator, and evidence writer
as a prepared FLASH demo. An identity camera frame is used for coordinates when
the source has no camera-to-robot calibration; no gripper pose is inferred.
Masks are explicit inputs so either an annotation or a segmentation model may
supply them without changing the tracker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.causal_service import CausalTrackingConfig
from lerobot.policies.flash.observation_dataset import camera_feature_keys
from lerobot.policies.flash.offline_perception import POSE_TRACE_SELECTION_VERSION, extract_pose_trace
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.scene_evidence_writer import SceneEvidenceWriter
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend


class RecordedMaskSource:
    """Align one Boolean instance mask per role and source camera frame."""

    def __init__(self, directories: dict[str, Path]):
        self.directories = directories

    def masks(self, frame: RGBDFrame) -> dict[str, np.ndarray]:
        result = {}
        for role, directory in self.directories.items():
            mask = cv2.imread(str(directory / f"{frame.frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != frame.depth.shape:
                raise ValueError(f"{role}: mask {frame.frame_id} is missing or does not align with RGB-D")
            result[role] = mask != 0
        return result


def extract(
    dataset_root: Path,
    mask_directories: dict[str, Path],
    config_bundle: Path,
    *,
    episode: int = 0,
    output: Path | None = None,
) -> Path:
    root = dataset_root.expanduser().resolve()
    source_path = root / "flash" / "v1" / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "perception_only":
        raise ValueError("select an observation-only RGB-D episode")
    bundle_path = config_bundle.expanduser().resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    vision_config = VisionPipelineConfig.from_dict(bundle["vision_pipeline"])
    causal_config = CausalTrackingConfig.from_config(bundle["causal_tracking"])
    directories = {role: path.expanduser().resolve() for role, path in mask_directories.items()}
    if not directories or any(not role or not path.is_dir() for role, path in directories.items()):
        raise ValueError("provide at least one named object with a mask directory")
    dataset = LeRobotDataset(source["repo_id"], root=root)
    rgb_key, depth_key = camera_feature_keys(dataset.meta.features, source)
    episode_rows = [
        index for index in range(dataset.num_frames) if int(dataset[index]["episode_index"]) == episode
    ]
    if not episode_rows:
        raise ValueError(f"episode {episode} is empty")
    intrinsics = np.asarray(source["camera_intrinsics"], dtype=np.float64)
    destination = output or root / "flash" / "v1" / "observation-evidence" / f"episode_{episode:06d}"
    identity = hashlib.sha256(
        source_path.read_bytes()
        + bundle_path.read_bytes()
        + json.dumps({role: str(path) for role, path in directories.items()}, sort_keys=True).encode()
        + str(POSE_TRACE_SELECTION_VERSION).encode()
    ).hexdigest()
    writer = SceneEvidenceWriter(
        destination, total_frames=len(episode_rows), roles=tuple(directories), source_sha256=identity
    )
    backend = build_vision_backend(vision_config, RecordedMaskSource(directories), local_reference=True)
    references = tuple(
        ObjectReference(
            role,
            f"observation-episode-{episode}:{role}",
            (
                ReferenceSnippet(
                    source["repo_id"], episode, 0, role, np.eye(4), str(directory), "camera-frame"
                ),
            ),
        )
        for role, directory in directories.items()
    )

    def frames():
        for frame_id, row_index in enumerate(episode_rows):
            row = dataset[row_index]
            rgb = (row[rgb_key].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            depth = row[depth_key].numpy().squeeze().astype(np.float32)
            source_time_s = float(row["flash.source_time_s"].item())
            yield RGBDFrame(
                frame_id,
                rgb,
                depth,
                intrinsics,
                0.001,
                np.eye(4),
                source_time_s,
                source_time_s,
                "camera-frame",
            )

    try:
        trace = extract_pose_trace(
            frames(),
            backend,
            references,
            causal_tracking=causal_config,
            on_frame_evidence=writer.record,
        )
    finally:
        backend.close()
    writer.finish()
    (destination / "observation.json").write_text(
        json.dumps(
            {
                "schema": "flash-observation-evidence-v1",
                "dataset_root": str(root),
                "episode": episode,
                "source_metadata_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "evidence_identity": identity,
                "coordinate_frame": "camera",
                "robot_actions_available": False,
                "gripper_prediction_available": False,
                "roles": list(directories),
                "config_bundle": str(bundle_path),
                "surface_reference_max_points_per_view": vision_config.surface_reference_max_points_per_view,
                "frames": len(trace),
                "observed_frames_by_role": {
                    role: sum(row[role].evidence.value == "observed" for row in trace) for role in directories
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("config_bundle", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--mask", action="append", required=True, metavar="ROLE=DIR")
    args = parser.parse_args()
    masks = {}
    for entry in args.mask:
        role, separator, path = entry.partition("=")
        if not separator or not role or not path or role in masks:
            parser.error("each --mask must be a unique ROLE=DIR")
        masks[role] = Path(path)
    print(extract(args.dataset_root, masks, args.config_bundle, episode=args.episode))


if __name__ == "__main__":
    main()
