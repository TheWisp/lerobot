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
from lerobot.policies.flash.appearance import Sam3MaskSource
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
    prompts_by_role: dict[str, str] | None = None,
    seed_points_by_role: dict[str, tuple[float, float]] | None = None,
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
    prompts = dict(prompts_by_role or {})
    seeds = dict(seed_points_by_role or {})
    if bool(directories) == bool(prompts):
        raise ValueError("provide either saved masks or SAM3 prompts for at least one object")
    if directories and (seeds or any(not role or not path.is_dir() for role, path in directories.items())):
        raise ValueError("saved masks need valid directories and cannot use SAM3 seed points")
    if set(seeds) - set(prompts):
        raise ValueError("each SAM3 seed point needs an object prompt")
    dataset = LeRobotDataset(source["repo_id"], root=root)
    rgb_key, depth_key = camera_feature_keys(dataset.meta.features, source)
    mask_source = (
        RecordedMaskSource(directories)
        if directories
        else Sam3MaskSource(
            prompts,
            camera_id=rgb_key,
            resolution=vision_config.sam_resolution,
            seed_points_by_role=seeds,
        )
    )
    roles = tuple(directories or prompts)
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
        + json.dumps(
            {
                "mask_directories": {role: str(path) for role, path in directories.items()},
                "prompts_by_role": prompts,
                "seed_points_by_role": seeds,
            },
            sort_keys=True,
        ).encode()
        + str(POSE_TRACE_SELECTION_VERSION).encode()
    ).hexdigest()
    writer = SceneEvidenceWriter(
        destination, total_frames=len(episode_rows), roles=roles, source_sha256=identity
    )
    backend = build_vision_backend(vision_config, mask_source, local_reference=True)
    references = tuple(
        ObjectReference(
            role,
            f"observation-episode-{episode}:{role}",
            (ReferenceSnippet(source["repo_id"], episode, 0, role, np.eye(4), str(prompt), "camera-frame"),),
        )
        for role, prompt in (directories or prompts).items()
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
                "roles": list(roles),
                "mask_source": "recorded" if directories else "sam3",
                "prompts_by_role": prompts,
                "seed_points_by_role": seeds,
                "config_bundle": str(bundle_path),
                "surface_reference_max_points_per_view": vision_config.surface_reference_max_points_per_view,
                "frames": len(trace),
                "observed_frames_by_role": {
                    role: sum(row[role].evidence.value == "observed" for row in trace) for role in roles
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
    parser.add_argument("--mask", action="append", default=[], metavar="ROLE=DIR")
    parser.add_argument("--prompt", action="append", default=[], metavar="ROLE=TEXT")
    parser.add_argument("--seed", action="append", default=[], metavar="ROLE=X,Y")
    parser.add_argument("--output", type=Path, help="Write a separate evidence sidecar for comparison")
    args = parser.parse_args()
    masks = {}
    for entry in args.mask:
        role, separator, path = entry.partition("=")
        if not separator or not role or not path or role in masks:
            parser.error("each --mask must be a unique ROLE=DIR")
        masks[role] = Path(path)
    prompts = {}
    for entry in args.prompt:
        role, separator, prompt = entry.partition("=")
        if not separator or not role or not prompt.strip() or role in prompts:
            parser.error("each --prompt must be a unique ROLE=TEXT")
        prompts[role] = prompt.strip()
    seeds = {}
    for entry in args.seed:
        role, separator, coordinates = entry.partition("=")
        try:
            x, y = (float(value) for value in coordinates.split(","))
        except ValueError:
            parser.error("each --seed must be ROLE=X,Y")
        if not separator or not role or role in seeds:
            parser.error("each --seed must have a unique role")
        seeds[role] = (x, y)
    print(
        extract(
            args.dataset_root,
            masks,
            args.config_bundle,
            episode=args.episode,
            output=args.output,
            prompts_by_role=prompts,
            seed_points_by_role=seeds,
        )
    )


if __name__ == "__main__":
    main()
