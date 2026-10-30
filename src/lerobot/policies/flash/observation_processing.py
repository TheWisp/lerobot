"""Run a LeRobot RGB-D episode through FLASH perception, with or without robot actions.

This uses the same surface backend, causal scene estimator, and evidence writer
as a prepared FLASH demo. An identity camera frame is used for coordinates when
the source has no camera-to-robot calibration. This pass never infers a grasp
from robot actions; it measures vision and scene-group evidence only.
Masks are explicit inputs so either an annotation or a segmentation model may
supply them without changing the tracker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .appearance import Sam3MaskSource
from .causal_service import CausalTrackingConfig
from .geometry import transform
from .observation_dataset import (
    CameraSample,
    camera_feature_keys,
    depth_scale_m,
    episode_camera_samples,
    robot_actions_available,
    scene_from_camera,
)
from .offline_perception import POSE_TRACE_SELECTION_VERSION, extract_pose_trace
from .perception import ObjectReference, ReferenceSnippet, RGBDFrame
from .scene_evidence_writer import SCENE_EVIDENCE_CAPTURE_VERSION, SceneEvidenceWriter
from .vision_pipeline import VisionPipelineConfig, build_vision_backend


def default_processing_config() -> dict:
    """Editable prototype defaults for the observation processing UI.

    The causal values match the rig's documented trial-config example. They
    are processing parameters, not object or task rules, and every run stores
    the exact supplied config for later review.
    """
    return {
        "vision_pipeline": VisionPipelineConfig().to_dict(),
        "causal_tracking": asdict(
            CausalTrackingConfig(
                history_frames=120,
                queue_frames=60,
                min_inliers=12,
                max_frame_gap_s=0.1,
                fit_threshold_m=0.005,
            )
        ),
    }


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


def _camera_pose_candidates(
    audit_path: Path,
    root: Path,
    source_path: Path,
    mask_directories: dict[str, Path],
    episode: int,
    samples: list[CameraSample],
    recorded_pose_trace_sha256: str,
) -> tuple[list[np.ndarray | None], list[dict]]:
    """Load a controlled camera-pose comparison, never a production calibration.

    The audit's alternative must beat the source on held-out RGB tracks. It
    remains a hypothesis: neither those tracks nor a low surface residual are
    independent ground truth for a robot action.
    """
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema") != "flash-camera-pose-audit-v1"
        or audit.get("source") != str(root)
        or audit.get("episode") != episode
        or audit.get("source_metadata_sha256") != hashlib.sha256(source_path.read_bytes()).hexdigest()
        or audit.get("recorded_pose_trace_sha256") != recorded_pose_trace_sha256
        or not isinstance(audit.get("frames"), list)
        or len(audit["frames"]) != len(samples)
    ):
        raise ValueError("camera-pose audit does not match this recorded episode")
    mask_hash = hashlib.sha256()
    for frame_id in range(len(samples)):
        for role, directory in sorted(mask_directories.items()):
            mask_hash.update(role.encode() + b"\0" + (directory / f"{frame_id:06d}.png").read_bytes())
    if audit.get("mask_sha256") != mask_hash.hexdigest():
        raise ValueError("camera-pose audit masks no longer match the selected masks")
    candidates = []
    selections = []
    for frame_id, (sample, entry) in enumerate(zip(samples, audit["frames"], strict=True)):
        audited_time = entry.get("time_s") if isinstance(entry, dict) else None
        if (
            not isinstance(audited_time, (int, float))
            or entry.get("frame") != frame_id
            or not np.isclose(audited_time, sample.source_time_s, atol=1e-6)
        ):
            raise ValueError("camera-pose audit frame times no longer match this episode")
        fit = entry.get("pnp")
        if fit is None:
            candidates.append(None)
            selections.append({"frame": frame_id, "selected": "source", "reason": "no candidate"})
            continue
        source_error = fit.get("heldout_source") or {}
        fit_error = fit.get("heldout_fit") or {}
        source_median = source_error.get("median_px")
        fit_median = fit_error.get("median_px")
        useful = (
            isinstance(source_median, (int, float))
            and isinstance(fit_median, (int, float))
            and np.isfinite(source_median)
            and np.isfinite(fit_median)
            and fit_median < source_median
            and isinstance(fit.get("training_inliers"), (int, float))
            and fit["training_inliers"] >= 6
        )
        candidates.append(transform(fit["candidate_world_from_camera"]) if useful else None)
        selections.append(
            {
                "frame": frame_id,
                "selected": "background_rgb_candidate" if useful else "source",
                "source_heldout_median_px": source_median,
                "candidate_heldout_median_px": fit_median,
                "training_inliers": fit.get("training_inliers"),
            }
        )
    return candidates, selections


def _extract_to_directory(
    dataset_root: Path,
    mask_directories: dict[str, Path],
    config_bundle: Path,
    *,
    episode: int = 0,
    output: Path | None = None,
    prompts_by_role: dict[str, str] | None = None,
    seed_points_by_role: dict[str, tuple[float, float]] | None = None,
    camera_pose_audit: Path | None = None,
) -> Path:
    root = dataset_root.expanduser().resolve()
    source_path = root / "flash" / "v1" / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") not in {"perception_only", "robot_recording"}:
        raise ValueError("select a registered RGB-D episode")
    bundle_path = config_bundle.expanduser().resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    try:
        config_reference = bundle_path.relative_to(root)
    except ValueError:
        config_reference = bundle_path
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
    has_robot_actions = robot_actions_available(dataset.meta.features)
    if source["status"] == "robot_recording" and not has_robot_actions:
        raise ValueError("robot recording needs both measured state and action features")
    rgb_key, depth_key = camera_feature_keys(dataset.meta.features, source)
    scale_m = depth_scale_m(source)
    mask_source = (
        RecordedMaskSource(directories)
        if directories
        else Sam3MaskSource(
            prompts,
            camera_id=rgb_key,
            resolution=vision_config.sam_resolution,
            seed_points_by_role=seeds,
            detection_correction=vision_config.detection_correction.control(offline=True),
        )
    )
    roles = tuple(directories or prompts)
    camera_samples, source_trace_sha256 = episode_camera_samples(root, dataset, source, episode)
    audit_path = camera_pose_audit.expanduser().resolve() if camera_pose_audit is not None else None
    if audit_path is not None and (not directories or source.get("camera_pose_feature") is None):
        raise ValueError("camera-pose comparison needs saved masks and a moving-camera source")
    candidate_poses, camera_selections = (
        _camera_pose_candidates(
            audit_path,
            root,
            source_path,
            directories,
            episode,
            camera_samples,
            hashlib.sha256(
                np.asarray(
                    dataset.episode_column(source["camera_pose_feature"], episode), dtype=np.float64
                ).tobytes()
            ).hexdigest(),
        )
        if audit_path is not None
        else (None, None)
    )
    audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest() if audit_path is not None else None
    revised_calibration_version = (
        f"background-rgb-candidate:{audit_sha256[:12]}" if audit_sha256 is not None else None
    )
    intrinsics = np.asarray(source["camera_intrinsics"], dtype=np.float64)
    destination = output or root / "flash" / "v1" / "observation-evidence" / f"episode_{episode:06d}"
    identity = hashlib.sha256(
        source_path.read_bytes()
        + (source_trace_sha256 or "").encode()
        + bundle_path.read_bytes()
        + (audit_path.read_bytes() if audit_path is not None else b"")
        + json.dumps(
            {
                "mask_directories": {role: str(path) for role, path in directories.items()},
                "prompts_by_role": prompts,
                "seed_points_by_role": seeds,
            },
            sort_keys=True,
        ).encode()
        + f":pose:{POSE_TRACE_SELECTION_VERSION}:scene:{SCENE_EVIDENCE_CAPTURE_VERSION}".encode()
    ).hexdigest()
    writer = SceneEvidenceWriter(
        destination, total_frames=len(camera_samples), roles=roles, source_sha256=identity
    )
    backend = build_vision_backend(vision_config, mask_source, local_reference=True)
    references = tuple(
        ObjectReference(
            role,
            f"observation-episode-{episode}:{role}",
            (
                ReferenceSnippet(
                    source["repo_id"],
                    episode,
                    0,
                    role,
                    np.eye(4),
                    str(prompt),
                    revised_calibration_version
                    if revised_calibration_version is not None
                    else "source-world"
                    if source.get("camera_pose_feature")
                    else "fixed-scene-camera"
                    if source.get("fixed_scene_from_camera")
                    else "camera-frame",
                ),
            ),
        )
        for role, prompt in (directories or prompts).items()
    )

    def frames():
        for frame_id, sample in enumerate(camera_samples):
            row = dataset[sample.row_index]
            rgb = (row[rgb_key].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            depth = row[depth_key].numpy().squeeze().astype(np.float32)
            scene_pose, calibration_version = scene_from_camera(row, dataset.meta.features, source)
            if candidate_poses is not None and candidate_poses[frame_id] is not None:
                scene_pose = candidate_poses[frame_id]
            if revised_calibration_version is not None:
                calibration_version = revised_calibration_version
            yield RGBDFrame(
                frame_id,
                rgb,
                depth,
                intrinsics,
                scale_m,
                scene_pose,
                sample.source_time_s,
                sample.source_time_s,
                calibration_version,
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
                "source_trace_sha256": source_trace_sha256,
                "source_row_indices": [sample.row_index for sample in camera_samples],
                "source_camera_frame_ids": [sample.camera_frame_id for sample in camera_samples],
                "evidence_identity": identity,
                "coordinate_frame": source.get("coordinate_frame")
                or (
                    "world"
                    if source.get("camera_pose_feature") or source.get("fixed_scene_from_camera")
                    else "camera"
                ),
                "robot_actions_available": has_robot_actions,
                "gripper_prediction_available": False,
                "camera_pose_comparison": camera_selections,
                "camera_pose_audit_sha256": audit_sha256,
                "roles": list(roles),
                "mask_source": "recorded" if directories else "sam3",
                "prompts_by_role": prompts,
                "seed_points_by_role": seeds,
                "config_bundle": str(config_reference),
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


def extract(
    dataset_root: Path,
    mask_directories: dict[str, Path],
    config_bundle: Path,
    *,
    episode: int = 0,
    output: Path | None = None,
    prompts_by_role: dict[str, str] | None = None,
    seed_points_by_role: dict[str, tuple[float, float]] | None = None,
    camera_pose_audit: Path | None = None,
) -> Path:
    """Publish a complete pass while preserving the previous evidence on failure."""
    root = dataset_root.expanduser().resolve()
    if camera_pose_audit is not None and output is None:
        raise ValueError("camera-pose comparison needs an explicit output sidecar")
    destination = (
        (output or root / "flash" / "v1" / "observation-evidence" / f"episode_{episode:06d}")
        .expanduser()
        .resolve()
    )
    normal_destination = (root / "flash" / "v1" / "observation-evidence" / f"episode_{episode:06d}").resolve()
    if camera_pose_audit is not None and destination == normal_destination:
        raise ValueError("camera-pose comparison cannot replace normal observation evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-pending-", dir=destination.parent))
    backup = None
    try:
        _extract_to_directory(
            root,
            mask_directories,
            config_bundle,
            episode=episode,
            output=staging,
            prompts_by_role=prompts_by_role,
            seed_points_by_role=seed_points_by_role,
            camera_pose_audit=camera_pose_audit,
        )
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}-previous-", dir=destination.parent))
            backup.rmdir()  # safe-destruct: empty path just created by this call
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if backup is not None:
                backup.rename(destination)
                backup = None
            raise
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # safe-destruct: only this call's temporary evidence directory
        if backup is not None and backup.exists() and destination.exists():
            # safe-destruct: replaced evidence after complete publication; cleanup failure must not undo success.
            shutil.rmtree(backup, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("config_bundle", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--mask", action="append", default=[], metavar="ROLE=DIR")
    parser.add_argument("--prompt", action="append", default=[], metavar="ROLE=TEXT")
    parser.add_argument("--seed", action="append", default=[], metavar="ROLE=X,Y")
    parser.add_argument("--output", type=Path, help="Write a separate evidence sidecar for comparison")
    parser.add_argument(
        "--camera-pose-audit",
        type=Path,
        help="Diagnostic only: compare held-out RGB camera candidates without replacing source data",
    )
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
            camera_pose_audit=args.camera_pose_audit,
        )
    )


if __name__ == "__main__":
    main()
