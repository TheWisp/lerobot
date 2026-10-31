"""Read-only reference-view review for recordings made by the sealed rig importer."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .reference_view_candidates import MeasuredReferenceView, review_reference_view_candidates
from .surface_observation import move, observation, spatial_sample
from .surface_reference import RegisteredSurfaceModel
from .trajectory import Evidence, PoseEstimate
from .vision_pipeline import VisionPipelineConfig


def review_sealed_reference_view(
    root: Path, bundle_path: Path, episode: int, role: str, frame_index: int
) -> dict:
    """Compare a selected view without changing the saved demo or active reference."""
    from .prepared import PreparedFlashSkill

    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("choose a nonnegative demo camera frame")
    prepared = PreparedFlashSkill.load(bundle_path)
    if prepared.skill.source_episode != episode:
        raise ValueError("selected view belongs to a different demo episode")
    reference = next((item for item in prepared.references if item.role == role), None)
    if reference is None:
        raise ValueError("selected object is not in this prepared demo")
    if len(reference.snippets) != 1:
        raise ValueError("this review currently needs a single active reference view")
    annotation = json.loads((bundle_path.parent / "annotation.json").read_text(encoding="utf-8"))
    report = json.loads((bundle_path.parent / "preparation-report.json").read_text(encoding="utf-8"))
    if annotation["episode"] != prepared.skill.source_episode:
        raise ValueError("prepared annotation belongs to another demo episode")
    cache_path = (root / report["pose_trace_cache"]).resolve()
    if not cache_path.is_relative_to((root / "flash" / "v1" / "preparation-cache").resolve()):
        raise ValueError("prepared pose trace is outside this dataset")
    evidence_root = bundle_path.parent / "scene-evidence"
    if not cache_path.is_file() or not evidence_root.is_dir():
        raise ValueError("this prepared version lacks the pose or mask evidence needed for review")
    intrinsics = np.loadtxt(root / "recording" / "cam_K.txt").reshape(3, 3)
    calibration = json.loads(Path(annotation["calibration_path"]).expanduser().read_text(encoding="utf-8"))
    base_from_camera = np.asarray(calibration["camera"]["T_base_cam"], dtype=float)
    config = VisionPipelineConfig.from_settings(annotation, backend="appearance")

    def measured(name: str, index: int) -> MeasuredReferenceView | None:
        path = evidence_root / f"{index:06d}.json.gz"
        if not path.is_file():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            evidence = json.load(stream)
        if evidence["frame_index"] != index:
            raise ValueError("prepared mask evidence has a different source frame")
        runs = evidence["masks_rle"].get(name)
        if runs is None:
            return None
        depth = cv2.imread(str(root / "recording" / "depth" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.shape != tuple(evidence["image_size"]):
            raise ValueError(f"frame {index} has no aligned recorded depth")
        mask = np.zeros(depth.size, dtype=bool)
        for start, length in runs:
            if (
                type(start) is not int
                or type(length) is not int
                or start < 0
                or length < 0
                or start + length > mask.size
            ):
                raise ValueError(f"frame {index} has invalid object mask runs")
            mask[start : start + length] = True
        mask = mask.reshape(depth.shape)
        depth_m = depth.astype(np.float32) * 0.001
        observed = observation(depth_m, mask, intrinsics, asdict(config.observation))
        return MeasuredReferenceView(
            index, depth_m, observed["valid_image"], observed["points"], int(mask.sum())
        )

    snippet = reference.snippets[0]
    first_mask_path = (root / snippet.mask_artifact).resolve()
    if not first_mask_path.is_relative_to((root / "flash" / "v1" / "masks").resolve()):
        raise ValueError("prepared reference mask is outside this dataset")
    first_mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
    first_depth = cv2.imread(
        str(root / "recording" / "depth" / f"{snippet.frame_index:06d}.png"), cv2.IMREAD_UNCHANGED
    )
    if first_mask is None or first_depth is None or first_mask.shape != first_depth.shape:
        raise ValueError("prepared reference has no aligned mask and depth")
    model = next((item for item in prepared.surface_models if item.role == role), None)
    if model is None:
        # Older prepared bundles predate saved surface models. Rebuild only the
        # first view from their immutable reference mask and recorded depth.
        depth_m = first_depth.astype(np.float32) * 0.001
        observed = observation(depth_m, first_mask.astype(bool), intrinsics, asdict(config.observation))
        if len(observed["points"]) < 3:
            raise ValueError("prepared first view has insufficient reliable depth")
        points = move(observed["points"], snippet.object_from_camera)
        limit = config.surface_reference_max_points_per_view
        model = RegisteredSurfaceModel(role, reference.version, (points[spatial_sample(points, limit)],))

    with np.load(cache_path, allow_pickle=False) as cache:
        count = int(cache["frames"])
        if frame_index >= count:
            raise ValueError("selected frame is outside this prepared pose trace")
        estimates = {}
        for index in range(count):
            evidence = Evidence(str(cache[f"{role}_evidence"][index]))
            estimates[index] = {
                role: PoseEstimate(
                    role,
                    cache[f"{role}_poses"][index] if evidence != Evidence.LOST else None,
                    float(index),
                    float(index),
                    evidence,
                    reference.version,
                    tuple(bool(x) for x in cache[f"{role}_translation_supported"][index]),
                    tuple(bool(x) for x in cache[f"{role}_rotation_supported"][index]),
                    estimator=str(cache[f"{role}_estimator"][index]),
                    source_frame_id=index,
                )
            }
    summaries, details = review_reference_view_candidates(
        estimates,
        (model,),
        {role: [frame_index]},
        measured,
        intrinsics,
        base_from_camera,
        config.surface_reference_max_points_per_view,
    )
    comparisons = details[0]["comparisons"]
    reverse_worsened = sorted(
        (
            row
            for row in comparisons
            if row["first_view"]["reference_to_depth_p90_m"] is not None
            and row["combined_views"]["reference_to_depth_p90_m"] is not None
        ),
        key=lambda row: row["combined_views"]["reference_to_depth_p90_m"]
        - row["first_view"]["reference_to_depth_p90_m"],
        reverse=True,
    )
    return {
        "reference_frame_index": snippet.frame_index,
        "first_view_points": len(model.view_points_object_m[0]),
        "candidate": summaries[0],
        "largest_reverse_conflicts": reverse_worsened[:5],
        "disclaimer": "Same pose trace registered and checked this view; review cannot establish independent accuracy.",
    }
