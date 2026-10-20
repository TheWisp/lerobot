"""Appearance-based pixel correspondences for the metric RGB-D pose backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from .continuity import PoseCoastLimit, PoseContinuityBackend
from .matched_rgbd import MatchedRGBDPoseBackend, ObjectPointModel, PixelMatches
from .perception import ObjectReference, PoseBackend, ReferenceSnippet, RGBDFrame
from .pose_gap import PoseGapBackend, PoseGapLimit
from .registration import transform_points, unproject_pixels, valid_depth_pixels
from .symmetry import AxialSymmetry


class PatchExtractor(Protocol):
    def extract(self, rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class MaskSource(Protocol):
    def masks(self, frame: RGBDFrame) -> dict[str, np.ndarray]: ...


class Sam3MaskSource:
    """Use LeRobot's shared SAM3 detector/tracker once per aligned live frame.

    Distinct roles need distinct prompts here. When two objects share the same
    description, a point-prompted instance source must supply their masks.
    """

    def __init__(
        self,
        prompts_by_role: dict[str, str],
        *,
        camera_id: str,
        device: str = "cuda",
        resolution: int | None = None,
        seed_points_by_role: dict[str, tuple[float, float]] | None = None,
        adapter_factory: Callable[..., object] | None = None,
    ) -> None:
        if (
            not prompts_by_role
            or not camera_id
            or any(
                not role or not prompt.strip() or "." in prompt for role, prompt in prompts_by_role.items()
            )
            or len(set(prompts_by_role.values())) != len(prompts_by_role)
        ):
            raise ValueError("SAM3 needs distinct nonempty role prompts without periods")
        self.prompts = dict(prompts_by_role)
        self.camera_id = camera_id
        self.device = device
        self.resolution = resolution
        self.adapter_factory = adapter_factory
        self.seed_points_by_role = dict(seed_points_by_role or {})
        if set(self.seed_points_by_role) - set(self.prompts):
            raise ValueError("point-seeded role lacks an object prompt")
        self._adapter = None
        self._last_frame_id: int | None = None
        self._calibration_version: str | None = None

    def masks(self, frame: RGBDFrame) -> dict[str, np.ndarray]:
        if self._adapter is None:
            if self.adapter_factory is None:
                from lerobot.overlays.adapters import Sam3TrackByDetectionAdapter

                factory = Sam3TrackByDetectionAdapter
            else:
                factory = self.adapter_factory
            self._adapter = factory(device=self.device, resolution=self.resolution)
            self._adapter.set_camera(self.camera_id)
            text_prompts = [
                prompt for role, prompt in self.prompts.items() if role not in self.seed_points_by_role
            ]
            self._adapter.set_control(
                {"prompt": " . ".join(text_prompts), "text_detection": bool(text_prompts)}
            )
            for sequence, (role, (x, y)) in enumerate(self.seed_points_by_role.items(), start=1):
                if not 0 <= x < frame.depth.shape[1] or not 0 <= y < frame.depth.shape[0]:
                    raise ValueError(f"{role}: point seed lies outside the aligned frame")
                self._adapter.set_control(
                    {
                        "click_seq": sequence,
                        "click_name": {self.camera_id: self.prompts[role]},
                        "clicks": {self.camera_id: [[x, y, 1]]},
                    }
                )
                # Each point has its own name. Apply it to this frame before
                # queuing the next seed, as the adapter stores one pending name.
                self._adapter.segment(np.ascontiguousarray(frame.rgb))
        if self._last_frame_id is not None and (
            frame.frame_id <= self._last_frame_id or frame.calibration_version != self._calibration_version
        ):
            self._adapter.reset()
        self._last_frame_id = frame.frame_id
        self._calibration_version = frame.calibration_version
        segmented = self._adapter.segment(np.ascontiguousarray(frame.rgb))
        return {
            role: np.asarray(segmented[prompt], dtype=bool)
            for role, prompt in self.prompts.items()
            if prompt in segmented
        }


@dataclass(frozen=True)
class ReferenceAppearanceFrame:
    snippet: ReferenceSnippet
    frame: RGBDFrame
    mask: np.ndarray


@dataclass(frozen=True)
class AppearanceModel:
    points: ObjectPointModel
    descriptors: np.ndarray

    def __post_init__(self) -> None:
        descriptors = _normalize(self.descriptors)
        if len(descriptors) != len(self.points.points_object_m):
            raise ValueError("one descriptor is required per object point")
        object.__setattr__(self, "descriptors", descriptors)


def _normalize(descriptors: np.ndarray) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("descriptors must be finite NxD")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero appearance descriptor")
    return values / norms


def build_appearance_model(
    reference: ObjectReference,
    views: tuple[ReferenceAppearanceFrame, ...],
    extractor: PatchExtractor,
) -> AppearanceModel:
    """Combine masked feature/depth samples from several registered demo views."""
    allowed = {(s.repo_id, s.episode, s.frame_index) for s in reference.snippets}
    point_groups, feature_groups = [], []
    for view in views:
        snippet, frame = view.snippet, view.frame
        if (snippet.repo_id, snippet.episode, snippet.frame_index) not in allowed:
            raise ValueError("appearance view is not part of the object reference")
        if snippet.calibration_version != frame.calibration_version or snippet.frame_index != frame.frame_id:
            raise ValueError("appearance view frame or calibration differs from its snippet")
        if view.mask.shape != frame.depth.shape or view.mask.dtype != bool or not view.mask.any():
            raise ValueError("appearance view needs a nonempty aligned mask")
        xy, descriptors = extractor.extract(frame.rgb, view.mask)
        xy = np.asarray(xy, dtype=float)
        descriptors = _normalize(descriptors)
        if xy.shape != (len(descriptors), 2):
            raise ValueError("feature pixels and descriptors differ in length")
        valid = valid_depth_pixels(frame.depth, xy)
        if not valid.any():
            continue
        camera_points = unproject_pixels(frame.depth, frame.intrinsics, xy[valid], frame.depth_m_per_unit)
        point_groups.append(transform_points(snippet.object_from_camera, camera_points))
        feature_groups.append(descriptors[valid])
    if not point_groups:
        raise ValueError("no reference features have aligned depth")
    return AppearanceModel(
        ObjectPointModel(reference.role, reference.version, np.concatenate(point_groups)),
        np.concatenate(feature_groups),
    )


class AppearancePixelMatcher:
    """One masked live patch per demo point, leaving geometric rejection to RGB-D fit."""

    def __init__(
        self,
        models: tuple[AppearanceModel, ...],
        extractor: PatchExtractor,
        mask_source: MaskSource,
        *,
        min_similarity: float,
    ) -> None:
        if not -1 <= min_similarity <= 1:
            raise ValueError("similarity floor must be a cosine score")
        self.models = {model.points.role: model for model in models}
        if not models or len(self.models) != len(models):
            raise ValueError("appearance models need unique object roles")
        self.extractor = extractor
        self.mask_source = mask_source
        self.min_similarity = min_similarity

    def matches(self, frame: RGBDFrame) -> tuple[PixelMatches, ...]:
        masks = self.mask_source.masks(frame)
        out = []
        for role, model in self.models.items():
            mask = masks.get(role)
            if mask is None or not mask.any():
                continue
            if mask.shape != frame.depth.shape or mask.dtype != bool:
                raise ValueError(f"{role}: live mask must align with RGB-D")
            xy, features = self.extractor.extract(frame.rgb, mask)
            xy = np.asarray(xy, dtype=float)
            features = _normalize(features)
            if xy.shape != (len(features), 2) or features.shape[1] != model.descriptors.shape[1]:
                raise ValueError(f"{role}: live features differ from the reference model")
            if not len(features):
                continue
            similarity = features @ model.descriptors.T
            nearest = np.argmax(similarity, axis=1)
            scores = similarity[np.arange(len(features)), nearest]
            chosen_model, chosen_live = [], []
            seen: set[int] = set()
            for live_index in np.argsort(-scores):
                model_index = int(nearest[live_index])
                if scores[live_index] < self.min_similarity:
                    break
                if model_index not in seen:
                    chosen_model.append(model_index)
                    chosen_live.append(int(live_index))
                    seen.add(model_index)
            out.append(
                PixelMatches(
                    role,
                    model.points.reference_version,
                    np.asarray(chosen_model, dtype=int),
                    xy[chosen_live],
                )
            )
        return tuple(out)


@dataclass(frozen=True)
class AppearanceBackendFactory:
    """Spawn-safe construction of the first live RGB-D pose backend.

    Keep only reference data and configuration in the parent. SAM3 and DINO
    weights load when this factory runs inside :class:`PoseProcess`.
    """

    models: tuple[AppearanceModel, ...]
    prompts_by_role: dict[str, str]
    camera_id: str
    min_similarity: float
    threshold_m: float
    min_inliers: int
    trials: int
    device: str = "cuda"
    sam_resolution: int | None = None
    dino_model_id: str = "facebook/dinov2-small"
    axial_symmetries: tuple[AxialSymmetry, ...] = ()
    pose_coast_limits: tuple[PoseCoastLimit, ...] = ()
    pose_gap_limits: tuple[PoseGapLimit, ...] = ()
    seed_points_by_role: dict[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if set(self.prompts_by_role) != {model.points.role for model in self.models}:
            raise ValueError("one text prompt is needed for each appearance model role")
        if {limit.role for limit in self.pose_coast_limits} - set(self.prompts_by_role):
            raise ValueError("pose-coasting role lacks an appearance model")
        if {limit.role for limit in self.pose_gap_limits} - set(self.prompts_by_role):
            raise ValueError("pose-gap role lacks an appearance model")

    def __call__(self) -> PoseBackend:
        from .dino_features import DinoPatchExtractor

        source = Sam3MaskSource(
            self.prompts_by_role,
            camera_id=self.camera_id,
            device=self.device,
            resolution=self.sam_resolution,
            seed_points_by_role=self.seed_points_by_role,
        )
        matcher = AppearancePixelMatcher(
            self.models,
            DinoPatchExtractor(model_id=self.dino_model_id, device=self.device),
            source,
            min_similarity=self.min_similarity,
        )
        backend = MatchedRGBDPoseBackend(
            tuple(model.points for model in self.models),
            matcher,
            threshold_m=self.threshold_m,
            min_inliers=self.min_inliers,
            trials=self.trials,
        )
        if self.axial_symmetries:
            from .symmetry import SymmetryCanonicalizingBackend

            backend = SymmetryCanonicalizingBackend(backend, self.axial_symmetries)
        if self.pose_gap_limits:
            backend = PoseGapBackend(backend, self.pose_gap_limits)
        if self.pose_coast_limits:
            model_centers = {
                model.points.role: np.median(model.points.points_object_m, axis=0) for model in self.models
            }
            backend = PoseContinuityBackend(
                backend,
                tuple(
                    replace(limit, center_object_m=model_centers[limit.role])
                    for limit in self.pose_coast_limits
                ),
            )
        return backend
