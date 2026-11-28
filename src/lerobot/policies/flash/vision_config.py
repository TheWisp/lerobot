"""Serializable settings for the shared FLASH vision pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from numbers import Integral

import numpy as np

from .surface_correspondence import SurfaceMotionConfig
from .surface_grounding import ShapeConfig
from .surface_observation import SurfaceObservationConfig


@dataclass(frozen=True)
class DetectionCorrectionConfig:
    """Independent text detection checks; the robot loop never waits for these.

    The same acceptance rule is used offline and live. Only the check cadence
    differs, and each detection is compared with tracking on its source frame.
    """

    enabled: bool = False
    offline_interval_frames: int = 16
    runtime_interval_frames: int = 60
    min_detector_score: float = 0.8
    min_tracker_score: float = 0.8
    min_mask_overlap: float = 0.5

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("detector correction enabled must be a boolean")
        if any(
            type(n) is not int or n < 1 for n in (self.offline_interval_frames, self.runtime_interval_frames)
        ):
            raise ValueError("detector correction intervals must be positive frame counts")
        if any(
            not np.isfinite(value) or not 0 <= value <= 1
            for value in (self.min_detector_score, self.min_tracker_score, self.min_mask_overlap)
        ):
            raise ValueError("detector correction scores and overlap must be in [0, 1]")

    def control(self, *, offline: bool) -> dict:
        return {
            "enabled": self.enabled,
            "interval_frames": self.offline_interval_frames if offline else self.runtime_interval_frames,
            "min_detector_score": self.min_detector_score,
            "min_tracker_score": self.min_tracker_score,
            "min_mask_overlap": self.min_mask_overlap,
        }


@dataclass(frozen=True)
class VisionPipelineConfig:
    backend: str = "surface"
    sam_resolution: int | None = 504
    min_similarity: float = 0.75
    registration_threshold_m: float = 0.005
    registration_min_inliers: int = 12
    registration_trials: int = 100
    visible_shape_fallback: bool = False
    parallel_surface_objects: bool = True
    pipeline_surface_frames: bool = True
    surface_object_workers: int = 2
    surface_reference_max_points_per_view: int = 15000
    compiled_surface_fit: bool = True
    reuse_surface_search: bool = True
    detection_correction: DetectionCorrectionConfig = field(default_factory=DetectionCorrectionConfig)
    observation: SurfaceObservationConfig = field(default_factory=SurfaceObservationConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    motion: SurfaceMotionConfig = field(default_factory=SurfaceMotionConfig)

    def __post_init__(self):
        for config in (self, self.observation, self.shape, self.motion):
            for setting in fields(config):
                value = getattr(config, setting.name)
                if (
                    type(setting.default) is int
                    and value is not None
                    and (not isinstance(value, Integral) or isinstance(value, bool))
                ):
                    raise ValueError(f"{setting.name} must be an integer")
        if self.sam_resolution is not None and (self.sam_resolution < 224 or self.sam_resolution % 14):
            raise ValueError("SAM resolution must be at least 224 and divisible by 14")
        if self.backend not in {"surface", "appearance"}:
            raise ValueError("unknown FLASH vision backend")
        if any(
            not isinstance(value, bool)
            for value in (
                self.parallel_surface_objects,
                self.pipeline_surface_frames,
                self.compiled_surface_fit,
                self.reuse_surface_search,
                self.shape.require_full_mask_coverage,
                self.shape.scene_depth_constraints,
                self.shape.prefer_observed_overlap,
                self.observation.filtered_depth_for_fitting,
            )
        ):
            raise ValueError("surface acceleration and coverage settings must be booleans")
        if self.surface_object_workers < 1:
            raise ValueError("surface_object_workers must be positive")
        if self.surface_reference_max_points_per_view < 3:
            raise ValueError("surface reference needs at least three points per view")
        if not np.isfinite(self.min_similarity) or not -1 <= self.min_similarity <= 1:
            raise ValueError("invalid appearance similarity")
        if not np.isfinite(self.registration_threshold_m) or self.registration_threshold_m <= 0:
            raise ValueError("registration distance must be positive")
        if self.registration_min_inliers < 3 or self.registration_trials < 1:
            raise ValueError("invalid registration sampling")

        for key, value in asdict(self.observation).items():
            if key == "filtered_depth_for_fitting":
                continue
            if value is not None and (
                not np.isfinite(value)
                or value < 0
                or (value == 0 and key not in ("mask_margin_px", "outer_mask_margin_px"))
            ):
                raise ValueError(f"invalid observation setting: {key}")
        for key, value in asdict(self.shape).items():
            if key in {
                "registration",
                "require_full_mask_coverage",
                "scene_depth_constraints",
                "prefer_observed_overlap",
            }:
                continue
            if (
                not np.isfinite(value)
                or value < 0
                or (value == 0 and key not in {"mask_boundary_tolerance_px", "mask_coverage_weight"})
            ):
                raise ValueError(f"invalid shape setting: {key}")
        if self.observation.min_surface_spacing_m > self.observation.surface_spacing_m:
            raise ValueError("minimum surface spacing exceeds maximum spacing")
        if self.shape.max_fit_points < 3:
            raise ValueError("surface fitting needs at least three samples")
        motion = self.motion
        if (
            any(
                not np.isfinite(value) or value <= 0
                for key, value in asdict(motion).items()
                if key != "covisibility_min"
            )
            or not 0 <= motion.covisibility_min <= 1
            or not 0 < motion.ransac_confidence < 1
            or motion.min_ransac_pairs < 4
            or motion.crop_min_px > motion.crop_max_px
            or 2 * motion.grid_step_px >= motion.input_size_px
        ):
            raise ValueError("invalid RGB motion settings")

    @classmethod
    def from_dict(cls, values: dict):
        values = dict(values)
        for key, kind in (
            ("detection_correction", DetectionCorrectionConfig),
            ("observation", SurfaceObservationConfig),
            ("shape", ShapeConfig),
            ("motion", SurfaceMotionConfig),
        ):
            if key in values and isinstance(values[key], dict):
                section = dict(values[key])
                if key == "shape":
                    # Old recordings remain labelled by their stored provenance.
                    # Reruns use the supported solver and save its actual settings.
                    section.pop("batch_jacobian", None)
                    section.pop("max_correction_iterations", None)
                    registration = dict(section.get("registration", {}))
                    if registration.get("backend") == "scipy":
                        registration["backend"] = "visible_icp"
                    section["registration"] = registration
                values[key] = kind(**section)
        return cls(**values)

    def to_dict(self):
        return asdict(self)
