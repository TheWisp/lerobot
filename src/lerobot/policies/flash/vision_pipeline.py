"""Shared RGB-D vision construction for preparation, execution and inspection.

Scheduling and group/attachment fusion live above this module. A backend choice
is saved with a prepared skill, not inferred from whether its caller is a demo,
a robot, or a debugger. GPU models load only when a factory is called.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral

import numpy as np

from .appearance import AppearanceModel, MaskSource, Sam3MaskSource, build_appearance_pose_backend
from .perception import PoseBackend, RGBDFrame
from .surface_correspondence import SurfaceMotionConfig
from .surface_grounding import ShapeConfig
from .surface_observation import SurfaceObservationConfig
from .surface_tracking import SurfaceTrackingSession, load_ufm
from .symmetry import AxialSymmetry
from .trajectory import Evidence, PoseEstimate


@dataclass(frozen=True)
class VisionPipelineConfig:
    backend: str = "surface"
    sam_resolution: int | None = 504
    min_similarity: float = 0.75
    registration_threshold_m: float = 0.005
    registration_min_inliers: int = 12
    registration_trials: int = 100
    visible_shape_fallback: bool = False
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
        if not np.isfinite(self.min_similarity) or not -1 <= self.min_similarity <= 1:
            raise ValueError("invalid appearance similarity")
        if not np.isfinite(self.registration_threshold_m) or self.registration_threshold_m <= 0:
            raise ValueError("registration distance must be positive")
        if self.registration_min_inliers < 3 or self.registration_trials < 1:
            raise ValueError("invalid registration sampling")

        for key, value in asdict(self.observation).items():
            if value is not None and (
                not np.isfinite(value) or value < 0 or (value == 0 and key != "mask_margin_px")
            ):
                raise ValueError(f"invalid observation setting: {key}")
        for key, value in asdict(self.shape).items():
            if not np.isfinite(value) or value < 0 or (value == 0 and key != "mask_boundary_tolerance_px"):
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
            ("observation", SurfaceObservationConfig),
            ("shape", ShapeConfig),
            ("motion", SurfaceMotionConfig),
        ):
            if key in values and isinstance(values[key], dict):
                values[key] = kind(**values[key])
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_settings(cls, settings: dict, *, backend: str = "surface"):
        """Read old registration fields once; an explicit saved config wins."""
        values = {
            key: settings[key]
            for key in (
                "sam_resolution",
                "min_similarity",
                "registration_threshold_m",
                "registration_min_inliers",
                "registration_trials",
            )
            if key in settings
        }
        values.update(settings.get("vision_pipeline") or {})
        values.setdefault("backend", backend)
        return cls.from_dict(values)


class FrameMaskCache:
    """One segmentation call per input, also used by the reference initializer."""

    def __init__(self, source: MaskSource):
        self.source = source
        self.frame = None
        self.current = {}
        self.elapsed_ms = 0.0

    def masks(self, frame: RGBDFrame):
        if frame is not self.frame:
            start = time.perf_counter()
            masks = self.source.masks(frame)
            self.elapsed_ms = (time.perf_counter() - start) * 1000
            for role, mask in masks.items():
                if np.asarray(mask).shape != frame.depth.shape or np.asarray(mask).dtype != bool:
                    raise ValueError(f"{role}: instance mask must align with RGB-D")
            self.current = masks
            self.frame = frame
        return self.current


class SurfacePoseBackend:
    """Temporal surface tracking in the *registered* demonstration object frame.

    The initializer establishes correspondence with that frame once. Tracking
    then transports the observed surface and the same frame together. A local
    inspection session explicitly has no cross-episode initializer: its object
    frame is chosen once at A. Such sessions are never created by a skill factory.
    """

    def __init__(self, masks, matcher, config, *, initializer=None, clock=time.monotonic):
        self.mask_source = masks
        self.matcher = matcher
        self.config = config
        self.initializer = initializer
        self.clock = clock
        self.last_masks = {}
        self.last_failure = {}
        self.last_details = {}
        self.references = {}
        self.sessions = {}
        self.reference_camera_from_object = {}
        self.previous_frame = None

    def initialize(self, references):
        self.references = {r.role: r for r in references}
        if len(self.references) != len(references) or not references:
            raise ValueError("surface tracker needs unique references")
        self.previous_frame = None
        self.sessions.clear()
        self.reference_camera_from_object.clear()
        if self.initializer is not None:
            self.initializer.initialize(references)

    def add_reference(self, reference):
        """Add a local inspection role while preserving every existing surface frame."""
        if self.initializer is not None:
            raise ValueError("prepared object registration cannot change during tracking")
        if reference.role in self.references:
            raise ValueError(f"{reference.role}: object role already exists")
        self.references[reference.role] = reference

    def update(self, frame):
        if not self.references:
            raise RuntimeError("initialize references before tracking")
        if self.previous_frame is not None:
            old = self.previous_frame
            if frame.frame_id <= old.frame_id or frame.source_time_s <= old.source_time_s:
                raise ValueError("surface frames must increase; reset the pipeline before seeking")
            if (
                frame.calibration_version != old.calibration_version
                or not np.array_equal(frame.intrinsics, old.intrinsics)
                or frame.depth.shape != old.depth.shape
            ):
                raise ValueError("camera geometry changed; reset the surface pipeline")
        self.previous_frame = frame
        self.last_masks = self.mask_source.masks(frame)
        if not set(self.last_masks) <= set(self.references):
            raise ValueError("mask source returned an unknown object role")
        seeds = {}
        if self.initializer is not None and set(self.sessions) != set(self.references):
            seeds = {e.role: e for e in self.initializer.update(frame)}
        self.last_failure = {}
        self.last_details = {}
        out = []
        for role, reference in self.references.items():
            seed = seeds.get(role)
            session = self.sessions.get(role)
            registered = (
                self.initializer is None
                or session is not None
                or (seed is not None and seed.evidence == Evidence.OBSERVED and seed.pose is not None)
            )
            pose = None
            if not registered:
                detail = {"status": "awaiting_reference_registration", "diagnostics": {}}
            else:
                if session is None:
                    session = SurfaceTrackingSession(
                        self.matcher,
                        observation_config=self.config.observation,
                        shape_config=self.config.shape,
                        motion_config=self.config.motion,
                    )
                mask = self.last_masks.get(role, np.zeros(frame.depth.shape, bool))
                detail = session.update(
                    frame.rgb, frame.metric_depth(), mask, frame.intrinsics, frame.frame_id
                )
                if "camera_from_reference" in detail:
                    if role not in self.sessions:
                        self.sessions[role] = session
                        self.reference_camera_from_object[role] = (
                            np.asarray(detail["camera_from_object"])
                            if self.initializer is None
                            else np.linalg.inv(frame.base_from_camera) @ seed.pose
                        )
                    pose = (
                        frame.base_from_camera
                        @ np.asarray(detail["camera_from_reference"])
                        @ self.reference_camera_from_object[role]
                    )
            diagnostics = detail["diagnostics"]
            accepted = pose is not None and (
                detail["status"] == "reference"
                or (
                    detail["status"] == "fitted"
                    and diagnostics.get("converged", False)
                    and diagnostics.get("inliers", 0) >= self.config.registration_min_inliers
                    and diagnostics.get("surface_p90_mm", float("inf"))
                    <= self.config.registration_threshold_m * 1000
                )
            )
            # The UI may inspect a rejected candidate; execution receives LOST.
            # Residuals are diagnostics, not a calibrated probability of correctness.
            detail["accepted"] = accepted
            detail["reference_version"] = reference.version
            detail["base_from_object"] = pose.tolist() if pose is not None else None
            self.last_details[role] = detail
            if not accepted:
                self.last_failure[role] = (
                    detail["status"] if pose is None else "surface proposal lacks accepted RGB-D support"
                )
            out.append(
                PoseEstimate(
                    role,
                    pose if accepted else None,
                    frame.source_time_s,
                    max(frame.available_time_s, self.clock()),
                    Evidence.OBSERVED if accepted else Evidence.LOST,
                    reference.version,
                    (accepted,) * 3,
                    (accepted,) * 3,
                    estimator="surface_rgbd",
                    last_direct_observation_s=frame.source_time_s if accepted else None,
                )
            )
        return tuple(out)


def build_vision_backend(
    config,
    mask_source,
    *,
    models=(),
    extractor=None,
    matcher=None,
    axial_symmetries=(),
    local_reference=False,
    clock=time.monotonic,
):
    """Single composition point; injectable models keep CPU tests deterministic."""
    masks = FrameMaskCache(mask_source)
    initializer = None
    if not local_reference:
        if not models or extractor is None:
            raise ValueError("prepared vision requires registered object models and an extractor")
        initializer = build_appearance_pose_backend(
            models,
            extractor,
            masks,
            min_similarity=config.min_similarity,
            threshold_m=config.registration_threshold_m,
            min_inliers=config.registration_min_inliers,
            trials=config.registration_trials,
            visible_shape_fallback=config.visible_shape_fallback,
            clock=clock,
        )
    if config.backend == "appearance":
        if initializer is None:
            raise ValueError("appearance tracking needs a prepared object reference")
        backend = initializer
    else:
        backend = SurfacePoseBackend(
            masks, load_ufm() if matcher is None else matcher, config, initializer=initializer, clock=clock
        )
    if axial_symmetries:
        from .symmetry import SymmetryCanonicalizingBackend

        backend = SymmetryCanonicalizingBackend(backend, axial_symmetries)
    return backend


@dataclass(frozen=True)
class VisionBackendFactory:
    """Serializable data only; SAM/DINO/UFM load inside the perception process."""

    models: tuple[AppearanceModel, ...]
    prompts_by_role: dict[str, str]
    camera_id: str
    config: VisionPipelineConfig
    dino_model_id: str = "facebook/dinov2-small"
    axial_symmetries: tuple[AxialSymmetry, ...] = ()
    seed_points_by_role: dict[str, tuple[float, float]] | None = None
    device: str = "cuda"

    def __call__(self) -> PoseBackend:
        from .dino_features import DinoPatchExtractor

        if set(self.prompts_by_role) != {m.points.role for m in self.models}:
            raise ValueError("each registered object needs one mask prompt")
        return build_vision_backend(
            self.config,
            Sam3MaskSource(
                self.prompts_by_role,
                camera_id=self.camera_id,
                device=self.device,
                resolution=self.config.sam_resolution,
                seed_points_by_role=self.seed_points_by_role,
            ),
            models=self.models,
            extractor=DinoPatchExtractor(model_id=self.dino_model_id, device=self.device),
            axial_symmetries=self.axial_symmetries,
        )
