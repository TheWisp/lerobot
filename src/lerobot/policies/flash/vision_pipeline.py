"""Shared RGB-D vision construction for preparation, execution and inspection.

Scheduling and group/attachment fusion live above this module. A backend choice
is saved with a prepared skill, not inferred from whether its caller is a demo,
a robot, or a debugger. GPU models load only when a factory is called.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral
from os import cpu_count
from threading import Lock

import numpy as np

from .appearance import AppearanceModel, MaskSource, Sam3MaskSource, build_appearance_pose_backend
from .perception import PoseBackend, RGBDFrame
from .pose_evidence import ConditionalSurfaceInformation, VisibleSurfaceSupport
from .surface_correspondence import SurfaceMotionConfig
from .surface_grounding import ShapeConfig
from .surface_observation import SurfaceObservationConfig, prepare_surface_frame
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
    parallel_surface_objects: bool = True
    pipeline_surface_frames: bool = True
    surface_object_workers: int = 2
    surface_reference_max_points_per_view: int = 15000
    compiled_surface_fit: bool = True
    reuse_surface_search: bool = True
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
                self.shape.batch_jacobian,
            )
        ):
            raise ValueError("surface acceleration settings must be booleans")
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
            if value is not None and (
                not np.isfinite(value) or value < 0 or (value == 0 and key != "mask_margin_px")
            ):
                raise ValueError(f"invalid observation setting: {key}")
        for key, value in asdict(self.shape).items():
            if key == "batch_jacobian":
                continue
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
        self.last_candidate_poses = {}
        self.last_motion_prior_poses = {}
        self.last_visible_support = {}
        self.last_surface_information = {}
        self.references = {}
        self.sessions = {}
        self.reference_camera_from_object = {}
        self.previous_frame = None
        self._surface_pool = None
        self._prepare_pool = None
        self._prepared_count = 0
        self._finished_count = 0
        self._matcher_lock = Lock()
        from .surface_acceleration import SearchPool, compiled_kernels

        self._fit_kernels = compiled_kernels() if config.compiled_surface_fit else None
        self._search_pool = (
            SearchPool(min(cpu_count() or 1, config.shape.batch_query_workers))
            if config.reuse_surface_search
            else None
        )

    def _require_drained(self):
        if self._prepared_count != self._finished_count:
            raise RuntimeError("finish pending surface frames before changing references")

    def initialize(self, references):
        self._require_drained()
        self.references = {r.role: r for r in references}
        if len(self.references) != len(references) or not references:
            raise ValueError("surface tracker needs unique references")
        self.previous_frame = None
        self.last_candidate_poses = {}
        self.last_motion_prior_poses = {}
        self.last_visible_support = {}
        self.last_surface_information = {}
        self.sessions.clear()
        self.reference_camera_from_object.clear()
        if self.initializer is not None:
            self.initializer.initialize(references)

    def add_reference(self, reference):
        """Add a local inspection role while preserving every existing surface frame."""
        self._require_drained()
        if self.initializer is not None:
            raise ValueError("prepared object registration cannot change during tracking")
        if reference.role in self.references:
            raise ValueError(f"{reference.role}: object role already exists")
        self.references[reference.role] = reference

    @property
    def can_pipeline(self):
        # Registration reads pose state, unlike frame-pair matching. Finish it
        # before allowing another frame to prepare concurrently.
        return self.config.pipeline_surface_frames and (
            self.initializer is None or len(self.reference_camera_from_object) == len(self.references)
        )

    def close(self):
        for pool in (self._prepare_pool, self._surface_pool):
            if pool is not None:
                pool.shutdown(wait=True)
        self._prepare_pool = self._surface_pool = None
        if self._search_pool is not None:
            self._search_pool.close()
            self._search_pool = None

    def update(self, frame):
        return self.finish(self.prepare(frame))

    def prepare(self, frame):
        if self._prepared_count - self._finished_count >= 2:
            raise RuntimeError("surface preparation is limited to one frame ahead")
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
        masks = {role: mask.copy() for role, mask in self.mask_source.masks(frame).items()}
        mask_ms = self.mask_source.elapsed_ms if isinstance(self.mask_source, FrameMaskCache) else None
        if not set(masks) <= set(self.references):
            raise ValueError("mask source returned an unknown object role")
        seeds = {}
        if self.initializer is not None and set(self.reference_camera_from_object) != set(self.references):
            seeds = {e.role: e for e in self.initializer.update(frame)}
        jobs = []
        unregistered = set()
        for role in self.references:
            seed = seeds.get(role)
            session = self.sessions.get(role)
            registered = (
                self.initializer is None
                or role in self.reference_camera_from_object
                or (seed is not None and seed.evidence == Evidence.OBSERVED and seed.pose is not None)
            )
            if not registered:
                unregistered.add(role)
                continue
            if session is None:
                session = SurfaceTrackingSession(
                    self.matcher,
                    observation_config=self.config.observation,
                    shape_config=self.config.shape,
                    motion_config=self.config.motion,
                    matcher_lock=self._matcher_lock,
                    fit_kernels=self._fit_kernels,
                    search_pool=self._search_pool,
                    information_correspondence_m=self.config.registration_threshold_m,
                )
                self.sessions[role] = session
            jobs.append((role, session, masks.get(role)))

        measured = {}
        if jobs:
            depth_m = frame.metric_depth()
            frame_filters = prepare_surface_frame(depth_m, asdict(self.config.observation))

            def measure(job):
                role, session, mask = job
                if mask is None:
                    mask = np.zeros(frame.depth.shape, bool)
                return (
                    role,
                    session,
                    session.prepare(
                        frame.rgb,
                        depth_m,
                        mask,
                        frame.intrinsics,
                        frame.frame_id,
                        frame_filters=frame_filters,
                    ),
                )

            if self.config.parallel_surface_objects and len(jobs) > 1:
                if self._prepare_pool is None:
                    self._prepare_pool = ThreadPoolExecutor(
                        max_workers=min(cpu_count() or 1, self.config.surface_object_workers),
                        thread_name_prefix="flash-surface-prepare",
                    )
                measured = {
                    role: (session, detail) for role, session, detail in self._prepare_pool.map(measure, jobs)
                }
            else:
                measured = {role: (session, detail) for role, session, detail in map(measure, jobs)}

        prepared = {
            "frame": frame,
            "masks": masks,
            "seeds": seeds,
            "jobs": measured,
            "unregistered": unregistered,
            "sequence": self._prepared_count,
            "mask_ms": mask_ms,
        }
        self._prepared_count += 1
        return prepared

    def finish(self, prepared):
        if prepared["sequence"] != self._finished_count:
            raise ValueError("surface frames must finish once and in preparation order")
        frame, seeds = prepared["frame"], prepared["seeds"]
        self.last_masks = prepared["masks"]
        self.last_mask_ms = prepared["mask_ms"]
        self.last_failure, self.last_details = {}, {}
        self.last_candidate_poses, self.last_motion_prior_poses, self.last_visible_support = {}, {}, {}
        self.last_surface_information = {}
        unregistered = prepared["unregistered"]

        def finish_job(job):
            role, (session, item) = job
            return role, (session, session.finish(item))

        jobs = prepared["jobs"].items()
        if self.config.parallel_surface_objects and len(prepared["jobs"]) > 1:
            if self._surface_pool is None:
                self._surface_pool = ThreadPoolExecutor(
                    max_workers=min(cpu_count() or 1, self.config.surface_object_workers),
                    thread_name_prefix="flash-surface-fit",
                )
            measured = dict(self._surface_pool.map(finish_job, jobs))
        else:
            measured = dict(map(finish_job, jobs))
        out = []
        for role, reference in self.references.items():
            seed = seeds.get(role)
            pose = None
            if role in unregistered:
                detail = {"status": "awaiting_reference_registration", "diagnostics": {}}
            else:
                session, detail = measured[role]
                if "camera_from_reference" in detail:
                    if role not in self.reference_camera_from_object:
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
            self.last_candidate_poses[role] = (
                np.asarray(pose, dtype=float).copy() if pose is not None else None
            )
            motion_prior_camera = detail.get("motion_only_prior_camera_from_reference")
            motion_prior = (
                frame.base_from_camera
                @ np.asarray(motion_prior_camera, dtype=float)
                @ self.reference_camera_from_object[role]
                if motion_prior_camera is not None and role in self.reference_camera_from_object
                else None
            )
            self.last_motion_prior_poses[role] = motion_prior
            depth_residual_mm = diagnostics.get("surface_p90_mm")
            mask = self.last_masks.get(role)
            surface_axis_camera = diagnostics.get("principal_surface_axis_camera")
            surface_axis_base = (
                frame.base_from_camera[:3, :3] @ np.asarray(surface_axis_camera, dtype=float)
                if surface_axis_camera is not None
                else None
            )
            if surface_axis_base is not None:
                surface_axis_base /= np.linalg.norm(surface_axis_base)
            split_axes_camera = diagnostics.get("principal_surface_split_axes_camera")
            split_axes_base = None
            if split_axes_camera is not None:
                split_axes_base = []
                for axis_camera in split_axes_camera:
                    axis_base = frame.base_from_camera[:3, :3] @ np.asarray(axis_camera, dtype=float)
                    axis_base /= np.linalg.norm(axis_base)
                    split_axes_base.append(tuple(float(value) for value in axis_base))
            self.last_visible_support[role] = VisibleSurfaceSupport(
                mask_pixels=int(np.count_nonzero(mask)) if mask is not None else 0,
                depth_points=diagnostics.get("surface_points"),
                motion_inliers=diagnostics.get("inliers"),
                depth_residual_p90_m=(
                    float(depth_residual_mm) / 1000
                    if depth_residual_mm is not None and np.isfinite(depth_residual_mm)
                    else None
                ),
                surface_fit_converged=diagnostics.get("converged"),
                reference_depth_pixels=diagnostics.get("reference_depth_pixels"),
                reference_depth_residual_p90_m=(
                    float(diagnostics["reference_depth_p90_mm"]) / 1000
                    if diagnostics.get("reference_depth_p90_mm") is not None
                    else None
                ),
                visible_lateral_spread_m=(
                    float(diagnostics["visible_lateral_spread_mm"]) / 1000
                    if diagnostics.get("visible_lateral_spread_mm") is not None
                    else None
                ),
                principal_surface_axis_base=(
                    tuple(float(value) for value in surface_axis_base)
                    if surface_axis_base is not None
                    else None
                ),
                principal_surface_spreads_m=(
                    tuple(float(value) for value in diagnostics["principal_surface_spreads_m"])
                    if surface_axis_base is not None
                    else None
                ),
                principal_surface_points=(
                    int(diagnostics["principal_surface_points"]) if surface_axis_base is not None else None
                ),
                principal_surface_split_axes_base=(
                    tuple(split_axes_base) if split_axes_base is not None else None
                ),
                principal_surface_split_gap_rad=diagnostics.get("principal_surface_split_gap_rad"),
            )
            conditional = diagnostics.get("conditional_surface_information_camera")
            if conditional is not None:
                camera_basis = np.asarray(conditional["twist_basis"], dtype=float).reshape(6, -1)
                base_basis = camera_basis.copy()
                base_rotation = frame.base_from_camera[:3, :3]
                base_basis[:3] = base_rotation @ camera_basis[:3]
                base_basis[3:] = base_rotation @ camera_basis[3:]
                pivot_base = (
                    base_rotation @ np.asarray(conditional["pivot_m"], dtype=float)
                    + frame.base_from_camera[:3, 3]
                )
                self.last_surface_information[role] = ConditionalSurfaceInformation(
                    tuple(float(value) for value in pivot_base),
                    tuple(tuple(float(value) for value in row) for row in base_basis),
                    tuple(float(value) for value in conditional["strength"]),
                    int(conditional["matched_reference_points"]),
                    int(conditional["reference_points"]),
                    float(conditional["correspondence_distance_m"]),
                )
            # A full rigid observation needs two independently extended visible
            # directions. If the smaller spread is below the fit's measured
            # mismatch, the image cannot distinguish orientations of this
            # patch at the claimed precision. Retain the raw proposal for
            # replay, while the scene estimator continues its prior prediction.
            insufficient_extent = detail["status"] == "fitted" and diagnostics.get(
                "visible_lateral_spread_mm", 0
            ) <= diagnostics.get("surface_p90_mm", float("inf"))
            accepted = pose is not None and (
                detail["status"] == "reference"
                or (
                    detail["status"] == "fitted"
                    and not insufficient_extent
                    and diagnostics.get("converged", False)
                    and diagnostics.get("inliers", 0) >= self.config.registration_min_inliers
                    and diagnostics.get("surface_p90_mm", float("inf"))
                    <= self.config.registration_threshold_m * 1000
                    and diagnostics.get("reference_depth_pixels", 0) >= self.config.registration_min_inliers
                    and diagnostics.get("reference_depth_p90_mm") is not None
                    and diagnostics["reference_depth_p90_mm"] <= self.config.registration_threshold_m * 1000
                )
            )
            # The UI may inspect a rejected candidate; execution receives LOST.
            # Residuals are diagnostics, not a calibrated probability of correctness.
            detail["accepted"] = accepted
            detail["reference_version"] = reference.version
            detail["base_from_object"] = pose.tolist() if pose is not None else None
            detail["motion_only_prior_base_from_object"] = (
                motion_prior.tolist() if motion_prior is not None else None
            )
            self.last_details[role] = detail
            if role not in unregistered:
                session.record_publication(accepted)
            if not accepted:
                if pose is None:
                    self.last_failure[role] = detail["status"]
                elif insufficient_extent:
                    self.last_failure[role] = "visible patch is too narrow for a full rigid pose"
                else:
                    self.last_failure[role] = "surface proposal lacks accepted RGB-D support"
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
                    source_frame_id=frame.frame_id,
                )
            )
        self._finished_count += 1
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
