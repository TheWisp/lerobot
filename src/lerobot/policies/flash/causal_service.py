"""Every-frame point motion alongside slower, replaceable pose estimation."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from math import isfinite
from queue import Full, Queue
from threading import Lock, Thread
from typing import Any

import numpy as np

from .causal_points import CausalRigidPointTracker, RigidPointMotion
from .geometry import inverse
from .perception import LatestPoseStore, RGBDFrame
from .pose_disagreement import (
    FreeSpaceComparison,
    FreeSpaceVerdict,
    PoseComparison,
    PoseComparisonVerdict,
    compare_pose_proposals,
    free_space_support,
)
from .registration import transform_points
from .scene_groups import (
    EnvironmentRelationship,
    ScenePointTracker,
    SceneTracks,
    establish_environment_relationship,
    seed_environment_relationship,
)
from .symmetry import AxialSymmetry
from .trajectory import Evidence, PoseEstimate

DEFAULT_MAX_POSE_DISAGREEMENT_TRANSLATION_M = 0.01
DEFAULT_MAX_POSE_DISAGREEMENT_ROTATION_RAD = 0.15


@dataclass(frozen=True)
class _Anchor:
    frame_id: int
    estimates: tuple[PoseEstimate, ...]
    masks: dict[str, np.ndarray]
    source_time_s: float | None = None
    calibration_version: str | None = None


@dataclass
class _PendingTrack:
    tracker: CausalRigidPointTracker
    anchor: PoseEstimate
    last_frame_id: int
    proposals: dict[int, PoseEstimate]


@dataclass(frozen=True)
class CausalTrackingConfig:
    history_frames: int
    queue_frames: int
    min_inliers: int
    max_frame_gap_s: float
    fit_threshold_m: float
    image_scale: float = 1.0
    depth_support_radius_px: int = 1
    scene_max_points: int = 800
    environment_context_radius_m: float = 0.25
    environment_min_motion_m: float = 0.008
    environment_prediction_max_s: float = 1.0
    environment_min_inliers: int = 12
    static_prediction_max_s: float | None = None
    max_pose_disagreement_translation_m: float = DEFAULT_MAX_POSE_DISAGREEMENT_TRANSLATION_M
    max_pose_disagreement_rotation_rad: float = DEFAULT_MAX_POSE_DISAGREEMENT_ROTATION_RAD

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CausalTrackingConfig:
        """Require each live tracker limit to be visible in the rig configuration."""
        expected = {field.name for field in fields(cls)}
        # Recordings written before this depth-quality setting remain readable.
        required = expected - {"depth_support_radius_px"}
        if not required <= set(config) or not set(config) <= expected:
            raise ValueError(f"causal tracking keys differ: {sorted(set(config) ^ required)}")
        return cls(**config)

    def __post_init__(self) -> None:
        if (
            self.history_frames < 2
            or self.queue_frames < 2
            or self.min_inliers < 3
            or not isfinite(self.max_frame_gap_s)
            or self.max_frame_gap_s <= 0
            or not isfinite(self.fit_threshold_m)
            or self.fit_threshold_m <= 0
            or not isfinite(self.image_scale)
            or not 0 < self.image_scale <= 1
            or isinstance(self.depth_support_radius_px, bool)
            or not isinstance(self.depth_support_radius_px, int)
            or self.depth_support_radius_px < 0
            or self.scene_max_points < 16
            or not isfinite(self.environment_context_radius_m)
            or self.environment_context_radius_m <= 0
            or not isfinite(self.environment_min_motion_m)
            or self.environment_min_motion_m <= 0
            or not isfinite(self.environment_prediction_max_s)
            or self.environment_prediction_max_s <= 0
            or self.environment_min_inliers < 3
            or not isfinite(self.max_pose_disagreement_translation_m)
            or self.max_pose_disagreement_translation_m <= 0
            or not isfinite(self.max_pose_disagreement_rotation_rad)
            or self.max_pose_disagreement_rotation_rad <= 0
            or (
                self.static_prediction_max_s is not None
                and (not isfinite(self.static_prediction_max_s) or self.static_prediction_max_s <= 0)
            )
        ):
            raise ValueError("invalid causal tracking configuration")


class CausalPoseService:
    """Consume each camera frame, replaying delayed anchors on retained frames.

    A slow estimator may skip frames, but this queue may not. The caller stops
    the attempt on overflow. An anchor older than retained frame history is
    refused instead of being transplanted onto a newer image.
    """

    def __init__(
        self,
        store: LatestPoseStore,
        *,
        history_frames: int = 60,
        queue_frames: int = 64,
        min_inliers: int = 12,
        max_frame_gap_s: float = 0.1,
        fit_threshold_m: float = 0.005,
        image_scale: float = 1.0,
        depth_support_radius_px: int = 1,
        scene_max_points: int = 800,
        environment_context_radius_m: float = 0.25,
        environment_min_motion_m: float = 0.008,
        environment_prediction_max_s: float = 1.0,
        environment_min_inliers: int = 12,
        static_prediction_max_s: float | None = None,
        max_pose_disagreement_translation_m: float = DEFAULT_MAX_POSE_DISAGREEMENT_TRANSLATION_M,
        max_pose_disagreement_rotation_rad: float = DEFAULT_MAX_POSE_DISAGREEMENT_ROTATION_RAD,
        object_centers_by_role: dict[str, np.ndarray] | None = None,
        object_points_by_role: dict[str, np.ndarray] | None = None,
        axial_symmetries: tuple[AxialSymmetry, ...] = (),
        on_layer: Callable[[PoseEstimate], None] | None = None,
        on_comparison: Callable[[str, PoseComparison], None] | None = None,
        on_point_comparison: Callable[[str, PoseComparison], None] | None = None,
        on_free_space_comparison: Callable[[str, FreeSpaceComparison], None] | None = None,
        on_direct_observation: Callable[[PoseEstimate], None] | None = None,
    ) -> None:
        if history_frames < 2 or queue_frames < 2:
            raise ValueError("causal pose history and queue need at least two frames")
        self.store = store
        self.history: deque[RGBDFrame] = deque(maxlen=history_frames)
        self.scene_history: deque[SceneTracks] = deque(maxlen=history_frames)
        self.queue: Queue[RGBDFrame | _Anchor | None] = Queue(maxsize=queue_frames)
        self.tracker_kwargs = {
            "min_inliers": min_inliers,
            "max_frame_gap_s": max_frame_gap_s,
            "fit_threshold_m": fit_threshold_m,
            "image_scale": image_scale,
            "depth_support_radius_px": depth_support_radius_px,
        }
        self.active: dict[str, tuple[CausalRigidPointTracker, PoseEstimate]] = {}
        self.pending: dict[str, _PendingTrack] = {}
        self.motion_history: dict[str, deque[tuple[int, RigidPointMotion, np.ndarray]]] = {}
        self.scene = ScenePointTracker(
            max_points=scene_max_points,
            max_frame_gap_s=max_frame_gap_s,
            image_scale=image_scale,
            depth_support_radius_px=depth_support_radius_px,
        )
        self.environment_context_radius_m = environment_context_radius_m
        self.environment_min_motion_m = environment_min_motion_m
        self.environment_prediction_max_s = environment_prediction_max_s
        self.environment_min_inliers = environment_min_inliers
        self.static_prediction_max_s = static_prediction_max_s
        self.fit_threshold_m = fit_threshold_m
        self.max_pose_disagreement_translation_m = max_pose_disagreement_translation_m
        self.max_pose_disagreement_rotation_rad = max_pose_disagreement_rotation_rad
        self.object_centers_by_role = {
            role: np.asarray(center, dtype=float).copy()
            for role, center in (object_centers_by_role or {}).items()
        }
        self.object_points_by_role = {
            role: np.asarray(points, dtype=float).copy()
            for role, points in (object_points_by_role or {}).items()
        }
        self.axial_symmetries = {item.role: item for item in axial_symmetries}
        if len(self.axial_symmetries) != len(axial_symmetries):
            raise ValueError("at most one axial symmetry per role")
        self.on_layer = on_layer
        self.on_comparison = on_comparison
        self.on_point_comparison = on_point_comparison
        self.on_free_space_comparison = on_free_space_comparison
        self.on_direct_observation = on_direct_observation
        if any(
            center.shape != (3,) or not np.isfinite(center).all()
            for center in self.object_centers_by_role.values()
        ):
            raise ValueError("object support centers must be finite 3D points")
        if any(
            points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all()
            for points in self.object_points_by_role.values()
        ):
            raise ValueError("object support models must contain finite 3D points")
        self.observed: dict[str, tuple[PoseEstimate, SceneTracks, tuple[np.ndarray, ...]]] = {}
        self.candidate_relationships: dict[str, EnvironmentRelationship] = {}
        self.relationships: dict[str, EnvironmentRelationship] = {}
        self._expired_verified_roles: set[str] = set()
        self._attachment_lock = Lock()
        self._attached_role: str | None = None
        self._attachment_changed_roles: set[str] = set()
        self._thread = Thread(target=self._run, name="flash-causal-points", daemon=True)
        self._lock = Lock()
        self._error: Exception | None = None
        self._started = False
        self.frames = 0
        self.anchors = 0
        self.missed_anchors = 0
        self.superseded_proposals = 0
        self.published = 0
        self.published_by_role: dict[str, int] = {}
        self.direct_published = 0
        self.anchor_outcomes_by_role: dict[str, dict[str, int]] = {}
        self.static_published = 0
        self.max_processing_lag_s = 0.0
        self.latest_processed_source_time_s: float | None = None
        self.environment_relationships = 0
        self.environment_relationships_by_role: dict[str, int] = {}
        self.environment_published = 0
        self.environment_published_by_role: dict[str, int] = {}
        self.environment_candidate_published_by_role: dict[str, int] = {}
        self.environment_fit_failures_by_role: dict[str, int] = {}
        self.environment_expired_by_role: dict[str, int] = {}
        self.direct_group_disagreements_by_role: dict[str, dict[str, int]] = {}
        self.last_direct_group_comparison: dict[str, PoseComparison] = {}
        self.direct_point_disagreements_by_role: dict[str, dict[str, int]] = {}
        self.last_direct_point_comparison: dict[str, PoseComparison] = {}
        self.direct_free_space_counts_by_role: dict[str, dict[str, int]] = {}
        self.last_direct_free_space_comparison: dict[str, FreeSpaceComparison] = {}
        self.frame_compute_total_s = 0.0
        self.frame_compute_max_s = 0.0
        self.scene_update_total_s = 0.0
        self.environment_track_total_s = 0.0
        self.object_track_total_s = 0.0
        self.anchor_compute_total_s = 0.0
        self.anchor_compute_max_s = 0.0
        self.max_queue_age_s = 0.0
        self.replay_compute_total_s = 0.0
        self.replay_compute_max_s = 0.0

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def start(self) -> None:
        if self._started:
            raise RuntimeError("causal pose service already started")
        self._started = True
        self._thread.start()

    def submit_frame(self, frame: RGBDFrame) -> None:
        self._submit(frame)

    def submit_anchor(
        self,
        frame_id: int,
        estimates: tuple[PoseEstimate, ...],
        masks: dict[str, np.ndarray],
        *,
        source_time_s: float | None = None,
        calibration_version: str | None = None,
    ) -> None:
        self._submit(_Anchor(frame_id, estimates, masks, source_time_s, calibration_version))

    def process_offline(
        self,
        frame: RGBDFrame,
        estimates: tuple[PoseEstimate, ...],
        masks: dict[str, np.ndarray],
        *,
        attached_role: str | None = None,
    ) -> dict[str, PoseEstimate]:
        """Apply one recorded frame through the same scene estimator as live.

        Offline extraction has every camera frame and its direct proposal in
        source order. It can therefore run synchronously without a worker queue
        or artificial perception delay. This does not reproduce the scheduling
        of delayed live proposals; it shares their pose arbitration and group
        tracking rules.
        """
        if self._started:
            raise RuntimeError("offline pose processing cannot share a running worker")
        self.set_attached_role(attached_role)
        self._frame(frame)
        self._anchor(
            _Anchor(
                frame.frame_id,
                estimates,
                masks,
                frame.source_time_s,
                frame.calibration_version,
            )
        )
        self._drain_pending()
        return self.store.snapshot()[1]

    def set_attached_role(self, role: str | None) -> None:
        """Withdraw support assumptions when an object is grasped or released."""
        with self._attachment_lock:
            if role != self._attached_role:
                self._attachment_changed_roles.update(
                    item for item in (self._attached_role, role) if item is not None
                )
                self._attached_role = role

    def _sync_attachment(self) -> str | None:
        with self._attachment_lock:
            role = self._attached_role
            changed = self._attachment_changed_roles.copy()
            self._attachment_changed_roles.clear()
        for item in changed:
            self.relationships.pop(item, None)
            self.candidate_relationships.pop(item, None)
            self.observed.pop(item, None)
            self._expired_verified_roles.discard(item)
        return role

    def _submit(self, item: RGBDFrame | _Anchor) -> None:
        if self.error is not None:
            raise RuntimeError("causal pose service failed") from self.error
        try:
            self.queue.put_nowait(item)
        except Full as exc:
            raise RuntimeError("causal point-frame queue overflow") from exc

    def _run(self) -> None:
        try:
            while (item := self.queue.get()) is not None:
                started_s = time.perf_counter()
                if isinstance(item, RGBDFrame):
                    self.max_queue_age_s = max(self.max_queue_age_s, started_s - item.source_time_s)
                    self._frame(item)
                    elapsed_s = time.perf_counter() - started_s
                    self.frame_compute_total_s += elapsed_s
                    self.frame_compute_max_s = max(self.frame_compute_max_s, elapsed_s)
                else:
                    self._anchor(item)
                    elapsed_s = time.perf_counter() - started_s
                    self.anchor_compute_total_s += elapsed_s
                    self.anchor_compute_max_s = max(self.anchor_compute_max_s, elapsed_s)
                # Catch a delayed object-point seed up only in the gaps
                # between incoming camera events. Each step is interruptible
                # by the next queued frame, so replay cannot starve the
                # current scene state for the duration of its whole history.
                self._drain_pending(while_queue_empty=True)
        except Exception as exc:
            with self._lock:
                self._error = exc

    def _drain_pending(self, *, while_queue_empty: bool = False) -> None:
        while self.pending and (not while_queue_empty or self.queue.empty()):
            attached_role = self._sync_attachment()
            replay_role = next(
                (role for role in self.pending if role == attached_role or role not in self.relationships),
                None,
            )
            if replay_role is None:
                break
            replay_started_s = time.perf_counter()
            if not self._replay_one(replay_role):
                break
            elapsed_s = time.perf_counter() - replay_started_s
            self.replay_compute_total_s += elapsed_s
            self.replay_compute_max_s = max(self.replay_compute_max_s, elapsed_s)

    def _frame(self, frame: RGBDFrame) -> None:
        attached_role = self._sync_attachment()
        if self.history and frame.frame_id <= self.history[-1].frame_id:
            raise ValueError("causal camera frames must increase")
        self.history.append(frame)
        scene_started_s = time.perf_counter()
        scene = self.scene.update(frame)
        self.scene_update_total_s += time.perf_counter() - scene_started_s
        self.scene_history.append(scene)
        self.frames += 1
        self.max_processing_lag_s = max(self.max_processing_lag_s, time.perf_counter() - frame.source_time_s)
        self.latest_processed_source_time_s = frame.source_time_s
        object_motion_roles: set[str] = set()
        object_started_s = time.perf_counter()
        for role, (tracker, anchor) in list(self.active.items()):
            if role in self.relationships and role != attached_role:
                continue
            try:
                motion = tracker.update(frame)
            except ValueError:
                del self.active[role]
                self.motion_history.pop(role, None)
                continue
            if motion.base_motion_from_seed is not None:
                self.motion_history[role].append(
                    (frame.frame_id, motion, tracker.current_pixels(frame.depth.shape))
                )
                object_motion_roles.add(role)
                self._publish(role, anchor, motion)
        self.object_track_total_s += time.perf_counter() - object_started_s
        environment_started_s = time.perf_counter()
        for role, relationship in list(self.relationships.items()) + list(
            self.candidate_relationships.items()
        ):
            confirmed = role in self.relationships
            source = self.relationships if confirmed else self.candidate_relationships
            last_group_s = relationship.last_group_measurement_s or relationship.anchor_time_s
            if frame.source_time_s - last_group_s > self.environment_prediction_max_s:
                del source[role]
                if confirmed:
                    self._expired_verified_roles.add(role)
                self.environment_expired_by_role[role] = self.environment_expired_by_role.get(role, 0) + 1
                continue
            if role in object_motion_roles or role == attached_role:
                continue
            tracked = relationship.track(
                scene, fit_threshold_m=self.fit_threshold_m, min_inliers=self.environment_min_inliers
            )
            if tracked is None:
                self.environment_fit_failures_by_role[role] = (
                    self.environment_fit_failures_by_role.get(role, 0) + 1
                )
                # An unverified support hypothesis has no stronger claim than
                # the last direct object view. If its points disappear, the
                # configured bounded stationary prior may bridge the gap.
                # A verified moving group must never be replaced this way.
                if not confirmed and (observed := self.observed.get(role)) is not None:
                    self._publish_static(role, observed[0], frame)
                continue
            pose, advanced = tracked
            source[role] = advanced
            observed = self.observed.get(role)
            if observed is None or observed[0].reference_version != relationship.reference_version:
                continue
            anchor = observed[0]
            estimate = PoseEstimate(
                role,
                pose,
                frame.source_time_s,
                max(frame.available_time_s, time.perf_counter()),
                Evidence.PREDICTED,
                relationship.reference_version,
                anchor.translation_supported,
                anchor.rotation_supported,
                estimator="environment_group_motion" if confirmed else "environment_support_hypothesis",
                last_direct_observation_s=anchor.source_time_s,
            )
            previous_version, _ = self.store.snapshot()
            if self.store.publish((estimate,)) != previous_version:
                if self.on_layer is not None:
                    self.on_layer(estimate)
                self.environment_published += 1
                self.environment_published_by_role[role] = self.environment_published_by_role.get(role, 0) + 1
                if not confirmed:
                    self.environment_candidate_published_by_role[role] = (
                        self.environment_candidate_published_by_role.get(role, 0) + 1
                    )
        self.environment_track_total_s += time.perf_counter() - environment_started_s
        # A missing visual fit is not evidence that a stationary object moved.
        # Carry the last direct pose for a bounded interval, with explicit
        # prediction provenance. Once co-motion has been verified, only that
        # relationship may carry the pose: a static prior must not hide a
        # moved support surface or a failed group fit.
        for role, (anchor, _, _) in self.observed.items():
            if (
                role != attached_role
                and role not in object_motion_roles
                and role not in self.relationships
                and role not in self.candidate_relationships
            ):
                self._publish_static(role, anchor, frame)

    def _publish_static(self, role: str, anchor: PoseEstimate, frame: RGBDFrame) -> None:
        if self.static_prediction_max_s is None or role in self._expired_verified_roles:
            return
        age = frame.source_time_s - anchor.source_time_s
        if not 0 < age <= self.static_prediction_max_s:
            return
        current = self.store.snapshot()[1].get(role)
        if current is not None:
            if current.source_time_s > frame.source_time_s:
                return
            if current.source_time_s == frame.source_time_s and (
                current.evidence != Evidence.PREDICTED
                or (
                    current.last_direct_observation_s is not None
                    and current.last_direct_observation_s >= anchor.source_time_s
                )
            ):
                return
        assert anchor.pose is not None
        before, _ = self.store.snapshot()
        after = self.store.publish(
            (
                PoseEstimate(
                    role,
                    anchor.pose,
                    frame.source_time_s,
                    max(frame.available_time_s, time.perf_counter()),
                    Evidence.PREDICTED,
                    anchor.reference_version,
                    anchor.translation_supported,
                    anchor.rotation_supported,
                    estimator="stationary_prior",
                    last_direct_observation_s=anchor.source_time_s,
                ),
            )
        )
        if after != before:
            if self.on_layer is not None:
                self.on_layer(self.store.snapshot()[1][role])
            self.static_published += 1

    def _anchor(self, result: _Anchor) -> None:
        attached_role = self._sync_attachment()
        source = next((frame for frame in self.history if frame.frame_id == result.frame_id), None)
        scene = next((item for item in self.scene_history if item.frame_id == result.frame_id), None)
        if (
            source is None
            or scene is None
            or (result.source_time_s is not None and result.source_time_s != source.source_time_s)
            or (
                result.calibration_version is not None
                and result.calibration_version != source.calibration_version
            )
        ):
            self.missed_anchors += 1
            return
        for estimate in result.estimates:
            self._count_anchor_outcome(estimate.role, "received")
            mask = result.masks.get(estimate.role)
            if mask is None:
                self._count_anchor_outcome(estimate.role, "missing_mask")
                continue
            if estimate.pose is None or estimate.evidence == Evidence.LOST:
                self._count_anchor_outcome(estimate.role, "lost_pose")
                continue
            if estimate.source_time_s != source.source_time_s:
                self._count_anchor_outcome(estimate.role, "time_mismatch")
                continue
            symmetry = self.axial_symmetries.get(estimate.role)
            if symmetry is not None and not estimate.symmetry_canonicalized:
                try:
                    active = self.active.get(estimate.role)
                    reference = active[1].pose if active is not None else None
                    estimate = replace(
                        estimate,
                        pose=symmetry.canonicalize(estimate.pose, reference_pose=reference),
                        symmetry_canonicalized=True,
                    )
                except ValueError:
                    self._count_anchor_outcome(estimate.role, "direct_symmetry_invalid")
                    continue
            self._record_direct_free_space(estimate, source)
            current = self.store.snapshot()[1].get(estimate.role)
            if current is not None:
                last_direct_s = current.last_direct_observation_s
                if last_direct_s is None and current.evidence == Evidence.OBSERVED:
                    last_direct_s = current.source_time_s
                if last_direct_s is not None and last_direct_s > estimate.source_time_s:
                    # The full scene estimator will replay all later evidence.
                    # Until then, replaying only point motion from this older
                    # proposal would erase the later direct observation.
                    self.superseded_proposals += 1
                    self._count_anchor_outcome(estimate.role, "superseded_by_direct")
                    continue
            if (
                estimate.has_direct_translation
                and estimate.role != attached_role
                and self._direct_conflicts_with_verified_group(estimate, source, scene, mask)
            ):
                self._count_anchor_outcome(estimate.role, "direct_group_unresolved")
                continue
            if self._direct_conflicts_with_point_motion(estimate, source, mask):
                self._count_anchor_outcome(estimate.role, "direct_point_unresolved")
                continue
            if estimate.has_direct_translation and estimate.role != attached_role:
                self._observe_environment(estimate, scene, result.masks)
            self._count_anchor_outcome(estimate.role, "direct_accepted")
            if self.on_layer is not None:
                self.on_layer(estimate)
            # The newest published state may already be a later point-motion
            # prediction. Preserve the accepted source-frame observation as a
            # separate event so a held attachment can correct at image time.
            if estimate.evidence == Evidence.OBSERVED and self.on_direct_observation is not None:
                self.on_direct_observation(estimate)
            # The pose process only submits source-frame proposals. This is
            # the sole owner that can publish an accepted absolute pose.
            if current is not None and current.source_time_s > estimate.source_time_s:
                # A prediction for a later camera frame cannot become an
                # observation at this older frame. Rebase that prediction on
                # the new direct evidence below, at its real source time.
                self._count_anchor_outcome(estimate.role, "direct_rebased")
            else:
                before, _ = self.store.snapshot()
                if self.store.publish((estimate,)) != before:
                    self.direct_published += 1
                    self._count_anchor_outcome(estimate.role, "direct_published")
                else:
                    self._count_anchor_outcome(estimate.role, "direct_not_published")
            if estimate.role in self.relationships and estimate.role != attached_role:
                # The verified scene group already updates this stationary
                # object's pose every frame. Keep a recent direct-image seed
                # ready for independent motion or a later grasp, but do not
                # spend the live camera budget replaying the same interval in
                # a second tracker.
                standby = CausalRigidPointTracker(**self.tracker_kwargs)
                seeded = standby.seed(source, mask)
                if seeded >= standby.min_inliers:
                    self.active.pop(estimate.role, None)
                    self.pending[estimate.role] = _PendingTrack(standby, estimate, source.frame_id, {})
                    self.motion_history[estimate.role] = deque(maxlen=self.history.maxlen)
                    self._count_anchor_outcome(estimate.role, "standby_point_seeded")
                else:
                    self._count_anchor_outcome(estimate.role, "standby_point_seed_failed")
                continue
            active = self.active.get(estimate.role)
            if active is not None and active[1].reference_version == estimate.reference_version:
                history = {
                    frame_id: (motion, pixels)
                    for frame_id, motion, pixels in self.motion_history[estimate.role]
                }
                from_source = history.get(source.frame_id)
                from_latest = history.get(self.history[-1].frame_id)
                if from_source is not None and from_latest is not None:
                    source_motion, source_pixels = from_source
                    inside = mask[source_pixels[:, 1], source_pixels[:, 0]]
                    if int(inside.sum()) >= active[0].min_inliers:
                        # Both transforms are measured from the same surviving
                        # point identities. Rebase the old seed on this new
                        # source-frame observation, then carry it to the latest
                        # frame without rerunning optical flow over history.
                        seed_pose = inverse(source_motion.base_motion_from_seed) @ estimate.pose
                        corrected = replace(estimate, pose=seed_pose)
                        self.active[estimate.role] = (active[0], corrected)
                        self._publish(estimate.role, corrected, from_latest[0])
                        self._count_anchor_outcome(estimate.role, "reused_point_track")
                        continue
            pending = self.pending.get(estimate.role)
            if pending is not None and pending.anchor.reference_version == estimate.reference_version:
                motion_at_source = next(
                    (
                        motion
                        for frame_id, motion, _ in self.motion_history[estimate.role]
                        if frame_id == source.frame_id
                    ),
                    None,
                )
                if motion_at_source is not None and motion_at_source.base_motion_from_seed is not None:
                    seed_pose = inverse(motion_at_source.base_motion_from_seed) @ estimate.pose
                    pending.anchor = replace(estimate, pose=seed_pose)
                elif source.frame_id > pending.last_frame_id:
                    pending.proposals[source.frame_id] = estimate
                self._count_anchor_outcome(estimate.role, "pending_point_track")
                if self.history[-1].frame_id > source.frame_id and estimate.role not in self.relationships:
                    self._publish_static(estimate.role, estimate, self.history[-1])
                continue
            tracker = CausalRigidPointTracker(**self.tracker_kwargs)
            seeded = tracker.seed(source, mask)
            if seeded < tracker.min_inliers:
                self._count_anchor_outcome(estimate.role, "point_seed_failed")
                if self.history[-1].frame_id > source.frame_id and estimate.role not in self.relationships:
                    self._publish_static(estimate.role, estimate, self.history[-1])
                continue
            self._count_anchor_outcome(estimate.role, "point_seeded")
            self.motion_history[estimate.role] = deque(maxlen=self.history.maxlen)
            self.motion_history[estimate.role].append(
                (
                    source.frame_id,
                    RigidPointMotion(
                        np.eye(4), source.source_time_s, source.available_time_s, seeded, seeded, seeded, 0.0
                    ),
                    tracker.current_pixels(source.depth.shape),
                )
            )
            self.anchors += 1
            self.pending[estimate.role] = _PendingTrack(tracker, estimate, source.frame_id, {})
            # A point track may fail before catching up to the newest camera
            # frame. Keep the last directly measured pose explicitly predicted
            # until another source supports a newer state.
            if self.history[-1].frame_id > source.frame_id and estimate.role not in self.relationships:
                self._publish_static(estimate.role, estimate, self.history[-1])

    def _replay_one(self, role: str) -> bool:
        pending = self.pending[role]
        next_frame = next((frame for frame in self.history if frame.frame_id > pending.last_frame_id), None)
        if next_frame is None:
            if pending.last_frame_id == self.history[-1].frame_id:
                self.active[role] = (pending.tracker, pending.anchor)
                del self.pending[role]
            else:
                # Its history was evicted before a causal replay could catch
                # up. A future direct view must seed a new track.
                del self.pending[role]
                self.motion_history.pop(role, None)
                self._count_anchor_outcome(role, "point_history_expired")
            return False
        try:
            motion = pending.tracker.update(next_frame)
        except ValueError:
            del self.pending[role]
            self.motion_history.pop(role, None)
            self._count_anchor_outcome(role, "point_replay_lost")
            return False
        pending.last_frame_id = next_frame.frame_id
        if motion.base_motion_from_seed is not None:
            self.motion_history[role].append(
                (next_frame.frame_id, motion, pending.tracker.current_pixels(next_frame.depth.shape))
            )
            if (proposal := pending.proposals.pop(next_frame.frame_id, None)) is not None:
                pending.anchor = replace(
                    proposal,
                    pose=inverse(motion.base_motion_from_seed) @ proposal.pose,
                )
            self._publish(role, pending.anchor, motion)
        if next_frame.frame_id == self.history[-1].frame_id:
            self.active[role] = (pending.tracker, pending.anchor)
            del self.pending[role]
            self._count_anchor_outcome(role, "point_replay_caught_up")
        return True

    def _count_anchor_outcome(self, role: str, outcome: str) -> None:
        counts = self.anchor_outcomes_by_role.setdefault(role, {})
        counts[outcome] = counts.get(outcome, 0) + 1

    def _direct_conflicts_with_verified_group(
        self, estimate: PoseEstimate, frame: RGBDFrame, scene: SceneTracks, mask: np.ndarray
    ) -> bool:
        """Keep an unresolved partial view from replacing a verified group anchor.

        A direct fit can supersede the group when the aligned RGB-D surface
        favors it. If neither fit dominates, retain the group's causal track
        and wait for more evidence instead of promoting the disputed fit to a
        stationary prior when the support group later expires.
        """
        relationship = self.relationships.get(estimate.role)
        points = self.object_points_by_role.get(estimate.role)
        if (
            relationship is None
            or points is None
            or estimate.pose is None
            or frame.source_time_s < relationship.anchor_time_s
        ):
            return False
        group_pose = relationship.predict(
            scene, fit_threshold_m=self.fit_threshold_m, min_inliers=self.environment_min_inliers
        )
        if group_pose is None or not self._pose_disagrees(estimate.role, group_pose, estimate.pose):
            return False
        center = points.mean(axis=0)
        radius = float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1))))
        comparison = compare_pose_proposals(
            frame,
            mask,
            points,
            estimate.pose,
            group_pose,
            max_agreement_offset_m=self.fit_threshold_m,
            max_agreement_angle_rad=float(np.arctan2(self.fit_threshold_m, max(radius, np.finfo(float).eps))),
            min_masked_depth_points=self.environment_min_inliers,
        )
        self.last_direct_group_comparison[estimate.role] = comparison
        verdict = comparison.verdict.value
        counts = self.direct_group_disagreements_by_role.setdefault(estimate.role, {})
        counts[verdict] = counts.get(verdict, 0) + 1
        if self.on_comparison is not None:
            self.on_comparison(estimate.role, comparison)
        return comparison.verdict not in (
            PoseComparisonVerdict.AGREE,
            PoseComparisonVerdict.CANDIDATE_FAVORED,
        )

    def _direct_conflicts_with_point_motion(
        self, estimate: PoseEstimate, frame: RGBDFrame, mask: np.ndarray
    ) -> bool:
        """Do not reset a supported point track from a disputed direct fit.

        A delayed direct result is compared at its own camera frame. Its
        point-motion alternative is retained only while enough surviving
        tracked pixels still lie in that frame's object mask. A missing or
        unsupported point prior never prevents direct reacquisition.
        """
        active = self.active.get(estimate.role)
        points = self.object_points_by_role.get(estimate.role)
        if (
            active is None
            or points is None
            or not estimate.has_direct_translation
            or active[1].reference_version != estimate.reference_version
        ):
            return False
        tracked = next(
            (
                (motion, pixels)
                for frame_id, motion, pixels in self.motion_history.get(estimate.role, ())
                if frame_id == frame.frame_id
            ),
            None,
        )
        if tracked is None:
            return False
        motion, pixels = tracked
        if (
            motion.base_motion_from_seed is None
            or motion.source_time_s != frame.source_time_s
            or int(mask[pixels[:, 1], pixels[:, 0]].sum()) < active[0].min_inliers
        ):
            return False
        assert active[1].pose is not None and estimate.pose is not None
        predicted = motion.base_motion_from_seed @ active[1].pose
        symmetry = self.axial_symmetries.get(estimate.role)
        if symmetry is not None:
            try:
                predicted = symmetry.canonicalize(predicted, reference_pose=estimate.pose)
            except ValueError:
                self._count_anchor_outcome(estimate.role, "point_symmetry_invalid")
                return False
        center = points.mean(axis=0)
        radius = float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1))))
        comparison = compare_pose_proposals(
            frame,
            mask,
            points,
            estimate.pose,
            predicted,
            max_agreement_offset_m=self.fit_threshold_m,
            max_agreement_angle_rad=float(np.arctan2(self.fit_threshold_m, max(radius, np.finfo(float).eps))),
            min_masked_depth_points=active[0].min_inliers,
        )
        self.last_direct_point_comparison[estimate.role] = comparison
        counts = self.direct_point_disagreements_by_role.setdefault(estimate.role, {})
        counts[comparison.verdict.value] = counts.get(comparison.verdict.value, 0) + 1
        if self.on_point_comparison is not None:
            self.on_point_comparison(estimate.role, comparison)
        return comparison.verdict not in (
            PoseComparisonVerdict.AGREE,
            PoseComparisonVerdict.CANDIDATE_FAVORED,
        )

    def _record_direct_free_space(self, estimate: PoseEstimate, frame: RGBDFrame) -> None:
        """Record an independent contradiction without yet changing pose ownership."""
        points = self.object_points_by_role.get(estimate.role)
        if points is None or not estimate.has_direct_translation:
            return
        assert estimate.pose is not None
        support = free_space_support(
            frame,
            points,
            estimate.pose,
            depth_tolerance_m=self.fit_threshold_m,
            max_local_depth_span_m=self.max_pose_disagreement_translation_m,
        )
        if support.valid_depth_pixels < self.tracker_kwargs["min_inliers"]:
            verdict = FreeSpaceVerdict.INSUFFICIENT_DEPTH
        elif support.median_signed_gap_m is not None and (
            support.median_signed_gap_m > self.max_pose_disagreement_translation_m
        ):
            verdict = FreeSpaceVerdict.CONTRADICTED
        else:
            verdict = FreeSpaceVerdict.NOT_CONTRADICTED
        comparison = FreeSpaceComparison(verdict, support, self.max_pose_disagreement_translation_m)
        self.last_direct_free_space_comparison[estimate.role] = comparison
        counts = self.direct_free_space_counts_by_role.setdefault(estimate.role, {})
        counts[verdict.value] = counts.get(verdict.value, 0) + 1
        if self.on_free_space_comparison is not None:
            self.on_free_space_comparison(estimate.role, comparison)

    def _observe_environment(
        self, estimate: PoseEstimate, scene: SceneTracks, masks: dict[str, np.ndarray]
    ) -> None:
        """Verify co-motion before a group can predict a hidden object."""
        role = estimate.role
        # An independent direct view can re-establish the stationary prior
        # after a formerly verified support group lost all measurable points.
        self._expired_verified_roles.discard(role)
        existing = self.relationships.get(role) or self.candidate_relationships.get(role)
        if existing is not None:
            predicted = existing.predict(
                scene, fit_threshold_m=self.fit_threshold_m, min_inliers=self.environment_min_inliers
            )
            if (
                predicted is None
                or estimate.pose is None
                or self._pose_disagrees(role, predicted, estimate.pose)
            ):
                self.relationships.pop(role, None)
                self.candidate_relationships.pop(role, None)
            elif scene.source_time_s > existing.anchor_time_s:
                current_ids = {int(point_id): index for index, point_id in enumerate(scene.ids)}
                keep = [int(point_id) for point_id in existing.ids if int(point_id) in current_ids]
                if len(keep) >= self.environment_min_inliers:
                    relationship = EnvironmentRelationship(
                        np.asarray(keep, dtype=np.int64),
                        scene.base_xyz[[current_ids[point_id] for point_id in keep]].copy(),
                        estimate.pose,
                        scene.source_time_s,
                        estimate.reference_version,
                        max(existing.last_group_measurement_s or existing.anchor_time_s, scene.source_time_s),
                    )
                    if role in self.candidate_relationships:
                        self.relationships[role] = relationship
                        self.candidate_relationships.pop(role, None)
                        self.environment_relationships += 1
                        self.environment_relationships_by_role[role] = (
                            self.environment_relationships_by_role.get(role, 0) + 1
                        )
                    else:
                        self.relationships[role] = relationship
        if role not in self.relationships and role not in self.candidate_relationships:
            previous = self.observed.get(role)
            if previous is not None:
                old_estimate, old_scene, old_masks = previous
                if old_estimate.pose is not None and estimate.pose is not None:
                    candidate = establish_environment_relationship(
                        old_scene,
                        scene,
                        old_estimate.pose,
                        estimate.pose,
                        old_masks,
                        context_radius_m=self.environment_context_radius_m,
                        fit_threshold_m=self.fit_threshold_m,
                        min_motion_m=self.environment_min_motion_m,
                        min_inliers=self.environment_min_inliers,
                        reference_version=estimate.reference_version,
                        allow_stationary=True,
                        object_center_object_m=self.object_centers_by_role.get(role),
                    )
                    if candidate is not None:
                        self.candidate_relationships[role] = candidate
            if role not in self.candidate_relationships:
                seed = seed_environment_relationship(
                    scene,
                    estimate.pose,
                    tuple(masks.values()),
                    context_radius_m=self.environment_context_radius_m,
                    min_inliers=self.environment_min_inliers,
                    reference_version=estimate.reference_version,
                    object_center_object_m=self.object_centers_by_role.get(role),
                )
                if seed is not None:
                    self.candidate_relationships[role] = seed
        self.observed[role] = (estimate, scene, tuple(mask.copy() for mask in masks.values()))

    def _pose_disagrees(self, role: str, predicted: np.ndarray, observed: np.ndarray) -> bool:
        local_center = self.object_centers_by_role.get(role)
        if local_center is None:
            local_center = np.zeros(3)
        before = transform_points(predicted, local_center[None])[0]
        after = transform_points(observed, local_center[None])[0]
        if np.linalg.norm(before - after) > self.max_pose_disagreement_translation_m:
            return True
        turn = predicted[:3, :3].T @ observed[:3, :3]
        return (
            float(np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1)))
            > self.max_pose_disagreement_rotation_rad
        )

    def _publish(self, role: str, anchor: PoseEstimate, motion) -> None:
        if motion.base_motion_from_seed is None:
            return
        assert anchor.pose is not None
        pose = motion.base_motion_from_seed @ anchor.pose
        symmetry = self.axial_symmetries.get(role)
        if symmetry is not None:
            try:
                previous = self.store.snapshot()[1].get(role)
                reference = previous.pose if previous is not None else anchor.pose
                pose = symmetry.canonicalize(pose, reference_pose=reference)
            except ValueError:
                self._count_anchor_outcome(role, "point_symmetry_invalid")
                return
        current = PoseEstimate(
            role,
            pose,
            motion.source_time_s,
            max(motion.available_time_s, time.perf_counter()),
            # The current RGB-D frame measures point motion. It does not
            # independently recognize the object's absolute pose or identity;
            # those still come from the older anchor and can drift or be
            # corrupted by a bad seed mask. Keep that distinction visible to
            # the planner and the recorded evidence stream.
            Evidence.PREDICTED,
            anchor.reference_version,
            anchor.translation_supported,
            anchor.rotation_supported,
            estimator="object_point_motion",
            last_direct_observation_s=anchor.source_time_s,
            symmetry_canonicalized=symmetry is not None,
        )
        before, _ = self.store.snapshot()
        after = self.store.publish((current,))
        if after != before:
            if self.on_layer is not None:
                self.on_layer(current)
            self.published += 1
            self.published_by_role[role] = self.published_by_role.get(role, 0) + 1

    def close(self, timeout_s: float = 5.0) -> None:
        if not self._started:
            return
        try:
            self.queue.put(None, timeout=timeout_s)
        except Full as exc:
            raise RuntimeError("causal pose service did not drain") from exc
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("causal pose service did not stop")
        self._started = False


def build_causal_pose_service(
    store: LatestPoseStore,
    config: CausalTrackingConfig,
    *,
    object_centers_by_role: dict[str, np.ndarray] | None = None,
    object_points_by_role: dict[str, np.ndarray] | None = None,
    axial_symmetries: tuple[AxialSymmetry, ...] = (),
    on_layer: Callable[[PoseEstimate], None] | None = None,
    on_comparison: Callable[[str, PoseComparison], None] | None = None,
    on_point_comparison: Callable[[str, PoseComparison], None] | None = None,
    on_free_space_comparison: Callable[[str, FreeSpaceComparison], None] | None = None,
    on_direct_observation: Callable[[PoseEstimate], None] | None = None,
) -> CausalPoseService:
    """Construct the identical estimator policy for recorded and live frames."""
    return CausalPoseService(
        store,
        **asdict(config),
        object_centers_by_role=object_centers_by_role,
        object_points_by_role=object_points_by_role,
        axial_symmetries=axial_symmetries,
        on_layer=on_layer,
        on_comparison=on_comparison,
        on_point_comparison=on_point_comparison,
        on_free_space_comparison=on_free_space_comparison,
        on_direct_observation=on_direct_observation,
    )
