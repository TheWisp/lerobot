"""Every-frame point motion alongside slower, replaceable pose estimation."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, replace
from math import isfinite
from queue import Full, Queue
from threading import Lock, Thread

import numpy as np

from .causal_points import CausalRigidPointTracker, RigidPointMotion
from .geometry import inverse
from .perception import LatestPoseStore, RGBDFrame
from .scene_groups import (
    EnvironmentRelationship,
    ScenePointTracker,
    SceneTracks,
    establish_environment_relationship,
)
from .trajectory import Evidence, PoseEstimate


@dataclass(frozen=True)
class _Anchor:
    frame_id: int
    estimates: tuple[PoseEstimate, ...]
    masks: dict[str, np.ndarray]
    source_time_s: float | None = None
    calibration_version: str | None = None


@dataclass(frozen=True)
class CausalTrackingConfig:
    history_frames: int
    queue_frames: int
    min_inliers: int
    max_frame_gap_s: float
    fit_threshold_m: float
    image_scale: float = 1.0
    scene_max_points: int = 800
    environment_context_radius_m: float = 0.25
    environment_min_motion_m: float = 0.008
    environment_prediction_max_s: float = 1.0
    environment_min_inliers: int = 12
    static_prediction_max_s: float | None = None

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
            or self.scene_max_points < 16
            or not isfinite(self.environment_context_radius_m)
            or self.environment_context_radius_m <= 0
            or not isfinite(self.environment_min_motion_m)
            or self.environment_min_motion_m <= 0
            or not isfinite(self.environment_prediction_max_s)
            or self.environment_prediction_max_s <= 0
            or self.environment_min_inliers < 3
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
        scene_max_points: int = 800,
        environment_context_radius_m: float = 0.25,
        environment_min_motion_m: float = 0.008,
        environment_prediction_max_s: float = 1.0,
        environment_min_inliers: int = 12,
        static_prediction_max_s: float | None = None,
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
        }
        self.active: dict[str, tuple[CausalRigidPointTracker, PoseEstimate]] = {}
        self.motion_history: dict[str, deque[tuple[int, RigidPointMotion, np.ndarray]]] = {}
        self.scene = ScenePointTracker(
            max_points=scene_max_points,
            max_frame_gap_s=max_frame_gap_s,
            image_scale=image_scale,
        )
        self.environment_context_radius_m = environment_context_radius_m
        self.environment_min_motion_m = environment_min_motion_m
        self.environment_prediction_max_s = environment_prediction_max_s
        self.environment_min_inliers = environment_min_inliers
        self.static_prediction_max_s = static_prediction_max_s
        self.fit_threshold_m = fit_threshold_m
        self.observed: dict[str, tuple[PoseEstimate, SceneTracks, tuple[np.ndarray, ...]]] = {}
        self.candidate_relationships: dict[str, EnvironmentRelationship] = {}
        self.relationships: dict[str, EnvironmentRelationship] = {}
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
        self.environment_published = 0
        self.frame_compute_total_s = 0.0
        self.frame_compute_max_s = 0.0
        self.anchor_compute_total_s = 0.0
        self.anchor_compute_max_s = 0.0
        self.max_queue_age_s = 0.0

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
        except Exception as exc:
            with self._lock:
                self._error = exc

    def _frame(self, frame: RGBDFrame) -> None:
        if self.history and frame.frame_id <= self.history[-1].frame_id:
            raise ValueError("causal camera frames must increase")
        self.history.append(frame)
        scene = self.scene.update(frame)
        self.scene_history.append(scene)
        self.frames += 1
        self.max_processing_lag_s = max(self.max_processing_lag_s, time.perf_counter() - frame.source_time_s)
        self.latest_processed_source_time_s = frame.source_time_s
        object_motion_roles: set[str] = set()
        for role, (tracker, anchor) in list(self.active.items()):
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
        for role, relationship in list(self.relationships.items()):
            if frame.source_time_s - relationship.anchor_time_s > self.environment_prediction_max_s:
                del self.relationships[role]
                continue
            if role in object_motion_roles:
                continue
            pose = relationship.predict(
                scene, fit_threshold_m=self.fit_threshold_m, min_inliers=self.environment_min_inliers
            )
            if pose is None:
                continue
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
                estimator="environment_group_motion",
                last_direct_observation_s=relationship.anchor_time_s,
            )
            previous_version, _ = self.store.snapshot()
            if self.store.publish((estimate,)) != previous_version:
                self.environment_published += 1
        # A missing visual fit is not evidence that a stationary object moved.
        # Carry the last direct pose for a bounded interval, with explicit
        # prediction provenance. Once co-motion has been verified, only that
        # relationship may carry the pose: a static prior must not hide a
        # moved support surface or a failed group fit.
        for role, (anchor, _, _) in self.observed.items():
            if role not in object_motion_roles and role not in self.relationships:
                self._publish_static(role, anchor, frame)

    def _publish_static(self, role: str, anchor: PoseEstimate, frame: RGBDFrame) -> None:
        if self.static_prediction_max_s is None:
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
            self.static_published += 1

    def _anchor(self, result: _Anchor) -> None:
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
            if estimate.has_direct_translation:
                self._observe_environment(estimate, scene, result.masks)
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
            tracker = CausalRigidPointTracker(**self.tracker_kwargs)
            seeded = tracker.seed(source, mask)
            if seeded < tracker.min_inliers:
                self._count_anchor_outcome(estimate.role, "point_seed_failed")
                if self.history[-1].frame_id > source.frame_id and estimate.role not in self.relationships:
                    self._publish_static(estimate.role, estimate, self.history[-1])
                continue
            self._count_anchor_outcome(estimate.role, "point_seeded")
            self.active[estimate.role] = (tracker, estimate)
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
            for frame in self.history:
                if frame.frame_id <= source.frame_id:
                    continue
                try:
                    motion = tracker.update(frame)
                except ValueError:
                    del self.active[estimate.role]
                    self.motion_history.pop(estimate.role, None)
                    self._count_anchor_outcome(estimate.role, "point_replay_lost")
                    break
                if motion.base_motion_from_seed is not None:
                    self.motion_history[estimate.role].append(
                        (frame.frame_id, motion, tracker.current_pixels(frame.depth.shape))
                    )
                self._publish(estimate.role, estimate, motion)
            # A point track may fail before catching up to the newest camera
            # frame. Keep the last directly measured pose explicitly predicted
            # until another source supports a newer state.
            if self.history[-1].frame_id > source.frame_id and estimate.role not in self.relationships:
                self._publish_static(estimate.role, estimate, self.history[-1])

    def _count_anchor_outcome(self, role: str, outcome: str) -> None:
        counts = self.anchor_outcomes_by_role.setdefault(role, {})
        counts[outcome] = counts.get(outcome, 0) + 1

    def _observe_environment(
        self, estimate: PoseEstimate, scene: SceneTracks, masks: dict[str, np.ndarray]
    ) -> None:
        """Verify co-motion before a group can predict a hidden object."""
        role = estimate.role
        existing = self.relationships.get(role) or self.candidate_relationships.get(role)
        if existing is not None:
            predicted = existing.predict(
                scene, fit_threshold_m=self.fit_threshold_m, min_inliers=self.environment_min_inliers
            )
            if predicted is None or estimate.pose is None or self._pose_disagrees(predicted, estimate.pose):
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
                    )
                    if role in self.candidate_relationships:
                        self.relationships[role] = relationship
                        self.candidate_relationships.pop(role, None)
                        self.environment_relationships += 1
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
                    )
                    if candidate is not None:
                        self.candidate_relationships[role] = candidate
        self.observed[role] = (estimate, scene, tuple(mask.copy() for mask in masks.values()))

    def _pose_disagrees(self, predicted: np.ndarray, observed: np.ndarray) -> bool:
        if np.linalg.norm(predicted[:3, 3] - observed[:3, 3]) > 2 * self.fit_threshold_m:
            return True
        turn = predicted[:3, :3].T @ observed[:3, :3]
        return float(np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1))) > 0.15

    def _publish(self, role: str, anchor: PoseEstimate, motion) -> None:
        if motion.base_motion_from_seed is None:
            return
        assert anchor.pose is not None
        current = PoseEstimate(
            role,
            motion.base_motion_from_seed @ anchor.pose,
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
        )
        before, _ = self.store.snapshot()
        after = self.store.publish((current,))
        if after != before:
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
