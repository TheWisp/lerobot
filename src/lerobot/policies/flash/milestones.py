"""Evidence-based milestone checks, run outside the motor dispatcher."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import numpy as np

from .geometry import inverse
from .progress import MilestoneUpdate, Verdict
from .skill import MilestoneSpec
from .trajectory import PoseEstimate


class PoseMilestonePublisher:
    """Turn each newly published pose snapshot into session evidence.

    Call from the perception collector, outside the fixed-rate motor thread.
    Milestones without a pose relation remain the responsibility of another
    evidence source, such as a gripper or task-specific outcome verifier.
    """

    def __init__(
        self,
        milestones: tuple[MilestoneSpec, ...],
        publish: Callable[[MilestoneUpdate], None],
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.milestones = tuple(item for item in milestones if item.pose_relation is not None)
        self.publish = publish
        self.clock = clock
        self._last_sent: dict[str, tuple[str, Verdict]] = {}

    def __call__(self, pose_version: int, poses: Mapping[str, PoseEstimate]) -> None:
        now_s = self.clock()
        for milestone in self.milestones:
            update = verify_pose_relation(milestone, poses, pose_version=pose_version, now_s=now_s)
            identity = (update.source_id, update.verdict)
            if self._last_sent.get(milestone.name) != identity:
                self.publish(update)
                self._last_sent[milestone.name] = identity


def verify_pose_relation(
    milestone: MilestoneSpec,
    poses: Mapping[str, PoseEstimate],
    *,
    pose_version: int,
    now_s: float,
) -> MilestoneUpdate:
    """Check one annotated relation using only fresh independent observations.

    The evidence identity uses the two source-frame times, so an unrelated
    pose-store update cannot count the same camera evidence twice. Expiration
    gets a distinct identity so it can revoke a previously true verdict.
    Missing, predicted, stale or partly supported evidence yields UNKNOWN.
    The tolerances belong to the skill annotation, not to the tracker.
    """
    criterion = milestone.pose_relation
    if criterion is None:
        raise ValueError(f"{milestone.name} has no pose relation criterion")
    if pose_version < 0 or not np.isfinite(now_s):
        raise ValueError("pose version and verification time must be valid")
    moving = poses.get(criterion.moving_role)
    reference = poses.get(criterion.reference_role)
    if moving is not None and reference is not None:
        source_id = "|".join(
            f"{pose.role}:{pose.source_time_s!r}:{pose.reference_version}:"
            f"{pose.evidence.value}:{pose.translation_supported!r}:{pose.rotation_supported!r}"
            for pose in (moving, reference)
        )
        if any(now_s - pose.source_time_s > criterion.max_pose_age_s for pose in (moving, reference)):
            source_id += "|expired"
        elif any(now_s < pose.available_time_s for pose in (moving, reference)):
            source_id += "|unavailable"
    else:
        source_id = f"poses:{pose_version}"
    source_time_s = max(
        (pose.source_time_s for pose in (moving, reference) if pose is not None), default=now_s
    )

    def update(verdict: Verdict, reason: str) -> MilestoneUpdate:
        return MilestoneUpdate(milestone.name, verdict, source_id, source_time_s, reason)

    if moving is None or reference is None:
        return update(Verdict.UNKNOWN, "one or both object poses are missing")
    if (
        moving.reference_version != criterion.moving_version
        or reference.reference_version != criterion.reference_version
    ):
        return update(Verdict.UNKNOWN, "object frame version differs from milestone annotation")
    if not reference.usable(now_s, criterion.max_pose_age_s, False):
        return update(Verdict.UNKNOWN, "reference pose is not freshly and fully observed")
    moving_rotation = (criterion.rotation_tolerance_rad is not None,) * 3
    if not moving.usable(now_s, criterion.max_pose_age_s, False, rotation=moving_rotation):
        return update(Verdict.UNKNOWN, "moving pose lacks fresh observed required axes")

    actual = inverse(reference.pose) @ moving.pose
    error = inverse(criterion.reference_from_moving) @ actual
    translation_error_m = float(np.linalg.norm(error[:3, 3]))
    rotation_error_rad = float(np.arccos(np.clip((np.trace(error[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
    if translation_error_m > criterion.translation_tolerance_m or (
        criterion.rotation_tolerance_rad is not None and rotation_error_rad > criterion.rotation_tolerance_rad
    ):
        return update(
            Verdict.FALSE,
            f"relative pose error: {translation_error_m:.4f} m, {rotation_error_rad:.4f} rad",
        )
    return update(
        Verdict.TRUE,
        f"relative pose verified: {translation_error_m:.4f} m, {rotation_error_rad:.4f} rad",
    )
