"""Evidence-based milestone checks, run outside the motor dispatcher."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping

import numpy as np

from .geometry import inverse
from .progress import MilestoneUpdate, Verdict
from .skill import MilestoneSpec
from .trajectory import Evidence, PoseEstimate, required_rotation_axes


def _usable_direct_components(
    pose: PoseEstimate, now_s: float, max_age_s: float, rotation: tuple[bool, bool, bool]
) -> bool:
    """A direct centre measurement can verify a position-only relation.

    Symmetry may make the complete pose predicted while the current RGB-D
    frame still measures its translation. It cannot verify any orientation
    that was filled from a prior or from the gripper.
    """
    return (
        pose.pose is not None
        and pose.evidence != Evidence.LOST
        and 0 <= now_s - pose.source_time_s <= max_age_s
        and now_s >= pose.available_time_s
        and pose.supports(rotation=rotation)
        and (pose.evidence == Evidence.OBSERVED or (pose.direct_translation_observed and not any(rotation)))
    )


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
    Missing, stale or partly supported evidence yields UNKNOWN. A directly
    measured translation may verify a position-only relation even when an
    irrelevant rotation was predicted; a wholly predicted pose never can.
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
        # A delayed estimator can revise a pose on the same RGB-D frame. Its
        # source time stays fixed, but a changed transform must supersede the
        # earlier verdict. Unchanged poses keep the same ID across unrelated
        # pose-store revisions, so they cannot count as fresh evidence.
        def pose_identity(pose: PoseEstimate) -> str:
            transform_id = (
                hashlib.blake2b(pose.pose.tobytes(order="C"), digest_size=16).hexdigest()
                if pose.pose is not None
                else "none"
            )
            return (
                f"{pose.role}:{pose.source_frame_id}:{pose.source_time_s!r}:"
                f"{pose.reference_version}:{pose.evidence.value}:{pose.direct_translation_observed}:"
                f"{pose.translation_supported!r}:{pose.rotation_supported!r}:{transform_id}"
            )

        source_id = "|".join(pose_identity(pose) for pose in (moving, reference))
        if any(now_s - pose.source_time_s > criterion.max_pose_age_s for pose in (moving, reference)):
            source_id += "|expired"
        elif any(now_s < pose.available_time_s for pose in (moving, reference)):
            source_id += "|unavailable"
    else:
        source_id = f"poses:{pose_version}"
    source_time_s = max(
        (pose.source_time_s for pose in (moving, reference) if pose is not None), default=now_s
    )
    oldest_source_time_s = min(
        (pose.source_time_s for pose in (moving, reference) if pose is not None), default=now_s
    )

    def update(verdict: Verdict, reason: str) -> MilestoneUpdate:
        return MilestoneUpdate(
            milestone.name, verdict, source_id, source_time_s, reason, oldest_source_time_s
        )

    if moving is None or reference is None:
        return update(Verdict.UNKNOWN, "one or both object poses are missing")
    if (
        moving.reference_version != criterion.moving_version
        or reference.reference_version != criterion.reference_version
    ):
        return update(Verdict.UNKNOWN, "object frame version differs from milestone annotation")
    local_point = np.asarray(
        criterion.moving_point_m if criterion.moving_point_m is not None else (0, 0, 0), dtype=float
    )
    local_direction = (
        np.asarray(criterion.moving_direction, dtype=float)
        if criterion.moving_direction is not None
        else None
    )
    expected = criterion.reference_from_moving
    reference_point = expected[:3, :3] @ local_point + expected[:3, 3]
    reference_direction = expected[:3, :3] @ local_direction if local_direction is not None else None
    if criterion.rotation_tolerance_rad is not None:
        reference_rotation = moving_rotation = (True,) * 3
    else:
        reference_rotation = required_rotation_axes(reference_point, reference_direction)
        moving_rotation = required_rotation_axes(local_point, local_direction)
    if not _usable_direct_components(reference, now_s, criterion.max_pose_age_s, reference_rotation):
        return update(Verdict.UNKNOWN, "reference pose lacks fresh observed required axes")
    if not _usable_direct_components(moving, now_s, criterion.max_pose_age_s, moving_rotation):
        return update(Verdict.UNKNOWN, "moving pose lacks fresh observed required axes")

    actual = inverse(reference.pose) @ moving.pose
    actual_point = actual[:3, :3] @ local_point + actual[:3, 3]
    translation_error_m = float(np.linalg.norm(actual_point - reference_point))
    if local_direction is not None:
        actual_direction = actual[:3, :3] @ local_direction
        expected_direction = reference_direction
        rotation_error_rad = float(np.arccos(np.clip(actual_direction @ expected_direction, -1.0, 1.0)))
        rotation_limit = criterion.direction_tolerance_rad
    else:
        error = inverse(expected) @ actual
        rotation_error_rad = float(np.arccos(np.clip((np.trace(error[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        rotation_limit = criterion.rotation_tolerance_rad
    if translation_error_m > criterion.translation_tolerance_m or (
        rotation_limit is not None and rotation_error_rad > rotation_limit
    ):
        return update(
            Verdict.FALSE,
            f"relative pose error: {translation_error_m:.4f} m, {rotation_error_rad:.4f} rad",
        )
    return update(
        Verdict.TRUE,
        f"relative pose verified: {translation_error_m:.4f} m, {rotation_error_rad:.4f} rad",
    )
