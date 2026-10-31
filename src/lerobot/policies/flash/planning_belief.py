"""Atomic, read-only planning projection of scene and held-object evidence."""

from __future__ import annotations

from threading import Lock

from .trajectory import AttachmentState, PoseEstimate


class PlanningBeliefStore:
    """Publish one consistent scene/attachment pair to the motor coordinator.

    The scene estimator remains the pose owner. Its asynchronous callbacks
    prepare attachment corrections first, then publish this short-lock
    projection. A planner that is already running keeps its earlier immutable
    snapshot until the next scheduling tick.
    """

    def __init__(self, *, max_attachment_age_s: float | None) -> None:
        self._lock = Lock()
        self._pose_version = -1
        self._poses: dict[str, PoseEstimate] = {}
        self._attachment_revision = -1
        self._active_role: str | None = None
        self._attachment: AttachmentState | None = None
        self._max_attachment_age_s = max_attachment_age_s

    def publish_scene(
        self,
        pose_version: int,
        poses: dict[str, PoseEstimate],
        attachment_revision: int,
        active_role: str | None,
        attachment: AttachmentState | None,
    ) -> None:
        """Install a collector snapshot after its held evidence has been applied."""
        with self._lock:
            if pose_version >= self._pose_version:
                self._pose_version = pose_version
                self._poses = dict(poses)
            self._publish_attachment_locked(attachment_revision, active_role, attachment)

    def publish_attachment(
        self,
        attachment_revision: int,
        active_role: str | None,
        attachment: AttachmentState | None,
    ) -> None:
        """Publish a gripper event without inventing a new camera pose version."""
        with self._lock:
            self._publish_attachment_locked(attachment_revision, active_role, attachment)

    def _publish_attachment_locked(
        self,
        revision: int,
        active_role: str | None,
        attachment: AttachmentState | None,
    ) -> None:
        if revision >= self._attachment_revision:
            self._attachment_revision = revision
            self._active_role = active_role
            self._attachment = attachment

    def snapshot(
        self, now_s: float, held_role: str | None
    ) -> tuple[int, dict[str, PoseEstimate], AttachmentState | None]:
        """Return one version of both inputs; expire a held relation by time."""
        with self._lock:
            version = self._pose_version
            poses = dict(self._poses)
            attachment = self._attachment if held_role == self._active_role else None
        if attachment is not None and (
            now_s < attachment.available_time_s
            or now_s < attachment.source_time_s
            or (
                self._max_attachment_age_s is not None
                and now_s - attachment.source_time_s > self._max_attachment_age_s
            )
        ):
            attachment = None
        return version, poses, attachment

    def pose_snapshot(self) -> tuple[int, dict[str, PoseEstimate]]:
        """Return exactly the pose version the planner can read for recording."""
        with self._lock:
            return self._pose_version, dict(self._poses)
