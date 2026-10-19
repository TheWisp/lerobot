"""Bind demonstrated gripper events to one live rigid-object attachment."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from .execution import MotorTick
from .perception import HeldAttachmentUpdater, LatestPoseStore
from .skill import AttachmentAction, FlashSkill
from .trajectory import AttachmentState, Evidence, PoseEstimate


class AttachmentLifecycle:
    """Translate accepted motor events into explicit attachment assumptions.

    An attach event seeds the latest fresh pregrasp observation, if one exists.
    This is an assumption until a later independent held observation reaches
    the updater. A detach event withdraws that assumption; release success is
    still decided by a separate milestone, not by sending a gripper command.
    """

    def __init__(
        self,
        skill: FlashSkill,
        poses: LatestPoseStore,
        updaters: Mapping[str, HeldAttachmentUpdater],
        *,
        max_pregrasp_age_s: float,
    ) -> None:
        if not isfinite(max_pregrasp_age_s) or max_pregrasp_age_s <= 0:
            raise ValueError("maximum pregrasp pose age must be positive")
        self.transitions = {transition.event_id: transition for transition in skill.attachment_transitions}
        self.poses = poses
        self.updaters = dict(updaters)
        for transition in self.transitions.values():
            updater = self.updaters.get(transition.role)
            if updater is None or updater.reference_version != transition.reference_version:
                raise ValueError(f"{transition.event_id}: missing matching held-object updater")
        self.max_pregrasp_age_s = max_pregrasp_age_s
        self.active_role: str | None = None
        self.last_event: str | None = None

    def on_motor_tick(self, tick: MotorTick) -> None:
        for updater in self.updaters.values():
            updater.on_motor_tick(tick)
        receipt = tick.receipt
        event_id = receipt.requested.event_id if receipt is not None else None
        transition = self.transitions.get(event_id) if event_id is not None else None
        if transition is None:
            return
        if transition.action == AttachmentAction.ATTACH:
            if self.active_role is not None:
                raise RuntimeError(f"cannot attach {transition.role} while holding {self.active_role}")
            _, poses = self.poses.snapshot()
            seed = poses.get(transition.role)
            if seed is not None and not self._fresh_seed(
                seed, transition.reference_version, receipt.dispatched_s
            ):
                seed = None
            self.updaters[transition.role].activate(receipt.dispatched_s, seed)
            self.active_role = transition.role
        else:
            if self.active_role != transition.role:
                raise RuntimeError(f"cannot detach {transition.role}: it is not the active attachment")
            self.updaters[transition.role].deactivate()
            self.active_role = None
        self.last_event = event_id

    def _fresh_seed(self, pose: PoseEstimate, version: str, at_s: float) -> bool:
        return (
            pose.reference_version == version
            and pose.evidence == Evidence.OBSERVED
            and pose.pose is not None
            and pose.supports()
            and pose.available_time_s <= at_s
            and 0 <= at_s - pose.source_time_s <= self.max_pregrasp_age_s
        )

    def on_pose_update(self, version: int, poses: dict[str, PoseEstimate]) -> bool:
        return self.active_role is not None and self.updaters[self.active_role].on_pose_update(version, poses)

    def state(self, now_s: float) -> AttachmentState | None:
        return self.updaters[self.active_role].state(now_s) if self.active_role is not None else None
