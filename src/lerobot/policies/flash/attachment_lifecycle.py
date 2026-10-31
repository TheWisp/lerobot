"""Bind demonstrated gripper events to one live rigid-object attachment."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from math import isfinite
from threading import Lock

import numpy as np

from .execution import MotorTick
from .perception import HeldAttachmentUpdater, LatestPoseStore, RGBDFrame
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
        # Motor samples retain the relation that was valid at that source
        # time. Camera processing may trail a release, so the present active
        # role cannot answer a historical frame's prediction.
        self._prediction_lock = Lock()
        self._prediction_samples = {
            role: deque(maxlen=updater.history.capacity) for role, updater in self.updaters.items()
        }
        self._role_events: dict[str, deque[tuple[float, bool, int]]] = {
            role: deque(maxlen=updater.history.capacity + 1) for role, updater in self.updaters.items()
        }
        self._attachment_epoch = 0
        self._last_direct_by_role: dict[str, float | None] = dict.fromkeys(self.updaters)

    def on_motor_tick(self, tick: MotorTick) -> None:
        for updater in self.updaters.values():
            updater.on_motor_tick(tick)
        if self.active_role is not None:
            role = self.active_role
            updater = self.updaters[role]
            state = updater.state(tick.observed_at_s)
            joints = np.array([float(tick.observed[f"{name}.pos"]) for name in updater.arm_joint_names])
            with self._prediction_lock:
                self._prediction_samples[role].append(
                    (
                        tick.observed_at_s,
                        joints,
                        state,
                        self._last_direct_by_role[role],
                        self._attachment_epoch,
                    )
                )
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
            grasp_joints = [
                float(tick.observed[f"{name}.pos"]) for name in self.updaters[transition.role].arm_joint_names
            ]
            self.updaters[transition.role].activate(receipt.dispatched_s, seed, grasp_joints)
            self.active_role = transition.role
            with self._prediction_lock:
                self._attachment_epoch += 1
                self._last_direct_by_role[transition.role] = (
                    seed.source_time_s
                    if seed is not None and seed.evidence == Evidence.OBSERVED
                    else seed.last_direct_observation_s
                    if seed is not None
                    else None
                )
                self._role_events[transition.role].append(
                    (receipt.dispatched_s, True, self._attachment_epoch)
                )
        else:
            if self.active_role != transition.role:
                raise RuntimeError(f"cannot detach {transition.role}: it is not the active attachment")
            self.updaters[transition.role].deactivate()
            self.active_role = None
            with self._prediction_lock:
                self._role_events[transition.role].append(
                    (receipt.dispatched_s, False, self._attachment_epoch)
                )
        self.last_event = event_id

    def _fresh_seed(self, pose: PoseEstimate, version: str, at_s: float) -> bool:
        return (
            pose.reference_version == version
            and pose.evidence in (Evidence.OBSERVED, Evidence.PREDICTED)
            and pose.pose is not None
            and pose.supports()
            and pose.available_time_s <= at_s
            and 0 <= at_s - pose.source_time_s <= self.max_pregrasp_age_s
        )

    def on_pose_update(
        self,
        version: int,
        poses: dict[str, PoseEstimate],
        *,
        frame: RGBDFrame | None = None,
        mask: np.ndarray | None = None,
    ) -> bool:
        with self._prediction_lock:
            role = self.active_role
            epoch = self._attachment_epoch
        if role is None:
            return False
        updater = self.updaters[role]
        accepted = updater.on_pose_update(version, poses, frame=frame, mask=mask)
        held = poses.get(role)
        if held is not None and held.evidence == Evidence.OBSERVED:
            # A delayed visual correction belongs to its image time, not the
            # motor tick on which the worker happened to accept it.
            with self._prediction_lock:
                if self.active_role != role or self._attachment_epoch != epoch:
                    return accepted
                if accepted:
                    self._last_direct_by_role[role] = held.source_time_s
                samples = self._prediction_samples[role]
                for index, (time_s, joints, _state, direct_s, sample_epoch) in enumerate(samples):
                    if sample_epoch == epoch and time_s >= held.source_time_s:
                        samples[index] = (
                            time_s,
                            joints,
                            updater.state(time_s),
                            held.source_time_s if accepted else direct_s,
                            sample_epoch,
                        )
        return accepted

    def prediction_at(self, role: str, frame: RGBDFrame) -> PoseEstimate | None:
        """Predict a held pose for one camera time using bounded measured motor history."""
        updater = self.updaters.get(role)
        if updater is None:
            return None
        with self._prediction_lock:
            events = self._role_events[role]
            event = next((item for item in reversed(events) if item[0] <= frame.source_time_s), None)
            if event is None or not event[1]:
                return None
            sample = next(
                (
                    item
                    for item in reversed(self._prediction_samples[role])
                    if item[0] <= frame.source_time_s and item[4] == event[2]
                ),
                None,
            )
        if sample is None:
            return None
        motor_time_s, joints, state, last_direct_s, _ = sample
        if (
            state is None
            or frame.source_time_s - motor_time_s > updater.history.max_sample_gap_s
            or (
                updater.max_attachment_age_s is not None
                and frame.source_time_s - state.source_time_s > updater.max_attachment_age_s
            )
        ):
            return None
        pose = updater.kinematics.forward_kinematics(joints) @ state.gripper_from_object
        return PoseEstimate(
            role,
            pose,
            frame.source_time_s,
            frame.available_time_s,
            Evidence.PREDICTED,
            state.reference_version,
            (True,) * 3,
            (True,) * 3,
            estimator="gripper_motion",
            source_frame_id=frame.frame_id,
            last_direct_observation_s=last_direct_s,
        )

    def state(self, now_s: float) -> AttachmentState | None:
        return self.updaters[self.active_role].state(now_s) if self.active_role is not None else None
