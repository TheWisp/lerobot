"""Bind demonstrated gripper events to one live rigid-object attachment."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from threading import Lock

import numpy as np

from .execution import MotorTick
from .perception import HeldAttachmentUpdater, LatestPoseStore, RGBDFrame
from .skill import AttachmentAction, FlashSkill
from .trajectory import AttachmentState, Evidence, PoseEstimate


@dataclass
class _VisualEvent:
    pose: PoseEstimate
    frame: RGBDFrame
    mask: np.ndarray
    ee_at_capture: np.ndarray
    state_after: dict[str, object]
    last_direct_after: float | None


@dataclass
class _EpochJournal:
    role: str
    start_s: float
    end_s: float | None
    checkpoint_s: float
    checkpoint: dict[str, object]
    checkpoint_last_direct_s: float | None
    observations: deque[_VisualEvent]


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
        pose_snapshot_at: Callable[[float], tuple[int, dict[str, PoseEstimate]]] | None = None,
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
        self.pose_snapshot_at = pose_snapshot_at
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
        self._belief_revision = 0
        self._last_direct_by_role: dict[str, float | None] = dict.fromkeys(self.updaters)
        self._epochs: dict[int, _EpochJournal] = {}

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
            _, poses = (
                self.pose_snapshot_at(receipt.dispatched_s)
                if self.pose_snapshot_at is not None
                else self.poses.snapshot()
            )
            seed = poses.get(transition.role)
            if seed is not None and not self._fresh_seed(
                seed, transition.reference_version, receipt.dispatched_s
            ):
                seed = None
            grasp_joints = [
                float(tick.observed[f"{name}.pos"]) for name in self.updaters[transition.role].arm_joint_names
            ]
            self.updaters[transition.role].activate(receipt.dispatched_s, seed, grasp_joints)
            with self._prediction_lock:
                self.active_role = transition.role
                self._attachment_epoch += 1
                self._belief_revision += 1
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
                self._epochs[self._attachment_epoch] = _EpochJournal(
                    transition.role,
                    receipt.dispatched_s,
                    None,
                    receipt.dispatched_s,
                    self.updaters[transition.role].snapshot(),
                    self._last_direct_by_role[transition.role],
                    deque(),
                )
        else:
            if self.active_role != transition.role:
                raise RuntimeError(f"cannot detach {transition.role}: it is not the active attachment")
            self.updaters[transition.role].deactivate()
            with self._prediction_lock:
                self.active_role = None
                self._belief_revision += 1
                self._role_events[transition.role].append(
                    (receipt.dispatched_s, False, self._attachment_epoch)
                )
                self._epochs[self._attachment_epoch].end_s = receipt.dispatched_s
                retained_epochs = {
                    sample[4] for samples in self._prediction_samples.values() for sample in samples
                }
                for old_epoch, journal in tuple(self._epochs.items()):
                    if journal.end_s is not None and old_epoch not in retained_epochs:
                        del self._epochs[old_epoch]
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

    def epoch_at(self, role: str, source_time_s: float) -> int | None:
        """Find the gripper event epoch that actually covered an image time."""
        with self._prediction_lock:
            return next(
                (
                    epoch
                    for epoch, journal in reversed(tuple(self._epochs.items()))
                    if journal.role == role
                    and journal.start_s <= source_time_s
                    and (journal.end_s is None or source_time_s < journal.end_s)
                ),
                None,
            )

    def is_active_at(self, role: str, source_time_s: float) -> bool:
        with self._prediction_lock:
            journal = self._epochs.get(self._attachment_epoch)
            return bool(
                self.active_role == role
                and journal is not None
                and journal.start_s <= source_time_s
                and journal.end_s is None
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
        held = poses.get(role)
        ee = (
            updater._ee_at(held.source_time_s)
            if held is not None
            and held.evidence == Evidence.OBSERVED
            and frame is not None
            and mask is not None
            else None
        )
        accepted = updater.on_pose_update(version, poses, frame=frame, mask=mask, ee_at_capture=ee)
        if held is not None and held.evidence == Evidence.OBSERVED:
            # A delayed visual correction belongs to its image time, not the
            # motor tick on which the worker happened to accept it.
            with self._prediction_lock:
                if self.active_role != role or self._attachment_epoch != epoch:
                    return accepted
                self._belief_revision += 1
                if accepted:
                    self._last_direct_by_role[role] = held.source_time_s
                if frame is not None and mask is not None and ee is not None:
                    journal = self._epochs.get(epoch)
                    if (
                        journal is not None
                        and held.source_time_s >= journal.start_s
                        and (
                            not journal.observations
                            or held.source_time_s > journal.observations[-1].pose.source_time_s
                        )
                    ):
                        if len(journal.observations) == updater.history.capacity:
                            expired = journal.observations.popleft()
                            journal.checkpoint_s = expired.pose.source_time_s
                            journal.checkpoint = expired.state_after
                            journal.checkpoint_last_direct_s = expired.last_direct_after
                        journal.observations.append(
                            _VisualEvent(
                                held,
                                frame,
                                mask,
                                ee,
                                updater.snapshot(),
                                self._last_direct_by_role[role],
                            )
                        )
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

    def reconcile_source_order(
        self,
        role: str,
        first_s: float,
        last_s: float,
        observations: tuple[tuple[PoseEstimate, RGBDFrame, np.ndarray], ...],
    ) -> bool:
        """Rebuild held corrections after a committed scene decision replay.

        The scene service supplies all accepted direct observations in its
        retained source-time window. Each attachment epoch is replayed from a
        checkpoint made before that window. Motor samples are revised only
        after the complete new visual history is ready.
        """
        updater = self.updaters.get(role)
        if updater is None:
            return True
        with self._prediction_lock:
            epochs = tuple(
                (epoch, journal)
                for epoch, journal in self._epochs.items()
                if journal.role == role
                and journal.start_s <= last_s
                and (journal.end_s is None or journal.end_s > first_s)
            )
        for epoch, journal in epochs:
            first_held_s = max(first_s, journal.start_s)
            if first_held_s < journal.checkpoint_s:
                return False
            retained = [
                item for item in journal.observations if not first_held_s <= item.pose.source_time_s <= last_s
            ]
            existing = {item.frame.frame_id: item for item in journal.observations}
            revised = []
            for pose, frame, mask in observations:
                if (
                    pose.role != role
                    or not journal.start_s <= pose.source_time_s
                    or (journal.end_s is not None and pose.source_time_s >= journal.end_s)
                    or not first_held_s <= pose.source_time_s <= last_s
                ):
                    continue
                old = existing.get(frame.frame_id)
                ee = old.ee_at_capture if old is not None else updater._ee_at(pose.source_time_s)
                if ee is None:
                    return False
                revised.append(_VisualEvent(pose, frame, mask, ee, {}, None))
            ordered = sorted((*retained, *revised), key=lambda item: item.pose.source_time_s)
            shadow = updater.replay_copy()
            shadow.restore(journal.checkpoint)
            last_direct = journal.checkpoint_last_direct_s
            rebuilt = []
            for item in ordered:
                accepted = shadow.on_pose_update(
                    0,
                    {role: item.pose},
                    frame=item.frame,
                    mask=item.mask,
                    ee_at_capture=item.ee_at_capture,
                )
                if accepted:
                    last_direct = item.pose.source_time_s
                rebuilt.append(
                    _VisualEvent(
                        item.pose,
                        item.frame,
                        item.mask,
                        item.ee_at_capture,
                        shadow.snapshot(),
                        last_direct,
                    )
                )
            with self._prediction_lock:
                # Motor ticks may append while replay computes; revise every
                # currently retained sample in one short critical section.
                journal.observations = deque(rebuilt)
                samples = self._prediction_samples[role]
                for index, (time_s, joints, _state, _direct_s, sample_epoch) in enumerate(samples):
                    if sample_epoch != epoch or time_s < first_held_s:
                        continue
                    latest = next(
                        (item for item in reversed(rebuilt) if item.pose.source_time_s <= time_s), None
                    )
                    shadow.restore(latest.state_after if latest is not None else journal.checkpoint)
                    samples[index] = (
                        time_s,
                        joints,
                        shadow.state(time_s),
                        latest.last_direct_after if latest is not None else journal.checkpoint_last_direct_s,
                        sample_epoch,
                    )
                if self.active_role == role and self._attachment_epoch == epoch:
                    shadow.restore(rebuilt[-1].state_after if rebuilt else journal.checkpoint)
                    updater.restore(shadow.snapshot())
                    self._last_direct_by_role[role] = last_direct
                    self._belief_revision += 1
        return True

    def planning_state(self, now_s: float) -> tuple[int, str | None, AttachmentState | None]:
        """Read one relation revision for an atomic planning projection."""
        with self._prediction_lock:
            role = self.active_role
            state = self.updaters[role].state(now_s) if role is not None else None
            return self._belief_revision, role, state

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
