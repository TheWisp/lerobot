"""Prepare future FLASH chunks while the fixed-rate session owns the robot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import floor

import numpy as np

from .perception import LatestPoseStore
from .planner import CartesianKinematics
from .planner_worker import PlanFailureKind, PlannerWorker, PlanRequest
from .progress import Phase
from .session import FlashSession
from .skill import MotionMode, Relation
from .trajectory import AttachmentState, ConnectionTrajectory, Evidence, PoseEstimate, spatial_goal


@dataclass(frozen=True)
class PlanBufferConfig:
    lead_ticks: int
    horizon_ticks: int
    refill_at_ticks: int
    max_reference_shift_m: float
    max_reference_turn_rad: float
    max_wait_for_evidence_s: float = 2.0

    def __post_init__(self) -> None:
        if (
            self.lead_ticks < 1
            or self.horizon_ticks <= self.lead_ticks + 1
            or not 0 < self.refill_at_ticks < self.horizon_ticks
            or not np.isfinite(self.max_reference_shift_m)
            or self.max_reference_shift_m <= 0
            or not np.isfinite(self.max_reference_turn_rad)
            or self.max_reference_turn_rad <= 0
            or not np.isfinite(self.max_wait_for_evidence_s)
            or self.max_wait_for_evidence_s <= 0
        ):
            raise ValueError("invalid FLASH planning buffer or reference-change limits")


class PlanningCoordinator:
    """Submit short future IK chunks after each motor tick, without solving IK.

    The caller starts the PlannerWorker separately and calls ``step`` at the
    motor cadence. The worker's publication callback must be the session's
    ``publish_plan``. A changed reference inside the configured envelope
    replans the unissued suffix; a larger jump interrupts instead of sending
    the known-stale queue. The coordinator does not claim to verify grasp,
    configure cameras, or record an episode.
    """

    def __init__(
        self,
        session: FlashSession,
        worker: PlannerWorker,
        poses: LatestPoseStore,
        kinematics: CartesianKinematics,
        arm_joint_names: tuple[str, ...],
        config: PlanBufferConfig,
        attachment_at: Callable[[float], AttachmentState | None],
    ) -> None:
        self.session = session
        self.worker = worker
        self.poses = poses
        self.kinematics = kinematics
        self.arm_joint_names = arm_joint_names
        self.config = config
        self.attachment_at = attachment_at
        self._next_revision = 0
        self._pending_revision: int | None = None
        self._planned_reference: PoseEstimate | None = None
        self._planned_attachment_time_s: float | None = None
        self._planned_interval_index: int | None = None
        self._connection: ConnectionTrajectory | None = None
        self._start_anchor_interval_index: int | None = None
        self._start_anchor_pose: np.ndarray | None = None
        self._missing_evidence_key: tuple[int, str] | None = None
        self._missing_evidence_since_s: float | None = None

    def step(self):
        """Check live evidence, execute one motor tick, then refill the future queue."""
        # Re-read evidence before dispatch so a newly known large target move
        # can invalidate the old queue without sending one more stale action.
        self.schedule()
        receipt = self.session.step()
        self.schedule()
        return receipt

    def schedule(self) -> None:
        if self.worker.fatal_error is not None:
            self._fail(f"IK worker failed: {self.worker.fatal_error}")
            return
        phase = self.session.progress.phase
        if phase not in (Phase.WAITING, Phase.MOVING):
            return
        result = self.worker.last_result
        if (
            self._pending_revision is not None
            and result is not None
            and result.revision == self._pending_revision
        ):
            self._pending_revision = None
            if result.error:
                if result.failure in (
                    PlanFailureKind.MISSING_EVIDENCE,
                    PlanFailureKind.INFEASIBLE,
                    PlanFailureKind.INVALID,
                ):
                    self._fail(f"IK plan failed: {result.error}")
                    return
                self._planned_reference = None
            elif result.connection is not None:
                # The worker may have lengthened this generated path to meet
                # joint-speed limits. Subsequent chunks must share that clock.
                self._connection = result.connection
        interval = self.session.progress.interval
        _, live_poses = self.poses.snapshot()
        now_s = self.session.motor.clock()
        if interval.relation == Relation.EE_TO_START:
            interval_index = self.session.progress.interval_index
            if self._start_anchor_interval_index != interval_index:
                observed = self.session.motor.last_observed
                if observed is None or any(f"{name}.pos" not in observed for name in self.arm_joint_names):
                    return
                joints = np.array([float(observed[f"{name}.pos"]) for name in self.arm_joint_names])
                self._start_anchor_pose = self.kinematics.forward_kinematics(joints)
                self._start_anchor_interval_index = interval_index
            reference = PoseEstimate(
                interval.reference_role,
                self._start_anchor_pose,
                now_s,
                now_s,
                Evidence.OBSERVED,
                interval.reference_version,
                (True, True, True),
                (True, True, True),
            )
        else:
            reference = live_poses.get(interval.reference_role)
        limits = self.worker.planner.limits
        if reference is None or not reference.usable(
            now_s, limits.max_pose_age_s, limits.allow_predicted_reference
        ):
            # A queued first command must not start after the evidence that
            # produced it has disappeared while the interval waits to enter.
            if phase == Phase.WAITING and self.session.actions.snapshot().pending:
                self.session.actions.cancel_unissued("reference lost before interval entry")
                self._planned_reference = None
            if phase == Phase.MOVING:
                self._fail(f"{interval.reference_role} lost its usable pose during motion")
            else:
                self._wait_for_evidence(f"{interval.reference_role} usable pose", now_s)
            return
        self._clear_missing_evidence(f"{interval.reference_role} usable pose")
        if interval.reference_version != reference.reference_version:
            self._fail(f"{interval.reference_role} pose uses the wrong object frame")
            return
        attachment = self.attachment_at(now_s) if interval.relation == Relation.HELD_TO_TARGET else None
        if interval.relation == Relation.HELD_TO_TARGET and attachment is None:
            if phase == Phase.MOVING:
                self._fail("held-object attachment expired during motion")
            else:
                self._wait_for_evidence("held-object attachment", now_s)
            return
        self._clear_missing_evidence("held-object attachment")
        if attachment is not None and attachment.reference_version != interval.held_reference_version:
            self._fail("held-object attachment uses the wrong object frame")
            return
        changed_interval = self._planned_interval_index != self.session.progress.interval_index
        new_image = (
            changed_interval
            or self._planned_reference is None
            or (
                interval.relation != Relation.EE_TO_START
                and self._planned_reference.source_time_s != reference.source_time_s
            )
            or self._planned_attachment_time_s != (attachment.source_time_s if attachment else None)
        )
        snapshot = self.session.actions.snapshot()
        next_tick = self.session.motor.metrics.ticks
        pending = [
            action
            for action in snapshot.pending
            if action.tick >= next_tick and action.interval == interval.name
        ]
        if phase == Phase.WAITING and pending:
            if (
                not changed_interval
                and self._planned_reference is not None
                and self._reference_changed_too_far(self._planned_reference, reference, interval, attachment)
            ):
                self.session.actions.cancel_unissued("reference changed before interval entry")
                pending = []
            else:
                # Preserve a complete first chunk until its scheduled tick.
                # Replanning on every camera timestamp can otherwise cancel
                # it indefinitely when lead time exceeds perception period.
                return
        if phase == Phase.MOVING and len(pending) > self.config.refill_at_ticks and not new_image:
            return
        if (
            phase == Phase.MOVING
            and new_image
            and not changed_interval
            and self._planned_reference is not None
            and self._reference_changed_too_far(self._planned_reference, reference, interval, attachment)
        ):
            self._fail(
                f"{interval.reference_role} changed the demonstrated goal beyond the replanning envelope"
            )
            return
        # A fresh camera frame must not repeatedly supersede an IK solve
        # before it can publish its first usable chunk. The current request
        # retains a bounded pose snapshot; a large change was rejected above.
        # Once that solve completes, the next step can replan from the newer
        # observation without ever emptying the motor queue.
        if self._pending_revision is not None:
            return
        if phase == Phase.WAITING:
            start_tick = next_tick + self.config.lead_ticks
            boundary = None
            interval_start_s = self.session.motor.epoch_s + start_tick * self.session.motor.period_s
        else:
            start_tick = next_tick + self.config.lead_ticks
            if pending and pending[-1].tick < start_tick - 1:
                start_tick = pending[-1].tick + 1
            if any(action.event_id and action.tick < start_tick for action in pending):
                # The old chunk already carries an imminent grasp/release.
                # Solving a suffix beyond it before the event is dispatched
                # would falsely classify that still-queued event as missed.
                return
            boundary = snapshot.boundary_before(start_tick)
            interval_start_s = self.session.progress.interval_start_s
            if boundary is None or interval_start_s is None:
                self._fail("no continuous joint boundary for an active interval replan")
                return
        state = boundary if boundary is not None else self.session.motor.last_observed
        if state is None or any(f"{name}.pos" not in state for name in self.arm_joint_names):
            return
        seed = np.array([float(state[f"{name}.pos"]) for name in self.arm_joint_names])
        if not np.isfinite(seed).all():
            self._fail("motor observation has non-finite joint state")
            return
        due_s = self.session.motor.epoch_s + start_tick * self.session.motor.period_s
        evidence_deadline_s = reference.source_time_s + limits.max_pose_age_s
        if attachment is not None:
            evidence_deadline_s = min(
                evidence_deadline_s,
                attachment.source_time_s + limits.effective_attachment_age_s,
            )
        evidence_ticks = floor((evidence_deadline_s - due_s) / self.session.motor.period_s) + 1
        if evidence_ticks <= 0:
            # A future refill can fall past the pose's age limit while the
            # current committed queue remains supported. Keep dispatching
            # that queue and retry on the next perception result. The
            # current-pose check above still stops motion when it ages out.
            return
        count = min(self.config.horizon_ticks, evidence_ticks)
        connection = None
        if interval.mode == MotionMode.CONNECTION:
            connection = (
                ConnectionTrajectory.begin(
                    interval,
                    self.kinematics.forward_kinematics(seed),
                    reference.pose,
                    attachment.gripper_from_object if attachment is not None else None,
                    due_s,
                )
                if new_image or self._connection is None
                else self._connection
            )
        revision = max(self._next_revision, snapshot.revision + 1)
        request = PlanRequest(
            revision=revision,
            interval=interval,
            start_tick=start_tick,
            count=count,
            epoch_s=self.session.motor.epoch_s,
            period_s=self.session.motor.period_s,
            interval_start_s=interval_start_s,
            seed_deg=seed,
            reference=replace(reference, pose=reference.pose.copy()),
            attachment=(
                replace(attachment, gripper_from_object=attachment.gripper_from_object.copy())
                if attachment is not None
                else None
            ),
            connection=connection,
            issued_events=snapshot.issued_events,
        )
        self.worker.submit(request)
        self._next_revision = revision + 1
        self._pending_revision = revision
        self._planned_reference = reference
        self._planned_attachment_time_s = attachment.source_time_s if attachment else None
        self._planned_interval_index = self.session.progress.interval_index
        self._connection = connection

    def _reference_changed_too_far(
        self, prior: PoseEstimate, current: PoseEstimate, interval, attachment: AttachmentState | None
    ) -> bool:
        if prior.reference_version != current.reference_version or current.pose is None:
            return True
        held = attachment.gripper_from_object if attachment is not None else None
        for sample in interval.samples:
            old_goal = spatial_goal(interval, sample, prior.pose, held)
            new_goal = spatial_goal(interval, sample, current.pose, held)
            shift = np.linalg.norm(old_goal[:3, 3] - new_goal[:3, 3])
            turn = np.arccos(np.clip((np.trace(old_goal[:3, :3].T @ new_goal[:3, :3]) - 1) / 2, -1, 1))
            if shift > self.config.max_reference_shift_m or turn > self.config.max_reference_turn_rad:
                return True
        return False

    def _fail(self, reason: str) -> None:
        self.session.actions.cancel_unissued(reason)
        self.session.progress.interrupt(self.session.motor.clock(), reason)

    def _clear_missing_evidence(self, kind: str) -> None:
        if self._missing_evidence_key is not None and self._missing_evidence_key[1] == kind:
            self._missing_evidence_key = None
            self._missing_evidence_since_s = None

    def _wait_for_evidence(self, kind: str, now_s: float) -> None:
        key = (self.session.progress.interval_index, kind)
        if self._missing_evidence_key != key:
            self._missing_evidence_key = key
            self._missing_evidence_since_s = now_s
        elif (
            self._missing_evidence_since_s is not None
            and now_s - self._missing_evidence_since_s >= self.config.max_wait_for_evidence_s
        ):
            self._fail(f"{kind} unavailable at interval entry")
