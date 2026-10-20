"""Compose one FLASH attempt without giving perception or IK the robot handle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from .attachment_lifecycle import AttachmentLifecycle
from .execution import MotorTick, RobotIO
from .milestones import PoseMilestonePublisher
from .perception import HeldAttachmentUpdater, LatestPoseStore, ObjectReference, PoseBackend
from .perception_service import PerceptionService, RGBDSource
from .perception_worker import PoseProcess
from .planner import CartesianKinematics, JointPlanner, PlannerLimits
from .planner_worker import PlannerWorker
from .planning_coordinator import PlanBufferConfig, PlanningCoordinator
from .progress import Phase, Verdict
from .recording import RecordedTick, TickRecordAssembler
from .session import FlashSession, PoseChange
from .skill import AttachmentAction, FlashSkill, Relation
from .trajectory import PoseEstimate


class AttemptRecorder(Protocol):
    @property
    def error(self) -> Exception | None: ...

    def start(self) -> None: ...

    def submit(self, record: RecordedTick) -> None: ...

    def finish(self, outcome: str, reason: str) -> None: ...


@dataclass(frozen=True)
class AttachmentRuntimeLimits:
    history_capacity: int
    max_motor_sample_gap_s: float
    max_attachment_age_s: float
    max_update_shift_m: float
    max_update_turn_rad: float
    max_pregrasp_age_s: float


@dataclass(frozen=True)
class AttemptResult:
    outcome: str
    reason: str
    phase: Phase
    motor_ticks: int


class FlashRuntime:
    """Own the worker lifecycle and keep motor stepping on its own clock.

    The caller connects a LeRobot robot and RGB-D camera, provides a regular
    episode writer through ``recorder``, then calls ``start`` and repeatedly
    calls ``step`` until the progress phase is terminal. Outcome verification
    remains an explicit caller decision; reaching the last motion sample is
    not a success verdict.
    """

    def __init__(
        self,
        *,
        skill: FlashSkill,
        robot: RobotIO,
        source: RGBDSource,
        pose_backend_factory: Callable[[], PoseBackend],
        references: tuple[ObjectReference, ...],
        kinematics: CartesianKinematics,
        arm_joint_names: tuple[str, ...],
        gripper_joint_name: str,
        planner_limits: PlannerLimits,
        buffer_config: PlanBufferConfig,
        attachment_limits: AttachmentRuntimeLimits,
        recorder: AttemptRecorder,
        rgb_feature: str,
        depth_feature: str | None,
        max_camera_age_s: float,
        fps: float = 30.0,
    ) -> None:
        self._validate_bindings(skill, references)
        self.skill = skill
        self.recorder = recorder
        self.poses = LatestPoseStore()
        updaters = {
            transition.role: HeldAttachmentUpdater(
                transition.role,
                transition.reference_version,
                kinematics,
                arm_joint_names,
                history_capacity=attachment_limits.history_capacity,
                max_motor_sample_gap_s=attachment_limits.max_motor_sample_gap_s,
                max_attachment_age_s=attachment_limits.max_attachment_age_s,
                max_update_shift_m=attachment_limits.max_update_shift_m,
                max_update_turn_rad=attachment_limits.max_update_turn_rad,
            )
            for transition in skill.attachment_transitions
        }
        self.attachment = AttachmentLifecycle(
            skill, self.poses, updaters, max_pregrasp_age_s=attachment_limits.max_pregrasp_age_s
        )

        def on_tick(tick: MotorTick) -> None:
            self.attachment.on_motor_tick(tick)
            self.recorder.submit(self.assembler.assemble(tick))

        self.session = FlashSession(skill, robot, on_tick=on_tick, fps=fps)
        self.milestones = PoseMilestonePublisher(
            skill.milestones, self.session.post_evidence, clock=self.session.motor.clock
        )
        self._last_pose_version = 0
        self._last_pose_times: dict[str, float] = {}

        def on_pose_update(version: int, poses: dict[str, PoseEstimate]) -> None:
            # Perception may publish for many seconds while the first observed
            # pose is being seeded. No motor session consumes evidence until
            # startup completes; the next collector pass publishes the latest
            # snapshot once the session is ready.
            if not self._ready:
                return
            self.attachment.on_pose_update(version, poses)
            if version != self._last_pose_version:
                changed = frozenset(
                    role
                    for role, pose in poses.items()
                    if pose.source_time_s > self._last_pose_times.get(role, float("-inf"))
                )
                if changed:
                    self.session.post_evidence(
                        PoseChange(changed, f"pose:{version}", max(poses[r].source_time_s for r in changed))
                    )
                    self._last_pose_times.update({role: poses[role].source_time_s for role in changed})
                self._last_pose_version = version
            # Revisit the same source frame to withdraw a true gate when it expires.
            self.milestones(version, poses)

        pose_process = PoseProcess(pose_backend_factory, references, self.poses)
        self.perception = PerceptionService(
            source,
            pose_process,
            self.poses,
            clock=self.session.motor.clock,
            on_pose_update=on_pose_update,
        )
        self.assembler = TickRecordAssembler(
            self.perception.frames,
            self.poses,
            rgb_feature=rgb_feature,
            depth_feature=depth_feature,
            max_camera_age_s=max_camera_age_s,
            progress_events=lambda: self.session.progress.events,
            clock=self.session.motor.clock,
        )
        planner = JointPlanner(kinematics, arm_joint_names, gripper_joint_name, planner_limits)
        self.planner = PlannerWorker(planner, self.session.publish_plan)
        self.coordinator = PlanningCoordinator(
            self.session,
            self.planner,
            self.poses,
            kinematics,
            arm_joint_names,
            buffer_config,
            self.attachment.state,
        )
        self._perception_started = False
        self._planner_started = False
        self._recorder_started = False
        self._ready = False
        self._started_once = False

    @staticmethod
    def _validate_bindings(skill: FlashSkill, references: tuple[ObjectReference, ...]) -> None:
        available = {(reference.role, reference.version) for reference in references}
        required = {
            (interval.reference_role, interval.reference_version)
            for interval in skill.intervals
            if interval.relation != Relation.EE_TO_START
        }
        for interval in skill.intervals:
            if interval.relation != Relation.HELD_TO_TARGET:
                continue
            if not interval.held_role or not interval.held_reference_version:
                raise ValueError(f"{interval.name}: held role and frame version are required for execution")
            required.add((interval.held_role, interval.held_reference_version))
            if not any(
                transition.action == AttachmentAction.ATTACH
                and transition.role == interval.held_role
                and transition.reference_version == interval.held_reference_version
                for transition in skill.attachment_transitions
            ):
                raise ValueError(f"{interval.name}: no matching demonstrated attach event")
        if any(version is None for _, version in required) or not required <= available:
            raise ValueError(f"skill object frames lack live references: {required - available!r}")

    def start(self, *, pose_start_timeout_s: float, first_pose_timeout_s: float) -> None:
        if self._started_once:
            raise RuntimeError("FLASH attempt cannot be started twice")
        self._started_once = True
        try:
            self.perception.start(timeout_s=pose_start_timeout_s)
            self._perception_started = True
            first_role = self.skill.intervals[0].reference_role
            self.perception.wait_ready(
                (first_role,),
                max_pose_age_s=self.planner.planner.limits.max_pose_age_s,
                timeout_s=first_pose_timeout_s,
            )
            # A demonstration may use more objects after the grasp. Establish
            # every referenced object frame before starting irreversible motion;
            # a support-constrained pose can be predicted in unconstrained axes.
            required_roles = tuple(
                dict.fromkeys(
                    interval.reference_role
                    for interval in self.skill.intervals
                    if interval.relation != Relation.EE_TO_START
                )
            )
            if len(required_roles) > 1:
                self.perception.wait_ready(
                    required_roles,
                    max_pose_age_s=self.planner.planner.limits.max_pose_age_s,
                    timeout_s=first_pose_timeout_s,
                    allow_predicted=True,
                )
            _, first_poses = self.poses.snapshot()
            if first_poses[first_role].reference_version != self.skill.intervals[0].reference_version:
                raise ValueError(f"{first_role}: live pose does not use the demonstrated object frame")
            self.planner.start()
            self._planner_started = True
            self.recorder.start()
            self._recorder_started = True
            # Heavy model loading and first pose seeding precede the motor epoch.
            self.session.motor.start()
            self._ready = True
        except Exception:
            self.close("failure", "FLASH startup failed")
            raise

    def step(self) -> None:
        if not self._ready:
            raise RuntimeError("FLASH attempt is not ready")
        error = self.perception.error
        if error is not None:
            self.session.actions.cancel_unissued("perception failed")
            self.session.progress.fail(self.session.motor.clock(), f"perception failed: {error}")
            return
        if self.recorder.error is not None:
            self.session.actions.cancel_unissued("recording failed")
            self.session.progress.fail(self.session.motor.clock(), f"recording failed: {self.recorder.error}")
            return
        if self.session.progress.phase in (Phase.FAILED, Phase.INTERRUPTED, Phase.MOTION_COMPLETE):
            return
        self.coordinator.step()

    def run(
        self,
        *,
        max_duration_s: float,
        pose_start_timeout_s: float,
        first_pose_timeout_s: float,
        outcome_verifier: Callable[[], tuple[Verdict, str]] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> AttemptResult:
        """Run one bounded attempt and save every issued motor tick.

        Without an independent verifier, motion completion is recorded as
        unknown, never as a claimed success. This synchronous method belongs
        in a background Run worker, not the GUI event loop.
        """
        if not isfinite(max_duration_s) or max_duration_s <= 0:
            raise ValueError("attempt duration must be finite and positive")
        outcome, reason = "failure", "FLASH attempt did not start"
        try:
            self.start(
                pose_start_timeout_s=pose_start_timeout_s,
                first_pose_timeout_s=first_pose_timeout_s,
            )
            deadline_s = self.session.motor.clock() + max_duration_s
            while self.session.progress.phase not in (
                Phase.MOTION_COMPLETE,
                Phase.SUCCEEDED,
                Phase.FAILED,
                Phase.INTERRUPTED,
            ):
                if stop_requested is not None and stop_requested():
                    self.session.actions.cancel_unissued("operator stopped attempt")
                    self.session.progress.interrupt(self.session.motor.clock(), "operator stopped attempt")
                    break
                if self.session.motor.clock() >= deadline_s:
                    self.session.actions.cancel_unissued("attempt duration expired")
                    self.session.progress.fail(self.session.motor.clock(), "attempt duration expired")
                    break
                self.step()
            phase = self.session.progress.phase
            if phase == Phase.MOTION_COMPLETE:
                verdict, reason = (
                    outcome_verifier()
                    if outcome_verifier is not None
                    else (Verdict.UNKNOWN, "unverified outcome")
                )
                self.session.finish(verdict, reason)
                outcome = {
                    Verdict.TRUE: "success",
                    Verdict.FALSE: "failure",
                    Verdict.UNKNOWN: "unknown",
                }[verdict]
            elif phase == Phase.FAILED:
                outcome, reason = "failure", self.session.progress.events[-1]["reason"]
            elif phase == Phase.SUCCEEDED:
                outcome, reason = "success", self.session.progress.events[-1]["reason"]
            else:
                outcome, reason = "interrupted", self.session.progress.events[-1]["reason"]
            return AttemptResult(
                outcome, reason, self.session.progress.phase, self.session.motor.metrics.ticks
            )
        except Exception as exc:
            outcome, reason = "failure", f"FLASH attempt error: {exc}"
            if self._ready:
                self.session.actions.cancel_unissued(reason)
                self.session.progress.fail(self.session.motor.clock(), reason)
            raise
        finally:
            if self._ready:
                self.close(outcome, reason)

    def close(self, outcome: str, reason: str) -> None:
        self._ready = False
        try:
            try:
                if self._perception_started:
                    self.perception.close()
            finally:
                try:
                    if self._planner_started:
                        self.planner.close()
                finally:
                    if self._recorder_started:
                        self.recorder.finish(outcome, reason)
        finally:
            self._perception_started = self._planner_started = self._recorder_started = False
