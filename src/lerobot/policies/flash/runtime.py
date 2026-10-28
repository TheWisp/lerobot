"""Compose one FLASH attempt without giving perception or IK the robot handle."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import ceil, isfinite
from queue import Empty, SimpleQueue
from typing import Protocol

import numpy as np

from .attachment_lifecycle import AttachmentLifecycle
from .causal_service import CausalTrackingConfig, build_causal_pose_service
from .execution import MotorTick, RobotIO
from .live_telemetry import FlashLiveTelemetry
from .milestones import PoseMilestonePublisher
from .perception import HeldAttachmentUpdater, LatestPoseStore, ObjectReference, PoseBackend, RGBDFrame
from .perception_service import PerceptionService, RGBDSource
from .perception_worker import PoseProcess
from .planner import CartesianKinematics, JointPlanner, PlannerLimits
from .planner_worker import PlannerWorker
from .planning_coordinator import PlanBufferConfig, PlanningCoordinator
from .progress import Phase, Verdict
from .recording import RecordedTick, TickRecordAssembler
from .session import FlashSession, PoseChange
from .skill import AttachmentAction, FlashSkill, Relation
from .symmetry import AxialSymmetry
from .trajectory import PoseEstimate


class AttemptRecorder(Protocol):
    @property
    def error(self) -> Exception | None: ...

    def start(self) -> None: ...

    def submit(self, record: RecordedTick) -> None: ...

    def finish(self, outcome: str, reason: str) -> None: ...


@dataclass(frozen=True)
class AttachmentRuntimeLimits:
    """One policy for grasp seeding, measured corrections, and optional expiry."""

    history_capacity: int
    max_motor_sample_gap_s: float
    max_attachment_age_s: float | None
    max_update_shift_m: float
    max_update_turn_rad: float
    max_pregrasp_age_s: float

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> AttachmentRuntimeLimits:
        """Read the attachment section of a rig's FLASH trial configuration."""
        required = {
            "motor_history_capacity",
            "max_motor_sample_gap_s",
            "max_unverified_age_s",
            "max_observation_shift_m",
            "max_observation_turn_deg",
            "max_pregrasp_pose_age_s",
        }
        if set(config) != required:
            raise ValueError(f"attachment policy keys differ: {sorted(set(config) ^ required)}")
        age = config["max_unverified_age_s"]
        values = {key: config[key] for key in required if key != "max_unverified_age_s"}
        if age is not None and (
            isinstance(age, bool) or not isinstance(age, (int, float)) or not isfinite(age) or age <= 0
        ):
            raise ValueError("attachment max_unverified_age_s must be positive or null")
        if (
            isinstance(values["motor_history_capacity"], bool)
            or not isinstance(values["motor_history_capacity"], int)
            or values["motor_history_capacity"] < 2
        ):
            raise ValueError("attachment motor_history_capacity must be at least two")
        for key, value in values.items():
            if key == "motor_history_capacity":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"attachment {key} must be finite and positive")
        return cls(
            int(values["motor_history_capacity"]),
            float(values["max_motor_sample_gap_s"]),
            None if age is None else float(age),
            float(values["max_observation_shift_m"]),
            np.deg2rad(float(values["max_observation_turn_deg"])),
            float(values["max_pregrasp_pose_age_s"]),
        )


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
        causal_tracking: CausalTrackingConfig | None = None,
        object_centers_by_role: dict[str, np.ndarray] | None = None,
        object_points_by_role: dict[str, np.ndarray] | None = None,
        live_telemetry: FlashLiveTelemetry | None = None,
        grasp_symmetries: tuple[AxialSymmetry, ...] = (),
    ) -> None:
        self._validate_bindings(skill, references)
        self.skill = skill
        self.arm_joint_names = arm_joint_names
        self.recorder = recorder
        self.max_camera_age_s = max_camera_age_s
        self.live_telemetry = live_telemetry
        self.poses = LatestPoseStore()
        accepted_direct: SimpleQueue[tuple[PoseEstimate, RGBDFrame | None, np.ndarray | None]] = SimpleQueue()

        def queue_held_direct(estimate: PoseEstimate, frame: RGBDFrame, mask: np.ndarray) -> None:
            # Before startup there is no consumer and no active attachment;
            # retaining full RGB-D frames here would grow memory unbounded.
            if self._ready and self.attachment.active_role == estimate.role:
                accepted_direct.put((estimate, frame, mask))

        self.causal = (
            build_causal_pose_service(
                self.poses,
                causal_tracking,
                object_centers_by_role=object_centers_by_role,
                object_points_by_role=object_points_by_role,
                axial_symmetries=grasp_symmetries,
                on_layer=live_telemetry.layer if live_telemetry is not None else None,
                on_comparison=live_telemetry.comparison if live_telemetry is not None else None,
                on_point_comparison=(live_telemetry.point_comparison if live_telemetry is not None else None),
                on_free_space_comparison=(
                    live_telemetry.free_space_comparison if live_telemetry is not None else None
                ),
                on_direct_observation_with_frame=queue_held_direct,
                on_anchor_decision=(live_telemetry.anchor_decision if live_telemetry is not None else None),
                on_scene_support=(live_telemetry.scene_support if live_telemetry is not None else None),
            )
            if causal_tracking is not None
            else None
        )
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
                object_center_object_m=(object_centers_by_role or {}).get(transition.role),
                object_points_object_m=(object_points_by_role or {}).get(transition.role),
                min_visual_support_points=causal_tracking.min_inliers if causal_tracking is not None else 3,
            )
            for transition in skill.attachment_transitions
        }
        self.attachment = AttachmentLifecycle(
            skill, self.poses, updaters, max_pregrasp_age_s=attachment_limits.max_pregrasp_age_s
        )

        def on_tick(tick: MotorTick) -> None:
            prior_attachment = self.attachment.active_role
            self.attachment.on_motor_tick(tick)
            if self.causal is not None and self.attachment.active_role != prior_attachment:
                self.causal.set_attached_role(self.attachment.active_role)
            if self.live_telemetry is not None:
                role = self.attachment.active_role
                updater = self.attachment.updaters[role] if role is not None else None
                self.live_telemetry.held_attachment(
                    role,
                    updater.state(tick.observed_at_s) if updater is not None else None,
                    np.array([tick.observed[f"{name}.pos"] for name in self.arm_joint_names])
                    if updater is not None
                    else None,
                    tick.observed_at_s,
                    independently_observed=updater.has_postgrasp_observation
                    if updater is not None
                    else False,
                )
                self.live_telemetry.motor_tick(tick)
            self.recorder.submit(self.assembler.assemble(tick))

        self.session = FlashSession(
            skill,
            robot,
            on_tick=on_tick,
            fps=fps,
            # An empty queue holds the last command while fresh perception or
            # IK catches up. The coordinator separately stops motion once its
            # pose expires; a fixed two-tick timeout would fail before that
            # evidence boundary during a normal asynchronous update.
            max_starved_ticks=max(1, ceil(planner_limits.max_pose_age_s * fps)),
            max_milestone_wait_s=buffer_config.max_wait_for_evidence_s,
            max_splice_delta={
                f"{name}.pos": speed / fps
                for name, speed in zip(arm_joint_names, planner_limits.joint_speed_deg_s, strict=True)
            },
        )
        self.milestones = PoseMilestonePublisher(
            skill.milestones, self.session.post_evidence, clock=self.session.motor.clock
        )
        self._last_pose_version = 0
        self._last_pose_by_role: dict[str, PoseEstimate] = {}

        def on_pose_update(version: int, poses: dict[str, PoseEstimate]) -> None:
            if self.live_telemetry is not None:
                self.live_telemetry.accepted(poses)
            direct_observations = []
            while True:
                try:
                    direct_observations.append(accepted_direct.get_nowait())
                except Empty:
                    break
            # Perception may publish for many seconds while the first observed
            # pose is being seeded. No motor session consumes evidence until
            # startup completes; the next collector pass publishes the latest
            # snapshot once the session is ready.
            if not self._ready:
                return
            for observed, frame, mask in direct_observations:
                self.attachment.on_pose_update(version, {observed.role: observed}, frame=frame, mask=mask)
            self.attachment.on_pose_update(version, poses)
            if version != self._last_pose_version:
                changed = frozenset(
                    role for role, pose in poses.items() if self._last_pose_by_role.get(role) is not pose
                )
                if changed:
                    self.session.post_evidence(
                        PoseChange(changed, f"pose:{version}", max(poses[r].source_time_s for r in changed))
                    )
                    self._last_pose_by_role.update({role: poses[role] for role in changed})
                self._last_pose_version = version
            # Revisit the same source frame to withdraw a true gate when it expires.
            self.milestones(version, poses)

        causal = self.causal

        def on_pose_result(result) -> None:
            if self.live_telemetry is not None:
                self.live_telemetry.proposals(
                    result.estimates, result.masks, result.frame_id, result.proposals
                )
            assert causal is not None
            causal.submit_anchor(
                result.frame_id,
                result.estimates,
                result.masks,
                source_time_s=result.source_time_s,
                calibration_version=result.calibration_version,
                proposals=result.proposals,
            )

        pose_process = PoseProcess(
            pose_backend_factory,
            references,
            self.poses if causal is None else None,
            on_result=on_pose_result if causal is not None else None,
            require_masks=causal is not None,
        )
        self.perception = PerceptionService(
            source,
            pose_process,
            self.poses,
            clock=self.session.motor.clock,
            on_pose_update=on_pose_update,
            on_frame=live_telemetry.frame if live_telemetry is not None else None,
            causal_frames=causal.submit_frame if causal is not None else None,
            causal_error=(lambda: causal.error) if causal is not None else None,
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
            grasp_symmetries,
        )
        self._perception_started = False
        self._causal_started = False
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
            if self.causal is not None:
                self.causal.start()
                self._causal_started = True
            self.perception.start(timeout_s=pose_start_timeout_s)
            self._perception_started = True
            first_object_interval = next(
                (interval for interval in self.skill.intervals if interval.relation != Relation.EE_TO_START),
                None,
            )
            if first_object_interval is not None:
                self.perception.wait_ready(
                    (first_object_interval.reference_role,),
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
            if self.causal is not None:
                # A first point-track seed may replay retained frames. Let
                # that work catch the live camera before starting the motor
                # clock; otherwise an old prediction can expire mid-approach
                # even though the camera itself is fresh.
                deadline_s = self.session.motor.clock() + first_pose_timeout_s
                while self.session.motor.clock() < deadline_s:
                    if self.perception.error is not None:
                        raise RuntimeError(
                            "perception failed while catching up to the live camera"
                        ) from self.perception.error
                    source_s = self.causal.latest_processed_source_time_s
                    _, current_poses = self.poses.snapshot()
                    now_s = self.session.motor.clock()
                    if (
                        source_s is not None
                        and 0 <= now_s - source_s <= self.max_camera_age_s
                        and all(
                            role in current_poses
                            and current_poses[role].usable(
                                now_s, self.planner.planner.limits.max_pose_age_s, True
                            )
                            for role in required_roles
                        )
                    ):
                        break
                    time.sleep(0.01)
                else:
                    raise RuntimeError("causal scene tracking did not catch up to the live camera")
            _, first_poses = self.poses.snapshot()
            if first_object_interval is not None:
                first_role = first_object_interval.reference_role
                if first_poses[first_role].reference_version != first_object_interval.reference_version:
                    raise ValueError(f"{first_role}: live pose does not use the demonstrated object frame")
            if self.coordinator.grasp_symmetries:
                initial = self.session.motor.robot.get_state_observation()
                self.coordinator.prepare_grasp_choices(
                    first_poses,
                    np.array([float(initial[f"{name}.pos"]) for name in self.arm_joint_names]),
                )
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
                    if self._causal_started and self.causal is not None:
                        self.causal.close()
                finally:
                    try:
                        if self._planner_started:
                            self.planner.close()
                    finally:
                        if self._recorder_started:
                            self.recorder.finish(outcome, reason)
        finally:
            self._perception_started = self._causal_started = self._planner_started = (
                self._recorder_started
            ) = False
