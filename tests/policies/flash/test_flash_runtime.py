"""Attempt wiring must establish fresh object frames before motor deadlines begin."""

from dataclasses import replace

import numpy as np
import pytest

from lerobot.policies.flash.causal_service import CausalTrackingConfig
from lerobot.policies.flash.execution import MissedEventError, TimedAction
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet
from lerobot.policies.flash.planner import PlannerLimits
from lerobot.policies.flash.planning_coordinator import PlanBufferConfig
from lerobot.policies.flash.progress import Phase
from lerobot.policies.flash.runtime import AttachmentRuntimeLimits, FlashRuntime
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample
from lerobot.policies.flash.symmetry import AxialSymmetry
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def test_attachment_policy_has_one_validated_configuration_boundary():
    config = {
        "motor_history_capacity": 120,
        "max_motor_sample_gap_s": 0.12,
        "max_unverified_age_s": None,
        "max_observation_shift_m": 0.015,
        "max_observation_turn_deg": 15,
        "max_pregrasp_pose_age_s": 3.1,
    }
    policy = AttachmentRuntimeLimits.from_config(config)
    assert policy.max_attachment_age_s is None
    assert policy.max_update_shift_m == 0.015
    assert policy.max_update_turn_rad == pytest.approx(np.deg2rad(15))
    with pytest.raises(ValueError, match="keys differ"):
        AttachmentRuntimeLimits.from_config({**config, "unknown_limit": 1})
    with pytest.raises(ValueError, match="positive or null"):
        AttachmentRuntimeLimits.from_config({**config, "max_unverified_age_s": 0})


def identity():
    return np.eye(4)


def reference(role, version):
    return ObjectReference(
        role,
        version,
        (ReferenceSnippet("demo", 0, 0, role, identity(), "mask.png", "cal-v1"),),
    )


def skill(relation=Relation.EE_TO_OBJECT):
    held = identity() if relation == Relation.HELD_TO_TARGET else None
    samples = (
        SpatialSample(0, identity(), identity(), held),
        SpatialSample(1, identity(), identity(), held),
    )
    return FlashSkill(
        "demo",
        0,
        (
            Interval(
                "move",
                MotionMode.DEMONSTRATED,
                relation,
                "target",
                samples,
                reference_version="target-v1",
                held_reference_version="held-v1" if held is not None else None,
                held_role="held" if held is not None else None,
            ),
        ),
    )


class FakeKinematics:
    def forward_kinematics(self, joints):
        return identity()

    def inverse_kinematics(self, seed, target):
        return seed


class FakeRobot:
    def get_state_observation(self):
        return {"arm.pos": 0.0}

    def send_action(self, action):
        return action


class FakeRecorder:
    error = None

    def __init__(self):
        self.started = False
        self.finished = None

    def start(self):
        self.started = True

    def submit(self, record):
        pass

    def finish(self, outcome, reason):
        self.finished = (outcome, reason)


def runtime(test_skill, references, recorder, *, causal_tracking=None, grasp_symmetries=()):
    return FlashRuntime(
        skill=test_skill,
        robot=FakeRobot(),
        source=object(),
        pose_backend_factory=lambda: None,
        references=references,
        kinematics=FakeKinematics(),
        arm_joint_names=("arm",),
        gripper_joint_name="gripper",
        planner_limits=PlannerLimits((100,), 0.01, 0.1, 0.5),
        buffer_config=PlanBufferConfig(2, 8, 3, 0.05, 0.2),
        attachment_limits=AttachmentRuntimeLimits(8, 0.05, 1, 0.01, 0.1, 0.2),
        recorder=recorder,
        rgb_feature="camera",
        depth_feature=None,
        max_camera_age_s=0.2,
        causal_tracking=causal_tracking,
        grasp_symmetries=grasp_symmetries,
    )


def test_runtime_gives_causal_scene_estimator_exclusive_pose_publication():
    attempt = runtime(
        skill(),
        (reference("target", "target-v1"),),
        FakeRecorder(),
        causal_tracking=CausalTrackingConfig(60, 64, 12, 0.1, 0.005),
    )
    assert attempt.causal is not None
    assert attempt.perception.worker.store is None
    assert attempt.perception.worker.on_result is not None


def test_runtime_rejects_held_motion_without_demonstrated_attach_event():
    with pytest.raises(ValueError, match="no matching demonstrated attach event"):
        runtime(
            skill(Relation.HELD_TO_TARGET),
            (reference("target", "target-v1"), reference("held", "held-v1")),
            FakeRecorder(),
        )


def test_motor_epoch_starts_after_pose_readiness(monkeypatch):
    recorder = FakeRecorder()
    attempt = runtime(skill(), (reference("target", "target-v1"),), recorder)
    clock = [10.0]
    attempt.session.motor.clock = lambda: clock[0]
    events = []

    def start_perception(timeout_s):
        clock[0] = 11.0
        events.append("perception")

    def wait_ready(roles, *, max_pose_age_s, timeout_s):
        assert roles == ("target",)
        clock[0] = 12.0
        attempt.poses.publish(
            (
                PoseEstimate(
                    "target", identity(), 12.0, 12.0, Evidence.OBSERVED, "target-v1", (True,) * 3, (True,) * 3
                ),
            )
        )
        events.append("pose ready")

    monkeypatch.setattr(attempt.perception, "start", start_perception)
    monkeypatch.setattr(attempt.perception, "wait_ready", wait_ready)
    monkeypatch.setattr(attempt.perception, "close", lambda: events.append("perception closed"))
    monkeypatch.setattr(attempt.planner, "start", lambda: events.append("planner"))
    monkeypatch.setattr(attempt.planner, "close", lambda: events.append("planner closed"))
    attempt.start(pose_start_timeout_s=2, first_pose_timeout_s=2)
    assert events[:3] == ["perception", "pose ready", "planner"]
    assert attempt.session.motor.epoch_s == 12.0
    assert recorder.started
    attempt.close("unknown", "test completed")
    assert recorder.finished == ("unknown", "test completed")


def test_startup_selects_declared_grasp_symmetry_before_motor_epoch(monkeypatch):
    symmetry = AxialSymmetry("target", np.array([0.0, 0.0, 1.0]), np.zeros(3), identity())
    attempt = runtime(
        skill(),
        (reference("target", "target-v1"),),
        FakeRecorder(),
        grasp_symmetries=(symmetry,),
    )

    def wait_ready(roles, *, max_pose_age_s, timeout_s):
        attempt.poses.publish(
            (
                PoseEstimate(
                    "target",
                    identity(),
                    1.0,
                    1.0,
                    Evidence.OBSERVED,
                    "target-v1",
                    (True,) * 3,
                    (True,) * 3,
                ),
            )
        )

    monkeypatch.setattr(attempt.perception, "start", lambda timeout_s: None)
    monkeypatch.setattr(attempt.perception, "wait_ready", wait_ready)
    monkeypatch.setattr(attempt.perception, "close", lambda: None)
    monkeypatch.setattr(attempt.planner, "start", lambda: None)
    monkeypatch.setattr(attempt.planner, "close", lambda: None)
    attempt.start(pose_start_timeout_s=2, first_pose_timeout_s=2)
    assert attempt.coordinator.grasp_choices["target"].angle_rad == 0
    assert attempt.session.motor.epoch_s is not None
    attempt.close("unknown", "test completed")


def test_runtime_wires_joint_continuity_for_late_free_space_deadlines():
    attempt = runtime(skill(), (reference("target", "target-v1"),), FakeRecorder())
    queue = attempt.session.actions
    queue.publish(
        0,
        [
            TimedAction(i, {"arm.pos": 0.5 * i, "gripper.pos": 1.0}, "move", may_skip_if_late=True)
            for i in range(4)
        ],
    )
    assert queue.take(0)[0] is not None
    queue.note_issued({"arm.pos": 0.0, "gripper.pos": 1.0})
    # This is the real queue constructed by FlashRuntime. Previously it had
    # no continuity limit and aborted after a perfectly bounded motor slip.
    assert queue.expire_before(3) == 2

    unsafe = runtime(skill(), (reference("target", "target-v1"),), FakeRecorder()).session.actions
    unsafe.publish(
        0,
        [
            TimedAction(i, {"arm.pos": 0.5 * i, "gripper.pos": float(i)}, "move", may_skip_if_late=True)
            for i in range(4)
        ],
    )
    unsafe.take(0)
    unsafe.note_issued({"arm.pos": 0.0, "gripper.pos": 0.0})
    with pytest.raises(MissedEventError, match="jump at gripper"):
        unsafe.expire_before(3)


def test_startup_pose_updates_do_not_overflow_unconsumed_evidence():
    attempt = runtime(skill(), (reference("target", "target-v1"),), FakeRecorder())
    callback = attempt.perception.on_pose_update
    assert callback is not None
    for version in range(1, 201):
        pose = PoseEstimate(
            "target",
            identity(),
            float(version),
            float(version),
            Evidence.OBSERVED,
            "target-v1",
            (True,) * 3,
            (True,) * 3,
        )
        callback(version, {"target": pose})
    assert attempt.session._evidence.empty()

    attempt._ready = True
    callback(200, {"target": pose})
    assert attempt.session._evidence.qsize() == 1


def test_startup_acquires_later_reference_before_any_motion(monkeypatch):
    first = skill().intervals[0]
    second = replace(first, name="place", reference_role="ring", reference_version="ring-v1")
    attempt = runtime(
        FlashSkill("demo", 0, (first, second)),
        (reference("target", "target-v1"), reference("ring", "ring-v1")),
        FakeRecorder(),
    )
    waits = []

    def wait_ready(roles, *, max_pose_age_s, timeout_s, allow_predicted=False):
        waits.append((roles, allow_predicted))
        if len(waits) == 1:
            attempt.poses.publish(
                (
                    PoseEstimate(
                        "target",
                        identity(),
                        1.0,
                        1.0,
                        Evidence.OBSERVED,
                        "target-v1",
                        (True,) * 3,
                        (True,) * 3,
                    ),
                    PoseEstimate(
                        "ring",
                        identity(),
                        1.0,
                        1.0,
                        Evidence.PREDICTED,
                        "ring-v1",
                        (True,) * 3,
                        (True,) * 3,
                    ),
                )
            )

    monkeypatch.setattr(attempt.perception, "start", lambda timeout_s: None)
    monkeypatch.setattr(attempt.perception, "wait_ready", wait_ready)
    monkeypatch.setattr(attempt.perception, "close", lambda: None)
    monkeypatch.setattr(attempt.planner, "start", lambda: waits.append("planner started"))
    monkeypatch.setattr(attempt.planner, "close", lambda: None)
    attempt.start(pose_start_timeout_s=2, first_pose_timeout_s=2)
    assert waits == [(("target",), False), (("target", "ring"), True), "planner started"]
    attempt.close("unknown", "test completed")


def test_motion_completion_without_outcome_verifier_remains_unknown(monkeypatch):
    recorder = FakeRecorder()
    attempt = runtime(skill(), (reference("target", "target-v1"),), recorder)

    def start(**_kwargs):
        attempt._ready = True
        attempt.session.motor.start()

    def complete_motion():
        attempt.session.progress.phase = Phase.MOTION_COMPLETE

    monkeypatch.setattr(attempt, "start", start)
    monkeypatch.setattr(attempt.coordinator, "step", complete_motion)
    monkeypatch.setattr(
        attempt, "close", lambda outcome, reason: setattr(recorder, "finished", (outcome, reason))
    )
    result = attempt.run(max_duration_s=1, pose_start_timeout_s=1, first_pose_timeout_s=1)
    assert result.outcome == "unknown"
    assert result.phase == Phase.MOTION_COMPLETE
    assert recorder.finished == ("unknown", "unverified outcome")


def test_operator_stop_interrupts_before_dispatch_and_records_reason(monkeypatch):
    recorder = FakeRecorder()
    attempt = runtime(skill(), (reference("target", "target-v1"),), recorder)

    def start(**_kwargs):
        attempt._ready = True
        attempt.session.motor.start()

    monkeypatch.setattr(attempt, "start", start)
    monkeypatch.setattr(
        attempt, "close", lambda outcome, reason: setattr(recorder, "finished", (outcome, reason))
    )
    result = attempt.run(
        max_duration_s=1,
        pose_start_timeout_s=1,
        first_pose_timeout_s=1,
        stop_requested=lambda: True,
    )
    assert result.phase == Phase.INTERRUPTED
    assert result.motor_ticks == 0
    assert recorder.finished == ("interrupted", "operator stopped attempt")
