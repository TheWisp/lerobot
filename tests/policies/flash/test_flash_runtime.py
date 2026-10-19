"""Attempt wiring must establish fresh object frames before motor deadlines begin."""

import numpy as np
import pytest

from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet
from lerobot.policies.flash.planner import PlannerLimits
from lerobot.policies.flash.planning_coordinator import PlanBufferConfig
from lerobot.policies.flash.progress import Phase
from lerobot.policies.flash.runtime import AttachmentRuntimeLimits, FlashRuntime
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


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


def runtime(test_skill, references, recorder):
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
    )


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
