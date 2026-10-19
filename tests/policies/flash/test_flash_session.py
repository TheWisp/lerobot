"""End-to-end state transitions through the fixed motor dispatcher."""

import numpy as np

from lerobot.policies.flash.execution import TimedAction
from lerobot.policies.flash.progress import MilestoneUpdate, Phase, Verdict
from lerobot.policies.flash.session import FlashSession, PoseChange
from lerobot.policies.flash.skill import (
    FlashSkill,
    Interval,
    MilestoneSpec,
    MotionMode,
    Relation,
    SpatialSample,
)


def pose():
    return np.eye(4)


def skill(*, gate=None):
    interval = Interval(
        "contact",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "target",
        (SpatialSample(0, pose(), pose()), SpatialSample(0.1, pose(), pose())),
        entry_milestone=gate,
    )
    milestones = (MilestoneSpec(gate, ("target",)),) if gate else ()
    return FlashSkill("demo/source", 0, (interval,), milestones=milestones)


class FakeRobot:
    def __init__(self):
        self.calls = []

    def get_state_observation(self):
        self.calls.append("read")
        return {"arm.pos": 0.0}

    def send_action(self, action):
        self.calls.append("send")
        return dict(action)


def test_session_runs_at_fixed_ticks_and_requires_explicit_outcome():
    now = [0.0]
    robot = FakeRobot()
    receipts = []
    session = FlashSession(
        skill(),
        robot,
        on_receipt=receipts.append,
        clock=lambda: now[0],
        sleep=lambda dt: now.__setitem__(0, now[0] + dt),
    )
    session.publish_plan(
        0, [TimedAction(i, {"arm.pos": float(i)}, "contact", interval_complete=i == 3) for i in range(4)]
    )
    for _ in range(4):
        session.step()
    assert now[0] == 0.1
    assert session.progress.phase == Phase.MOTION_COMPLETE
    assert session.motor.metrics.commanded == 4
    assert len(receipts) == 4
    assert robot.calls == ["read", "send"] * 4
    session.finish(Verdict.TRUE, "verified by an independent outcome check")
    assert session.progress.phase == Phase.SUCCEEDED


def test_target_change_invalidates_release_gate_before_motor_send():
    now = [0.0]
    robot = FakeRobot()
    session = FlashSession(
        skill(gate="aligned"), robot, clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt)
    )
    session.post_evidence(MilestoneUpdate("aligned", Verdict.TRUE, "camera:1", 0.0))
    session.post_evidence(PoseChange(frozenset({"target"}), "pose:2", 0.01))
    session.publish_plan(
        0, [TimedAction(0, {"arm.pos": 1.0}, "contact", event_id="release", requires_milestone="aligned")]
    )
    assert session.step() is None
    assert session.progress.phase == Phase.WAITING
    assert session.motor.metrics.idle == 1
    assert "send" not in robot.calls
    assert any(e["type"] == "plan_cancelled" and e["events"] == ("release",) for e in session.progress.events)


def test_release_gate_rejection_keeps_tick_but_never_sends_release():
    now = [0.0]
    robot = FakeRobot()
    ticks = []
    session = FlashSession(
        skill(gate="aligned"),
        robot,
        on_tick=ticks.append,
        clock=lambda: now[0],
        sleep=lambda dt: now.__setitem__(0, now[0] + dt),
    )
    session.post_evidence(MilestoneUpdate("aligned", Verdict.TRUE, "camera:1", 0.0))
    session.publish_plan(
        0,
        [
            TimedAction(0, {"arm.pos": 0.1}, "contact"),
            TimedAction(1, {"arm.pos": 0.2}, "contact", event_id="release", requires_milestone="aligned"),
        ],
    )
    assert session.step() is not None
    session.post_evidence(PoseChange(frozenset({"target"}), "camera:2", 0.01))
    assert session.step() is None
    assert session.progress.phase == Phase.INTERRUPTED
    assert [tick.action_source for tick in ticks] == ["commanded", "held_goal"]
    assert ticks[1].idle_reason == "milestone gate rejected"
    assert robot.calls.count("send") == 1


def test_late_contact_tick_fails_instead_of_bursting_old_commands():
    now = [0.0]
    robot = FakeRobot()
    session = FlashSession(
        skill(), robot, clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt)
    )
    session.publish_plan(0, [TimedAction(i, {"arm.pos": float(i)}, "contact") for i in range(4)])
    now[0] = 0.085  # two 30 Hz deadlines have already passed
    assert session.step() is None
    assert session.progress.phase == Phase.FAILED
    assert session.motor.metrics.commanded == 0
    assert "send" not in robot.calls
    assert "missed required" in session.progress.events[-1]["reason"]
