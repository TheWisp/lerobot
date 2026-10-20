"""End-to-end state transitions through the fixed motor dispatcher."""

import threading

import numpy as np
import pytest

from lerobot.policies.flash.execution import PlanRejectedError, TimedAction
from lerobot.policies.flash.planner import JointPlanner, PlannerLimits
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
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


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


def test_future_contact_chunk_waits_to_start_its_own_demo_clock():
    now = [0.0]
    robot = FakeRobot()
    session = FlashSession(
        skill(), robot, clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt)
    )
    session.publish_plan(
        0,
        [TimedAction(i, {"arm.pos": float(i)}, "contact", interval_complete=i == 5) for i in range(2, 6)],
    )
    assert session.step() is None
    assert session.step() is None
    assert session.progress.phase == Phase.WAITING
    assert session.step().tick == 2
    assert session.progress.interval_start_s == 2 / 30
    assert robot.calls.count("send") == 1


def test_waiting_tick_cannot_idle_across_a_concurrent_planner_publication():
    now = [0.0]
    session = FlashSession(
        skill(), FakeRobot(), clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt)
    )
    inspected = threading.Event()
    continue_idle = threading.Event()
    original_has_due = session.actions.has_due

    def paused_has_due(tick):
        result = original_has_due(tick)
        inspected.set()
        assert continue_idle.wait(1)
        return result

    session.actions.has_due = paused_has_due
    step_errors = []
    publication_errors = []

    def step():
        try:
            session.step()
        except Exception as exc:
            step_errors.append(exc)

    def publish():
        try:
            session.publish_plan(0, [TimedAction(0, {"arm.pos": 1.0}, "contact")])
        except PlanRejectedError as exc:
            publication_errors.append(exc)

    motor = threading.Thread(target=step)
    motor.start()
    assert inspected.wait(1)
    planner = threading.Thread(target=publish)
    planner.start()
    assert planner.is_alive()  # publication waits for the motor's idle decision
    continue_idle.set()
    motor.join(1)
    planner.join(1)
    assert not motor.is_alive() and not planner.is_alive()
    assert not step_errors
    assert len(publication_errors) == 1  # tick zero has already been skipped
    assert session.motor.metrics.idle == 1


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
    # A rejected release was never sent, but an interrupted attempt cannot
    # accept a late chunk without an explicitly permitted recovery transition.
    with pytest.raises(PlanRejectedError, match="no longer active"):
        session.publish_plan(1, [TimedAction(2, {"arm.pos": 0.2}, "contact", event_id="release")])


def test_late_contact_start_shifts_the_unissued_sequence_without_dropping_it():
    now = [0.0]
    robot = FakeRobot()
    session = FlashSession(
        skill(), robot, clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt)
    )
    session.publish_plan(0, [TimedAction(i, {"arm.pos": float(i)}, "contact") for i in range(4)])
    now[0] = 0.085  # two 30 Hz deadlines have already passed
    first = session.step()
    assert (first.tick, first.requested.action["arm.pos"]) == (2, 0.0)
    assert session.progress.phase == Phase.MOVING
    assert session.motor.metrics.skipped_ticks == 2
    assert session.motor.metrics.commanded == 1
    assert robot.calls.count("send") == 1


def test_compiled_release_gate_blocks_first_gripper_opening_tick():
    class StaticKinematics:
        def inverse_kinematics(self, seed, target):
            return np.array([0.0])

        def forward_kinematics(self, joints):
            return pose()

    class CaptureRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.sent = []

        def send_action(self, action):
            self.sent.append(dict(action))
            return super().send_action(action)

    interval = Interval(
        "place",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "target",
        (
            SpatialSample(0.0, pose(), pose(), gripper=10.0),
            SpatialSample(
                0.05, pose(), pose(), gripper=20.0, event_id="release", requires_milestone="aligned"
            ),
            SpatialSample(0.1, pose(), pose(), gripper=20.0),
        ),
        reference_version="target-v1",
    )
    annotated = FlashSkill("demo/source", 0, (interval,), milestones=(MilestoneSpec("aligned", ("target",)),))
    reference = PoseEstimate(
        "target", pose(), 0.0, 0.0, Evidence.OBSERVED, "target-v1", (True,) * 3, (True,) * 3
    )
    planner = JointPlanner(StaticKinematics(), ("arm",), "gripper", PlannerLimits((100.0,), 0.001, 0.01, 2.0))
    commands = planner.plan(
        interval,
        start_tick=0,
        count=4,
        epoch_s=0.0,
        period_s=1 / 30,
        interval_start_s=0.0,
        seed_deg=np.array([0.0]),
        reference=reference,
        attachment=None,
    )
    now = [0.0]
    robot = CaptureRobot()
    session = FlashSession(
        annotated, robot, clock=lambda: now[0], sleep=lambda dt: now.__setitem__(0, now[0] + dt)
    )
    session.publish_plan(0, commands)
    assert session.step() is not None
    assert session.step() is not None
    assert session.step() is None
    assert [action["gripper.pos"] for action in robot.sent] == [10.0, 10.0]
    assert session.progress.phase == Phase.INTERRUPTED
