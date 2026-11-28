"""Behavioral invariants for direct spatial imitation."""

import pytest

from lerobot.policies.flash.execution import (
    ActionQueue,
    MissedEventError,
    MotorLoop,
    PlanRejectedError,
    TimedAction,
)


def test_retarget_keeps_future_prefix_and_required_event():
    queue = ActionQueue()
    queue.publish(
        0,
        [
            TimedAction(0, {"joint.pos": 0}, "grasp"),
            TimedAction(1, {"joint.pos": 1}, "grasp", event_id="close"),
        ],
    )
    assert queue.take(0)[0].action["joint.pos"] == 0
    with pytest.raises(PlanRejectedError, match="drops"):
        queue.publish(1, [TimedAction(1, {"joint.pos": 2}, "grasp")])
    queue.publish(1, [TimedAction(1, {"joint.pos": 2}, "grasp", event_id="close")])
    assert queue.take(1)[0].event_id == "close"
    with pytest.raises(PlanRejectedError, match="repeats"):
        queue.publish(2, [TimedAction(2, {"joint.pos": 3}, "grasp", event_id="close")])

    prefix_queue = ActionQueue()
    prefix_queue.publish(
        4, [TimedAction(0, {"joint.pos": 0}, "carry"), TimedAction(1, {"joint.pos": 1}, "carry")]
    )
    prefix_queue.publish(5, [TimedAction(1, {"joint.pos": 2}, "carry")])
    assert prefix_queue.take(0)[1] == 4
    assert prefix_queue.take(1)[1] == 5


def test_delayed_replan_cannot_lose_the_start_of_a_demonstrated_interval():
    queue = ActionQueue()
    queue.skip(0)
    with pytest.raises(PlanRejectedError, match="expired before required samples"):
        queue.publish(
            0,
            [TimedAction(i, {"joint.pos": float(i)}, "contact") for i in range(4)],
        )
    assert queue.revision == -1
    queue.publish(0, [TimedAction(i, {"joint.pos": float(i)}, "contact") for i in range(1, 4)])
    assert queue.take(1)[0].action["joint.pos"] == 1.0


def test_queue_rejects_a_retarget_jump_in_robot_api_action_space():
    queue = ActionQueue(max_splice_delta={"joint.pos": 0.5})
    queue.publish(0, [TimedAction(0, {"joint.pos": 0.2}, "carry")])
    assert queue.take(0)[0] is not None
    queue.note_issued({"joint.pos": 0.1})
    with pytest.raises(PlanRejectedError, match="jumps"):
        queue.publish(1, [TimedAction(1, {"joint.pos": 1.0}, "carry")])
    queue.publish(1, [TimedAction(1, {"joint.pos": 0.4}, "carry")])
    assert queue.take(1)[0].action["joint.pos"] == 0.4


def test_motor_overrun_delays_free_space_sequence_without_a_joint_jump():
    class FakeRobot:
        def get_state_observation(self):
            return {"joint.pos": 0.0}

        def send_action(self, action):
            return action

    now = [0.0]
    queue = ActionQueue(max_splice_delta={"joint.pos": 0.31})
    queue.publish(
        0,
        [TimedAction(i, {"joint.pos": i * 0.1}, "carry", may_skip_if_late=True) for i in range(6)],
    )
    ticks = []
    loop = MotorLoop(
        FakeRobot(),
        queue,
        lambda receipt: None,
        on_tick=ticks.append,
        clock=lambda: now[0],
        sleep=lambda dt: None,
    )
    loop.start()
    assert loop.tick().tick == 0
    now[0] = 0.12
    receipt = loop.tick()
    assert receipt.tick == 3
    assert receipt.requested.action["joint.pos"] == pytest.approx(0.1)
    assert loop.metrics.skipped_ticks == 2
    assert loop.metrics.commanded == 2
    assert ticks[-1].skipped_before == 2
    now[0] = 0.134
    assert loop.tick().requested.action["joint.pos"] == pytest.approx(0.2)


def test_motor_metrics_separate_post_dispatch_callbacks_from_bus_io():
    class FakeRobot:
        def get_state_observation(self):
            return {"joint.pos": 0.0}

        def send_action(self, action):
            return action

    now = [0.0]
    queue = ActionQueue()
    queue.publish(0, [TimedAction(0, {"joint.pos": 0.1}, "carry")])

    def on_receipt(_receipt):
        now[0] += 0.02

    def on_tick(_tick):
        now[0] += 0.03

    loop = MotorLoop(
        FakeRobot(), queue, on_receipt, on_tick=on_tick, clock=lambda: now[0], sleep=lambda _dt: None
    )
    loop.start()
    loop.tick()
    assert loop.metrics.write_s == [0.0]
    assert loop.metrics.max_receipt_callback_s == pytest.approx(0.02)
    assert loop.metrics.max_tick_callback_s == pytest.approx(0.03)


def test_one_missed_motor_deadline_defers_required_samples_without_a_joint_jump():
    class FakeRobot:
        def get_state_observation(self):
            return {"joint.pos": 0.0}

        def send_action(self, action):
            return action

    now = [0.0]
    queue = ActionQueue()
    queue.publish(
        0,
        [
            TimedAction(0, {"joint.pos": 0.0}, "contact"),
            TimedAction(1, {"joint.pos": 0.5}, "contact", event_id="grasp"),
            TimedAction(2, {"joint.pos": 1.0}, "contact", interval_complete=True),
        ],
    )
    loop = MotorLoop(FakeRobot(), queue, lambda receipt: None, clock=lambda: now[0], sleep=lambda dt: None)
    loop.start()
    assert loop.tick().tick == 0
    now[0] = 0.07  # Tick 1 is missed, but only by one motor period.
    second = loop.tick()
    assert (second.tick, second.requested.event_id, second.requested.action["joint.pos"]) == (
        2,
        "grasp",
        0.5,
    )
    assert queue.snapshot().issued_events == frozenset({"grasp"})
    now[0] = 0.101
    final = loop.tick()
    assert (final.tick, final.requested.interval_complete, final.requested.action["joint.pos"]) == (
        3,
        True,
        1.0,
    )
    assert loop.metrics.skipped_ticks == 1


def test_multiple_missed_deadlines_preserve_required_contact_event_order():
    class FakeRobot:
        def get_state_observation(self):
            return {"joint.pos": 0.0}

        def send_action(self, action):
            return action

    now = [0.0]
    queue = ActionQueue(max_splice_delta={"joint.pos": 0.1})
    queue.publish(
        0,
        [
            TimedAction(0, {"joint.pos": 0.0}, "contact"),
            TimedAction(1, {"joint.pos": 0.1}, "contact", event_id="grasp"),
            TimedAction(2, {"joint.pos": 0.2}, "contact", interval_complete=True),
        ],
    )
    loop = MotorLoop(FakeRobot(), queue, lambda receipt: None, clock=lambda: now[0], sleep=lambda dt: None)
    loop.start()
    assert loop.tick().tick == 0
    now[0] = 0.14
    grasp = loop.tick()
    assert (grasp.tick, grasp.requested.event_id, grasp.requested.action["joint.pos"]) == (
        4,
        "grasp",
        0.1,
    )
    now[0] = 0.168
    completion = loop.tick()
    assert (completion.tick, completion.requested.interval_complete) == (5, True)
    assert loop.metrics.skipped_ticks == 3


def test_motor_overrun_rejects_an_unsafe_free_space_jump():
    queue = ActionQueue(max_splice_delta={"joint.pos": 0.1})
    queue.publish(
        0,
        [
            TimedAction(i, {"joint.pos": value}, "carry", may_skip_if_late=True)
            for i, value in enumerate((0, 0.1, 0.5))
        ],
    )
    queue.take(0)
    queue.note_issued({"joint.pos": 0})
    with pytest.raises(MissedEventError, match="would jump"):
        queue.expire_before(2)


def test_slow_state_read_delays_contact_sequence_without_dropping_its_first_sample():
    now = [0.0]

    class SlowReadRobot:
        def __init__(self):
            self.sent = []

        def get_state_observation(self):
            now[0] += 0.08
            return {"joint.pos": 0.0}

        def send_action(self, action):
            self.sent.append(action)
            return action

    robot = SlowReadRobot()
    queue = ActionQueue()
    queue.publish(0, [TimedAction(i, {"joint.pos": float(i)}, "contact") for i in range(4)])
    loop = MotorLoop(robot, queue, lambda receipt: None, clock=lambda: now[0], sleep=lambda dt: None)
    loop.start()
    receipt = loop.tick()
    assert receipt.tick == 2
    assert receipt.requested.action["joint.pos"] == 0.0
    assert robot.sent == [{"joint.pos": 0.0}]
    assert loop.metrics.skipped_ticks == 2


def test_direct_queue_take_cannot_drop_a_required_contact_sample():
    queue = ActionQueue()
    queue.publish(0, [TimedAction(i, {"joint.pos": float(i)}, "contact") for i in range(4)])
    with pytest.raises(MissedEventError, match="missed required contact sample"):
        queue.take(2)


def test_motor_tick_reads_state_and_records_accepted_action_without_camera_wait():
    class FakeRobot:
        def get_state_observation(self):
            calls.append("state")
            return {"joint.pos": 0.0}

        def send_action(self, action):
            calls.append("send")
            return {"joint.pos": min(action["joint.pos"], 1.0)}

    calls = []
    receipts = []
    ticks = []
    now = [0.0]
    queue = ActionQueue()
    queue.publish(0, [TimedAction(0, {"joint.pos": 2.0}, "move")])
    loop = MotorLoop(
        FakeRobot(),
        queue,
        receipts.append,
        on_tick=ticks.append,
        clock=lambda: now[0],
        sleep=lambda dt: now.__setitem__(0, now[0] + dt),
    )
    receipt = loop.tick()
    assert calls == ["state", "send"]
    assert receipt.accepted["joint.pos"] == 1.0
    assert receipts == [receipt]
    assert ticks[0].observed_at_s == 0.0
    assert loop.metrics.commanded == 1
    assert loop.tick() is None
    assert now[0] == pytest.approx(1 / 30)
    assert loop.metrics.starved == 1
