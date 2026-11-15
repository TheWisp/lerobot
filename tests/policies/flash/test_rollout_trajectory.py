"""Adaptation -> dispatched receipt -> saved Cartesian path, without hardware."""

import json
from dataclasses import replace

import numpy as np

from lerobot.policies.flash.demo_document import atomic_json
from lerobot.policies.flash.demo_execution import plan_demo_range
from lerobot.policies.flash.execution import ActionQueue, MotorLoop
from lerobot.policies.flash.rollout_recording import RecordedRobot, RunJournal
from lerobot.policies.flash.rollout_trajectory import read_trajectory
from tests.policies.flash.test_demo_execution import evidence, plan_for, ready_state, robot  # noqa: F401


def test_joint_transition_uses_request_before_compensation_and_saved_calibration(tmp_path, robot):  # noqa: F811
    model = robot.cartesian_model()
    plan = plan_for(robot)
    atomic_json(tmp_path / "plan.json", plan)
    requested = ready_state(model)
    accepted = {**requested, model.joint_names[0]: 20.0}

    class ClippingRobot:
        def send_action(self, action):
            return accepted.copy()

    journal = RunJournal(tmp_path)
    journal.emit("run.started", mode="robot")
    RecordedRobot(ClippingRobot(), journal).send_action(requested)
    journal.close()
    # Reading the rollout cannot depend on current calibration/profile files.
    robot.config.geometry_path.unlink()
    trajectory = read_trajectory(tmp_path)
    assert len(trajectory["points"]) == 1
    point = trajectory["points"][0]
    camera_from_base = np.linalg.inv(plan["base_from_camera"])
    expected = (camera_from_base @ model.forward(requested))[:3, 3]
    np.testing.assert_allclose(point["position"], expected)
    assert not np.allclose(point["position"], (camera_from_base @ model.forward(accepted))[:3, 3])
    assert point["source"] == "joint_transition"
    assert point["relative_s"] > 0
    assert point["time_s"] - trajectory["origin_s"] == point["relative_s"]
    robot.connect.assert_not_called()


def test_dispatched_adapted_targets_survive_replanning_and_legacy_journals(tmp_path, robot):  # noqa: F811
    model = robot.cartesian_model()
    state = ready_state(model)
    plan = plan_for(robot)
    plan["keyframes"].append({**plan["keyframes"][0], "frame": 1})
    atomic_json(tmp_path / "plan.json", plan)

    class CompensatingRobot:
        def get_state_observation(self):
            return state.copy()

        def send_action(self, action):
            return {**action, model.joint_names[1]: action[model.joint_names[1]] + 8}

    class Clock:
        now = 0.0

        def sleep(self, delay):
            self.now += delay

    clock = Clock()
    queue = ActionQueue()
    journal = RunJournal(tmp_path)
    journal.emit("run.started", mode="robot")
    motor = MotorLoop(
        RecordedRobot(CompensatingRobot(), journal),
        queue,
        on_receipt=lambda _: None,
        on_tick=lambda tick: journal.emit("control.tick", tick=tick),
        clock=lambda: clock.now,
        sleep=clock.sleep,
    )
    expected = []
    camera_from_base = np.linalg.inv(plan["base_from_camera"])
    try:
        for revision, offset in enumerate((0.01, 0.03, 0.05)):
            pose = np.eye(4)
            pose[:3, 3] = [offset, 0, 0]
            planned = plan_demo_range(plan, model, 0, state, evidence(pose), 1.0)
            commands = [a for a in planned.actions if a.interval.startswith("demo:")]
            commands = [replace(a, tick=revision + i) for i, a in enumerate(commands)]
            queue.publish(revision, commands)
            if revision < 2:
                # Replanning replaces the unissued suffix. The first receipt
                # can also beat the publication's journal event in production.
                receipt = motor.tick()
                target = (
                    np.asarray(plan["base_from_camera"]) @ pose @ plan["keyframes"][0]["reference_from_tool"]
                )
                np.testing.assert_allclose(receipt.requested.tool_target, target)
                expected.append((camera_from_base @ target)[:3, 3])
                assert not np.allclose(
                    expected[-1], (camera_from_base @ model.forward(receipt.accepted))[:3, 3]
                )
            journal.emit("plan.published", revision=revision, evidence=planned.evidence, actions=commands)
    finally:
        journal.close()
    robot.config.geometry_path.unlink()
    result = read_trajectory(tmp_path)
    assert len(result["points"]) == 2  # No replaced or merely planned targets.
    np.testing.assert_allclose([p["position"] for p in result["points"]], expected)
    assert all(p["source"] == "adapted_target" for p in result["points"])

    # Existing runs have no explicit target field, but preserve the exact
    # dispatched revision and reference transform needed to recover it.
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    for event in events:
        if event["kind"] == "control.tick":
            del event["tick"]["receipt"]["requested"]["tool_target"]
        elif event["kind"] == "plan.published":
            for action in event["actions"]:
                del action["tool_target"]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    assert read_trajectory(tmp_path) == result
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()


def test_fit_passes_through_commands_without_crossing_failure(tmp_path, robot):  # noqa: F811
    model = robot.cartesian_model()
    atomic_json(tmp_path / "plan.json", plan_for(robot))
    rows = [{"kind": "run.started", "time_s": 10.0}]
    for i, q in enumerate([0, 10, 20, 15]):
        action = {**ready_state(model), model.joint_names[0]: q}
        rows.append({"kind": "robot.command_requested", "time_s": 11.0 + i, "requested": action})
        rows.append(
            {
                "kind": "robot.command_accepted",
                "time_s": 11.0 + i,
                "accepted": action,
            }
        )
    rows += [
        {"kind": "robot.command_failed", "time_s": 15.0},
        {"kind": "robot.command_requested", "time_s": 16.0, "requested": ready_state(model)},
        {"kind": "robot.command_accepted", "time_s": 16.0, "accepted": ready_state(model)},
        {"kind": "robot.command_requested", "time_s": 17.0, "requested": ready_state(model)},
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps({"sequence": i, **row}) + "\n" for i, row in enumerate(rows)) + '{"kind":'
    )
    result = read_trajectory(tmp_path)
    assert len(result["points"]) == 5
    assert len(result["spans"]) == 2
    points = np.array([point["position"] for point in result["points"][:4]])
    curve = np.array(result["spans"][0]["positions"])
    np.testing.assert_allclose(curve[::4], points)
    for i in range(3):
        assert (curve[i * 4 : (i + 1) * 4 + 1] >= np.minimum(points[i], points[i + 1]) - 1e-12).all()
        assert (curve[i * 4 : (i + 1) * 4 + 1] <= np.maximum(points[i], points[i + 1]) + 1e-12).all()
    assert result["points"][-1]["relative_s"] == 6


def test_empty_run_needs_no_model_or_robot(tmp_path):
    (tmp_path / "events.jsonl").write_text('{"kind":"run.started","time_s":1}\n')
    assert read_trajectory(tmp_path)["points"] == []
