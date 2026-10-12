"""Measured reset orchestration on the calibrated idealized fixture; no GUI calls."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import test_planned_reset as fixtures
from guarded_execution import ExecutionStoppedError
from live_reset import MeasuredReset
from planned_reset import ResetPlanner


@pytest.fixture(scope="module")
def planned():
    fixtures.TransferTests.setUpClass()
    scene = fixtures.TransferTests.scene
    target = copy.deepcopy(scene.bodies["gamepad"])
    target.pose[0, 3] += 0.020
    return scene, ResetPlanner(scene).transfer("gamepad", target, fixtures.TransferTests.start)


def fake_run(planned, drifting=False, change_reference=False, tilted_release=False):
    scene, plan = planned
    scene = copy.copy(scene)
    executor = SimpleNamespace(
        scene=scene, identity="owner", stopped=False, q=plan.segments[0].joints[0].copy(), phases=[]
    )

    def run(segment):
        executor.phases.append(segment.phase)
        executor.q = segment.joints[-1].copy()

    def stop():
        executor.stopped = True

    executor.run, executor.stop = run, stop
    source = plan.source.pose
    nominal_hold = plan.segments[4].attached[1]
    calls = []
    actual_hold = nominal_hold.copy()
    actual_hold[0, 3] += 0.004

    def measure(name, token, predicted):
        calls.append(token)
        tip = scene.robot.tip(executor.q)
        if len(calls) == 1:
            pose = tip @ nominal_hold
        elif executor.phases[-1] == "release":
            pose = tip @ actual_hold
            if tilted_release:
                angle = np.deg2rad(20)
                rotate = np.array(
                    [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
                )
                pose[:3, :3] = rotate @ pose[:3, :3]
        else:
            hold = actual_hold.copy()
            if drifting:
                hold[0, 3] += len(calls) * 0.003
            pose = tip @ hold
        motion = pose @ np.linalg.inv(source)
        q = dict(zip(scene.robot.names, executor.q, strict=True))
        return {
            "reference": ["changed" if change_reference and len(calls) > 1 else "demo", 0],
            "motion_base": motion,
            "hold_from_reference": np.linalg.inv(tip) @ motion,
            "q_obs": q,
            "q_cmd": q,
        }

    initial = {"reference": ["demo", 0], "motion_base": np.eye(4)}
    runner = MeasuredReset(scene, executor, measure, lambda _: None)
    return runner, executor, calls, plan, initial


def test_reset_uses_measured_hold_after_lift_and_checks_release_before_retreat(planned):
    runner, executor, calls, plan, initial = fake_run(planned)
    actual = runner.transfer(plan, initial)
    np.testing.assert_allclose(actual[:3, 3], plan.destination.pose[:3, 3], atol=0.0005)
    assert len(calls) == 4 and not executor.stopped
    assert executor.phases == [
        "open",
        "approach",
        "grasp_descent",
        "close",
        "lift",
        "carry",
        "place_descent",
        "release",
        "retreat",
    ]
    np.testing.assert_allclose(runner.scene.bodies["gamepad"].pose, actual)


def test_changed_reference_freezes_before_carry(planned):
    runner, executor, _, plan, initial = fake_run(planned, change_reference=True)
    with pytest.raises(ExecutionStoppedError, match="reference changed"):
        runner.transfer(plan, initial)
    assert executor.stopped and executor.phases[-1] == "lift"


def test_repeated_slip_cannot_proceed_to_placement(planned):
    runner, executor, _, plan, initial = fake_run(planned, drifting=True)
    with pytest.raises((ExecutionStoppedError, RuntimeError)):
        runner.transfer(plan, initial)
    assert executor.stopped and "place_descent" not in executor.phases


def test_tilted_release_is_not_reported_as_a_successful_reset(planned):
    runner, executor, _, plan, initial = fake_run(planned, tilted_release=True)
    with pytest.raises(ExecutionStoppedError, match="tilted"):
        runner.transfer(plan, initial)
    assert executor.stopped and executor.phases[-1] == "release"
