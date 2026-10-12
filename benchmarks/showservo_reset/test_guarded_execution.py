"""Fake-GUI regressions: checked execution never opens a serial connection."""

from types import SimpleNamespace

import numpy as np
import pytest
from guarded_execution import ExecutionStoppedError, GuiExecutor, checked_samples
from motion_planner import NoPathError
from planned_reset import Segment


class Robot:
    names = ["x", "y", "z", "gripper"]

    @staticmethod
    def tip(q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q[:3]) / 100
        return pose


def scene(valid=lambda q: True):
    return SimpleNamespace(robot=Robot(), check=lambda q, **kw: valid(q), reason="test obstacle")


def segment(q=None, **kw):
    return Segment("carry", np.asarray(q if q is not None else [[0, 0, 0, 80], [1, 0, 0, 80]], float), **kw)


def test_resampling_checks_interior_and_bounds_tip_travel():
    seg = segment()
    checked = checked_samples(seg, scene(), (seg.joints[0], seg.joints[0]))
    assert np.max(np.abs(np.diff(checked.joints[:, 0]))) <= 0.2
    with pytest.raises(NoPathError, match="obstacle"):
        checked_samples(seg, scene(lambda q: not 0.1 < q[0] < 0.4), (seg.joints[0], seg.joints[0]))


def test_bridge_from_actual_arrival_is_checked_instead_of_ramping_to_nominal():
    seg = segment()
    actual = seg.joints[0] + [-0.5, 0, 0, 0]
    checked = checked_samples(seg, scene(), (actual, seg.joints[0]))
    np.testing.assert_allclose(checked.joints[0], actual)
    with pytest.raises(NoPathError, match="obstacle"):
        checked_samples(seg, scene(lambda q: not -0.4 < q[0] < -0.1), (actual, seg.joints[0]))
    with pytest.raises(NoPathError, match="replan"):
        checked_samples(seg, scene(), (actual + [-1, 0, 0, 0], seg.joints[0]))


def test_held_jaws_are_checked_over_actual_opening_to_closing_command():
    seg = segment(attached=("gamepad", np.eye(4)))
    actual = seg.joints[0] + [0, 0, 0, -4]
    with pytest.raises(NoPathError, match="obstacle"):
        checked_samples(seg, scene(lambda q: q[-1] >= 79 or q[0] < 0.3), (actual, seg.joints[0]))
    checked = checked_samples(seg, scene(), (actual, seg.joints[0]))
    assert np.all(checked.joints[:, -1] == 80), "the closing command must not become the observed opening"


def test_gripper_change_uses_stationary_actual_arm_without_blended_arm_motion():
    seg = Segment("release", np.array([[0, 0, 0, 80], [0, 0, 0, 76]], float))
    actual = seg.joints[0] + [0.3, 0, 0, 0]
    checked = checked_samples(seg, scene(), (actual, seg.joints[0]))
    assert np.all(checked.joints[:, 0] == 0.3)
    assert checked.joints[-1, -1] == 76


class FakeGui:
    def __init__(self):
        self.calls = []
        self.q = np.array([0, 0, 0, 80.0])
        self.path = None
        self.age = 0.01
        self.rows = None
        self.i = 0

    def state(self):
        q = dict(zip(Robot.names, self.q, strict=True))
        return {
            "connected": True,
            "halted": False,
            "read_age_s": self.age,
            "q_obs": q,
            "q_cmd": q,
            "path": self.path,
        }

    def __call__(self, route, body=None):
        self.calls.append((route, body))
        if route == "start":
            self.rows = np.asarray(body["samples"])
            self.path = {"id": "owned", "status": "running"}
            self.i = 0
        if route == "heartbeat":
            self.i += 1
            self.q = self.rows[self.i]
            self.path = {"id": "owned", "status": "done" if self.i == len(self.rows) - 1 else "running"}
        return self.state()


def test_gui_execution_preserves_joint_plan_and_requires_a_finished_handoff(monkeypatch):
    monkeypatch.setattr("guarded_execution.time.sleep", lambda _: None)
    gui = FakeGui()
    logs = []
    executor = GuiExecutor(scene(), logs.append, call=gui)
    observed, commanded = executor.run(segment())
    np.testing.assert_allclose(observed, [1, 0, 0, 80])
    assert executor.identity == "owned" and logs
    assert "finish" not in [n for n, _ in gui.calls]
    executor.finish()
    assert gui.calls[-1] == ("finish", {"id": "owned"})
    assert executor.identity is None


def test_stale_telemetry_prevents_any_path_start():
    gui = FakeGui()
    gui.age = 1
    executor = GuiExecutor(scene(), lambda _: None, call=gui)
    with pytest.raises(ExecutionStoppedError, match="stale"):
        executor.run(segment())
    assert [n for n, _ in gui.calls] == ["state"]


def test_observed_collision_stops_and_never_renews_lease(monkeypatch):
    monkeypatch.setattr("guarded_execution.time.sleep", lambda _: None)
    gui = FakeGui()
    original = gui.__call__

    def changed(route, body=None):
        result = original(route, body)
        if route == "heartbeat":
            result["q_obs"]["y"] = 1
        return result

    # Keep the fault in the next state snapshot, too.
    def with_drift(route, body=None):
        result = changed(route, body)
        if any(n == "heartbeat" for n, _ in gui.calls):
            result["q_obs"]["y"] = 1
        return result

    executor = GuiExecutor(scene(lambda q: q[1] < 0.5), lambda _: None, call=with_drift)
    with pytest.raises(ExecutionStoppedError, match="observed arm"):
        executor.run(segment())
    routes = [n for n, _ in gui.calls]
    assert routes.count("heartbeat") == 1 and routes[-1] == "stop"
    assert "finish" not in routes
