"""Read-only reset sensing uses fresh frames and rejects movement/stale poses."""

import io
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import jog, pregrasp, reset_observation as obs
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES


class Kin:
    def forward_kinematics(self, q):
        t = np.eye(4)
        t[:3, 3] = np.asarray(q[:3]) * 0.01
        return t


@pytest.fixture
def rig(monkeypatch):
    j = jog._Jog(robot=object(), kin=Kin())
    j.q_obs = dict.fromkeys(MOTOR_NAMES, 0.0)
    j.q_cmd = j.q_obs.copy()
    j.last_read_at = time.monotonic()
    j.target = np.eye(4)
    monkeypatch.setattr(jog, "_jog", j)
    monkeypatch.setattr(pregrasp, "_state", pregrasp._State())
    pregrasp._state.demo = SimpleNamespace(name="demo", objects={"object": {"status": "done", "frame": 7}})
    pregrasp._state.worker.proc = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(pregrasp, "_t_base_cam", lambda: np.eye(4))
    monkeypatch.setattr(pregrasp, "_object_points", lambda *a: np.array([[0.01, 0.02, 0.03]]))
    count = [0]

    async def frame():
        count[0] += 1
        j.last_read_at = time.monotonic()
        return np.full((4, 5, 3), count[0], np.uint8), np.ones((4, 5)), {"fx": 10}

    async def check(rgb):
        return {"moved": False}

    async def locate(*a, **kw):
        assert kw["track"] is False, "measurement must not start a following controller"
        d = np.eye(4)
        d[0, 3] = 0.004
        return {"ok": True, "delta": d, "inliers": 100, "card_points": 100}

    monkeypatch.setattr(pregrasp, "_frame", frame)
    monkeypatch.setattr(pregrasp, "_camera_check_async", check)
    monkeypatch.setattr(pregrasp, "_locate", locate)
    app = FastAPI()
    app.include_router(obs.router)
    with TestClient(app) as client:
        yield j, client, count


def test_snapshot_keeps_rgbd_encoders_and_calibration_together(rig):
    j, client, count = rig
    response = client.get("/api/reset/frame.npz")
    assert response.status_code == 200, response.text
    data = np.load(io.BytesIO(response.content), allow_pickle=False)
    assert count[0] == 1
    assert np.all(data["rgb"] == 1)
    np.testing.assert_array_equal(data["q_obs"], np.zeros(len(MOTOR_NAMES)))
    np.testing.assert_array_equal(data["motor_names"], MOTOR_NAMES)
    np.testing.assert_array_equal(data["t_base_cam"], np.eye(4))


def test_object_pose_is_measured_in_reference_coordinates_and_never_moves_arm(rig):
    j, client, count = rig
    result = client.post("/api/reset/object_pose", json={"object": "object", "click": [2, 2]})
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["reference"] == ["demo", 7] and data["accepted"] == 5 and count[0] == 5
    assert np.asarray(data["motion_base"])[0, 3] == pytest.approx(0.004)
    np.testing.assert_allclose(data["motion_base"], data["hold_from_reference"])
    assert j.thread is None and j.q_cmd == j.q_obs


def test_snapshot_rejects_motion_during_camera_capture(rig, monkeypatch):
    j, client, _ = rig
    original = pregrasp._frame

    async def moved():
        result = await original()
        j.q_obs[MOTOR_NAMES[0]] = 0.4
        j.target = j.kin.forward_kinematics(np.array(list(j.q_obs.values())))
        return result

    monkeypatch.setattr(pregrasp, "_frame", moved)
    result = client.get("/api/reset/frame.npz")
    assert result.status_code == 409 and "moved" in result.text


def test_path_observation_requires_the_owner_and_actual_arrival(rig):
    j, client, _ = rig
    j.mode = "path"
    j.guarded_path = SimpleNamespace(identity="owner", status="running")
    assert client.get("/api/reset/frame.npz?path_id=owner").status_code == 409
    j.guarded_path.status = "done"
    assert client.get("/api/reset/frame.npz").status_code == 409
    assert client.get("/api/reset/frame.npz?path_id=someone").status_code == 409
    result = client.get("/api/reset/frame.npz?path_id=owner")
    assert result.status_code == 200, result.text


@pytest.mark.parametrize("fault", ["stale", "follow", "act", "target", "freeze"])
def test_moving_or_unavailable_arm_cannot_supply_reset_pose(rig, fault):
    j, client, _ = rig
    if fault == "stale":
        j.last_read_at -= 1
    if fault == "follow":
        pregrasp._state.track.follow = True
    if fault == "act":
        pregrasp._state.act.on = True
    if fault == "target":
        j.target[0, 3] += 0.02
    if fault == "freeze":
        j.halted = True
    assert client.get("/api/reset/frame.npz").status_code == 409


def test_weak_finds_and_camera_calibration_drift_are_not_measurements(rig, monkeypatch):
    _, client, _ = rig

    async def weak(*a, **kw):
        return {"object": "object", "ok": True, "delta": np.eye(4), "inliers": 1, "card_points": 100}

    monkeypatch.setattr(pregrasp, "_locate", weak)
    result = client.post("/api/reset/object_pose", json={"object": "object", "click": [2, 2]})
    assert result.status_code == 409 and "consistent strong" in result.text

    async def moved(rgb):
        return {"moved": True}

    monkeypatch.setattr(pregrasp, "_camera_check_async", moved)
    result = client.get("/api/reset/frame.npz")
    assert result.status_code == 409 and "calibration" in result.text
