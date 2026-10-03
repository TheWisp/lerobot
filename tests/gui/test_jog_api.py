# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Jog API: the bounded pose walk, the motor->URDF conversion, and the guards that need no arm."""

import math

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import jog
from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES


def _pose(xyz, rot=np.eye(3)):
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = xyz
    return pose


def test_step_pose_is_bounded_per_tick_and_arrives():
    ref, target = _pose([0, 0, 0]), _pose([0.1, 0, 0])
    stepped = jog._step_pose(ref, target)
    assert np.linalg.norm(stepped[:3, 3]) == pytest.approx(jog.MAX_LINEAR_M_S / jog.HZ)
    for _ in range(200):
        ref = jog._step_pose(ref, target)
    assert np.allclose(ref, target)


def test_step_pose_bounds_rotation_and_finishes_it():
    from scipy.spatial.transform import Rotation

    target = _pose([0, 0, 0], Rotation.from_euler("z", 40, degrees=True).as_matrix())
    ref = _pose([0, 0, 0])
    stepped = jog._step_pose(ref, target)
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(stepped[:3, :3]).as_rotvec()))
    assert ang == pytest.approx(math.degrees(jog.MAX_ANGULAR_RAD_S / jog.HZ))
    for _ in range(100):
        ref = jog._step_pose(ref, target)
    assert np.allclose(ref, target, atol=1e-9)


def test_urdf_rad_applies_sign_and_offset():
    q = dict.fromkeys(MOTOR_NAMES, 10.0)
    out = jog._urdf_rad(LEFT_ARM_ALIGNMENT, q)
    a = LEFT_ARM_ALIGNMENT["shoulder_lift"]
    assert out["S2"] == pytest.approx(math.radians(a.sign * 10.0 + a.offset_deg))
    assert out["S1"] == pytest.approx(math.radians(-10.0))


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(jog.router)
    return TestClient(app)


def test_guards_without_an_arm(client):
    assert client.get("/api/jog/meta").json() == {"available": False}
    assert client.get("/api/jog/state").json() == {"connected": False}
    assert (
        client.post("/api/jog/target", json={"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]}).status_code
        == 409
    )
    assert client.post("/api/jog/connect", json={"profile": "x", "arm": "middle"}).status_code == 422
    assert client.post(
        "/api/jog/connect", json={"profile": "no-such-profile-xyz", "arm": "left"}
    ).status_code in (404, 500)
    assert client.post("/api/jog/disconnect").json() == {"status": "ok"}


def test_step_pose_honours_the_speed_it_is_given():
    ref, target = _pose([0, 0, 0]), _pose([0.1, 0, 0])
    stepped = jog._step_pose(ref, target, max_linear_m_s=0.3)
    assert np.linalg.norm(stepped[:3, 3]) == pytest.approx(0.3 / jog.HZ)


def test_limits_are_validated_and_survive_without_an_arm(client):
    r = client.post("/api/jog/limits", json={"linear_mm_s": 120, "angular_deg_s": 60})
    assert r.status_code == 200
    assert r.json()["linear_mm_s"] == pytest.approx(120)
    assert r.json()["angular_deg_s"] == pytest.approx(60)
    assert client.post("/api/jog/limits", json={"linear_mm_s": 5000}).status_code == 422
    assert client.post("/api/jog/limits", json={"angular_deg_s": 0}).status_code == 422
    # A rejected value leaves the accepted ones in place.
    assert jog._jog.max_linear_m_s == pytest.approx(0.12)
    assert jog._jog.max_angular_rad_s == pytest.approx(math.radians(60))


def test_cap_rotation_stops_on_the_same_axis_at_the_cap():
    from scipy.spatial.transform import Rotation

    r_obs = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    far = _pose([0.1, 0.2, 0.3], Rotation.from_euler("x", 110, degrees=True).as_matrix())
    out, clamped = jog._cap_rotation(far, r_obs, math.radians(60))
    assert clamped
    assert np.allclose(out[:3, 3], far[:3, 3])
    rel = Rotation.from_matrix(out[:3, :3] @ r_obs.T).as_rotvec()
    assert np.degrees(np.linalg.norm(rel)) == pytest.approx(60)
    assert np.allclose(rel / np.linalg.norm(rel), [1, 0, 0])
    near = _pose([0, 0, 0], Rotation.from_euler("x", 50, degrees=True).as_matrix())
    same, clamped = jog._cap_rotation(near, r_obs, math.radians(60))
    assert not clamped and same is near


def test_rotation_cap_limit_is_validated(client):
    assert client.post("/api/jog/limits", json={"rotation_cap_deg": 45}).json()[
        "rotation_cap_deg"
    ] == pytest.approx(45)
    assert client.post("/api/jog/limits", json={"rotation_cap_deg": 200}).status_code == 422


def test_gripper_request_is_validated_and_needs_an_arm(client):
    assert client.post("/api/jog/gripper", json={"pos": 50}).status_code == 409
    assert client.post("/api/jog/gripper", json={"pos": 120}).status_code in (409, 422)


class _FakeBus:
    def read(self, *args, **kwargs):
        return 30


class _FakeRobot:
    def __init__(self, q):
        self.q, self.sent, self.bus = dict(q), [], _FakeBus()

    def get_observation(self):
        return {f"{m}.pos": v for m, v in self.q.items()}

    def send_action(self, action):
        self.sent.append(dict(action))
        return action


class _FakeKin:
    def forward_kinematics(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q[:3], dtype=float) / 1000.0
        return pose

    def inverse_kinematics(self, seed, pose):
        return np.asarray(seed, dtype=float).copy()


def test_handing_the_arm_back_after_an_act_keeps_the_grasps_closing():
    q = dict.fromkeys(MOTOR_NAMES, 0.0)
    q["gripper"] = 80.2  # the jaws stopped by the object
    j = jog._Jog(robot=_FakeRobot(q), kin=_FakeKin(), arm="left", workspace_min=(-1.0, -1.0, -1.0))
    j.mode, j.q_target = "joints", {**q, "gripper": 80.9}  # the act's last command closes past contact
    try:
        jog._joints_stop(j)
        assert j.mode == "cartesian" and j.q_target is None
        assert j.q_cmd["gripper"] == 80.9 and j.grip_target == 80.9, (
            "the observed opening would relax the grasp"
        )
    finally:
        jog._stop_loop(j)
