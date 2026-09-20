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
