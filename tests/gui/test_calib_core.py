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

"""Touch-calibration arithmetic on synthetic data, and the router's guards without devices."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

from lerobot.gui.api import _calib_core as core, calib

INTR = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}


def _pose(rot: Rotation, t) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = t
    return pose


def test_tool_point_recovers_the_fingertip_from_touches_at_different_orientations():
    rng = np.random.default_rng(0)
    d_true = np.array([0.021, -0.031, 0.095])  # fingertip in the anchor frame
    point = np.array([0.25, -0.05, 0.02])  # the touched corner, base frame
    poses = []
    for eul in ([0, 0, 0], [30, 0, 0], [0, 35, 0], [-20, 15, 40], [10, -30, -25]):
        rot = Rotation.from_euler("xyz", eul, degrees=True)
        p = point - rot.apply(d_true) + rng.normal(0, 0.0005, 3)  # half a millimetre of touch noise
        poses.append(_pose(rot, p))
    out = core.solve_tool_point(poses)
    assert np.allclose(out["offset_m"], d_true, atol=0.002)
    assert np.allclose(out["point_m"], point, atol=0.002)
    assert out["max_m"] < 0.002 and out["n"] == 5


def test_tool_point_refuses_touches_that_share_an_orientation():
    rot = Rotation.from_euler("x", 20, degrees=True)
    poses = [_pose(rot, [0.2 + 0.01 * i, 0, 0]) for i in range(4)]
    with pytest.raises(ValueError, match="too alike"):
        core.solve_tool_point(poses)
    with pytest.raises(ValueError, match="at least"):
        core.solve_tool_point(poses[:2])


def test_rigid_fit_recovers_a_known_transform_and_reports_scale_one():
    rng = np.random.default_rng(1)
    r_true = Rotation.from_euler("xyz", [170, 5, -30], degrees=True).as_matrix()
    t_true = np.array([0.3, -0.1, 0.6])
    cam = rng.uniform(-0.2, 0.2, size=(6, 3)) + [0, 0, 0.8]
    base = (r_true @ cam.T).T + t_true + rng.normal(0, 0.001, cam.shape)
    out = core.rigid_fit(cam, base)
    tf = np.asarray(out["transform"])
    assert np.allclose(tf[:3, :3], r_true, atol=0.01)
    assert np.allclose(tf[:3, 3], t_true, atol=0.005)
    assert out["scale"] == pytest.approx(1.0, abs=0.01)
    assert out["max_m"] < 0.005
    # A biased range source shows up in the scale, not silently in the residual alone.
    out_scaled = core.rigid_fit(cam * 1.03, base)
    assert out_scaled["scale"] == pytest.approx(1 / 1.03, abs=0.01)


def test_rigid_fit_rejects_collinear_points():
    cam = np.array([[0, 0, 1.0], [0, 0, 1.1], [0, 0, 1.2]])
    with pytest.raises(ValueError, match="collinear"):
        core.rigid_fit(cam, cam + 0.1)


def test_corners_lift_through_the_marker_plane_not_the_corner_pixels():
    # A tilted plane in front of the camera; the depth image has the corners
    # themselves zeroed (holes), as a stereo sensor tends to at edges.
    normal = np.array([0.2, -0.1, -1.0])
    normal /= np.linalg.norm(normal)
    c0 = np.array([0.0, 0.0, 0.8])
    h, w = INTR["height"], INTR["width"]
    vs, us = np.mgrid[0:h, 0:w]
    rays = core.pixel_rays(np.stack([us.ravel(), vs.ravel()], axis=1), INTR)
    t = (c0 @ normal) / (rays @ normal)
    depth = (rays * t[:, None])[:, 2].reshape(h, w).astype(np.float32)
    corners_px = np.array([[300, 200], [360, 205], [355, 265], [295, 260]], dtype=float)
    for u, v in corners_px.astype(int):
        depth[v - 2 : v + 3, u - 2 : u + 3] = 0.0
    corners, diag = core.corners_from_depth_plane(corners_px, depth, INTR)
    expect = core.pixel_rays(corners_px, INTR)
    expect *= ((c0 @ normal) / (expect @ normal))[:, None]
    assert np.allclose(corners, expect, atol=1e-4)
    assert diag["pixels_used"] > 30 and diag["plane_rms_m"] < 1e-5


def test_pnp_lift_matches_the_projection_it_came_from():
    side = 0.04
    r = Rotation.from_euler("xyz", [180, 10, 20], degrees=True).as_matrix()
    t = np.array([0.05, -0.02, 0.7])
    half = side / 2
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    cam = (r @ obj.T).T + t
    px = np.stack(
        [INTR["fx"] * cam[:, 0] / cam[:, 2] + INTR["cx"], INTR["fy"] * cam[:, 1] / cam[:, 2] + INTR["cy"]], 1
    )
    out = core.corners_from_pnp(px, side, INTR)
    assert np.allclose(out, cam, atol=1e-3)


def test_detect_markers_finds_a_rendered_marker_with_ordered_corners():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    canvas = np.full((400, 400), 255, np.uint8)
    canvas[100:300, 100:300] = cv2.aruco.generateImageMarker(dictionary, 7, 200)
    found = core.detect_markers(canvas)
    assert [m["id"] for m in found] == [7]
    corners = np.asarray(found[0]["corners_px"])
    assert np.allclose(corners[0], [100, 100], atol=1.5)  # top-left first
    assert np.allclose(corners[2], [299, 299], atol=1.5)


def test_calibration_round_trips_and_yields_a_tip_offset(tmp_path):
    path = core.calibration_path(tmp_path, "white_left")
    assert core.load_calibration(path) == {}
    assert core.tip_offset_from_calibration({}) is None
    core.save_calibration(path, {"tool_point": {"offset_m": [0.01, 0.02, 0.03]}})
    data = core.load_calibration(path)
    assert "saved_at" in data
    off = core.tip_offset_from_calibration(data)
    assert np.allclose(off[:3, 3], [0.01, 0.02, 0.03]) and np.allclose(off[:3, :3], np.eye(3))


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(calib.router)
    return TestClient(app)


def test_router_guards_without_an_arm_or_camera(client):
    st = client.get("/api/calib/state").json()
    assert st["arm_connected"] is False and st["tool"]["touches"] == []
    assert client.post("/api/calib/tool/touch").status_code == 409
    assert client.post("/api/calib/tool/solve").status_code == 422
    assert client.post("/api/calib/tool/save").status_code == 409
    assert client.post("/api/calib/markers", json={"dictionary": "DICT_4X4_50"}).status_code == 409
    assert client.post("/api/calib/markers", json={"dictionary": "nope"}).status_code == 422
    assert client.get("/api/calib/markers.jpg").status_code == 404
    assert client.post("/api/calib/camera/touch", json={"marker_id": 1, "corner": 0}).status_code == 409
    assert client.post("/api/calib/camera/solve", json={"source": "depth"}).status_code == 422
