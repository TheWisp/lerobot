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


def test_marker_sheet_is_a_pdf_whose_markers_detect_at_the_requested_size(client):
    r = client.get("/api/calib/markers/sheet.pdf?side_mm=40&count=4")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"
    assert client.get("/api/calib/markers/sheet.pdf?side_mm=5").status_code == 422
    assert client.get("/api/calib/markers/sheet.pdf?side_mm=80&count=12").status_code == 422
    # The panel's default sheet must fit: eight markers at 40 mm.
    assert client.get("/api/calib/markers/sheet.pdf?side_mm=40&count=8").status_code == 200
    # The same sheet as an image: the markers come back in order, at 40 mm on a 300 dpi page.
    sheet = core.marker_sheet_image("DICT_4X4_50", 40.0, 4)
    found = core.detect_markers(np.asarray(sheet))
    assert [m["id"] for m in found] == [0, 1, 2, 3]
    side_px = np.linalg.norm(np.diff(np.asarray(found[0]["corners_px"])[:2], axis=0))
    assert side_px == pytest.approx(40 * 300 / 25.4, rel=0.01)


def test_refine_recovers_joint_zero_corrections_fingertip_and_camera_pose():
    """A synthetic arm whose zeros are off by known degrees: the joint fit finds them from the touches."""
    pytest.importorskip("pinocchio")
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES, TIP_OFFSET

    kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    inv_tip = np.linalg.inv(TIP_OFFSET)
    idx = {m: i for i, m in enumerate(MOTOR_NAMES)}
    dq_true = {"shoulder_lift": 4.0, "elbow_flex": 6.0, "wrist_flex": -3.0}
    d_true = np.array([-0.008, -0.090, 0.005])
    t_true = np.eye(4)
    t_true[:3, :3] = Rotation.from_euler("xyz", [200, 5, 90], degrees=True).as_matrix()
    t_true[:3, 3] = [-0.18, -0.05, 0.40]

    def fk_anchor(q, dq):
        qq = np.array(q, dtype=float)
        for m, v in dq.items():
            qq[idx[m]] += v
        return kin.forward_kinematics(qq) @ inv_tip

    def fingertip(q):  # where the real (zero-shifted) arm's fingertip is for encoder reading q
        a = fk_anchor(q, dq_true)
        return a[:3, 3] + a[:3, :3] @ d_true

    rng = np.random.default_rng(3)
    seed = np.array([0.0, -40.0, 70.0, 0.0, -40.0, -10.0, 90.0])
    # Corner touches: six configurations, the camera seeing the true fingertip.
    cam_touches = []
    for i in range(6):
        q = seed + rng.uniform(-25, 25, 7) * [1, 1, 1, 0.5, 1, 1, 0]
        corner_cam = t_true[:3, :3].T @ (fingertip(q) - t_true[:3, 3])
        a_model = fk_anchor(q, {})
        cam_touches.append(
            {
                "marker_id": i,
                "corner": 0,
                "q_obs": dict(zip(MOTOR_NAMES, q, strict=True)),
                "cam_depth_m": corner_cam.tolist(),
                "base_m": (a_model[:3, 3] + a_model[:3, :3] @ TIP_OFFSET[:3, 3]).tolist(),
            }
        )
    # Tool touches: distinct wrist orientations whose true fingertip lands on one point. The model's IK
    # gives a configuration for each target anchor; the real arm reads that configuration minus dq.
    point = fingertip(seed)
    tool_touches = []
    for _ in range(6):
        q = seed + rng.uniform(-30, 30, 7) * [0.3, 0.5, 0.5, 1, 1, 1, 0]
        target = fk_anchor(q, dq_true).copy()
        target[:3, 3] = point - target[:3, :3] @ d_true
        # The IK is iterative; drive it until the anchor lands, or skip the orientation.
        q_model = np.array(q, dtype=float)
        for _ in range(30):
            try:
                q_model = np.array(kin.inverse_kinematics(q_model, target @ TIP_OFFSET), dtype=float)
            except Exception:
                break
            if np.linalg.norm(fk_anchor(q_model, {})[:3, 3] - target[:3, 3]) < 2e-4:
                break
        else:
            continue
        if np.linalg.norm(fk_anchor(q_model, {})[:3, 3] - target[:3, 3]) >= 2e-4:
            continue
        for m, v in dq_true.items():
            q_model[idx[m]] -= v
        tool_touches.append(
            {"q_obs": dict(zip(MOTOR_NAMES, q_model, strict=True)), "anchor": fk_anchor(q_model, {}).tolist()}
        )
    assert len(tool_touches) >= 3, "the synthetic tool touches need three reachable orientations"

    out = core.refine_kinematics(fk_anchor, MOTOR_NAMES, tool_touches, cam_touches)
    for m, v in dq_true.items():
        assert out["joint_zero_deg"][m] == pytest.approx(v, abs=0.3), m
    assert np.allclose(out["offset_m"], d_true, atol=0.002)
    tf = np.asarray(out["transform"])
    assert np.allclose(tf[:3, :3], t_true[:3, :3], atol=0.01) and np.allclose(
        tf[:3, 3], t_true[:3, 3], atol=0.003
    )
    assert out["camera_rms_m"] < 0.002 and out["tool_rms_m"] < 0.003
    assert out["before"]["camera_rms_m"] > out["camera_rms_m"]


def test_corrected_alignment_folds_motor_side_zero_into_the_offset():
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT

    out = core.corrected_alignment(LEFT_ARM_ALIGNMENT, {"elbow_flex": 2.0})
    a, b = LEFT_ARM_ALIGNMENT["elbow_flex"], out["elbow_flex"]
    assert b.sign == a.sign and b.offset_deg == pytest.approx(a.offset_deg + a.sign * 2.0)
    assert out["shoulder_pan"] == LEFT_ARM_ALIGNMENT["shoulder_pan"]
    assert core.joint_zero_from_calibration({}) == {}


def test_marker_drift_flags_a_moved_camera_and_tolerates_hidden_stickers():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    canvas = np.full((480, 848), 255, np.uint8)
    for mid, (x, y) in {0: (60, 60), 1: (600, 60), 2: (60, 320), 3: (600, 320)}.items():
        canvas[y : y + 80, x : x + 80] = cv2.aruco.generateImageMarker(dictionary, mid, 80)
    reference = {m["id"]: dict(enumerate(m["corners_px"])) for m in core.detect_markers(canvas)}
    assert sorted(reference) == [0, 1, 2, 3]
    # The same frame again: nothing moved.
    still = core.marker_drift(reference, core.detect_markers(canvas))
    assert still["checked"] and not still["moved"] and still["max_px"] < 0.5 and still["n_corners"] == 16
    # The whole image shifted by three pixels: the camera or the tray moved.
    moved = core.marker_drift(reference, core.detect_markers(np.roll(canvas, 3, axis=1)))
    assert moved["checked"] and moved["moved"] and 2.5 < moved["max_px"] < 3.5
    # Two stickers hidden under objects: the check runs on the visible ones and still passes.
    partial = canvas.copy()
    partial[300:420, 40:160] = 255
    partial[300:420, 580:700] = 255
    part = core.marker_drift(reference, core.detect_markers(partial))
    assert part["checked"] and not part["moved"] and part["missing"] == [2, 3] and part["n_corners"] == 8
    # Fewer than three corners visible: nothing is claimed either way.
    one = core.marker_drift({0: {0: reference[0][0]}}, core.detect_markers(canvas))
    assert not one["checked"] and not one["moved"]


def test_marker_reference_prefers_the_saved_detection_and_falls_back_to_the_touches():
    full = {
        "markers_px": {"4": [[1, 2], [3, 4], [5, 6], [7, 8]]},
        "touches": [{"marker_id": 9, "corner": 1, "pixel": [0, 0]}],
    }
    assert core.marker_reference(full) == {4: {0: [1.0, 2.0], 1: [3.0, 4.0], 2: [5.0, 6.0], 3: [7.0, 8.0]}}
    old = {
        "touches": [
            {"marker_id": 9, "corner": 1, "pixel": [10, 20]},
            {"marker_id": 9, "corner": 3, "pixel": [30, 40]},
        ]
    }
    assert core.marker_reference(old) == {9: {1: [10.0, 20.0], 3: [30.0, 40.0]}}
