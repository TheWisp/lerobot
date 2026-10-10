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

"""Teach-and-transport arithmetic on a synthetic textured object that slides across a flat table."""

import json
import pathlib

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

from lerobot.gui.api import _pregrasp_core as core, pregrasp

INTR = {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0, "width": 848, "height": 480}


def _scene(rng, shift=(0, 0)):
    """A flat table at 0.45 m with a textured patch; the patch is shifted by whole pixels."""
    h, w = 480, 848
    rgb = np.full((h, w, 3), 128, np.uint8)
    patch = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    y0, x0 = 200 + shift[1], 300 + shift[0]
    rgb[y0 : y0 + 120, x0 : x0 + 160] = patch
    depth = np.full((h, w), 0.45, np.float32)
    return rgb, depth


def test_keypoints_and_registration_recover_a_pure_slide():
    rng = np.random.default_rng(5)
    rgb0, depth0 = _scene(rng)
    rng = np.random.default_rng(5)  # same texture, moved
    dx, dy = 40, -25
    rgb1, depth1 = _scene(rng, shift=(dx, dy))
    teach = core.keypoints_in_box(rgb0, depth0, INTR, (300, 200, 460, 320))
    assert teach["valid"].all() and len(teach["uv"]) >= 8
    out = core.register(teach, rgb1, depth1, INTR)
    assert out["ok"], out
    d = np.asarray(out["delta_cam"])
    expect = np.array([dx * 0.45 / INTR["fx"], dy * 0.45 / INTR["fy"], 0.0])
    assert np.allclose(d[:3, 3], expect, atol=0.002)
    assert np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())) < 1.0
    assert out["rms_m"] < 0.002 and abs(out["scale"] - 1.0) < 0.02
    assert out["n_inliers_3d"] >= 8


def test_registration_abstains_on_an_unrelated_frame():
    rng = np.random.default_rng(6)
    rgb0, depth0 = _scene(rng)
    teach = core.keypoints_in_box(rgb0, depth0, INTR, (300, 200, 460, 320))
    rgb_other, depth_other = _scene(np.random.default_rng(99))
    out = core.register(teach, rgb_other, depth_other, INTR)
    assert not out["ok"] and out["reason"]


def test_keypoints_refuse_a_box_without_depth():
    rng = np.random.default_rng(7)
    rgb, depth = _scene(rng)
    depth[:] = 0.0
    with pytest.raises(ValueError, match="depth"):
        core.keypoints_in_box(rgb, depth, INTR, (300, 200, 460, 320))


def test_transport_conjugates_the_camera_motion_into_the_base_frame():
    t_bc = np.eye(4)
    t_bc[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()  # camera looking down
    t_bc[:3, 3] = [0.1, 0.2, 0.5]
    delta_cam = np.eye(4)
    delta_cam[:3, 3] = [0.03, 0.0, 0.0]  # object slid along camera x
    pose = np.eye(4)
    pose[:3, 3] = [0.25, -0.1, 0.05]
    out = core.transport_pose(t_bc, delta_cam, pose)
    # camera x maps to base x under a 180 deg roll, so the pose slides +30 mm in base x
    assert np.allclose(out[:3, 3], [0.28, -0.1, 0.05])
    assert np.allclose(out[:3, :3], pose[:3, :3])
    summary = core.motion_summary(delta_cam)
    assert summary["translation_mm"] == pytest.approx(30.0) and summary["rotation_deg"] == pytest.approx(0.0)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(pregrasp.router)
    return TestClient(app)


def test_router_guards_without_devices(client):
    st = client.get("/api/pregrasp/state").json()
    assert st["teach"] is None and st["test"] is None
    assert client.post("/api/pregrasp/teach/capture", json={"box": [0, 0, 10, 10]}).status_code == 409
    assert client.post("/api/pregrasp/demo/record/start").status_code == 409
    assert client.post("/api/pregrasp/test/capture").status_code == 409
    assert client.post("/api/pregrasp/go", json={"hover_mm": 10}).status_code == 409
    assert client.get("/api/pregrasp/teach.jpg").status_code == 404


def _shape_scene(shift=(0, 0), size=(60, 90), height=0.02, z_table=0.45):
    """A flat table with one plain raised block (no texture), pixel-shifted."""
    h, w = 480, 848
    rgb = np.full((h, w, 3), 128, np.uint8)
    depth = np.full((h, w), z_table, np.float32)
    y0, x0 = 220 + shift[1], 380 + shift[0]
    depth[y0 : y0 + size[0], x0 : x0 + size[1]] = z_table - height
    return rgb, depth


def test_shape_mode_finds_a_plain_block_moved_across_the_table():
    rgb0, depth0 = _shape_scene()
    teach = core.shape_teach(depth0, INTR, (360, 200, 490, 300))
    assert teach["mode"] == "shape" and teach["n_points"] > 1000
    assert teach["height_m"] == pytest.approx(0.02, abs=0.003)
    assert teach["elongation"] > 1.3  # 60x90 footprint has a direction
    dx, dy = 120, -60
    _rgb1, depth1 = _shape_scene(shift=(dx, dy))
    out = core.shape_register(teach, depth1, INTR)
    assert out["ok"], out
    d = np.asarray(out["delta_cam"])
    # A shift of the block's footprint by (dx, dy) pixels at the block's own depth.
    z = 0.45 - 0.02
    assert np.allclose(d[:3, 3][:2], [dx * z / INTR["fx"], dy * z / INTR["fy"]], atol=0.003)
    assert abs(out["yaw_deg"]) < 2.0 and not out["symmetric"]


def test_shape_mode_abstains_when_nothing_similar_stands_on_the_table():
    _rgb0, depth0 = _shape_scene()
    teach = core.shape_teach(depth0, INTR, (360, 200, 490, 300))
    _rgb1, depth1 = _shape_scene(size=(20, 20), height=0.05)  # a different, taller, smaller thing
    out = core.shape_register(teach, depth1, INTR)
    assert not out["ok"] and out["reason"]


def test_a_plain_block_has_no_texture_to_teach_by():
    rgb, depth = _shape_scene()
    with pytest.raises(ValueError, match="textured"):
        core.keypoints_in_box(rgb, depth, INTR, (360, 200, 490, 300))


def test_weak_texture_is_taught_by_shape_and_strong_texture_keeps_a_shape_fallback():
    # A raised block with a faint pattern: a few SIFT points, below the texture floor.
    rng = np.random.default_rng(11)
    rgb, depth = _shape_scene()
    faint = (128 + rng.integers(-6, 7, size=(60, 90, 3))).astype(np.uint8)
    rgb[220:280, 380:470] = faint
    try:
        kp = core.keypoints_in_box(rgb, depth, INTR, (360, 200, 490, 300))
        weak = int(kp["valid"].sum()) < core.MIN_TEXTURE_POINTS
    except ValueError:
        weak = True
    assert weak, "a faint pattern must not count as texture"
    # A strongly textured raised block keeps a shape model next to its keypoints and the
    # shape model alone still finds the block after it moved.
    rgb2, depth2 = _shape_scene()
    rgb2[200:300, 360:490] = rng.integers(0, 255, size=(100, 130, 3), dtype=np.uint8)
    kp2 = core.keypoints_in_box(rgb2, depth2, INTR, (360, 200, 490, 300))
    assert int(kp2["valid"].sum()) >= core.MIN_TEXTURE_POINTS
    shape = core.shape_teach(depth2, INTR, (360, 200, 490, 300))
    _rgb3, depth3 = _shape_scene(shift=(100, 40))
    out = core.shape_register(shape, depth3, INTR)
    assert out["ok"] and out["mode"] == "shape"


def _rect_scene(
    angle_deg=0.0, centre=(430, 250), size_px=(50, 110), height=0.02, colour=(230, 200, 40), z_table=0.45
):
    """A plain rectangular block of a given colour on a flat table, turned by ``angle_deg`` in the image."""
    import cv2

    h, w = 480, 848
    rgb = np.full((h, w, 3), 128, np.uint8)
    depth = np.full((h, w), z_table, np.float32)
    box = cv2.boxPoints(
        ((float(centre[0]), float(centre[1])), (float(size_px[1]), float(size_px[0])), float(angle_deg))
    )
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(box).astype(np.int32), 1)
    rgb[mask > 0] = colour
    depth[mask > 0] = z_table - height
    return rgb, depth


def test_shape_mode_measures_the_turn_of_a_rectangle_from_its_footprint():
    rgb0, depth0 = _rect_scene(0.0)
    teach = core.shape_teach(depth0, INTR, (350, 190, 510, 310), rgb0)
    rgb1, depth1 = _rect_scene(35.0, centre=(560, 300))
    out = core.shape_register(teach, depth1, INTR, rgb1)
    assert out["ok"] and not out["symmetric"], out
    # The image turn maps to a turn about the table normal; the camera looks straight down here, so
    # the magnitude matches and only the sign depends on the frame handedness.
    assert abs(abs(out["yaw_deg"]) - 35.0) < 3.0
    assert out["footprint_iou"] > 0.8
    # Round footprint: no turn reported.
    rgb_r, depth_r = _rect_scene(0.0, size_px=(80, 80))
    teach_r = core.shape_teach(depth_r, INTR, (350, 170, 510, 330), rgb_r)
    _rgb_r2, depth_r2 = _rect_scene(20.0, size_px=(80, 80), centre=(560, 300))
    out_r = core.shape_register(teach_r, depth_r2, INTR, _rgb_r2)
    # A square turned 20 deg matches at 20, 110, -70 and -160; the smallest consistent turn is 20.
    assert out_r["ok"] and abs(abs(out_r["yaw_deg"]) - 20.0) < 3.0


def test_colour_gate_keeps_the_taught_object_apart_from_a_touching_neighbour():
    rgb0, depth0 = _rect_scene(0.0, colour=(230, 200, 40))
    teach = core.shape_teach(depth0, INTR, (350, 190, 510, 310), rgb0)
    assert teach["colour"] is not None, "a yellow block on a grey table is a colour cue"
    # At find time a green block of the same size touches the yellow one.
    rgb1, depth1 = _rect_scene(0.0, centre=(560, 300), colour=(230, 200, 40))
    rgb_g, depth_g = _rect_scene(0.0, centre=(670, 300), colour=(40, 200, 60))
    rgb1[depth_g < 0.449] = rgb_g[depth_g < 0.449]
    depth1[depth_g < 0.449] = depth_g[depth_g < 0.449]
    out = core.shape_register(teach, depth1, INTR, rgb1)
    assert out["ok"] and out["colour_used"], out
    assert abs(out["n_points"] - teach["n_points"]) < 0.2 * teach["n_points"]
    z = 0.45 - 0.02
    assert np.allclose(
        np.asarray(out["delta_cam"])[:3, 3][:2], [130 * z / INTR["fx"], 50 * z / INTR["fy"]], atol=0.004
    )


class _FakeProc:
    def poll(self):
        return None

    def terminate(self):
        pass


def test_worker_protocol_round_trips_a_teach_and_a_find(client):
    """The server side of the worker protocol, with the worker played by the test."""
    import io
    import json

    pregrasp._state.worker.proc = _FakeProc()  # "running" without spawning anything
    try:
        rgb, depth = _rect_scene(0.0)
        job = pregrasp._queue_job("teach", "yellow block", rgb, depth, INTR)
        with pregrasp._state.lock:
            pregrasp._state.teach_job = job.id
        # No job is handed out twice, and the frame round-trips intact.
        r = client.get("/api/pregrasp/worker/job", params={"wait": 0})
        assert r.status_code == 200 and r.json()["id"] == job.id and r.json()["concept"] == "yellow block"
        assert client.get("/api/pregrasp/worker/job", params={"wait": 0}).status_code == 204
        fr = np.load(io.BytesIO(client.get("/api/pregrasp/worker/frame.npz", params={"id": job.id}).content))
        assert fr["rgb"].shape == rgb.shape and json.loads(str(fr["intr"]))["fx"] == INTR["fx"]
        # The worker's teach result becomes the taught object.
        mask = depth < 0.449
        uv = np.argwhere(mask)[::50][:, ::-1].astype(float)
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            meta=json.dumps(
                {
                    "ok": True,
                    "n_points": len(uv),
                    "radius_mm": 40.0,
                    "shape_class": "general",
                    "yaw_observable": True,
                }
            ),
            mask=mask,
            uv=uv,
            xyz=np.zeros((len(uv), 3)),
        )
        assert (
            client.post(
                "/api/pregrasp/worker/result", params={"id": job.id}, content=buf.getvalue()
            ).status_code
            == 200
        )
        st = client.get("/api/pregrasp/state").json()
        assert (
            st["teach"]["mode"] == "features"
            and st["teach"]["concept"] == "yellow block"
            and not st["teach_pending"]
        )
        assert client.get("/api/pregrasp/teach.jpg").status_code == 200
        # A find result with a delta lands as the test; without an arm there is no transport, and it says so.
        with pregrasp._state.lock:
            pregrasp._state.teach.tip_pose = np.eye(4)
        job2 = pregrasp._queue_job("find", "yellow block", rgb, depth, INTR)
        with pregrasp._state.lock:
            pregrasp._state.find_job = job2.id
        delta = np.eye(4)
        delta[:3, 3] = [0.03, 0.0, 0.0]
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            meta=json.dumps(
                {
                    "ok": True,
                    "n_matches": 40,
                    "n_inliers": 30,
                    "rms_m": 0.002,
                    "scale": 1.01,
                    "shape_class": "general",
                    "yaw_observable": True,
                }
            ),
            mask=mask,
            live_uv=uv,
            delta=delta,
        )
        assert (
            client.post(
                "/api/pregrasp/worker/result", params={"id": job2.id}, content=buf.getvalue()
            ).status_code
            == 200
        )
        st = client.get("/api/pregrasp/state").json()
        assert st["test"]["mode"] == "features" and not st["find_pending"]
        assert st["test"]["ok"] is False and "arm" in st["test"]["reason"]
        # Stopping the worker turns the poll into "gone".
        assert client.post("/api/pregrasp/worker/stop").status_code == 200
        assert client.get("/api/pregrasp/worker/job", params={"wait": 0}).status_code == 410
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.teach = None
            pregrasp._state.test = None


def test_snap_to_table_yaw_keeps_the_turn_and_the_centroid_but_drops_the_axis_tilt():
    n = np.array([0.1, -0.2, -1.0])
    n /= np.linalg.norm(n)  # a table seen from a camera looking down at a slight angle
    # A true 90 deg turn about the normal, reported by a noisy fit as 90 deg about an axis tilted by 25 deg.
    seed = np.array([1.0, 0.0, 0.0])
    e1 = seed - np.dot(seed, n) * n
    e1 /= np.linalg.norm(e1)
    tilted = Rotation.from_rotvec(e1 * np.radians(25.0)).apply(n)
    r_bad = Rotation.from_rotvec(tilted * np.radians(90.0)).as_matrix()
    c = np.array([0.02, -0.05, 0.45])
    c_new = c + np.array([0.08, 0.03, 0.0])
    delta_bad = np.eye(4)
    delta_bad[:3, :3] = r_bad
    delta_bad[:3, 3] = c_new - r_bad @ c
    out = core.snap_to_table_yaw(delta_bad, n, c)
    d = out["delta"]
    assert abs(abs(out["yaw_deg"]) - 90.0) < 3.0 and abs(out["tilt_deg"] - 25.0) < 1.0
    # The snapped turn is about the normal, and the centroid still lands where the fit put it.
    rv = Rotation.from_matrix(d[:3, :3]).as_rotvec()
    assert abs(abs(np.dot(rv / np.linalg.norm(rv), n)) - 1.0) < 1e-6
    assert np.allclose(d[:3, :3] @ c + d[:3, 3], c_new)
    # A pure yaw is left alone.
    r_ok = Rotation.from_rotvec(n * np.radians(40.0)).as_matrix()
    delta_ok = np.eye(4)
    delta_ok[:3, :3] = r_ok
    out2 = core.snap_to_table_yaw(delta_ok, n, c)
    assert abs(out2["yaw_deg"] - 40.0) < 1e-6 and out2["tilt_deg"] < 1e-6
    assert np.allclose(out2["delta"], delta_ok)


def test_options_toggle_flat(client):
    assert client.post("/api/pregrasp/options", json={"flat": False}).json()["flat"] is False
    assert client.get("/api/pregrasp/state").json()["flat"] is False
    assert client.post("/api/pregrasp/options", json={"flat": True}).json()["flat"] is True
    assert client.post("/api/pregrasp/options", json={}).json()["flat"] is True, (
        "an option not given is left alone"
    )
    assert client.post("/api/pregrasp/options", json={"flat": False}).json()["flat"] is False


def test_compose_with_face_takes_the_axis_from_the_faces_and_the_turn_from_the_fit():
    n_teach = np.array([0.05, -0.1, -1.0])
    n_teach /= np.linalg.norm(n_teach)
    # The block turned 90 deg about its own face normal and the face itself tilted by 12 deg.
    seed = np.array([1.0, 0.0, 0.0])
    e1 = seed - np.dot(seed, n_teach) * n_teach
    e1 /= np.linalg.norm(e1)
    r_face_tilt = Rotation.from_rotvec(e1 * np.radians(12.0)).as_matrix()
    n_find = r_face_tilt @ n_teach
    r_true = Rotation.from_rotvec(n_find * np.radians(90.0)).as_matrix() @ r_face_tilt
    c = np.array([0.0, -0.03, 0.45])
    c_new = c + np.array([0.05, 0.02, 0.0])
    # The fit got the turn but wandered 30 deg in the axis (as a thin cloud lets it).
    wander = Rotation.from_rotvec(e1 * np.radians(30.0)).as_matrix()
    r_fit = wander @ r_true
    delta_fit = np.eye(4)
    delta_fit[:3, :3] = r_fit
    delta_fit[:3, 3] = c_new - r_fit @ c
    out = core.compose_with_face(delta_fit, n_teach, n_find, c)
    d = out["delta"]
    assert abs(out["face_tilt_deg"] - 12.0) < 0.1
    assert np.allclose(d[:3, :3] @ n_teach, n_find, atol=1e-6)  # the taught face lands on the found face
    assert np.allclose(d[:3, :3] @ c + d[:3, 3], c_new)  # the centroid lands where the fit put it
    assert out["fit_axis_tilt_deg"] > 10.0  # the wander is reported
    # The composed rotation is the true one up to a few degrees of in-plane error from the wander.
    err = Rotation.from_matrix(d[:3, :3] @ r_true.T).magnitude()
    assert np.degrees(err) < 20.0
    # With no wander the composition reproduces the true motion exactly.
    delta_true = np.eye(4)
    delta_true[:3, :3] = r_true
    delta_true[:3, 3] = c_new - r_true @ c
    out2 = core.compose_with_face(delta_true, n_teach, n_find, c)
    assert np.allclose(out2["delta"], delta_true, atol=1e-6)
    assert abs(abs(out2["yaw_deg"]) - 90.0) < 1e-6


def test_turn_and_lean_split_a_base_motion_into_what_the_operator_can_see():
    up = np.array([0.0, 0.0, 1.0])
    yaw = np.eye(4)
    yaw[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    out = core.turn_and_lean(yaw, up)
    assert abs(out["turn_deg"] - 30.0) < 1e-6 and out["lean_deg"] < 1e-6
    tip = np.eye(4)
    tip[:3, :3] = Rotation.from_euler("x", 10, degrees=True).as_matrix()
    out = core.turn_and_lean(tip, up)
    assert abs(out["lean_deg"] - 10.0) < 1e-6 and abs(out["turn_deg"]) < 1e-6


def test_face_usable_wants_one_dominant_plane_with_enough_points():
    good = {"planarity": 0.6, "n": 1500, "n_plane": 900, "dominance": 2.5}
    assert core.face_usable(good)
    assert not core.face_usable(None)
    assert not core.face_usable({**good, "planarity": 0.2})  # the face is a sliver of the cloud
    assert not core.face_usable({**good, "n_plane": 50})  # too few points to beat the fit's own axis
    assert not core.face_usable({**good, "dominance": 1.2})  # two faces compete: which one is the face?
    assert core.face_usable({"planarity": 0.9, "n": 400})  # a card from before the consensus fit still counts


def test_go_refuses_when_the_markers_say_the_camera_moved(client):
    with pregrasp._state.lock:
        pregrasp._state.test = pregrasp._Test(
            at="now",
            rgb=np.zeros((4, 4, 3), np.uint8),
            result={
                "ok": True,
                "camera_check": {"checked": True, "moved": True, "max_px": 6.3, "tol_px": 2.0},
            },
            transported=np.eye(4),
        )
    try:
        r = client.post("/api/pregrasp/go", json={"hover_mm": 20})
        assert r.status_code == 409 and "moved" in r.json()["detail"] and "6.3" in r.json()["detail"]
    finally:
        with pregrasp._state.lock:
            pregrasp._state.test = None


def test_a_certified_fit_on_a_sliver_of_the_card_is_not_trusted():
    assert core.find_trusted(290, 399) == (True, "")
    ok, why = core.find_trusted(8, 399)
    assert not ok and "8 of the card's 399" in why
    # A small card is held to the absolute floor, not the share.
    assert core.find_trusted(20, 60)[0] and not core.find_trusted(19, 60)[0]


def test_a_find_matching_a_small_share_of_the_demo_view_is_weak_and_shown_so():
    """A find of the gamepad matching 32 of its demo view's 400 points put the grasp 20 degrees off and missed; finds
    matching 116 or more were within 6 degrees. The share sets strong or weak, and the live view says which."""
    from lerobot.gui.api import pregrasp

    assert core.find_strength(32, 400) == (False, 0.08)
    strong, share = core.find_strength(116, 400)
    assert strong and share == pytest.approx(0.29)
    assert core.find_strength(60, 400)[0] and not core.find_strength(59, 400)[0], "the bar is 15% of the card"
    assert core.find_strength(30, None) == (None, None), "a find made before the card's size was reported"
    weak = pregrasp._find_badge({"ok": True, "inliers": 32, "card_points": 400})
    assert (
        weak is not None and weak[0].startswith("find: weak, 32 of 400 points") and weak[1] == (0, 165, 255)
    )
    strong_badge = pregrasp._find_badge({"ok": True, "inliers": 310, "card_points": 400})
    assert strong_badge == ("find: strong, 310 of 400 points", (60, 230, 60))
    assert pregrasp._find_badge({"ok": True, "inliers": 310}) is None, "no badge without the card's size"
    assert pregrasp._find_badge({"ok": False, "reason": "no match"}) is None


def test_a_job_the_worker_never_answers_stops_pending(client):
    pregrasp._state.worker.proc = _FakeProc()
    try:
        rgb, depth = _rect_scene(0.0)
        job = pregrasp._queue_job("teach", "yellow block", rgb, depth, INTR)
        with pregrasp._state.lock:
            pregrasp._state.teach_job = job.id
        assert client.get("/api/pregrasp/state").json()["teach_pending"]
        job.created -= pregrasp.JOB_TIMEOUT_S + 1
        st = client.get("/api/pregrasp/state").json()
        assert not st["teach_pending"] and "never taken" in st["worker"]["log"]
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.teach_job = None
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()


def test_a_clicked_pixel_teaches_without_a_name_and_rides_on_the_job(client, monkeypatch):
    """A click on the camera view is a teach: the worker gets the pixel, SAM3 segments what is under it."""
    rgb, depth = _rect_scene(0.0)

    async def frame():
        return rgb, depth, INTR

    monkeypatch.setattr(pregrasp, "_frame", frame)
    pregrasp._state.worker.proc = _FakeProc()
    try:
        with pregrasp._state.lock:
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()
        assert (
            client.post("/api/pregrasp/teach/capture", json={"mode": "features", "concept": ""}).status_code
            == 422
        )
        assert (
            client.post("/api/pregrasp/teach/capture", json={"mode": "features", "click": [1]}).status_code
            == 422
        )
        r = client.post("/api/pregrasp/teach/capture", json={"mode": "features", "click": [40, 30]})
        assert r.status_code == 200 and r.json()["pending"]
        job = client.get("/api/pregrasp/worker/job", params={"wait": 0}).json()
        assert job["kind"] == "teach" and job["click"] == [40, 30] and job["concept"] == "clicked object"
        # A typed name and a click together keep the name.
        client.post(
            "/api/pregrasp/teach/capture",
            json={"mode": "features", "concept": "aa battery", "click": [40, 30]},
        )
        job = client.get("/api/pregrasp/worker/job", params={"wait": 0}).json()
        assert job["click"] == [40, 30] and job["concept"] == "aa battery"
    finally:
        pregrasp._state.worker.proc = None


def test_a_successful_teach_starts_the_track_when_the_camera_is_live(client, monkeypatch):
    """The guided flow has no "start tracking" step: a taught object is tracked from that moment."""
    import io
    import json

    from lerobot.gui.api import showservo

    started = []
    monkeypatch.setattr(pregrasp, "_begin_track", lambda: started.append(True) or True)
    pregrasp._state.worker.proc = _FakeProc()
    try:
        rgb, depth = _rect_scene(0.0)
        mask = depth < 0.449
        uv = np.argwhere(mask)[::50][:, ::-1].astype(float)

        def teach_result():
            job = pregrasp._queue_job("teach", "yellow block", rgb, depth, INTR)
            with pregrasp._state.lock:
                pregrasp._state.teach_job = job.id
            client.get("/api/pregrasp/worker/job", params={"wait": 0})
            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                meta=json.dumps(
                    {
                        "ok": True,
                        "n_points": len(uv),
                        "radius_mm": 40.0,
                        "shape_class": "general",
                        "yaw_observable": True,
                    }
                ),
                mask=mask,
                uv=uv,
                xyz=np.zeros((len(uv), 3)),
            )
            assert (
                client.post(
                    "/api/pregrasp/worker/result", params={"id": job.id}, content=buf.getvalue()
                ).status_code
                == 200
            )

        monkeypatch.setattr(showservo, "live_camera", lambda: None)
        teach_result()
        assert started == [], "no camera, nothing to track"
        monkeypatch.setattr(showservo, "live_camera", lambda: object())
        teach_result()
        assert started == [True], "a live camera and a taught object: the track starts by itself"
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.teach = None


def test_track_results_update_the_live_pose_and_an_occluded_frame_holds_it(client):
    import io
    import json

    pregrasp._state.worker.proc = _FakeProc()
    try:
        rgb, depth = _rect_scene(0.0)
        keypoints = {
            "mode": "features",
            "concept": "yellow block",
            "n_points": 40,
            "xyz": np.zeros((40, 3)) + [0.0, 0.0, 0.45],
            "uv": np.zeros((40, 2)),
            "mask": depth < 0.449,
            "radius_mm": 40.0,
            "shape_class": "general",
            "yaw_observable": True,
            "face": None,
        }
        with pregrasp._state.lock:
            pregrasp._state.teach = pregrasp._Teach(
                at="t", box=(0, 0, 0, 0), rgb=rgb, depth_m=depth, intr=INTR, keypoints=keypoints
            )
            tr = pregrasp._state.track
            tr.on, tr.algo = True, "dino"
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()
        job = pregrasp._queue_job("track", "yellow block", rgb, depth, INTR, algo="dino", compress=False)
        with pregrasp._state.lock:
            tr.job = job.id
        assert client.get("/api/pregrasp/worker/job", params={"wait": 0}).json()["algo"] == "dino"
        fr = np.load(io.BytesIO(client.get("/api/pregrasp/worker/frame.npz", params={"id": job.id}).content))
        assert fr["rgb"].shape == rgb.shape
        delta = np.eye(4)
        delta[:3, 3] = [0.02, 0.0, 0.0]
        buf = io.BytesIO()
        meta = {"ok": True, "state": "tracking", "algo": "dino", "ms": 20.0, "n_matches": 40, "n_inliers": 30}
        np.savez(buf, meta=json.dumps(meta), live_uv=np.zeros((30, 2)), delta=delta)
        assert (
            client.post(
                "/api/pregrasp/worker/result", params={"id": job.id}, content=buf.getvalue()
            ).status_code
            == 200
        )
        st = client.get("/api/pregrasp/state").json()
        assert st["track"]["last"]["state"] == "tracking" and st["track"]["last"]["axis_source"] == "fit"
        assert st["test"]["ok"] and abs(st["test"]["motion"]["translation_mm"] - 20.0) < 1e-6
        assert client.get("/api/pregrasp/track/live.jpg").status_code == 200
        # An occluded frame leaves the last pose in place; only the status changes.
        job2 = pregrasp._queue_job("track", "yellow block", rgb, depth, INTR, algo="dino", compress=False)
        with pregrasp._state.lock:
            tr.job = job2.id
        buf = io.BytesIO()
        np.savez(
            buf,
            meta=json.dumps({"ok": False, "state": "occluded", "algo": "dino", "ms": 5.0, "n_matches": 3}),
            live_uv=np.zeros((3, 2)),
        )
        assert (
            client.post(
                "/api/pregrasp/worker/result", params={"id": job2.id}, content=buf.getvalue()
            ).status_code
            == 200
        )
        st = client.get("/api/pregrasp/state").json()
        assert st["track"]["last"]["state"] == "occluded" and st["test"]["ok"]
        # Starting needs a camera; stopping always works.
        assert client.post("/api/pregrasp/track/start", json={"algo": "dino"}).status_code == 409
        assert client.post("/api/pregrasp/track/start", json={"algo": "p2p"}).status_code == 409
        assert client.post("/api/pregrasp/track/start", json={"algo": "nope"}).status_code == 422
        assert client.post("/api/pregrasp/track/stop").status_code == 200
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.teach = None
            pregrasp._state.test = None
            pregrasp._state.track = pregrasp._Track()


def test_an_act_recording_keeps_every_tracker_frame_with_the_points_its_pose_was_fitted_on(client, tmp_path):
    """The live tracker flipped the cube's orientation mid-act and its log could not say on which points."""
    import io
    import json

    pregrasp._state.worker.proc = _FakeProc()
    try:
        rgb, depth = _rect_scene(0.0)
        keypoints = {
            "mode": "features",
            "concept": "cube",
            "n_points": 40,
            "xyz": np.zeros((40, 3)) + [0.004, 0.007, 0.43],  # the block's top, where its middle is seen from
            "uv": np.zeros((40, 2)),
            "mask": depth < 0.449,
            "radius_mm": 40.0,
            "shape_class": "general",
            "yaw_observable": True,
            "face": None,
        }
        run = pregrasp._Run(root=tmp_path / "act", meta={"demo": "d", "arm_t0": 100.0})
        with pregrasp._state.lock:
            pregrasp._state.teach = pregrasp._Teach(
                at="t", box=(0, 0, 0, 0), rgb=rgb, depth_m=depth, intr=INTR, keypoints=keypoints
            )
            tr = pregrasp._state.track
            tr.on, tr.algo = True, "p2p"
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()
            pregrasp._state.run = run
            pregrasp._state.act.step = "pre-grasp 1"
        # A grid over the block's top, so the view places it (the act's rule): a few points bunched together would not.
        fit_uv = np.array([[x, y] for x in (390, 417, 443, 470) for y in (235, 245, 255, 265)], np.float32)
        inliers = [True] * 15 + [False]
        frames = [
            ({"ok": True, "state": "tracking", "n_inliers": 30, "fit_points": 16, "fit_inliers": 15}, True),
            ({"ok": False, "state": "occluded", "fit_points": 0, "fit_inliers": 0}, False),
        ]
        for meta, posed in frames:
            job = pregrasp._queue_job("track", "cube", rgb, depth, INTR, algo="p2p", compress=False)
            with pregrasp._state.lock:
                tr.job = job.id
            arrays = {"live_uv": np.zeros((3, 2))}
            if posed:
                arrays.update(
                    delta=np.eye(4),
                    fit_uv=fit_uv,
                    fit_inlier=np.array(inliers),
                    mask=depth < 0.449,
                )
            buf = io.BytesIO()
            np.savez(buf, meta=json.dumps({"algo": "p2p", "ms": 20.0, "n_matches": 3, **meta}), **arrays)
            r = client.post("/api/pregrasp/worker/result", params={"id": job.id}, content=buf.getvalue())
            assert r.status_code == 200
            assert client.get("/api/pregrasp/state").status_code == 200, (
                "the fitted points stay out of the readout"
            )
        q = dict.fromkeys(("shoulder_pan", "shoulder_lift", "elbow_flex", "forearm_roll"), 1.0)
        q.update(wrist_flex=2.0, wrist_roll=3.0, gripper=50.0)
        run.target("pre-grasp 1", pose=np.eye(4))
        pregrasp._RUN_EXECUTOR.submit(lambda: None).result()  # the frames queued before it are on disk
        root = pregrasp._finish_run(run, [{"t": 0.5, "obs": q, "cmd": q}], lambda q: np.eye(4), {"ok": True})
        f0 = np.load(tmp_path / "act" / "frames" / "000000.npz")
        assert np.array_equal(f0["fit_uv"], fit_uv) and f0["fit_inlier"].tolist() == inliers
        assert "delta_used" in f0.files, "the motion the act would follow, next to the tracker's own"
        for name in ("000000.jpg", "000000_depth.png", "000000_mask.png", "000001.jpg", "000001.npz"):
            assert (tmp_path / "act" / "frames" / name).exists(), name
        summary = json.loads((tmp_path / "act" / "act.json").read_text())
        assert [f["step"] for f in summary["frames"]] == ["pre-grasp 1", "pre-grasp 1"]
        assert [f["used"] for f in summary["frames"]] == [True, False]
        assert summary["frames"][0]["fit_inliers"] == 15 and summary["targets"][0]["step"] == "pre-grasp 1"
        arm = np.load(tmp_path / "act" / "arm.npz")
        assert arm["t"].tolist() == [100.5] and arm["q_obs"][0, -1] == 50.0
        assert root == str(tmp_path / "act")
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.teach = None
            pregrasp._state.test = None
            pregrasp._state.run = None
            pregrasp._state.track = pregrasp._Track()
            pregrasp._state.act = pregrasp._Act()


def test_under_the_resting_prior_the_face_is_reported_but_the_surface_sets_the_axis():
    """A resting object's measured face tilt is the face's own noise; the surface it rests on sets the axis."""
    rgb, depth = _rect_scene(0.0)
    teach = pregrasp._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=rgb,
        depth_m=depth,
        intr=INTR,
        keypoints={
            "mode": "features",
            "concept": "c",
            "n_points": 40,
            "xyz": np.zeros((40, 3)) + [0, 0, 0.45],
        },
    )
    n_table = [0.0, 0.0, -1.0]
    # The fit turned 30 deg about an axis tilted 25 deg; the faces say the object tipped 12 deg.
    tilt = Rotation.from_euler("x", 25, degrees=True).as_matrix()
    r_fit = tilt @ Rotation.from_euler("z", 30, degrees=True).as_matrix() @ tilt.T
    delta = np.eye(4)
    delta[:3, :3] = r_fit
    n_face = (Rotation.from_euler("x", 12, degrees=True).as_matrix() @ np.array(n_table)).tolist()
    faces = {"planarity": 0.8, "n": 500, "n_plane": 400, "dominance": 4.0}
    r = {
        "delta": delta,
        "face_teach": {**faces, "normal": n_table},
        "face_find": {**faces, "normal": n_face},
        "table_teach": n_table,
        "table_find": n_table,
    }
    result: dict = {}
    assert pregrasp._compose_motion(result, r, teach, flat=True, t_bc=None) is None
    assert result["axis_source"] == "surface"
    assert result["surface_tilt_deg"] < 1e-6
    assert abs(result["face_tilt_deg"] - 12.0) < 1e-6
    # The turn survives about the table normal, as the in-plane turn the tilted fit implies.
    assert 25.0 < abs(result["yaw_deg"]) <= 30.0
    assert np.allclose(result["delta_cam"][:3, :3] @ np.array(n_table), n_table, atol=1e-9)
    # Without the opt-in the fit stands as it is; the face tilt is still reported.
    result = {}
    pregrasp._compose_motion(result, r, teach, flat=False, t_bc=None)
    assert result["axis_source"] == "fit" and abs(result["face_tilt_deg"] - 12.0) < 1e-6
    np.testing.assert_allclose(result["delta_cam"], delta)


def test_a_surface_that_tilted_between_frames_tilts_the_object_with_it():
    """The resting prior is about the surface, not about a level table: a ramp leans the object."""
    rgb, depth = _rect_scene(0.0)
    teach = pregrasp._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=rgb,
        depth_m=depth,
        intr=INTR,
        keypoints={
            "mode": "features",
            "concept": "c",
            "n_points": 40,
            "xyz": np.zeros((40, 3)) + [0, 0, 0.45],
        },
    )
    n_a = np.array([0.0, 0.0, -1.0])
    ramp = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    n_b = ramp @ n_a
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler("z", 15, degrees=True).as_matrix()  # the fit saw only the turn
    r = {"delta": delta, "table_teach": n_a.tolist(), "table_find": n_b.tolist()}
    result: dict = {}
    pregrasp._compose_motion(result, r, teach, flat=True, t_bc=None)
    assert result["axis_source"] == "surface" and abs(result["surface_tilt_deg"] - 20.0) < 1e-6
    assert np.allclose(result["delta_cam"][:3, :3] @ n_a, n_b, atol=1e-9)  # the bottom follows the surface


def test_the_pose_is_the_raw_fit_unless_the_resting_prior_is_opted_in():
    """Replayed against ground truth, every composition on top of the fit scored below it; the fit stands."""
    rgb, depth = _rect_scene(0.0)
    teach = pregrasp._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=rgb,
        depth_m=depth,
        intr=INTR,
        keypoints={
            "mode": "features",
            "concept": "c",
            "n_points": 40,
            "xyz": np.zeros((40, 3)) + [0, 0, 0.45],
        },
    )
    n = [0.0, 0.0, -1.0]
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(
        np.array([0.3, 0.2, 0.0]) * np.radians(40.0)
    ).as_matrix()  # a tilted turn
    delta[:3, 3] = [0.02, -0.01, 0.0]
    r = {
        "delta": delta,
        "table_teach": n,
        "table_find": n,
        "yaw_observable": False,
        "footprint_yaw_deg": 10.0,
        "shape_class": "rod",
    }
    out: dict = {}
    pregrasp._compose_motion(out, r, teach, flat=False, t_bc=None)
    assert out["axis_source"] == "fit" and "turn_source" not in out
    np.testing.assert_allclose(out["delta_cam"], delta)  # untouched: no rule, no prior
    opted: dict = {}
    pregrasp._compose_motion(opted, r, teach, flat=True, t_bc=None)
    assert opted["axis_source"] == "surface"
    assert np.allclose(opted["delta_cam"][:3, :3] @ np.array(n), n, atol=1e-9), (
        "the prior keeps the object on its surface"
    )
    assert "turn_source" not in opted, "no turn rule even under the prior"
    assert pregrasp._State().flat is False, "off unless opted in"


def test_transport_trajectory_carries_every_pose_by_the_same_base_motion():
    t_bc = np.eye(4)
    t_bc[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    t_bc[:3, 3] = [0.3, 0.0, 0.5]
    delta_cam = np.eye(4)
    delta_cam[:3, 3] = [0.02, 0.0, 0.0]
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, 0, 3] = np.linspace(0.2, 0.3, 5)
    out = core.transport_trajectory(t_bc, delta_cam, poses)
    assert out.shape == (5, 4, 4)
    for k in range(5):
        assert np.allclose(out[k], core.transport_pose(t_bc, delta_cam, poses[k]))


def _samples_and_history(n=60, hz=30.0, t0=1000.0):
    """A straight 60 mm fingertip move with the gripper closing at the end, and a tracker history that saw the object."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    samples = []
    for i in range(n):
        q = dict.fromkeys(MOTOR_NAMES, 0.0)
        q["shoulder_pan"] = i * 0.5
        q["gripper"] = 10.0 if i < n - 10 else 60.0
        samples.append({"t": i / hz, "obs": dict(q), "cmd": dict(q)})
    delta = np.eye(4)
    delta[:3, 3] = [0.05, 0.0, 0.0]
    history = [
        (t0 + i / 15.0, i % 7 != 3, None if i % 7 == 3 else delta) for i in range(int(n / hz * 15) + 1)
    ]
    return samples, history


def test_a_demo_is_built_from_the_joint_samples_and_the_tracker_history():
    samples, history = _samples_and_history()

    def fk(q):
        pose = np.eye(4)
        pose[0, 3] = 0.2 + q["shoulder_pan"] * 0.002  # 1 mm per sample, 59 mm over the demo
        return pose

    demo = pregrasp._demo_from_samples("d", "green cube", samples, history, fk, t0=1000.0)
    assert demo.n_points if False else len(demo.t) == 60 and demo.tips.shape == (60, 4, 4)
    assert abs(demo.fps - 30.0) < 0.5 and demo.grippers[0] == 10.0 and demo.grippers[-1] == 60.0
    assert demo.seen.mean() > 0.8 and np.allclose(demo.delta0[:3, 3], [0.05, 0.0, 0.0])
    assert abs(demo.tips[-1, 0, 3] - demo.tips[0, 0, 3] - 0.059) < 1e-9
    # With no tracker history the object is taken to sit where it was taught.
    cold = pregrasp._demo_from_samples("d", "green cube", samples, [], fk, t0=1000.0)
    assert not cold.seen.any() and np.allclose(cold.delta0, np.eye(4))


def test_demo_save_load_and_act_guards(client, tmp_path, monkeypatch):
    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    assert client.get("/api/pregrasp/demos").json() == {"demos": []}
    assert client.post("/api/pregrasp/act", json={}).status_code == 409
    assert client.post("/api/pregrasp/demo/save", json={}).status_code == 409
    assert client.post("/api/pregrasp/demo/record/start").status_code == 409
    assert client.post("/api/pregrasp/demo/load", json={"name": "nope"}).status_code == 404
    rgb, depth = _rect_scene(0.0)
    samples, history = _samples_and_history(n=12)

    def fk(q):
        pose = np.eye(4)
        pose[0, 3] = 0.2 + q["shoulder_pan"] * 0.002
        return pose

    demo = pregrasp._demo_from_samples("cube_push", "green cube", samples, history, fk, t0=1000.0)
    demo.taught = True
    teach = pregrasp._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=rgb,
        depth_m=depth,
        intr=INTR,
        keypoints={
            "mode": "features",
            "concept": "green cube",
            "n_points": 40,
            "xyz": np.zeros((40, 3)),
            "mask": depth < 0.449,
            "radius_mm": 14.0,
            "shape_class": "disc",
            "yaw_observable": False,
            "face": None,
        },
    )
    pregrasp._state.worker.proc = _FakeProc()
    try:
        with pregrasp._state.lock:
            pregrasp._state.teach, pregrasp._state.demo = teach, demo
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()
        r = client.post("/api/pregrasp/demo/save", json={"name": "cube push"})
        assert r.status_code == 200 and r.json()["name"] == "cube_push" and r.json()["n"] == 12
        root = tmp_path / "demos" / "cube_push"
        assert (root / pregrasp.DEMO_FILE).exists() and (root / "meta" / "info.json").exists()
        listed = client.get("/api/pregrasp/demos").json()["demos"]
        assert [d["name"] for d in listed] == ["cube_push"] and listed[0]["concept"] == "green cube"
        # Loading queues a teach from the saved frame and makes the demo current.
        with pregrasp._state.lock:
            pregrasp._state.demo = None
        refinds = []
        real_refind = pregrasp._start_refind
        pregrasp._start_refind = (
            refinds.append
        )  # after its own teach, the load finds the demo's objects again
        try:
            r = client.post("/api/pregrasp/demo/load", json={"name": "cube_push"})
        finally:
            pregrasp._start_refind = real_refind
        assert r.status_code == 200 and r.json()["teach_pending"]
        assert [d.name for d in refinds] == ["cube_push"], (
            "where the objects were last seen, after that teach"
        )
        job = client.get("/api/pregrasp/worker/job", params={"wait": 0}).json()
        assert job["kind"] == "teach" and job["concept"] == "green cube"
        st = client.get("/api/pregrasp/state").json()
        assert st["demo"]["name"] == "cube_push" and st["act"]["on"] is False
        # The act needs a found object.
        assert client.post("/api/pregrasp/act", json={}).status_code == 409
        assert client.post("/api/pregrasp/act/stop").status_code == 200
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.teach = None
            pregrasp._state.test = None
            pregrasp._state.demo = None
            pregrasp._state.teach_job = None
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()


def test_an_act_leaves_a_trial_row_the_operator_can_judge(client, tmp_path, monkeypatch):
    monkeypatch.setattr(pregrasp, "TRIALS_PATH", tmp_path / "trials.jsonl")
    monkeypatch.setattr(pregrasp, "_trials", None)
    rgb, depth = _rect_scene(0.0)
    delta = np.eye(4)
    delta[:3, 3] = [0.05, 0.0, 0.0]
    samples, history = _samples_and_history(n=6)
    demo = pregrasp._demo_from_samples("d1", "green cube", samples, history, lambda q: np.eye(4), t0=1000.0)
    with pregrasp._state.lock:
        pregrasp._state.teach = pregrasp._Teach(
            at="t",
            box=(0, 0, 0, 0),
            rgb=rgb,
            depth_m=depth,
            intr=INTR,
            keypoints={"mode": "features", "concept": "green cube", "n_points": 40, "xyz": np.zeros((40, 3))},
        )
        pregrasp._state.test = pregrasp._Test(
            at="now",
            rgb=rgb,
            result={
                "ok": True,
                "delta_cam": delta,
                "n_inliers": 30,
                "n_matches": 40,
                "axis_source": "surface",
                "yaw_deg": 12.0,
            },
        )
        pregrasp._state.demo = demo
        pregrasp._state.act.ok, pregrasp._state.act.step, pregrasp._state.act.progress = True, "done", 1.0
    try:
        row = pregrasp._record_trial()
        assert row["result"] == "done" and row["demo"] == "d1" and abs(row["centre_shift_mm"] - 50.0) < 1e-6
        rows = client.get("/api/pregrasp/trials").json()["rows"]
        assert len(rows) == 1 and rows[0]["object"] == "green cube"
        assert (
            client.post("/api/pregrasp/trials/verdict", json={"index": 0, "verdict": "missed"}).status_code
            == 200
        )
        assert (
            client.post("/api/pregrasp/trials/verdict", json={"index": 0, "verdict": "meh"}).status_code
            == 422
        )
        pregrasp._trials = None
        assert client.get("/api/pregrasp/trials").json()["rows"][0]["verdict"] == "missed"
        assert (
            client.post("/api/pregrasp/trials/verdict", json={"index": 0, "verdict": None}).status_code == 200
        )
        assert client.get("/api/pregrasp/trials").json()["rows"][0]["verdict"] is None, (
            "a mistaken verdict is cleared"
        )
    finally:
        with pregrasp._state.lock:
            pregrasp._state.teach = None
            pregrasp._state.test = None
            pregrasp._state.demo = None
            pregrasp._state.act = pregrasp._Act()


def test_footprint_yaw_prefers_the_previous_answer_among_a_squares_equal_peaks():
    # A square footprint turned 45 degrees: +45 and -45 overlay it equally well.
    s = 0.015
    square = np.array([[x, y] for x in np.linspace(-s, s, 16) for y in np.linspace(-s, s, 16)])
    t = np.radians(45.0)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    turned = square @ rot.T
    plus = core.footprint_yaw(square, turned, prefer_deg=40.0)
    minus = core.footprint_yaw(square, turned, prefer_deg=-40.0)
    assert 40.0 <= plus["yaw_deg"] <= 50.0 and -50.0 <= minus["yaw_deg"] <= -40.0
    assert abs(plus["iou"] - minus["iou"]) < 0.05  # the two answers are the same overlay
    # Without a preference the smaller turn wins, and a clear turn is not swayed by a preference.
    free = core.footprint_yaw(
        square,
        square
        @ np.array(
            [
                [np.cos(np.radians(20)), -np.sin(np.radians(20))],
                [np.sin(np.radians(20)), np.cos(np.radians(20))],
            ]
        ).T,
    )
    assert 15.0 <= free["yaw_deg"] <= 25.0
    swayed = core.footprint_yaw(square, turned, prefer_deg=120.0)
    assert (
        130.0 <= swayed["yaw_deg"] <= 140.0
        or -50.0 <= swayed["yaw_deg"] <= -40.0
        or 40.0 <= swayed["yaw_deg"] <= 50.0
    )


# ── the act: straight lines to the pre-grasp points, then the grasp exactly as recorded ───────


def test_mark_rules():
    p = core.keypoints_problem
    pre1, pre2, end3 = (
        {"t": 1.0, "kind": "pregrasp"},
        {"t": 2.0, "kind": "pregrasp"},
        {"t": 3.0, "kind": "grasp_end"},
    )
    assert p([], 0.0, 10.0) == "", "an empty list clears the marks"
    assert p([pre1], 0.0, 10.0) == "" and p([pre1, pre2, end3], 0.0, 10.0) == ""
    assert "pre-grasp" in p([end3], 0.0, 10.0)
    assert "after the last pre-grasp" in p([{"t": 3.5, "kind": "pregrasp"}, end3], 0.0, 10.0)
    assert "one end" in p([pre1, end3, {"t": 4.0, "kind": "grasp_end"}], 0.0, 10.0)
    assert "within the demo" in p([{"t": 11.0, "kind": "pregrasp"}], 0.0, 10.0)
    assert "one of" in p([{"t": 1.0, "kind": "release"}], 0.0, 10.0)


def test_place_mark_rules():
    p = core.keypoints_problem
    grasp = [
        {"t": 1.0, "kind": "pregrasp", "object": "cube"},
        {"t": 3.0, "kind": "grasp_end", "object": "cube"},
    ]
    pre_place, place_end = (
        {"t": 5.0, "kind": "preplace", "object": "box"},
        {"t": 7.0, "kind": "place_end", "object": "box"},
    )
    assert p([*grasp, pre_place, place_end], 0.0, 10.0) == ""
    assert p([*grasp, pre_place], 0.0, 10.0) == "", "no place end: the arm stops at the pre-place, holding"
    assert "grasp end first" in p([grasp[0], pre_place], 0.0, 10.0)
    assert "after the grasp end" in p([*grasp, dict(pre_place, t=2.0)], 0.0, 10.0)
    assert "at least one pre-place" in p([*grasp, place_end], 0.0, 10.0)
    assert "after the last pre-place" in p([*grasp, pre_place, dict(place_end, t=4.0)], 0.0, 10.0)
    assert "one end" in p([*grasp, pre_place, place_end, dict(place_end, t=8.0)], 0.0, 10.0)
    assert "follow one object" in p([grasp[0], dict(grasp[1], object="mug")], 0.0, 10.0)
    assert "follow one object" in p([*grasp, pre_place, dict(place_end, object="mug")], 0.0, 10.0)
    assert "another object" in p([*grasp, dict(pre_place, object="cube")], 0.0, 10.0)
    pose = {"t": 0.5, "kind": "pose", "object": "box"}
    assert p([*grasp, pose, pre_place, place_end], 0.0, 10.0) == "", (
        "the box's pose read before anything moved"
    )
    assert p([pose], 0.0, 10.0) == "", "a pose alone, set before the marks"
    assert "names its object" in p([*grasp, {"t": 0.5, "kind": "pose"}], 0.0, 10.0)
    assert "at one frame" in p([*grasp, pose, dict(pose, t=0.8)], 0.0, 10.0)
    assert "last pre-place" in p([*grasp, dict(pose, t=6.0), pre_place, place_end], 0.0, 10.0)
    assert "last pre-grasp" in p([*grasp, dict(pose, object="cube", t=2.0)], 0.0, 10.0)
    unnamed = [{"t": 1.0, "kind": "pregrasp"}, {"t": 3.0, "kind": "grasp_end"}]
    assert "both objects" in p([*unnamed, pre_place], 0.0, 10.0), (
        "the taught-before-the-demo object cannot be found in the gripper"
    )


def test_the_place_carries_the_held_object_onto_the_moved_target_however_it_sits_in_the_gripper():
    """The place is the demo's fingertip carried by the target's motion and corrected on the gripper's side by the
    change in the hold: the held object then ends where the demo put it relative to the target, even gripped
    elsewhere and at another angle than in the demo."""
    t, tips, grip, q = _demo_arrays()
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("z", 25, degrees=True).as_matrix()
    target[:3, 3] = [0.03, -0.01, 0.0]
    hold_demo, hold_now = np.eye(4), np.eye(4)
    hold_demo[:3, 3] = [0.0, 0.0, -0.02]  # the object 20 mm below the fingertip in the demo
    hold_now[:3, :3] = Rotation.from_euler(
        "x", 8, degrees=True
    ).as_matrix()  # gripped tilted and 6 mm off now
    hold_now[:3, 3] = [0.006, 0.0, -0.02]
    fix = hold_demo @ np.linalg.inv(hold_now)
    start = np.eye(4)
    start[:3, 3] = [0.15, 0.05, 0.12]
    kps = [
        {"t": float(t[20]), "kind": "pregrasp", "object": "cube"},
        {"t": float(t[40]), "kind": "grasp_end", "object": "cube"},
        {"t": float(t[60]), "kind": "preplace", "object": "box"},
        {"t": float(t[80]), "kind": "preplace", "object": "box"},
        {"t": float(t[110]), "kind": "place_end", "object": "box"},
    ]
    speed, lin = 0.5, 0.04
    plan = core.plan_place(kps, t, tips, grip, q, target, fix, start, 85.0, lin, np.radians(30), speed, 30.0)
    poses, grips, st = plan["poses"], plan["grips"], np.array(plan["stage"])
    a1, a2 = plan["arrive"]
    assert np.allclose(poses[0], start) and np.allclose(poses[a2], target @ tips[80] @ fix)
    assert _on_segment(poses[st == "pre-place 2"][:, :3, 3], poses[a1][:3, 3], poses[a2][:3, 3]), (
        "a straight line"
    )
    assert np.all(grips[st != "place"] == 85.0), "the grasp's closing held along the carry, never the demo's"
    p0, p1 = plan["place"]
    assert np.allclose(poses[p0 : p1 + 1], np.einsum("ij,njk,kl->nil", target, tips[81:111], fix))
    assert np.array_equal(grips[p0 : p1 + 1], grip[81:111]), "the recorded command, release included"
    # The point of the correction: the held object, gripped as it is now, ends where the demo left it on the target.
    assert np.allclose(poses[p1] @ hold_now, target @ tips[110] @ hold_demo)
    same = core.plan_place(
        kps, t, tips, grip, q, target, np.eye(4), start, 85.0, lin, np.radians(30), speed, 30.0
    )
    assert np.allclose(same["poses"][same["place"][0] :], np.einsum("ij,njk->nik", target, tips[81:111])), (
        "held as in the demo, the place is the demo carried by the target"
    )
    resumed = core.plan_place(
        kps, t, tips, grip, q, target, fix, poses[a1], 85.0, lin, np.radians(30), speed, 30.0, 1
    )
    assert resumed["arrive"] == [int(np.flatnonzero(np.array(resumed["stage"]) == "pre-place 2")[-1])]
    grasp = core.plan_pregrasp_grasp(
        kps, t, tips, grip, q, np.eye(4), start, 60.0, lin, np.radians(30), 80.0, speed, 30.0
    )
    then = core.plan_place(
        kps,
        t,
        tips,
        grip,
        q,
        target,
        fix,
        grasp["poses"][-1],
        grasp["grips"][-1],
        lin,
        np.radians(30),
        speed,
        30.0,
    )
    whole = core.join_plans(grasp, then)
    n = len(grasp["times"])
    assert len(whole["times"]) == n + len(then["times"]) - 1 and np.all(np.diff(whole["times"]) > 0)
    assert whole["grasp"] == grasp["grasp"] and whole["place"] == (
        then["place"][0] + n - 1,
        then["place"][1] + n - 1,
    )
    assert whole["arrive"] == grasp["arrive"] + [a + n - 1 for a in then["arrive"]]
    assert np.allclose(whole["poses"][whole["place"][1]], then["poses"][-1])


def test_the_grasp_check_tells_a_held_object_from_a_closing_on_nothing():
    held = core.grasp_held
    # Measured on the rig's acts: held ones stopped 0.89 to 10.56 short of the closing command, misses -0.06 to 0.48.
    assert held(100.0, 98.1, 100.0, 95.5, 1.0) == (True, pytest.approx(1.9))
    assert held(83.26, 82.37, 83.26, 82.03, 1.0)[0] is True, (
        "the pick-and-place demo squeezed the gamepad 1.23 short at its firm grip, and an act held it 0.89 short"
    )
    assert held(100.0, 99.52, 100.0, 95.4, 1.0)[0] is False, (
        "the widest-stopping closing on nothing, the cube left lying"
    )
    assert held(97.2, 90.3, 97.2, 90.3, 1.0)[0] is True
    assert held(100.0, 99.6, 100.0, 95.5, 1.0)[0] is False
    assert held(97.2, 96.9, 97.2, 90.3, 1.0)[0] is False
    assert held(100.0, 99.6, 100.0, 99.5, 1.0) == (None, pytest.approx(0.4)), (
        "a demo that barely squeezed cannot tell the two apart"
    )
    assert held(10.0, 13.0, 10.0, 14.0, -1.0)[0] is True, "a gripper that closes toward smaller readings"


def test_the_grip_is_firm_where_the_gripper_stops_short_of_its_command():
    """The stacking demo's closing, every third sample: the command rises from 62 to 95.1 over 7.63-8.23 s, the
    reading follows and stops at 90.3 at 8.33 s, 4.8 short, while the grasp's end mark comes at 9.16 s after a lift."""
    t = np.arange(0, 2.4, 1 / 30.0) + 7.0
    cmd = np.interp(t, [7.0, 7.6, 8.23, 9.5], [62.0, 62.1, 95.0, 95.1])
    obs = np.interp(t, [7.0, 7.63, 8.33, 9.4], [61.9, 62.1, 90.3, 90.3])
    i = core.firm_grip(t, cmd, obs, 0, len(t) - 1, 1.0)
    assert i is not None and t[i] == pytest.approx(8.33, abs=0.04)
    empty = np.minimum(cmd, obs + 100.0)  # nothing between the fingers: the reading reaches the command
    assert core.firm_grip(t, cmd, empty, 0, len(t) - 1, 1.0) is None
    idle = np.full_like(t, 62.0)  # still, but the command never closed: no grip
    assert core.firm_grip(t, idle, idle - 3.0, 0, len(t) - 1, 1.0) is None
    assert core.firm_grip(t, 100 - cmd, 100 - obs, 0, len(t) - 1, -1.0) == i, (
        "a gripper closing toward smaller readings"
    )


def test_a_hold_is_the_mean_of_the_views_that_agree_and_none_without_enough():
    centre = np.array([0.20, 0.0, 0.05])
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("y", 10, degrees=True).as_matrix()
    base[:3, 3] = [0.01, 0.0, -0.02]

    def view(turn_deg=0.0, shift=(0.0, 0.0, 0.0)):
        """The hold as one find sees it: turned about the object's own centre, as a find's noise turns it, and shifted."""
        c = (base @ np.append(centre, 1.0))[:3]
        turn = np.eye(4)
        turn[:3, :3] = Rotation.from_euler("z", turn_deg, degrees=True).as_matrix()
        turn[:3, 3] = c - turn[:3, :3] @ c
        h = turn @ base
        h[:3, 3] += shift
        return h

    views = [view(0.5), view(-0.5), view(0.0, (0.0004, 0.0, 0.0)), view(0.0, (-0.0004, 0.0, 0.0))]
    avg = core.average_hold([*views, view(20.0)], centre)
    assert avg["n"] == 4 and avg["views"] == 5, "the view 20 deg off is a bad find, left out"
    assert np.allclose(avg["hold"], base, atol=1e-4)
    assert avg["spread_deg"] < 1.0 and avg["spread_mm"] < 1.0
    assert core.average_hold([*views[:2], view(0.0, (0.03, 0.0, 0.0))], centre) is None, (
        "two that agree are not enough"
    )
    assert core.average_hold([], centre) is None


def _demo_arrays(n=120, hz=30.0):
    """A 30 Hz demo: the fingertip slides 30 mm, descends 40 mm and turns 20 deg; the gripper closes at 3 s."""
    t = np.arange(n) / hz
    tips = np.tile(np.eye(4), (n, 1, 1))
    tips[:, 0, 3] = 0.20 + 0.03 * t / t[-1]
    tips[:, 2, 3] = 0.08 - 0.04 * np.clip(t / 2.0, 0.0, 1.0)
    tips[:, :3, :3] = Rotation.from_euler("z", (20 * t / t[-1])[:, None], degrees=True).as_matrix()
    grip = np.where(t < 3.0, 60.0, 85.0)
    q = np.zeros((n, 7))
    q[:, 0] = 10.0 * t
    q[:, 6] = grip
    return t, tips, grip, q


def _on_segment(points, a, b):
    d = b - a
    s = (points - a) @ d / (d @ d)
    off = points - (a + np.outer(s, d))
    return bool(
        np.all(np.abs(off) < 1e-9)
        and np.all(np.diff(s) >= -1e-12)
        and s.min() >= -1e-12
        and s.max() <= 1 + 1e-12
    )


def test_the_plan_goes_straight_to_each_pregrasp_then_replays_the_grasp_as_recorded():
    t, tips, grip, q = _demo_arrays()
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    delta[:3, 3] = [0.01, -0.02, 0.0]
    start = np.eye(4)
    start[:3, 3] = [0.10, -0.10, 0.15]
    kps = [
        {"t": float(t[30]), "kind": "pregrasp"},
        {"t": float(t[60]), "kind": "pregrasp"},
        {"t": float(t[100]), "kind": "grasp_end"},
    ]
    speed, lin = 0.5, 0.04
    plan = core.plan_pregrasp_grasp(
        kps, t, tips, grip, q, delta, start, 95.0, lin, np.radians(30), 80.0, speed, 30.0
    )
    poses, grips, times, st = plan["poses"], plan["grips"], plan["times"], np.array(plan["stage"])
    a1, a2 = plan["arrive"]
    assert np.allclose(poses[0], start), "the plan starts where the arm is"
    assert np.allclose(poses[a1], delta @ tips[30]) and np.allclose(poses[a2], delta @ tips[60]), (
        "pre-grasps move with the object"
    )
    line1 = poses[st == "pre-grasp 1"][:, :3, 3]
    assert _on_segment(line1, start[:3, 3], (delta @ tips[30])[:3, 3]), (
        "a straight line from where the arm is"
    )
    assert _on_segment(
        poses[st == "pre-grasp 2"][:, :3, 3], (delta @ tips[30])[:3, 3], (delta @ tips[60])[:3, 3]
    )
    assert np.linalg.norm(np.diff(line1, axis=0), axis=1).max() <= lin * speed / 30.0 + 1e-12, (
        "at the walk's speed, scaled"
    )
    # The gripper walks to the pre-grasp's opening where the arm stands, then holds it along the line.
    g1 = grips[st == "pre-grasp 1"]
    moving = np.any(np.abs(line1 - start[:3, 3]) > 0, axis=1)
    assert (
        np.all(~moving[: int((~moving).sum())])
        and g1[~moving][-1] == grip[30]
        and np.all(g1[moving] == grip[30])
    )
    # The grasp is the demo sample for sample, carried by the object's motion, on the demo's clock.
    g = st == "grasp"
    assert np.allclose(poses[g], np.einsum("ij,njk->nik", delta, tips[61:101]))
    assert np.array_equal(grips[g], grip[61:101]), "the recorded gripper command, closing included"
    first = int(np.flatnonzero(g)[0])
    assert np.allclose(np.diff(times[first - 1 :]), np.diff(t[60:101]) / speed)
    assert np.allclose(plan["hints"][g], np.diff(q[60:101], axis=0)) and np.all(plan["hints"][~g] == 0.0)
    assert np.allclose(plan["floor_ref"][g], tips[61:101, 2, 3]) and np.all(np.isinf(plan["floor_ref"][~g]))
    short = core.plan_pregrasp_grasp(
        kps[:2], t, tips, grip, q, delta, start, 95.0, lin, np.radians(30), 80.0, speed, 30.0
    )
    assert short["grasp"] is None and np.allclose(short["poses"][-1], delta @ tips[60]), (
        "no grasp end: stop at the pre-grasp"
    )


class _StepKinematics:
    """A fake arm: the tip is the first three joints in millimetres; the IK closes half the gap per call
    (as a velocity-bounded QP does) and cannot reach past half a metre."""

    def __init__(self):
        self.calls = 0

    def forward_kinematics(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q[:3], dtype=float) / 1000.0
        return pose

    def inverse_kinematics(self, seed, pose):
        self.calls += 1
        q = np.asarray(seed, dtype=float).copy()
        q[:3] = np.clip(q[:3] + 0.5 * (np.asarray(pose[:3, 3]) * 1000.0 - q[:3]), -500.0, 500.0)
        return q


class _YawKinematics(_StepKinematics):
    """The fake arm with a wrist: joint 4 turns the tip about the vertical (degrees), closing half its gap per call."""

    def forward_kinematics(self, q):
        pose = super().forward_kinematics(q)
        pose[:3, :3] = Rotation.from_euler("z", float(q[4]), degrees=True).as_matrix()
        return pose

    def inverse_kinematics(self, seed, pose):
        q = super().inverse_kinematics(seed, pose)
        want = np.degrees(np.arctan2(pose[1, 0], pose[0, 0]))
        q[4] += 0.5 * ((want - q[4] + 180.0) % 360.0 - 180.0)
        return q


def test_a_landing_rule_names_the_turns_that_count_as_the_same_place():
    assert core.landing_turns("exact", 4) == [0.0]
    assert core.landing_turns("symmetry", 4) == [0.0, 90.0, 180.0, 270.0]
    assert core.landing_turns("symmetry", 1) == [0.0], "an object without symmetry lands only as shown"
    turns = core.landing_turns("turn", 4)
    assert len(turns) == round(360.0 / core.LANDING_STEP_DEG) and turns[0] == 0.0
    assert np.allclose(np.diff(turns), core.LANDING_STEP_DEG)
    with pytest.raises(AssertionError):
        core.landing_turns("anyhow", 1)


def test_a_landing_turns_about_the_object_where_its_motion_put_it():
    centre = np.array([0.10, -0.20, 0.03])
    turn = core.turn_about(centre, 90.0)
    assert np.allclose(turn @ np.r_[centre, 1.0], np.r_[centre, 1.0]), "the middle stays"
    assert np.allclose((turn @ np.r_[centre + [0.05, 0.0, 0.0], 1.0])[:3], centre + [0.0, 0.05, 0.0])
    motion = np.eye(4)
    motion[:3, :3] = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    motion[:3, 3] = [0.02, 0.01, 0.0]
    landed = core.landed(motion, centre, 90.0)
    assert np.allclose(landed[:3, :3] @ centre + landed[:3, 3], motion[:3, :3] @ centre + motion[:3, 3]), (
        "the turn is about where the object is now, so its middle lands where the object's motion put it"
    )
    assert Rotation.from_matrix(landed[:3, :3]).as_euler("xyz", degrees=True)[2] == pytest.approx(120.0)
    assert np.allclose(core.landed(motion, centre, 0.0), motion)


def test_a_solve_holds_a_joint_at_its_servo_range_and_says_what_that_costs():
    """An act of 2026-10-08 was refused for a place needing wrist_flex at -94 deg against its servo's -93. The solve
    never asks a joint past its range: it holds it there, and the residual says whether the pose is still reached."""
    kin = _YawKinematics()
    hi = np.array([np.nan, np.nan, np.nan, np.nan, 93.3, np.nan, np.nan])
    lo = -hi

    def pose(wrist_deg):
        return kin.forward_kinematics(np.array([150.0, 30.0, 40.0, 0.0, wrist_deg, 0.0, 85.0]))

    poses = np.stack([pose(80.0), pose(94.5), pose(120.0)])
    out = core.solve_plan_joints(kin, poses, np.full(3, 85.0), np.zeros((3, 7)), np.zeros(7), 6, lo, hi)
    assert out["q"][0, 4] == pytest.approx(80.0, abs=0.5) and out["held"][0] == -1, (
        "within range: solved as asked"
    )
    assert out["q"][1, 4] == pytest.approx(93.3) and out["held"][1] == 4, "held at the range"
    assert out["residual_deg"][1] == pytest.approx(1.2, abs=0.1), "what holding it costs"
    assert out["residual_deg"][1] <= core.ACT_REACH_TOL_DEG, "and the pose is still reached"
    assert out["q"][2, 4] == pytest.approx(93.3) and out["residual_deg"][2] > core.ACT_REACH_TOL_DEG, (
        "held, a pose 27 deg past the range is out of reach"
    )
    free = core.solve_plan_joints(kin, poses, np.full(3, 85.0), np.zeros((3, 7)), np.zeros(7), 6)
    assert free["q"][2, 4] == pytest.approx(120.0, abs=0.5) and (free["held"] == -1).all(), (
        "no ranges, no limit"
    )


def test_the_landing_taken_is_within_the_servos_with_joints_nearest_the_demos():
    """The demo set an object down with its wrist at 80 deg, near the servo's 93. The object it goes onto has turned 40
    deg about its middle since: landed as shown the wrist needs 120. Of the turns about that middle, the one that undoes
    the object's turn puts the arm back on the demo's joints."""
    kin = _YawKinematics()
    centre = np.array([0.15, 0.0, 0.0])
    demo_q = np.array([[150.0, 30.0, 40.0, 0.0, 80.0, 0.0, 85.0], [150.0, 30.0, 20.0, 0.0, 80.0, 0.0, 60.0]])
    tips = np.stack([kin.forward_kinematics(q) for q in demo_q])
    moved = core.turn_about(centre, 40.0)
    hi = np.array([np.nan, np.nan, np.nan, np.nan, 93.3, np.nan, np.nan])
    lo = -hi

    def poses_at(deg):
        return np.stack([core.landed(moved, centre, deg) @ tip for tip in tips])

    assert core.rank_landings(kin, [0.0], poses_at, demo_q, lo, hi, 6) == [], (
        "as shown the wrist goes past its servo"
    )
    assert len(core.rank_landings(kin, [0.0], poses_at, demo_q, None, None, 6)) == 1, (
        "the model alone reaches it"
    )
    ranked = core.rank_landings(kin, core.landing_turns("turn", 1), poses_at, demo_q, lo, hi, 6)
    cost, best = ranked[0]
    assert best == pytest.approx(320.0) and cost == pytest.approx(0.0, abs=0.1), (
        "the turn undoing the object's"
    )
    assert all(
        abs((120.0 + t + 180.0) % 360.0 - 180.0) <= 93.3 + core.ACT_REACH_TOL_DEG for _c, t in ranked
    ), "every turn kept is reached with the wrist held within its range"
    assert any(abs((120.0 + t + 180.0) % 360.0 - 180.0) > 93.3 for _c, t in ranked), (
        "a turn needing the wrist just past its range is kept, held there"
    )
    quarter = core.rank_landings(kin, core.landing_turns("symmetry", 4), poses_at, demo_q, lo, hi, 6)
    assert [t for _c, t in quarter] == [270.0, 180.0], (
        "the quarter turns the wrist reaches, the nearer to the demo first"
    )


def test_the_solve_continues_from_the_arm_and_reports_what_it_cannot_reach():
    kin = _StepKinematics()
    n = 40
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = 0.030 + np.arange(n) * 0.002
    grips = np.full(n, 70.0)
    hints = np.zeros((n, 7))
    hints[1:, 0] = 2.0  # the demo moved this joint as far per sample
    out = core.solve_plan_joints(kin, poses, grips, hints, np.zeros(7), grip_index=6)
    assert np.all(out["residual_m"] <= core.ACT_REACH_TOL_M) and np.all(out["q"][:, 6] == 70.0)
    # The first solve closes 30 mm from the arm's configuration by halves, to within the solve tolerance; every later
    # seed starts where the demo's joint change says and needs one call.
    assert kin.calls == 6 + (n - 1)
    assert np.all(np.abs(out["q"][:, 0] - poses[:, 0, 3] * 1000.0) <= core.ACT_SOLVE_TOL_M * 1000.0), (
        "solved past the reach tolerance"
    )
    assert out["step_deg"][0] == 0.0 and np.all(out["step_deg"][1:] <= 2.5)
    poses[-1, 0, 3] = 1.0
    out = core.solve_plan_joints(kin, poses, grips, hints, np.zeros(7), grip_index=6)
    assert out["residual_m"][-1] > core.ACT_REACH_TOL_M and np.all(
        out["residual_m"][:-1] <= core.ACT_REACH_TOL_M
    )


@pytest.mark.skipif(
    not __import__("lerobot.utils.import_utils", fromlist=["_pin_pink_available"])._pin_pink_available,
    reason="pin-pink (optional) not installed",
)
def test_on_the_so107_a_turned_object_keeps_the_grasp_reachable_without_a_jump():
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES

    kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    ready = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -45.0,
        "elbow_flex": 74.0,
        "forearm_roll": 0.0,
        "wrist_flex": -41.0,
        "wrist_roll": 0.0,
        "gripper": 95.0,
    }
    q0 = np.array([ready[m] for m in MOTOR_NAMES])
    n = 60
    t = np.arange(n) / 30.0
    q_demo = np.tile(q0, (n, 1))
    q_demo[:, MOTOR_NAMES.index("wrist_flex")] += np.linspace(0.0, -15.0, n)  # the demo lowers the wrist
    q_demo[:, MOTOR_NAMES.index("gripper")] = np.where(t < 1.5, 60.0, 85.0)
    tips = np.stack([kin.forward_kinematics(qd) for qd in q_demo])
    delta = np.eye(4)  # the object turned 15 deg about the vertical through the grasp and slid 20 mm
    delta[:3, :3] = Rotation.from_euler("z", 15, degrees=True).as_matrix()
    delta[:3, 3] = tips[30, :3, 3] - delta[:3, :3] @ tips[30, :3, 3] + np.array([0.02, 0.0, 0.0])
    gi = MOTOR_NAMES.index("gripper")
    kps = [{"t": float(t[20]), "kind": "pregrasp"}, {"t": float(t[59]), "kind": "grasp_end"}]
    plan = core.plan_pregrasp_grasp(
        kps,
        t,
        tips,
        q_demo[:, gi],
        q_demo,
        delta,
        kin.forward_kinematics(q0),
        95.0,
        0.04,
        np.radians(30),
        80.0,
        1.0,
        30.0,
    )
    out = core.solve_plan_joints(kin, plan["poses"], plan["grips"], plan["hints"], q0, gi)
    assert np.all(out["residual_m"] <= core.ACT_REACH_TOL_M) and np.all(
        out["residual_deg"] <= core.ACT_REACH_TOL_DEG
    )
    assert np.all(out["step_deg"] <= core.ACT_MAX_JOINT_STEP_DEG), (
        "no change of configuration between samples"
    )
    assert np.array_equal(out["q"][:, gi], plan["grips"])


def test_marks_are_validated_saved_beside_the_demo_and_loaded_back(client, tmp_path, monkeypatch):
    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    rgb, depth = _rect_scene(0.0)
    samples, history = _samples_and_history(n=30)
    demo = pregrasp._demo_from_samples(
        "marked", "green cube", samples, history, lambda q: np.eye(4), t0=1000.0
    )
    demo.taught = True
    demo.intr = dict(INTR)
    demo.frames = [(1000.0 + k / 10.0, np.full((48, 84, 3), 90 + k, np.uint8)) for k in range(10)]
    teach = pregrasp._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=rgb,
        depth_m=depth,
        intr=INTR,
        keypoints={
            "mode": "features",
            "concept": "green cube",
            "n_points": 40,
            "xyz": np.zeros((40, 3)),
            "mask": depth < 0.449,
        },
    )
    pregrasp._state.worker.proc = _FakeProc()
    try:
        with pregrasp._state.lock:
            pregrasp._state.teach, pregrasp._state.demo = teach, demo
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()
        assert client.post("/api/pregrasp/act", json={}).status_code == 409, "an unmarked demo is not acted"
        curve = client.get("/api/pregrasp/demo/curve").json()
        assert (
            curve["n"] == 30
            and curve["keypoints"] == []
            and curve["has_frames"] is True
            and "suggested" not in curve
        )
        assert curve["uv"] is None, "no camera calibration in the test: no projected path"
        assert (
            client.get("/api/pregrasp/demo/frame.jpg", params={"i": 7}).headers["content-type"]
            == "image/jpeg"
        )
        post = lambda kps: client.post("/api/pregrasp/demo/keypoints", json={"keypoints": kps})  # noqa: E731
        assert post([{"t": 0.5, "kind": "grasp_end"}]).status_code == 422, (
            "a grasp needs a pre-grasp to start from"
        )
        assert post([{"t": 0.5, "name": "x", "anchor": "object"}]).status_code == 422, (
            "the old shape is refused"
        )
        r = post(
            [{"t": 0.8, "kind": "grasp_end"}, {"t": 0.2, "kind": "pregrasp"}, {"t": 0.4, "kind": "pregrasp"}]
        )
        assert r.status_code == 200 and [k["kind"] for k in r.json()["keypoints"]] == [
            "pregrasp",
            "pregrasp",
            "grasp_end",
        ]
        assert client.post("/api/pregrasp/demo/save", json={"name": "marked"}).status_code == 200
        root = tmp_path / "demos" / "marked"
        assert (root / pregrasp.KEYPOINTS_FILE).exists()
        with pregrasp._state.lock:
            pregrasp._state.demo = None
        r = client.post("/api/pregrasp/demo/load", json={"name": "marked"})
        assert r.status_code == 200 and [k["t"] for k in r.json()["keypoints"]] == [0.2, 0.4, 0.8]
        # Marks written by the earlier editor are dropped when the demo loads, not misread.
        (root / pregrasp.KEYPOINTS_FILE).write_text(
            '{"keypoints": [{"t": 0.3, "name": "grasp", "anchor": "object"}]}'
        )
        r = client.post("/api/pregrasp/demo/load", json={"name": "marked"})
        assert r.status_code == 200 and r.json()["keypoints"] == []
        assert post([]).status_code == 200 and not (root / pregrasp.KEYPOINTS_FILE).exists(), (
            "clearing removes the sidecar"
        )
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.teach = None
            pregrasp._state.test = None
            pregrasp._state.demo = None
            pregrasp._state.teach_job = None
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()


def test_the_act_plan_names_what_it_cannot_do():
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    kin = _StepKinematics()
    samples, history = _samples_and_history(n=30)

    def fk(q):
        pose = np.eye(4)
        pose[0, 3] = q["shoulder_pan"] / 1000.0  # the fake arm's tip: pan is x in millimetres
        return pose

    demo = pregrasp._demo_from_samples("d", "green cube", samples, history, fk, t0=1000.0)
    demo.taught = True
    gi = MOTOR_NAMES.index("gripper")
    demo.q_obs = np.zeros((30, 7))
    demo.q_obs[:, 0] = [s["obs"]["shoulder_pan"] for s in samples]
    demo.q_obs[:, gi] = [s["obs"]["gripper"] for s in samples]
    demo.q_cmd = demo.q_obs.copy()
    demo.keypoints = [
        {"t": float(demo.t[9]), "kind": "pregrasp"},
        {"t": float(demo.t[18]), "kind": "grasp_end"},
    ]
    q_now = np.zeros(7)
    q_now[2] = 50.0  # the fake arm stands 50 mm up
    box = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    delta = np.eye(4)
    delta[0, 3] = 0.02  # the object slid 20 mm: reachable
    plan = pregrasp._plan_act(demo, delta, np.eye(4), kin, q_now, (0.04, np.radians(30)), box, 1.0)
    assert plan["ok"], plan["reason"]
    assert [m["label"] for m in plan["marks"]] == ["pre-grasp 1", "grasp"] and all(
        m["ok"] for m in plan["marks"]
    )
    assert plan["stage"][-1] == "grasp" and plan["q"].shape == (len(plan["times"]), 7)
    delta[0, 3] = 2.0  # two metres: past the fake arm's reach
    plan = pregrasp._plan_act(demo, delta, np.eye(4), kin, q_now, (0.04, np.radians(30)), box, 1.0)
    assert not plan["ok"] and plan["reason"].startswith("pre-grasp 1 is out of reach")
    delta[0, 3] = 0.02
    floor = ((-1.0, -1.0, 0.01), (1.0, 1.0, 1.0))  # a table 10 mm above where the demo's fingertip went
    plan = pregrasp._plan_act(demo, delta, np.eye(4), kin, q_now, (0.04, np.radians(30)), floor, 1.0)
    assert not plan["ok"] and plan["reason"] == "pre-grasp 1 would go 10 mm below the table"


def _gamepad_top(n_x=5, n_y=5, size=(0.072, 0.040), at=(0.0, 0.0, 0.43)):
    """Points spread over a gamepad's top face, camera frame: what a fully seen view fits to."""
    xs = np.linspace(-size[0] / 2, size[0] / 2, n_x)
    ys = np.linspace(-size[1] / 2, size[1] / 2, n_y)
    return np.array([[at[0] + x, at[1] + y, at[2]] for x in xs for y in ys])


def test_a_view_places_its_object_only_when_enough_is_seen_and_its_points_pin_the_pose():
    """Two acts on 2026-10-09 followed views of a gamepad the wrist covered but for a corner: 16 points bunched there
    fitted it 30 mm and 22 degrees off. A view counts only when enough of the object is seen and the points seen pin
    its middle within the act's reach tolerance."""
    middle = np.array([0.0, 0.0, 0.43])
    full = _gamepad_top()
    corner = _gamepad_top(
        4, 4, size=(0.010, 0.008), at=(0.030, 0.016, 0.43)
    )  # 16 points in a 10 x 8 mm corner
    assert core.placement_error(full, middle) < 0.4, (
        "a fully seen top fixes its middle to well under the noise"
    )
    assert core.placement_error(corner, middle) > 1.5, "a corner carries a turn's error out to the middle"
    assert core.placement_error(full[:2], middle) == float("inf"), "two points fix no rotation"
    line = np.array([[x, 0.0, 0.43] for x in np.linspace(-0.03, 0.03, 10)])
    assert core.placement_error(line, middle) == float("inf"), "nor do points in a line"
    assert core.placement_error(full * 1000, middle * 1000) == pytest.approx(
        core.placement_error(full, middle)
    ), "a number without units"

    assert core.view_places(1.0, full, middle, 0.97) == (True, "")
    ok, why = core.view_places(1.0, corner, middle, 0.97)
    assert not ok and "place its middle only to" in why, why
    ok, why = core.view_places(0.6, full, middle, 0.97)
    assert not ok and "only 60% of its points are seen" in why, (
        "partly covered: its points drift onto the cover"
    )
    assert core.view_places(1.0, None, None, 0.97) == (True, ""), (
        "a tracker without fit points: the share decides"
    )


def test_the_point_groups_carry_a_view_from_its_frame_to_their_newest():
    """The act's pose of an object no view places now: the last view that placed it, moved on with what it rests on
    since that view's frame (src/lerobot/showservo/docs/act_loop.md). The point groups' frames are their own; the
    view's frame falls between two of them."""

    def pose(x_mm, yaw_deg):
        m = np.eye(4)
        m[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
        m[:3, 3] = [x_mm / 1000.0, 0.05, 0.43]
        return m

    frames = [(10.0, pose(0, 0)), (10.2, pose(10, 10)), (10.4, pose(30, 30))]
    then = core.interp_rigid(
        pose(0, 0), pose(10, 10), 0.5
    )  # where they had it when the view's frame was read
    moved = core.carried_motion(frames, 10.1)
    assert np.allclose(moved, pose(30, 30) @ np.linalg.inv(then))
    assert np.allclose(moved @ then, pose(30, 30)), "the object as it was then lands where it is now"
    assert np.allclose(core.carried_motion(frames, 10.4), np.eye(4)), (
        "a view of their newest frame: nothing since"
    )
    assert np.allclose(core.carried_motion(frames, 11.0), np.eye(4)), "a view newer than all their frames"
    assert np.allclose(core.carried_motion(frames, 9.0), pose(30, 30) @ np.linalg.inv(pose(0, 0))), (
        "a view older than their first frame: from their first"
    )
    assert np.allclose(core.carried_motion([], 10.0), np.eye(4)), "without them the view is held"


def _turned_about(middle, rotvec_deg, shift=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A motion that turns by ``rotvec_deg`` about ``middle`` and then shifts, camera frame."""
    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    m[:3, 3] = np.asarray(middle) - m[:3, :3] @ np.asarray(middle) + np.asarray(shift)
    return m


def test_the_object_placed_onto_moves_only_on_a_view_that_places_it(monkeypatch):
    """The cube's track had the share test alone; it now has the same rule as the object picked."""
    rgb, depth = _rect_scene(0.0)
    found = {"object": "box", "ok": True, "delta": np.eye(4), "view": ["d", 0]}
    model = _gamepad_top(at=(0.004, 0.007, 0.43))  # its key points, in its own find's frame: the block's top
    share = {"name": "box", "ok": True, "lost": False, "n_visible": 50, "n_tracks": 50}
    moved = np.eye(4)
    moved[:3, 3] = [0.010, 0.0, 0.0]
    spread = np.array([[x, y] for x in (390, 417, 443, 470) for y in (235, 245, 255, 265)], np.float32)
    bunched = np.array([[x, y] for x in (466, 468, 470, 472) for y in (266, 268, 270, 272)], np.float32)
    with pregrasp._state.lock:
        pregrasp._state.located = {"box": found}
        pregrasp._state.target = pregrasp._TargetTrack(obj="box", anchor=np.eye(4), n_points=50)
        pregrasp._state.trust_share = pregrasp.TRUST_SHARE_DEFAULT
    try:
        for uv, expect_moved in ((bunched, False), (spread, True)):
            found["delta"], found["stamp"] = np.eye(4), 100.0
            r = {
                "others": [share],
                "other_delta_0": moved,
                "other_fit_uv_0": uv,
                "other_fit_inlier_0": np.ones(len(uv), bool),
                "other_model_0": model,
            }
            pregrasp._apply_others(r, rgb.shape, depth, INTR, stamp=101.5)
            assert np.allclose(found["delta"], moved) == expect_moved, pregrasp._state.target.last
            assert found["stamp"] == (101.5 if expect_moved else 100.0), (
                "the view's frame time comes with its pose"
            )
        assert pregrasp._state.target.last["state"] == "tracking"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.located = {}
            pregrasp._state.target = pregrasp._TargetTrack()


def test_the_act_follows_an_object_moved_during_the_approach_and_grasps_where_it_settled(
    tmp_path, monkeypatch
):
    import asyncio
    import time as _time

    from lerobot.gui.api import jog
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    gi = MOTOR_NAMES.index("gripper")
    kin = _StepKinematics()
    n = 30
    t = np.arange(n) / 30.0
    q_obs = np.zeros((n, 7))
    q_obs[:, 0] = 100.0 + np.arange(n)  # the demo's fingertip slides 1 mm per sample in x
    q_obs[:, 2] = np.linspace(60.0, 20.0, n)  # and comes down
    q_obs[:, gi] = np.where(np.arange(n) < 20, 60.0, 85.0)  # closing on the object two thirds in
    tips = np.stack([kin.forward_kinematics(q) for q in q_obs])
    demo = pregrasp._Demo(
        name="d",
        concept="gamepad",
        fps=30.0,
        t=t,
        tips=tips,
        grippers=q_obs[:, gi],
        q_obs=q_obs,
        q_cmd=q_obs.copy(),
        deltas=np.tile(np.eye(4), (n, 1, 1)),
        seen=np.ones(n, dtype=bool),
        delta0=np.eye(4),
        taught=True,
    )
    demo.keypoints = [{"t": float(t[10]), "kind": "pregrasp"}, {"t": float(t[25]), "kind": "grasp_end"}]
    rgb, depth = _rect_scene(0.0)
    teach = pregrasp._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=rgb,
        depth_m=depth,
        intr=INTR,
        keypoints={
            "mode": "features",
            "concept": "gamepad",
            "n_points": 40,
            "xyz": np.tile([0.11, 0.0, 0.03], (40, 1)),
        },
    )
    moved = np.eye(4)
    moved[:3, 3] = [0.020, 0.010, 0.0]  # the operator slides the gamepad 22 mm while the arm comes in

    sim = {
        "q": np.array([80.0, -20.0, 90.0, 0, 0, 0, 60.0]),
        "grip": 60.0,
        "targets": [],
        "streamed": [],
        "limits": [],
        "stopped": False,
    }

    def set_target_pose(pose):
        sim["targets"].append(np.array(pose))
        sim["q"][:3] += (
            np.asarray(pose)[:3, 3] * 1000.0 - sim["q"][:3]
        ) * 0.34  # the walk covers a third per tick
        if len(sim["targets"]) == 3:
            with pregrasp._state.lock:
                pregrasp._state.track.history.append((_time.time(), True, moved))

    async def joints_start(q_first):
        sim["q"] = np.array([q_first[m] for m in MOTOR_NAMES])

    async def joints_stop():
        sim["stopped"] = True

    def set_target_joints(q):
        sim["q"] = np.array([q[m] for m in MOTOR_NAMES])
        sim["streamed"].append(sim["q"].copy())

    monkeypatch.setattr(jog, "kinematics", lambda: kin)
    monkeypatch.setattr(
        jog,
        "current_tip_and_anchor",
        lambda: (
            kin.forward_kinematics(sim["q"]),
            np.eye(4),
            {m: float(sim["q"][k]) for k, m in enumerate(MOTOR_NAMES)},
        ),
    )
    monkeypatch.setattr(jog, "set_target_pose", set_target_pose)
    monkeypatch.setattr(jog, "current_status", lambda: {"connected": True, "halted": False, "holding": False})
    monkeypatch.setattr(jog, "current_gripper", lambda: sim["grip"])
    monkeypatch.setattr(jog, "set_gripper", lambda g: sim.__setitem__("grip", g))
    monkeypatch.setattr(jog, "walk_limits", lambda: (0.04, np.radians(30)))
    monkeypatch.setattr(jog, "set_walk_limits", lambda lin, ang: sim["limits"].append((lin, ang)))
    monkeypatch.setattr(jog, "workspace_box", lambda: ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    monkeypatch.setattr(jog, "joints_start", joints_start)
    monkeypatch.setattr(jog, "joints_stop", joints_stop)
    monkeypatch.setattr(jog, "set_target_joints", set_target_joints)
    from tests.gui.test_stream_objects import fake_playback

    fake_playback(monkeypatch, set_target_joints)
    monkeypatch.setattr(pregrasp, "_t_base_cam", lambda: np.eye(4))
    monkeypatch.setattr(pregrasp._state, "groups_with_acts", False)  # held poses: no point groups view here
    monkeypatch.setattr(pregrasp, "ACT_TICK_S", 0.002)
    monkeypatch.setattr(pregrasp, "ACT_STEP_TIMEOUT_S", 5.0)
    monkeypatch.setattr(pregrasp, "TRIALS_PATH", tmp_path / "trials.jsonl")
    monkeypatch.setattr(pregrasp, "_trials", None)
    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    arm_t0 = _time.time()
    monkeypatch.setattr(jog, "start_record", lambda: arm_t0)
    q_rec = {m: float(v) for m, v in zip(MOTOR_NAMES, sim["q"], strict=True)}
    monkeypatch.setattr(jog, "stop_record", lambda: [{"t": 0.0, "obs": q_rec, "cmd": q_rec}] * 3)
    monkeypatch.setattr(jog, "fk_tip", lambda q: np.eye(4))
    with pregrasp._state.lock:
        pregrasp._state.demo, pregrasp._state.teach = demo, teach
        pregrasp._state.test = pregrasp._Test(
            at="now",
            rgb=rgb,
            result={
                "ok": True,
                "delta_cam": np.eye(4),
                "table_teach": [0.0, 0.0, 1.0],
                "table_find": [0.0, 0.0, 1.0],
            },
        )
        pregrasp._state.track.on = True
        pregrasp._state.track.history = [(_time.time() - 1.0, True, np.eye(4))]
        pregrasp._state.track.last = {"state": "tracking"}
        pregrasp._state.act = pregrasp._Act(on=True, speed=4.0)

    async def run():
        async def tracker():  # the camera keeps seeing the gamepad where it was left
            while True:
                await asyncio.sleep(0.01)
                if len(sim["targets"]) >= 3:
                    with pregrasp._state.lock:
                        pregrasp._state.track.history.append((_time.time(), True, moved))

        feed = asyncio.create_task(tracker())
        try:
            await asyncio.wait_for(pregrasp._act_task(4.0), timeout=20.0)
        finally:
            feed.cancel()

    try:
        asyncio.run(run())
        act = pregrasp._state.act
        assert act.ok, act.reason
        assert np.allclose(sim["targets"][0], tips[10]), "the approach first aims where the object was"
        assert np.allclose(sim["targets"][-1], moved @ tips[10]), "then follows it to where it was moved"
        end = kin.forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - (moved @ tips[25])[:3, 3]) <= core.ACT_SOLVE_TOL_M, (
            "the grasp is replayed on the moved object"
        )
        assert sim["streamed"][-1][gi] == 85.0, "with the demo's closing"
        assert sim["limits"] == [(0.16, np.radians(30) * 4.0), (0.04, np.radians(30))], (
            "the walk sped up for the act, then restored"
        )
        assert sim["stopped"]
        # The act's recording: kept for analysis, named by its trial row.
        row = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[-1])
        rec = pathlib.Path(row["run"])
        assert rec.parent == tmp_path / "demos" / ".acts", "an unsaved demo's acts go under the demos folder"
        summary = json.loads((rec / "act.json").read_text())
        assert summary["result"]["ok"] and summary["demo"] == demo.name
        assert any("pose" in x for x in summary["targets"]) and any(
            "joints" in x for x in summary["targets"]
        ), "the approach's poses and the grasp's joints"
        arm = np.load(rec / "arm.npz")
        assert arm["q_obs"].shape == (3, 7) and arm["tip_obs"].shape == (3, 4, 4)
        assert pregrasp._state.run is None
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = pregrasp._state.teach = pregrasp._state.test = None
            pregrasp._state.track.on = False
            pregrasp._state.track.history = []
            pregrasp._state.track.last = {}
            pregrasp._state.act = pregrasp._Act()


def test_views_of_a_covered_object_move_neither_the_arm_nor_the_grasp(tmp_path, monkeypatch):
    """At 20:35 on 2026-10-09 the wrist covered all but a corner of the gamepad; the tracker's views of that corner,
    each fitted on all its points, were tens of degrees off, and the act moved the arm after every one and pushed the
    gamepad. Such views go through the tracker's own check here (enough points agree) and are turned away by the
    act's rule: the arm's every target and the grasp stay on the pose held."""
    import asyncio
    import time as _time

    from lerobot.gui.api import jog
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    gi = MOTOR_NAMES.index("gripper")
    kin = _StepKinematics()
    n = 30
    t = np.arange(n) / 30.0
    q_obs = np.zeros((n, 7))
    q_obs[:, 0] = 100.0 + np.arange(n)
    q_obs[:, 2] = np.linspace(60.0, 20.0, n)
    q_obs[:, gi] = np.where(np.arange(n) < 20, 60.0, 85.0)
    tips = np.stack([kin.forward_kinematics(q) for q in q_obs])
    demo = pregrasp._Demo(
        name="d",
        concept="gamepad",
        fps=30.0,
        t=t,
        tips=tips,
        grippers=q_obs[:, gi],
        q_obs=q_obs,
        q_cmd=q_obs.copy(),
        deltas=np.tile(np.eye(4), (n, 1, 1)),
        seen=np.ones(n, dtype=bool),
        delta0=np.eye(4),
        taught=True,
    )
    demo.keypoints = [{"t": float(t[10]), "kind": "pregrasp"}, {"t": float(t[25]), "kind": "grasp_end"}]
    rgb, depth = _rect_scene(0.0)
    teach = pregrasp._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=rgb,
        depth_m=depth,
        intr=INTR,
        keypoints={
            "mode": "features",
            "concept": "gamepad",
            "n_points": 40,
            "xyz": _gamepad_top(at=(0.004, 0.007, 0.43)),
        },
    )
    wrong = np.eye(4)  # what the corner's fit said: turned 20 degrees about x and 30 mm along it
    c, s = np.cos(np.radians(20.0)), np.sin(np.radians(20.0))
    wrong[:3, :3] = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    wrong[:3, 3] = [0.030, 0.0, 0.0]
    corner = np.array([[x, y] for x in (466, 468, 470, 472) for y in (266, 268, 270, 272)], np.float32)
    corner_view = {
        "ok": True,
        "state": "tracking",
        "algo": "p2p",
        "ms": 20.0,
        "n_inliers": 30,  # the tracker's own check passes: enough points agree
        "n_matches": 30,
        "n_tracks": 30,
        "delta": wrong,
        "fit_uv": corner,
        "fit_inlier": np.ones(len(corner), bool),
        "live_uv": np.zeros((len(corner), 2), np.float32),
    }
    sim = {"q": np.array([80.0, -20.0, 90.0, 0, 0, 0, 60.0]), "grip": 60.0, "targets": [], "streamed": []}

    def set_target_pose(pose):
        sim["targets"].append(np.array(pose))
        sim["q"][:3] += (np.asarray(pose)[:3, 3] * 1000.0 - sim["q"][:3]) * 0.34

    async def joints_start(q_first):
        sim["q"] = np.array([q_first[m] for m in MOTOR_NAMES])

    async def joints_stop():
        pass

    def set_target_joints(q):
        sim["q"] = np.array([q[m] for m in MOTOR_NAMES])
        sim["streamed"].append(sim["q"].copy())

    monkeypatch.setattr(jog, "kinematics", lambda: kin)
    monkeypatch.setattr(
        jog,
        "current_tip_and_anchor",
        lambda: (
            kin.forward_kinematics(sim["q"]),
            np.eye(4),
            {m: float(sim["q"][k]) for k, m in enumerate(MOTOR_NAMES)},
        ),
    )
    monkeypatch.setattr(jog, "set_target_pose", set_target_pose)
    monkeypatch.setattr(jog, "current_status", lambda: {"connected": True, "halted": False, "holding": False})
    monkeypatch.setattr(jog, "current_gripper", lambda: sim["grip"])
    monkeypatch.setattr(jog, "set_gripper", lambda g: sim.__setitem__("grip", g))
    monkeypatch.setattr(jog, "walk_limits", lambda: (0.04, np.radians(30)))
    monkeypatch.setattr(jog, "set_walk_limits", lambda lin, ang: None)
    monkeypatch.setattr(jog, "workspace_box", lambda: ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    monkeypatch.setattr(jog, "joints_start", joints_start)
    monkeypatch.setattr(jog, "joints_stop", joints_stop)
    monkeypatch.setattr(jog, "set_target_joints", set_target_joints)
    from tests.gui.test_stream_objects import fake_playback

    fake_playback(monkeypatch, set_target_joints)
    monkeypatch.setattr(pregrasp, "_t_base_cam", lambda: np.eye(4))
    monkeypatch.setattr(pregrasp._state, "groups_with_acts", False)  # held poses: no point groups view here
    monkeypatch.setattr(pregrasp, "ACT_TICK_S", 0.002)
    monkeypatch.setattr(pregrasp, "ACT_STEP_TIMEOUT_S", 5.0)
    monkeypatch.setattr(pregrasp, "TRIALS_PATH", tmp_path / "trials.jsonl")
    monkeypatch.setattr(pregrasp, "_trials", None)
    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    monkeypatch.setattr(jog, "start_record", lambda: _time.time())
    monkeypatch.setattr(jog, "stop_record", lambda: [])
    monkeypatch.setattr(jog, "fk_tip", lambda q: np.eye(4))
    with pregrasp._state.lock:
        pregrasp._state.demo, pregrasp._state.teach = demo, teach
        pregrasp._state.test = pregrasp._Test(at="now", rgb=rgb, result={"ok": True, "delta_cam": np.eye(4)})
        pregrasp._state.track.on = True
        pregrasp._state.track.history = [(_time.time() - 1.0, True, np.eye(4))]
        pregrasp._state.track.last = {"state": "tracking"}
        pregrasp._state.trust_share = pregrasp.TRUST_SHARE_DEFAULT
        pregrasp._state.act = pregrasp._Act(on=True, speed=4.0)
    reasons: list[str] = []

    async def run():
        async def tracker():  # once the arm is over it, the camera sees only the corner the wrist leaves
            while True:
                await asyncio.sleep(0.005)
                if not sim["targets"]:
                    continue
                job = pregrasp._Job(
                    id=f"corner-{len(reasons)}",
                    kind="track",
                    concept="gamepad",
                    rgb=rgb,
                    depth_m=depth,
                    intr=INTR,
                    created=_time.time(),
                    result=dict(corner_view),
                )
                with pregrasp._state.lock:
                    pregrasp._state.track.job = job.id
                await pregrasp._apply_track_result(job)
                reasons.append(pregrasp._state.track.last.get("reason") or "")

        feed = asyncio.create_task(tracker())
        try:
            await asyncio.wait_for(pregrasp._act_task(4.0), timeout=20.0)
        finally:
            feed.cancel()

    try:
        asyncio.run(run())
        act = pregrasp._state.act
        assert act.ok, act.reason
        assert reasons and all("place its middle only to" in why for why in reasons), reasons[:3]
        assert all(not ok for _w, ok, _d in pregrasp._state.track.history[1:]), "none of them became the pose"
        assert all(np.allclose(p, tips[10]) for p in sim["targets"]), "the arm never aimed where they put it"
        end = kin.forward_kinematics(sim["streamed"][-1])
        assert np.linalg.norm(end[:3, 3] - tips[25][:3, 3]) <= core.ACT_SOLVE_TOL_M, (
            "the grasp, on the pose held"
        )
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = pregrasp._state.teach = pregrasp._state.test = None
            pregrasp._state.track.on = False
            pregrasp._state.track.history = []
            pregrasp._state.track.last = {}
            pregrasp._state.track.job = None
            pregrasp._state.act = pregrasp._Act()


def test_the_act_refuses_to_start_while_the_tracker_has_lost_the_object(tmp_path, monkeypatch):
    import asyncio

    from lerobot.gui.api import jog
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    kin = _StepKinematics()
    samples, history = _samples_and_history(n=30)
    demo = pregrasp._demo_from_samples("d", "gamepad", samples, history, lambda q: np.eye(4), t0=1000.0)
    demo.keypoints = [{"t": float(demo.t[9]), "kind": "pregrasp"}]
    rgb, depth = _rect_scene(0.0)
    teach = pregrasp._Teach(
        at="t",
        box=(0, 0, 0, 0),
        rgb=rgb,
        depth_m=depth,
        intr=INTR,
        keypoints={"mode": "features", "concept": "gamepad", "n_points": 40, "xyz": np.zeros((40, 3))},
    )
    moves = []
    monkeypatch.setattr(jog, "kinematics", lambda: kin)
    monkeypatch.setattr(
        jog, "current_tip_and_anchor", lambda: (np.eye(4), np.eye(4), dict.fromkeys(MOTOR_NAMES, 0.0))
    )
    monkeypatch.setattr(jog, "set_target_pose", lambda pose: moves.append(pose))
    monkeypatch.setattr(pregrasp, "_t_base_cam", lambda: np.eye(4))
    monkeypatch.setattr(pregrasp._state, "groups_with_acts", False)  # held poses: no point groups view here
    with pregrasp._state.lock:
        pregrasp._state.demo, pregrasp._state.teach = demo, teach
        pregrasp._state.test = pregrasp._Test(
            at="a while ago", rgb=rgb, result={"ok": True, "delta_cam": np.eye(4)}
        )
        pregrasp._state.track.on = True
        pregrasp._state.track.last = {"state": "lost"}
        pregrasp._state.act = pregrasp._Act(on=True)
    try:
        asyncio.run(pregrasp._act_task(1.0))
        act = pregrasp._state.act
        assert (
            act.ok is False
            and "does not see the object (lost)" in act.reason
            and "load the demo again" in act.reason
        )
        assert moves == [], "the arm did not move on the stale pose"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = pregrasp._state.teach = pregrasp._state.test = None
            pregrasp._state.track.on = False
            pregrasp._state.track.last = {}
            pregrasp._state.act = pregrasp._Act()


class _FakeCamera:
    """A RealSense stand-in at the rig's size and rate: each read waits for the next frame."""

    def __init__(self, hz=30.0):
        self.period, self.k = 1.0 / hz, 0

    def color_intrinsics(self):
        return dict(INTR)

    def read_color_and_aligned_depth(self):
        import time as _time

        _time.sleep(self.period)
        self.k += 1
        rgb = np.full((480, 848, 3), self.k % 255, np.uint8)
        depth = np.full((480, 848), 450 + self.k % 10, np.uint16)
        return rgb, depth


def test_the_camera_records_on_its_own_for_a_replay_and_never_beside_a_demo_recording(
    client, tmp_path, monkeypatch
):
    import time as _time

    from lerobot.gui.api import showservo

    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    monkeypatch.setattr(showservo, "live_camera", lambda: None)
    assert client.post("/api/pregrasp/camera/record/start").status_code == 409, "no camera"
    camera = _FakeCamera()
    monkeypatch.setattr(showservo, "live_camera", lambda: camera)
    try:
        r = client.post("/api/pregrasp/camera/record/start")
        assert r.status_code == 200, r.text
        out = pathlib.Path(r.json()["out"])
        assert client.post("/api/pregrasp/camera/record/start").status_code == 409, "one recording at a time"
        refused = client.post("/api/pregrasp/demo/record/start")
        assert refused.status_code == 409 and "on its own" in refused.json()["detail"], (
            "a demo would split its frames"
        )
        _time.sleep(0.3)
        stopped = client.post("/api/pregrasp/camera/record/stop").json()
        assert stopped["frames"] >= 3 and not stopped["error"]
        assert len(np.loadtxt(out / "times.txt")) == stopped["frames"]
        assert (out / "rgb" / "000000.jpg").exists() and (out / "depth" / "000000.png").exists()
        assert (out / "cam_K.txt").exists() and out.parent == tmp_path / "demos" / ".recordings"
        assert client.post("/api/pregrasp/camera/record/stop").status_code == 409
    finally:
        with pregrasp._state.lock:
            rec, pregrasp._state.camera_recording = pregrasp._state.camera_recording, None
        if rec is not None:
            rec.stop.set()


def test_a_demo_records_the_camera_stream_without_a_teach_and_keeps_it_through_save_and_load(
    client, tmp_path, monkeypatch
):
    import pathlib
    import time as _time

    from lerobot.gui.api import jog, showservo
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    monkeypatch.setattr(pregrasp, "_demos_root", lambda: tmp_path / "demos")
    camera = _FakeCamera()
    monkeypatch.setattr(showservo, "live_camera", lambda: camera)
    t0 = _time.time()
    monkeypatch.setattr(jog, "start_record", lambda: t0)
    samples = [
        {
            "t": i / 30.0,
            "obs": dict.fromkeys(MOTOR_NAMES, float(i)),
            "cmd": dict.fromkeys(MOTOR_NAMES, float(i)),
        }
        for i in range(30)
    ]
    monkeypatch.setattr(jog, "stop_record", lambda: samples)
    monkeypatch.setattr(jog, "fk_tip", lambda q: np.eye(4))
    with pregrasp._state.lock:
        pregrasp._state.teach = None
        pregrasp._state.demo = None
    try:
        r = client.post("/api/pregrasp/demo/record/start")
        assert r.status_code == 200 and r.json()["camera"] is True, (
            "no teach needed, and the camera stream starts"
        )
        _time.sleep(0.5)
        r = client.post("/api/pregrasp/demo/record/stop", json={"name": "streamed"})
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["stream_frames"] >= 5 and info["has_frames"] and info["concept"] == "demo"
        work = pathlib.Path(pregrasp._state.demo.recording)
        assert work.parent == tmp_path / "demos" / ".recordings"
        times = np.loadtxt(work / "times.txt")
        assert len(times) == info["stream_frames"] and np.all(np.diff(times) > 0), (
            "one stamp per frame, in order"
        )
        assert (
            (work / "rgb" / "000000.jpg").exists()
            and (work / "depth" / "000000.png").exists()
            and (work / "cam_K.txt").exists()
        )
        frame = client.get("/api/pregrasp/demo/frame.jpg", params={"i": 10})
        assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg", (
            "the editor plays the stream"
        )
        assert client.post("/api/pregrasp/demo/save", json={}).status_code == 200, (
            "a demo without a teach saves"
        )
        root = tmp_path / "demos" / "streamed"
        assert (root / pregrasp.DEMO_RECORDING / "times.txt").exists() and not work.exists(), (
            "the stream moved into the demo"
        )
        assert any((root / "videos").rglob("*.mp4")), "the dataset's video is made from the stream"
        with pregrasp._state.lock:
            pregrasp._state.demo = None
        r = client.post("/api/pregrasp/demo/load", json={"name": "streamed"})
        assert (
            r.status_code == 200
            and r.json()["teach_pending"] is False
            and r.json()["stream_frames"] == info["stream_frames"]
        )
        # Saving again under the same name keeps the stream.
        assert client.post("/api/pregrasp/demo/save", json={}).status_code == 200
        assert (root / pregrasp.DEMO_RECORDING / "times.txt").exists()
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None
            pregrasp._state.recording = None
            pregrasp._state.stream = None


def _box_tracks() -> np.ndarray:
    """25 tracked points spread over the top of :func:`_rect_scene`'s block, pixels."""
    return np.array(
        [[x, y] for x in (390, 410, 430, 450, 470) for y in (235, 242, 250, 258, 265)], np.float32
    )


def _lifted(uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """``uv``'s points in the camera frame, by ``depth`` under them."""
    z = depth[np.round(uv[:, 1]).astype(int), np.round(uv[:, 0]).astype(int)].astype(float)
    return np.stack(
        [(uv[:, 0] - INTR["cx"]) * z / INTR["fx"], (uv[:, 1] - INTR["cy"]) * z / INTR["fy"], z], axis=1
    )


def _nearer_at(depth: np.ndarray, uv: np.ndarray, rows: list[int], by: float = 0.12) -> np.ndarray:
    """``depth`` with something ``by`` nearer the camera over the tracks ``rows``, as the wrist beside the cube read."""
    out = depth.copy()
    for u, v in np.round(uv[rows]).astype(int):
        out[v - 2 : v + 3, u - 2 : u + 3] -= by
    return out


def test_a_point_counts_as_seen_only_while_its_depth_agrees_with_where_it_was():
    """The depth check on its own: a track counts when the colour tracker sees it, the depth image has a reading under
    it, and that reading puts it within the tolerance of where the object's motion takes it from where it was when the
    last view placed the object. The motion is the one most of the tracks agree on, so a move of the whole object
    counts without anything carrying it, while a few tracks read nearer, as beside the wrist, do not; with too few
    tracks to fit on, the motion given is taken. A track with no reading under it counts as unseen only with something
    nearer right beside it, as the wrist was; otherwise it is left out of the count. A track with no place yet counts
    when seen with a reading."""
    _rgb, depth = _rect_scene(0.0)
    uv, idx, vis = _box_tracks(), np.arange(25), np.ones(25, bool)
    count, judged, now, move = core.depth_seen(idx, uv, vis, depth, INTR, {}, np.eye(4), core.DEPTH_AGREE_M)
    assert (
        count == judged == 25 and np.allclose(now[3], _lifted(uv, depth)[3]) and np.allclose(move, np.eye(4))
    )
    expected = core.depth_expect({}, now, np.eye(4))
    wrist = _nearer_at(depth, uv, [4, 9])
    wrist[np.round(uv[[14, 24], 1]).astype(int), np.round(uv[[14, 24], 0]).astype(int)] = 0.0  # no reading
    unseen = vis.copy()
    unseen[0] = False
    count, judged, now, _move = core.depth_seen(
        idx, uv, unseen, wrist, INTR, expected, np.eye(4), core.DEPTH_AGREE_M
    )
    assert count == 20 and not {0, 4, 9, 14, 24} & set(now), "unseen, read nearer, or no reading: not counted"
    assert judged == 23, "a reading missing with nothing nearer beside it is left out"
    beside = wrist.copy()
    u, v = np.round(uv[14]).astype(int)
    beside[v - 6 : v - 3, u - 2 : u + 3] -= 0.12  # the wrist six pixels from track 14
    assert core.depth_seen(idx, uv, unseen, beside, INTR, expected, np.eye(4), core.DEPTH_AGREE_M)[1] == 24, (
        "with something nearer beside it, it counts as unseen"
    )
    assert core.depth_seen(idx, uv, vis, wrist, INTR, expected, np.eye(4), 0.15)[0] == 23, (
        "within the tolerance"
    )
    farther = depth + 0.03  # the whole block 30 mm farther
    count, _judged, _now, move = core.depth_seen(
        idx, uv, vis, farther, INTR, expected, np.eye(4), core.DEPTH_AGREE_M
    )
    assert count == 25 and np.linalg.norm(move[:3, 3]) > 0.02, (
        "a move every track agrees on counts, and is found"
    )
    count, _judged, now, _move = core.depth_seen(
        idx, uv, vis, _nearer_at(farther, uv, [4, 9, 14, 24]), INTR, expected, np.eye(4), core.DEPTH_AGREE_M
    )
    assert count == 21 and not {4, 9, 14, 24} & set(now), (
        "moved, with four read nearer: those four still do not"
    )
    pushed = np.eye(4)
    pushed[2, 3] = 0.03
    few = {q: expected[q] for q in range(core.DEPTH_FIT_MIN - 1)}
    assert core.depth_seen(idx, uv, vis, farther, INTR, few, np.eye(4), core.DEPTH_AGREE_M)[0] == 25 - len(
        few
    ), "too few tracks to fit on: the motion given, none, leaves those with a place 30 mm off"
    assert core.depth_seen(idx, uv, vis, farther, INTR, few, pushed, core.DEPTH_AGREE_M)[0] == 25, (
        "and the point groups' motion, when given, takes them there"
    )
    assert (
        core.depth_seen(idx + 100, uv, vis, wrist, INTR, expected, np.eye(4), core.DEPTH_AGREE_M)[0] == 23
    ), "tracks with no place yet count when seen with a reading"
    moved_on = core.depth_expect(expected, {1: np.zeros(3)}, pushed)
    assert np.allclose(moved_on[1], 0.0) and np.allclose(moved_on[2], expected[2] + [0.0, 0.0, 0.03])


def test_the_object_placed_onto_moves_only_on_a_view_whose_points_depth_agrees(client, monkeypatch, tmp_path):
    """The act of 2026-10-10 11:15: the wrist came up beside the cube, and the depth camera read the wrist, or nothing,
    under some of its tracked points while the colour tracker still saw them; those views tilted the cube. With the
    depth check, on by default, a point counts as seen only while its depth agrees with where it was when the last
    view placed the object: such a view no longer moves it, says why, and the act's record keeps the share counted.
    Off, colour alone says what is seen, as before."""
    rgb, depth = _rect_scene(0.0)
    uv = _box_tracks()
    model = _lifted(uv, depth)
    share = {"name": "box", "ok": True, "lost": False, "n_visible": 25, "n_tracks": 25, "session": 1}
    found = {"object": "box", "ok": True, "delta": np.eye(4), "view": ["d", 0]}
    monkeypatch.setattr(pregrasp._state, "located", {"box": found})
    monkeypatch.setattr(
        pregrasp._state, "target", pregrasp._TargetTrack(obj="box", anchor=np.eye(4), n_points=25)
    )
    monkeypatch.setattr(pregrasp._state, "trust_share", pregrasp.TRUST_SHARE_DEFAULT)
    monkeypatch.setattr(pregrasp._state, "depth_check", True)
    monkeypatch.setattr(pregrasp._state, "depth_tol_m", core.DEPTH_AGREE_M)
    monkeypatch.setattr(pregrasp._state, "groups_with_acts", False)
    run = pregrasp._Run(root=tmp_path, meta={})
    monkeypatch.setattr(pregrasp._state, "run", run)
    tilted = _turned_about(model.mean(axis=0), (10.0, 0.0, 0.0))

    def view(d, depth_img, stamp):
        r = {"others": [share], "other_delta_0": d, "other_fit_uv_0": uv, "other_fit_inlier_0": np.ones(25, bool),
             "other_model_0": model, "other_track_idx_0": np.arange(25), "other_track_uv_0": uv,
             "other_track_vis_0": np.ones(25, bool)}  # fmt: skip
        pregrasp._apply_others(r, rgb.shape, depth_img, INTR, stamp=stamp)
        return found["delta"]

    assert np.allclose(view(np.eye(4), depth, 1.0), np.eye(4)), "the first view places it"
    wrist = _nearer_at(depth, uv, [4, 9, 14, 24])
    assert np.allclose(view(tilted, wrist, 1.2), np.eye(4)), (
        "a view whose depth disagrees leaves it where it was"
    )
    assert "seen with a depth that agrees within 20 mm" in pregrasp._state.target.last["reason"]
    assert run.meta["target_track"][-1]["depth_seen"] == pytest.approx(0.84)
    assert np.allclose(view(tilted, depth, 1.3), tilted), "with every point's depth agreeing, the view counts"
    options = client.post("/api/pregrasp/options", json={"depth_check": False}).json()
    assert options["depth_check"] is False and options["depth_tol_mm"] == pytest.approx(20.0)
    assert np.allclose(view(np.eye(4), wrist, 1.4), np.eye(4)), "off: colour alone says what is seen"
    assert "depth_seen" not in run.meta["target_track"][-1]


def test_the_depth_check_follows_the_object_and_starts_over_with_a_new_session(monkeypatch):
    """A tray pushed between two views does not read as bad depth, with the point groups' motion of the object or
    without it: the points agree on the move. A new tracker session numbers its tracks afresh, so it drops the places
    kept from the last; kept, they belong to other points and the view would not count."""
    import collections

    rgb, depth = _rect_scene(0.0)
    uv = _box_tracks()
    model = _lifted(uv, depth)
    pushed = np.eye(4)
    pushed[2, 3] = 0.03
    monkeypatch.setattr(pregrasp._state, "trust_share", pregrasp.TRUST_SHARE_DEFAULT)
    monkeypatch.setattr(pregrasp._state, "depth_check", True)
    monkeypatch.setattr(pregrasp._state, "depth_tol_m", core.DEPTH_AGREE_M)
    monkeypatch.setattr(pregrasp._state, "run", None)
    feed = pregrasp._GroupsFeed()
    feed.frames["box"] = collections.deque([(1.0, np.eye(4)), (2.0, pushed)])
    monkeypatch.setattr(pregrasp._state, "groups_feed", feed)

    def act(groups: bool):
        found = {"object": "box", "ok": True, "delta": np.eye(4), "view": ["d", 0]}
        monkeypatch.setattr(pregrasp._state, "located", {"box": found})
        monkeypatch.setattr(
            pregrasp._state, "target", pregrasp._TargetTrack(obj="box", anchor=np.eye(4), n_points=25)
        )
        monkeypatch.setattr(pregrasp._state, "groups_with_acts", groups)

        def view(d, depth_img, stamp, session=1, numbers=np.arange(25)):
            share = {
                "name": "box",
                "ok": True,
                "lost": False,
                "n_visible": 25,
                "n_tracks": 25,
                "session": session,
            }
            r = {"others": [share], "other_delta_0": d, "other_fit_uv_0": uv, "other_fit_inlier_0": np.ones(25, bool),
                 "other_model_0": model, "other_track_idx_0": numbers, "other_track_uv_0": uv,
                 "other_track_vis_0": np.ones(25, bool)}  # fmt: skip
            pregrasp._apply_others(r, rgb.shape, depth_img, INTR, stamp=stamp)
            return found["delta"]

        return view

    view = act(groups=True)
    view(np.eye(4), depth, 1.0)
    assert np.allclose(view(pushed, depth + 0.03, 2.0), pushed), (
        "carried by the point groups, the points agree"
    )
    view = act(groups=False)
    view(np.eye(4), depth, 1.0)
    assert np.allclose(view(pushed, depth + 0.03, 2.0), pushed), "without them, the points agree on the move"
    renumbered = np.random.default_rng(1).permutation(25)  # a restarted session's numbers for the same points
    assert np.allclose(view(np.eye(4), depth, 2.1, numbers=renumbered), pushed), (
        "kept places, other points: refused"
    )
    assert np.allclose(view(np.eye(4), depth, 2.2, session=2, numbers=renumbered), np.eye(4)), (
        "a new session starts the places over"
    )


def test_a_view_of_the_picked_object_counts_only_the_points_whose_depth_agrees(client, monkeypatch):
    """The picked object's views take the same depth check as the cube's: once a view has placed it, a view with
    something nearer under some of its points, as the gripper coming down beside it gives, leaves its pose and says
    why; the next view whose depth agrees is taken. Each frame comes through the worker's result endpoint, as live:
    the server once dropped the picked object's tracks there, and the check never ran on it."""
    import io
    import json

    rgb, depth = _rect_scene(0.0)
    uv = _box_tracks()
    model = _lifted(uv, depth)
    teach = pregrasp._Teach(at="t", box=(0, 0, 0, 0), rgb=rgb, depth_m=depth, intr=INTR,
                            keypoints={"mode": "features", "concept": "gamepad", "n_points": 25, "xyz": model,
                                       "radius_mm": 40.0, "shape_class": "box", "yaw_observable": True})  # fmt: skip
    track = pregrasp._Track(on=True)
    monkeypatch.setattr(pregrasp._state, "teach", teach)
    monkeypatch.setattr(pregrasp._state, "track", track)
    monkeypatch.setattr(pregrasp._state, "test", None)
    monkeypatch.setattr(pregrasp._state, "run", None)
    monkeypatch.setattr(pregrasp._state, "trust_share", pregrasp.TRUST_SHARE_DEFAULT)
    monkeypatch.setattr(pregrasp._state, "depth_check", True)
    monkeypatch.setattr(pregrasp._state, "depth_tol_m", core.DEPTH_AGREE_M)
    monkeypatch.setattr(pregrasp._state, "groups_with_acts", False)
    monkeypatch.setattr(pregrasp, "_t_base_cam", lambda: np.eye(4))
    monkeypatch.setattr(pregrasp._state.worker, "jobs", {})

    def frame(depth_img, k):
        meta = {"ok": True, "state": "tracking", "algo": "p2p", "ms": 20.0, "n_inliers": 25, "n_matches": 25,
                "n_tracks": 25, "session": 1}  # fmt: skip
        job = pregrasp._Job(id=f"j{k}", kind="track", concept="gamepad", rgb=rgb, depth_m=depth_img, intr=INTR,
                            created=100.0 + k)  # fmt: skip
        pregrasp._state.worker.jobs[job.id] = job
        track.job = job.id
        buf = io.BytesIO()
        np.savez(buf, meta=json.dumps(meta), delta=np.eye(4), fit_uv=uv, fit_inlier=np.ones(25, bool), live_uv=uv,
                 track_idx=np.arange(25), track_uv=uv, track_vis=np.ones(25, bool))  # fmt: skip
        reply = client.post("/api/pregrasp/worker/result", params={"id": job.id}, content=buf.getvalue())
        assert reply.status_code == 200, reply.text
        return dict(track.last)

    assert frame(depth, 0)["state"] == "tracking"
    assert track.depths.t == 100.0 and len(track.depths.at) == 25, (
        "the points it placed the object by are kept"
    )
    last = frame(_nearer_at(depth, uv, [0, 1, 2]), 1)
    assert last["state"] == "untrusted" and "seen with a depth that agrees" in last["reason"], last
    assert last["depth_seen"] == pytest.approx(0.88)
    assert track.depths.t == 100.0, "a view not taken keeps the places as they were"
    assert frame(depth, 2)["state"] == "tracking"
    state = client.get("/api/pregrasp/state")
    assert state.status_code == 200, "the page's poll serves the pose taken without the tracks behind it"
