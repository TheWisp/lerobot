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
    assert client.post("/api/pregrasp/teach/mark").status_code == 409
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
    assert client.post("/api/pregrasp/options", json={"flat": False}).json() == {"flat": False}
    assert client.get("/api/pregrasp/state").json()["flat"] is False
    assert client.post("/api/pregrasp/options", json={"flat": True}).json() == {"flat": True}


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
        assert client.post("/api/pregrasp/track/start", json={"algo": "nope"}).status_code == 422
        assert client.post("/api/pregrasp/track/stop").status_code == 200
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.teach = None
            pregrasp._state.test = None
            pregrasp._state.track = pregrasp._Track()


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
    assert result["axis_source"] == "surface" and result["face_tilt_applied"] is False
    assert result["surface_tilt_deg"] < 1e-6
    assert abs(result["face_tilt_deg"] - 12.0) < 1e-6
    # The turn survives about the table normal, as the in-plane turn the tilted fit implies.
    assert 25.0 < abs(result["yaw_deg"]) <= 30.0
    assert np.allclose(result["delta_cam"][:3, :3] @ np.array(n_table), n_table, atol=1e-9)
    # With the prior off the face carries the axis and the tilt is applied.
    result = {}
    pregrasp._compose_motion(result, r, teach, flat=False, t_bc=None)
    assert result["axis_source"] == "face" and result["face_tilt_applied"] is True
    assert np.allclose(result["delta_cam"][:3, :3] @ np.array(n_table), n_face, atol=1e-6)


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


def test_demo_keyframes_find_the_grasp_where_the_gripper_closed_and_the_approach_before_it():
    # A fingertip path: 2 s descending toward the grasp point, 1 s still while the gripper closes, 1 s lift.
    hz = 30
    t = np.arange(0, 4.0, 1.0 / hz)
    n = len(t)
    p_grasp = np.array([0.30, 0.10, 0.02])
    tips = np.tile(np.eye(4), (n, 1, 1))
    for i, ti in enumerate(t):
        if ti < 2.0:
            tips[i, :3, 3] = p_grasp + (1.0 - ti / 2.0) * np.array([0.0, 0.0, 0.12])  # from 120 mm above
        elif ti < 3.0:
            tips[i, :3, 3] = p_grasp
        else:
            tips[i, :3, 3] = p_grasp + np.array([0.0, 0.0, (ti - 3.0) * 0.08])
    g = np.full(n, 10.0)
    closing = (t >= 2.3) & (t < 2.6)
    g[closing] = 10.0 + (t[closing] - 2.3) / 0.3 * 50.0
    g[t >= 2.6] = 60.0
    g += 0.3 * np.sin(t * 50.0)  # leader tremor
    kf = core.demo_keyframes(t, tips, g, approach_m=0.03)
    i_g, i_p = kf["grasp"]["index"], kf["pregrasp"]["index"]
    assert 2.0 <= t[i_g] < 2.35 and np.allclose(kf["grasp"]["pose"][:3, 3], p_grasp, atol=1e-9)
    assert 55.0 < kf["grasp"]["gripper"] <= 61.0 and abs(kf["pregrasp"]["gripper"] - 10.0) < 1.0
    assert t[i_p] < 2.0 and 0.029 < kf["approach_m"] < 0.033  # the last sample 30 mm out on the way in
    assert 0.07 < kf["lift_m"] < 0.09
    # A demo whose gripper never closed teaches nothing.
    with pytest.raises(ValueError):
        core.demo_keyframes(t, tips, np.full(n, 10.0) + 0.3 * np.sin(t * 50.0))
    # The closing direction is whatever the gripper did, not a convention.
    kf2 = core.demo_keyframes(t, tips, 100.0 - g)
    assert kf2["grasp"]["index"] == i_g and 39.0 <= kf2["grasp"]["gripper"] < 45.0


def test_run_and_demo_endpoints_guard_without_keyframes_or_an_arm(client):
    with pregrasp._state.lock:
        pregrasp._state.teach = None
    assert client.post("/api/pregrasp/run", json={}).status_code == 409
    assert client.post("/api/pregrasp/teach/from_demo", json={}).status_code == 409
    assert client.post("/api/pregrasp/teach/mark", json={"which": "grasp"}).status_code == 409
    assert client.post("/api/pregrasp/teach/mark", json={"which": "elbow"}).status_code == 422
    assert client.post("/api/pregrasp/run/stop").status_code == 200
