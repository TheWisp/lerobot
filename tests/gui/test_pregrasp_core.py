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
