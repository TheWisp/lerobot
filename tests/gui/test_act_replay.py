import json
import pathlib
import shutil
import sys

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import pregrasp

INTR = {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0}
H, W = 480, 848  # the rig's camera
CUBE = (slice(200, 260), slice(300, 360))


def _act(tmp_path: pathlib.Path, held: bool = True) -> pathlib.Path:
    """One recorded frame of an act at the rig's size, the camera as the base: the picked object's mask and points,
    the cube's mask and points, where the act held the cube (moved 30 mm along x from the demo's view, 40 px at 0.45
    m) unless ``held`` is False, the fingertip and the target it was walked to; and the demo the cube was designated
    in, flat at 0.45 m."""
    demo = tmp_path / "demo"
    (demo / "recording" / "depth").mkdir(parents=True)
    cube = np.zeros((H, W), bool)
    cube[CUBE] = True
    np.savez(demo / "objects.npz", names=json.dumps([["cube", 0, [330, 230], 4]]), o0_mask=cube)
    cv2.imwrite(str(demo / "recording" / "depth" / "000000.png"), np.full((H, W), 450, np.uint16))
    np.savetxt(demo / "recording" / "cam_K.txt", [[600, 0, 424], [0, 600, 240], [0, 0, 1]])
    act = tmp_path / "act"
    (act / "frames").mkdir(parents=True)
    cv2.imwrite(str(act / "frames" / "000000.jpg"), np.full((H, W, 3), 60, np.uint8))
    picked = np.zeros((H, W), np.uint8)
    picked[300:360, 600:700] = 255
    cv2.imwrite(str(act / "frames" / "000000_mask.png"), picked)
    arrays = {
        "track_uv": np.array([[620.0, 320.0], [680.0, 340.0]]),
        "track_vis": np.array([True, False]),
        "other_track_uv_0": np.array([[330.0, 230.0]]),
        "other_track_vis_0": np.array([True]),
        "other_mask_0": cube,
    }
    frame = {"i": 0, "t_frame": 10.0, "step": "place", "used": True, "state": "tracking"}
    frame.update(n_matches=2, n_tracks=2)
    if held:
        arrays["onto_held"] = np.eye(4)
        arrays["onto_held"][0, 3] = 0.03
        frame["onto_held_from"] = {"find_id": "f1", "stamp": 9.0, "tracked_at": None}
    np.savez(act / "frames" / "000000.npz", **arrays)
    tip = np.eye(4)
    tip[:3, 3] = [-0.15, 0.1, 0.45]  # pixel (224, 373)
    np.savez(act / "arm.npz", t=np.array([10.0]), tip_obs=tip[None])
    meta = {
        "demo": "pick_place",
        "demo_root": str(demo),
        "intr": INTR,
        "t_bc": np.eye(4).tolist(),
        "t_started": 9.0,
        "frames": [frame],
        "target": {"object": "cube"},
        "targets": [{"t": 9.8, "step": "pre-place 2", "pose": tip.tolist()}],
    }
    (act / "act.json").write_text(json.dumps(meta))
    return act


@pytest.fixture(scope="module")
def overlay():
    bench = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path.insert(0, str(bench))
    try:
        import act_overlay

        yield act_overlay
    finally:
        sys.path.remove(str(bench))


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(pregrasp.router)
    return TestClient(app)


def _near(img, x, y, colour, tol=40):
    return np.abs(img[y, x].astype(int) - np.array(colour)).max() <= tol


def _has(img, xs, y, colour):
    return any(_near(img, x, y, colour) for x in xs)


def test_a_replayed_frame_draws_what_the_act_recorded(overlay, tmp_path):
    """To judge an act afterwards from what it had: the masks and points it kept, where it held the place object as it
    recorded it, the fingertip and where it was told to go; where the act did not record the held pose, none is
    drawn rather than one rebuilt."""
    act = _act(tmp_path)
    meta = json.loads((act / "act.json").read_text())
    points = overlay.demo_surface(pathlib.Path(meta["demo_root"]), "cube")
    img = pregrasp._replay_draw(act, meta, 0, points)
    assert _near(img, 600, 330, pregrasp.REPLAY_PICKED), "the picked object's mask edge"
    assert _near(img, 620, 320, pregrasp.REPLAY_PICKED), "a picked point it saw"
    assert _near(img, 300, 215, pregrasp.REPLAY_ONTO), "the cube's mask edge"
    assert _has(img, range(396, 405), 215, pregrasp.FOUND_COLOUR), "where the act held the cube: moved 40 px"
    assert not _has(img, range(296, 305), 215, pregrasp.FOUND_COLOUR), "not where the demo's view had it"
    assert _near(img, 224, 373, (255, 255, 255)), "the fingertip"
    assert _near(img, 234, 373, pregrasp.REPLAY_TOLD), "where it was told to go"
    unrecorded = _act(tmp_path / "older", held=False)
    img = pregrasp._replay_draw(unrecorded, json.loads((unrecorded / "act.json").read_text()), 0, points)
    assert not _has(img, range(396, 405), 215, pregrasp.FOUND_COLOUR), "nothing rebuilt"
    with pytest.raises(pregrasp.HTTPException):
        pregrasp._replay_draw(act, meta, 1, points)


def test_the_replay_of_a_trial_lists_its_frames_and_serves_each_drawn(client, tmp_path, monkeypatch):
    """The Approach tab's replay asks by trial: the frames with their time and step, whether the act recorded where it
    held the place object, and each frame as a picture; a trial without a recording says so."""
    act = _act(tmp_path)
    monkeypatch.setattr(pregrasp, "_trials", [{"run": str(act)}, {"run": None}])
    r = client.get("/api/pregrasp/replay", params={"trial": 0})
    assert r.status_code == 200
    assert r.json() | {"result": None} == {
        "trial": 0,
        "n": 1,
        "frames": [{"i": 0, "t": 1.0, "step": "place"}],
        "result": None,
        "held_recorded": True,
    }
    r = client.get("/api/pregrasp/replay/frame.jpg", params={"trial": 0, "i": 0})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR).shape == (H, W, 3)
    assert client.get("/api/pregrasp/replay/frame.jpg", params={"trial": 0, "i": 1}).status_code == 404
    r = client.get("/api/pregrasp/replay", params={"trial": 1})
    assert r.status_code == 404 and r.json()["detail"] == "this act was not recorded"
    assert client.get("/api/pregrasp/replay", params={"trial": 2}).status_code == 404


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_an_act_replays_outside_the_gui_as_frames_and_a_video(overlay, tmp_path):
    act = _act(tmp_path)
    video = overlay.render(act, 5.0)
    assert video == act / "overlay.mp4" and video.stat().st_size > 0
    img = cv2.imread(str(act / "overlay" / "000000.jpg"))
    assert img.shape == (H, W, 3)
