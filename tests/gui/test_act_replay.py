import base64
import json
import pathlib
import shutil
import sys
import time

import cv2
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from lerobot.gui.api import manipulation
from lerobot.gui.api.manipulation.act_timeline import ActTimeline
from lerobot.showservo import groups_scene

INTR = {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0}
H, W = 480, 848  # the rig's camera
CUBE = (slice(200, 260), slice(300, 360))


def _act(tmp_path: pathlib.Path, held: bool = True, groups: bool = True, wired: bool = False) -> pathlib.Path:
    """One recorded frame of an act at the rig's size, the camera as the base: the picked object's mask and points,
    the cube's mask and points, where the act held the cube (moved 30 mm along x from the demo's view, 40 px at 0.45
    m) unless ``held`` is False, the fingertip and the target it was walked to, the point groups' picture (the cube's
    group 3 with its own track at (500, 120), a world track at (700, 100)) unless ``groups`` is False, and with
    ``wired`` their wireframe as the act's view recorded it (the cube's group linked 500-540 along y 120; the world
    700-740 along y 100, 600 mm deep on a 380-620 mm scale) and the surfaces it tinted (the cube's group's over x
    480-560, y 100-140); and the demo the cube was designated in, flat at 0.45 m."""
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
    (act / "camera" / "rgb").mkdir(parents=True)
    shutil.copy(act / "frames" / "000000.jpg", act / "camera" / "rgb" / "000000.jpg")
    (act / "camera" / "timeline.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "saved": 1,
                "rows": [
                    {
                        "slot": 0,
                        "t_mono": 0.0,
                        "t_wall": 10.0,
                        "camera_mono": 0.0,
                        "image": 0,
                        "status": "captured",
                    }
                ],
            }
        )
    )
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
    if groups:
        arrays["groups_points"] = np.array([[500, 120, 3, 1], [540, 120, 3, 1], [700, 100, 0, 1]], np.int32)
        frame["groups"] = {
            "stamp": 9.95,
            "base": 0,
            "quiet": [],
            "objects": {
                "cube": {"group": 3, "own": False, "why": "", "n_seen": 1, "n_grouped": 1, "own_points": [0]}
            },
            "groups": {"3": {"members": 2, "n_fit": 2, "rms_mm": 1.1, "supported": True}},
        }
        if wired:
            arrays["groups_points"] = np.vstack([arrays["groups_points"], [[740, 100, 0, 1]]]).astype(
                np.int32
            )
            arrays["groups_links"] = np.array([[0, 1], [2, 3]], np.int32)
            arrays["groups_joins"] = np.zeros((0, 2), np.int32)
            arrays["groups_depth"] = np.array([400, 410, 600, 600], np.uint16)
            frame["groups"]["depth_range"] = [380.0, 620.0]
            small = np.full((H // 4, W // 4), groups_scene.SURFACE_NONE, np.uint16)
            small[25:35, 120:140] = 3  # the cube's group's surface: x 480-560, y 100-140
            ok, png = cv2.imencode(".png", small)
            frame["groups"]["surfaces"], frame["groups"]["scale"] = (
                base64.b64encode(png.tobytes()).decode(),
                4,
            )
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
    app.include_router(manipulation.router)
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
    img = manipulation.replay._replay_draw(act, meta, 0, {"cube": points})
    assert _near(img, 600, 330, manipulation.runtime.REPLAY_COLOURS[0]), "the picked object's mask edge"
    assert _near(img, 620, 320, manipulation.runtime.REPLAY_COLOURS[0]), "a picked point it saw"
    assert _near(img, 300, 215, manipulation.runtime.REPLAY_COLOURS[1]), "the cube's mask edge"
    assert _has(img, range(396, 405), 215, manipulation.runtime.FOUND_COLOUR), (
        "where the act held the cube: moved 40 px"
    )
    assert not _has(img, range(296, 305), 215, manipulation.runtime.FOUND_COLOUR), (
        "not where the demo's view had it"
    )
    assert _near(img, 224, 373, (255, 255, 255)), "the fingertip"
    assert _near(img, 234, 373, manipulation.runtime.REPLAY_TOLD), "where it was told to go"
    cube_group = groups_scene.group_colour(3, 0, set())
    assert _near(img, 541, 120, cube_group), "an act from before the wireframes: the cube's group, as dots"
    assert not _near(img, 505, 120, (255, 255, 255)), "no ring: the live view drew none"
    assert _near(img, 700, 100, (255, 255, 255)) and not _near(img, 702, 100, (255, 255, 255)), (
        "the world, small"
    )
    unrecorded = _act(tmp_path / "older", held=False)
    img = manipulation.replay._replay_draw(
        unrecorded, json.loads((unrecorded / "act.json").read_text()), 0, {"cube": points}
    )
    assert not _has(img, range(396, 405), 215, manipulation.runtime.FOUND_COLOUR), "nothing rebuilt"
    with pytest.raises(HTTPException):
        manipulation.replay._replay_draw(act, meta, 1, {"cube": points})


MUG = (slice(100, 140), slice(150, 230))


def _partial_pose():
    """A learned surface at pixels 400..440, 240..280, moved 40 pixels right by its raw fit."""
    v, u = np.mgrid[240:281:3, 400:441:3]
    xyz = np.column_stack(
        [
            (u.ravel() - INTR["cx"]) * 0.45 / INTR["fx"],
            (v.ravel() - INTR["cy"]) * 0.45 / INTR["fy"],
            np.full(u.size, 0.45),
        ]
    )
    delta = np.eye(4)
    delta[0, 3] = 0.03
    return {"ok": True, "delta": delta, "model_xyz": xyz, "model_valid": np.ones(len(xyz), bool)}


def test_pose_outline_excludes_unconfirmed_points_without_filtering_occluded_surface():
    """A grasp added stray finite candidates to P2P's map. They are not an object outline;
    confirmed points still belong to the predicted surface even outside the visible mask."""
    r = _partial_pose()
    candidate = np.array([[(700 - INTR["cx"]) * 0.45 / INTR["fx"], 0.0075, 0.45]])
    r["model_xyz"] = np.vstack([r["model_xyz"], candidate])
    r["model_valid"] = np.r_[r["model_valid"], False]
    r["mask"] = np.zeros((H, W), bool)  # occlusion is not map invalidity
    original = r["model_xyz"].copy()
    img = np.zeros((H, W, 3), np.uint8)
    state = manipulation.rendering._draw_track_estimate(img, INTR, r, r, "picked")
    assert state == "available"
    assert _has(img, range(480, 485), 250, manipulation.runtime.POSE_ESTIMATE_COLOUR)
    assert not _has(img, range(735, 746), 250, manipulation.runtime.POSE_ESTIMATE_COLOUR)
    np.testing.assert_array_equal(r["model_xyz"], original), "retain raw candidate geometry for diagnosis"


@pytest.mark.parametrize("validity", [None, np.array([True]), np.zeros(196, bool)])
def test_pose_outline_reports_missing_or_unusable_surface_validity(validity):
    r = _partial_pose()
    r["model_valid"] = validity
    img = np.zeros((H, W, 3), np.uint8)
    state = manipulation.rendering._draw_track_estimate(img, INTR, r, r, "picked")
    if validity is None:
        assert state == "available (surface validity unrecorded)"
        assert _has(img, range(480, 485), 250, manipulation.runtime.POSE_ESTIMATE_COLOUR)
    else:
        expected = (
            "surface unavailable" if len(validity) == len(r["model_xyz"]) else "surface validity invalid"
        )
        assert state == f"available ({expected})"
        assert not _has(img, range(480, 485), 250, manipulation.runtime.POSE_ESTIMATE_COLOUR)


@pytest.mark.parametrize("used", [False, True])
def test_replay_shows_partial_pose_separately_from_following_acceptance(tmp_path, monkeypatch, used):
    """A held object can have a fit despite insufficient visibility for following; show its predicted surface,
    using the model and raw transform from that frame, independently of the mask, demo or accepted transform."""
    act = _act(tmp_path, held=False, groups=False)
    meta = json.loads((act / "act.json").read_text())
    meta["frames"][0].update(ok=True, used=used, fit_inliers=18, fit_points=25)
    arrays = dict(np.load(act / "frames" / "000000.npz"))
    arrays.update({k: v for k, v in _partial_pose().items() if k != "ok"}, delta_used=np.eye(4))
    np.savez(act / "frames" / "000000.npz", **arrays)
    labels = []
    monkeypatch.setattr(manipulation.replay, "_replay_text", lambda _img, text, *_args: labels.append(text))
    img = manipulation.replay._replay_draw(act, meta, 0, {"picked": arrays["model_xyz"] + 10})
    assert _has(img, range(480, 485), 250, manipulation.runtime.POSE_ESTIMATE_COLOUR), (
        "raw pose's projected surface"
    )
    assert not _has(img, range(398, 405), 250, manipulation.runtime.POSE_ESTIMATE_COLOUR), (
        "no stale accepted pose"
    )
    assert any("pose estimate: available; fit inliers 18 / 25" in s for s in labels)
    assert any(f"following {'accepted' if used else 'rejected'}" in s for s in labels)


@pytest.mark.parametrize("missing", ["ok", "delta", "model_xyz", "bad_delta"])
def test_replay_does_not_invent_a_missing_current_pose_or_surface(tmp_path, monkeypatch, missing):
    act = _act(tmp_path, held=False, groups=False)
    meta = json.loads((act / "act.json").read_text())
    r = _partial_pose()
    meta["frames"][0]["ok"] = missing != "ok"
    if missing == "bad_delta":
        r["delta"][0, 0] = np.nan
    else:
        r.pop(missing)
    np.savez(act / "frames" / "000000.npz", **{k: v for k, v in r.items() if k != "ok"})
    labels = []
    monkeypatch.setattr(manipulation.replay, "_replay_text", lambda _img, text, *_args: labels.append(text))
    img = manipulation.replay._replay_draw(act, meta, 0, {"picked": _partial_pose()["model_xyz"]})
    assert not _has(img, range(480, 485), 250, manipulation.runtime.POSE_ESTIMATE_COLOUR)
    state = "available (surface unavailable)" if missing == "model_xyz" else "unavailable"
    assert any(f"pose estimate: {state};" in s for s in labels)


def test_live_view_shows_a_partial_pose_when_following_rejects_it(monkeypatch):
    r = _partial_pose()
    teach = manipulation.runtime._Teach(
        at="now",
        box=(0, 0, 0, 0),
        rgb=np.zeros((H, W, 3), np.uint8),
        depth_m=np.full((H, W), 0.45),
        intr=INTR,
        keypoints={"concept": "picked", "xyz": r["model_xyz"]},
    )
    for key in ("demo", "test"):
        monkeypatch.setattr(manipulation.runtime._state, key, None)
    monkeypatch.setattr(manipulation.groups, "_paint_groups", lambda img: img)
    labels = []
    monkeypatch.setattr(manipulation.replay, "_replay_text", lambda _img, text, *_args: labels.append(text))
    jpeg = manipulation.rendering._render_live(teach.rgb, r, None, None, teach, {"state": "occluded"})
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert _has(img, range(480, 485), 250, manipulation.runtime.POSE_ESTIMATE_COLOUR)
    assert any("Blue raw P2P outline: available; control pose: rejected" in s for s in labels)


def test_a_replayed_frame_draws_every_object_the_act_followed_by_name(overlay, tmp_path):
    """An act on three objects records the others' names in the order their arrays are numbered and where it held
    each: the replay draws each in its own colour by name (the frame's order is not the act's), and each where the act
    held it, moved from the demo's view."""
    act = _act(tmp_path, held=False, groups=False)
    demo = pathlib.Path(json.loads((act / "act.json").read_text())["demo_root"])
    cube, mug = np.zeros((H, W), bool), np.zeros((H, W), bool)
    cube[CUBE], mug[MUG] = True, True
    np.savez(
        demo / "objects.npz",
        names=json.dumps([["cube", 0, [330, 230], 4], ["mug", 0, [190, 120], 1]]),
        o0_mask=cube,
        o1_mask=mug,
    )
    z = dict(np.load(act / "frames" / "000000.npz"))
    moved = {name: np.eye(4) for name in ("cube", "mug")}
    moved["cube"][0, 3], moved["mug"][1, 3] = 0.03, 0.03  # 40 px right; 40 px down
    z.update(other_mask_0=mug, other_mask_1=cube, held_0=moved["cube"], held_1=moved["mug"])
    for key in ("other_track_uv_0", "other_track_vis_0"):
        z.pop(key)
    np.savez(act / "frames" / "000000.npz", **z)
    meta = json.loads((act / "act.json").read_text())
    meta["objects"] = ["gamepad", "cube", "mug"]
    meta["frames"][0]["others"] = ["mug", "cube"]
    meta["frames"][0]["held"] = [{"object": "cube", "find_id": "f1"}, {"object": "mug", "find_id": "f2"}]
    points = {name: overlay.demo_surface(demo, name) for name in ("cube", "mug")}
    img = manipulation.replay._replay_draw(act, meta, 0, points)
    assert _near(img, 600, 330, manipulation.runtime.REPLAY_COLOURS[0]), "the session's object, first colour"
    assert _near(img, 300, 215, manipulation.runtime.REPLAY_COLOURS[1]), (
        "the cube's mask edge in its own colour"
    )
    assert _near(img, 150, 120, manipulation.runtime.REPLAY_COLOURS[2]), (
        "the mug's mask edge in its own colour"
    )
    assert _has(img, range(396, 405), 215, manipulation.runtime.FOUND_COLOUR), (
        "where the act held the cube: moved 40 px right"
    )
    assert _has(img, range(186, 195), 180, manipulation.runtime.FOUND_COLOUR) or _has(
        img, range(150, 230), 179, manipulation.runtime.FOUND_COLOUR
    ), "where the act held the mug: moved 40 px down"
    assert manipulation.replay._replay_objects(meta) == ["gamepad", "cube", "mug"]
    del meta["objects"]
    assert manipulation.replay._replay_objects(meta) == ["picked", "cube"], (
        "an act recorded before: its one place object"
    )


def test_a_replayed_frame_draws_the_groups_wireframe_the_act_recorded(overlay, tmp_path):
    """An act recorded with the groups' wireframe replays it as the camera view painted it: each link coloured by its
    depth on the recorded scale, a group that moved differently tinted over its surface, and no dot or ring at a
    track."""
    act = _act(tmp_path, wired=True)
    meta = json.loads((act / "act.json").read_text())
    img = manipulation.replay._replay_draw(
        act, meta, 0, {"cube": overlay.demo_surface(pathlib.Path(meta["demo_root"]), "cube")}
    )
    far = groups_scene.depth_colours([600.0], 380.0, 620.0)[0]
    assert _near(img, 720, 100, far), "the world's link, coloured by its depth"
    cube_group = np.array(groups_scene.group_colour(3, 0, set()), int)
    assert _near(img, 520, 135, (cube_group + 60) // 2, tol=4), "the cube's group's surface, tinted"
    assert not _near(img, 500, 115, (255, 255, 255)), "no ring around its own track"
    assert not _near(img, 500, 123, cube_group), "no dot: the cube's group was drawn as 3 px ones"


def test_a_replay_paints_the_point_groups_exactly_as_the_live_view_did(overlay, tmp_path):
    """The replay and the live view paint a picture with one function: where nothing else is drawn, a replayed frame
    is pixel for pixel the live view's painting of the picture the act recorded, tint and wireframe."""
    act = _act(tmp_path, wired=True)
    meta = json.loads((act / "act.json").read_text())
    z = np.load(act / "frames" / "000000.npz")
    g = meta["frames"][0]["groups"]
    live_picture = {  # what the live view had: the feed's picture, lists as they come over the wire
        **{k: g[k] for k in ("base", "quiet", "surfaces", "scale", "depth_range")},
        "points": z["groups_points"].tolist(),
        "links": z["groups_links"].tolist(),
        "joins": [],
        "depth": z["groups_depth"].tolist(),
    }
    raw = cv2.imread(str(act / "frames" / "000000.jpg"))
    live = manipulation.groups._paint_picture(raw.copy(), live_picture)
    replayed = manipulation.replay._replay_draw(
        act, meta, 0, {"cube": overlay.demo_surface(pathlib.Path(meta["demo_root"]), "cube")}
    )
    for ys, xs in (
        (slice(95, 106), slice(690, 751)),
        (slice(100, 141), slice(480, 561)),
        (slice(120, 360), slice(828, 848)),
    ):
        assert np.array_equal(live[ys, xs], replayed[ys, xs]), (ys, xs)


def test_the_replay_of_a_trial_lists_its_frames_and_serves_each_drawn(client, tmp_path, monkeypatch):
    """The Approach tab's replay asks by trial: the frames with their time and step, whether the act recorded where it
    held the place object, and each frame as a picture; a trial without a recording says so."""
    act = _act(tmp_path)
    monkeypatch.setattr(manipulation.runtime, "_trials", [{"run": str(act)}, {"run": None}])
    r = client.get("/api/pregrasp/replay", params={"trial": 0})
    assert r.status_code == 200
    assert r.json() | {"result": None} == {
        "trial": 0,
        "n": 1,
        "frames": [{"i": 0, "t": 0.0, "step": "place"}],
        "fps": 30.0,
        "result": None,
        "control_cadence": None,
        "held_recorded": True,
        "pose_verification_recorded": False,
        "pose_disagreement_frames": [],
    }
    r = client.get("/api/pregrasp/replay/frame.jpg", params={"trial": 0, "i": 0})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    drawn = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert drawn.shape == (H, W, 3)
    r = client.get("/api/pregrasp/replay/frame.jpg", params={"trial": 0, "i": 0, "pair": 1})
    pair = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR).astype(float)
    camera = cv2.imread(str(act / "camera" / "rgb" / "000000.jpg")).astype(float)
    assert pair.shape == (H, 2 * W, 3), (
        "the camera's frame and the drawn one side by side, as the export shows them"
    )
    assert np.abs(pair[:, :W] - camera).mean() < 3.0 and np.abs(pair[:, W:] - drawn).mean() < 3.0, (
        "as the camera saw it on the left, drawn on the right (to JPEG's rounding)"
    )
    assert client.get("/api/pregrasp/replay/frame.jpg", params={"trial": 0, "i": 1}).status_code == 404
    r = client.get("/api/pregrasp/replay", params={"trial": 1})
    assert r.status_code == 404 and r.json()["detail"] == "this act was not recorded"
    assert client.get("/api/pregrasp/replay", params={"trial": 2}).status_code == 404


def test_replay_requires_the_fixed_rate_timeline(client, tmp_path, monkeypatch):
    act = _act(tmp_path)
    (act / "camera" / "timeline.json").unlink()
    monkeypatch.setattr(manipulation.runtime, "_trials", [{"run": str(act)}])
    for path in ("/api/pregrasp/replay", "/api/pregrasp/replay/frame.jpg", "/api/pregrasp/replay/video.mp4"):
        r = client.get(path, params={"trial": 0, "i": 0})
        assert r.status_code == 404
        assert r.json()["detail"] == "this act has no fixed-rate camera timeline"


def test_act_recorder_rejects_missing_camera_before_recording(monkeypatch):
    from lerobot.gui.api import jog, showservo

    monkeypatch.setattr(showservo, "live_camera", lambda: None)
    monkeypatch.setattr(jog, "start_record", lambda: pytest.fail("arm recording started without a camera"))
    with pytest.raises(RuntimeError, match="fixed-rate Act recorder needs a live camera"):
        manipulation.recording._begin_run(None, 1.0, np.eye(4), np.eye(4), {})


def _second_frame(act: pathlib.Path, t: float) -> None:
    """A second asynchronous result and camera image on the fixed 30 Hz timeline."""
    for suffix in (".jpg", "_mask.png", ".npz"):
        shutil.copy(act / "frames" / f"000000{suffix}", act / "frames" / f"000001{suffix}")
    meta = json.loads((act / "act.json").read_text())
    meta["frames"].append({**meta["frames"][0], "i": 1, "t_frame": t, "step": "settling"})
    (act / "act.json").write_text(json.dumps(meta))
    timeline_path = act / "camera" / "timeline.json"
    timeline = json.loads(timeline_path.read_text())
    last = round((t - 10.0) * 30)
    shutil.copy(act / "frames" / "000001.jpg", act / "camera" / "rgb" / f"{last:06d}.jpg")
    timeline["rows"].extend(
        {
            "slot": i,
            "t_mono": i / 30,
            "t_wall": 10.0 + i / 30,
            "camera_mono": i / 30 if i == last else 0.0,
            "image": last if i == last else 0,
            "status": "captured" if i == last else "held_camera_frame",
        }
        for i in range(1, last + 1)
    )
    timeline["saved"] = 2
    timeline_path.write_text(json.dumps(timeline))


def test_replay_indexes_disagreements_and_displays_saved_measurements_below_the_scene(
    client, tmp_path, monkeypatch
):
    act = _act(tmp_path)
    _second_frame(act, 10.5)
    meta = json.loads((act / "act.json").read_text())
    meta["frames"][1]["pose_verification"] = [
        {
            "object": "cube",
            "accepted": True,
            "disagreement": True,
            "groups": {"state": "disagreement", "rotation_deg": 12.0, "centre_mm": 0.2},
            "surface": {"state": "inconclusive"},
        }
    ]
    (act / "act.json").write_text(json.dumps(meta))
    monkeypatch.setattr(manipulation.runtime, "_trials", [{"run": str(act)}])
    data = client.get("/api/pregrasp/replay", params={"trial": 0}).json()
    assert data["pose_verification_recorded"] and data["pose_disagreement_frames"] == [15]
    captured = []
    original = manipulation.replay._replay_text

    def capture(im, text, *args):
        captured.append(text)
        return original(im, text, *args)

    monkeypatch.setattr(manipulation.replay, "_replay_text", capture)
    rows = json.loads((act / "camera" / "timeline.json").read_text())["rows"]
    pictures = [manipulation.replay._timeline_pair(act, meta, rows[i], {}) for i in (0, 15)]
    assert pictures[0].shape == pictures[1].shape
    assert pictures[0].shape[0] > H and pictures[0].shape[1] == 2 * W
    assert any("DISAGREEMENT 12.0 deg / 0.2 mm; P2P accepted" in t for t in captured)
    assert "Pose verification: not recorded" in captured
    assert any("RGB-D refit: inconclusive" in t for t in captured)
    raw = cv2.imread(str(act / "camera/rgb/000015.jpg"))
    np.testing.assert_array_equal(pictures[1][:H, :W], raw)
    assert np.any(pictures[1][H:, W:, 2] > 240), "the warning is visible below the image"


def _frame_count(path: pathlib.Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    return n


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_trials_replay_exports_as_a_video_at_the_acts_own_pace(client, tmp_path, monkeypatch):
    """To keep or share an act's replay: every frame as the camera saw it, and beside it with what the replay draws on
    it, each shown from its own time until the next one's, so the video plays at the act's pace; a trial without a
    recording has none."""
    act = _act(tmp_path)
    _second_frame(act, 10.5)  # half a second after the first
    monkeypatch.setattr(manipulation.runtime, "_trials", [{"run": str(act)}, {"run": None}])
    r = client.get("/api/pregrasp/replay/video.mp4", params={"trial": 0})
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert 'filename="act_act_replay.mp4"' in r.headers["content-disposition"]
    got = tmp_path / "got.mp4"
    got.write_bytes(r.content)
    assert _frame_count(got) == 16, "half a second, at the act's pace"
    first = cv2.VideoCapture(str(got)).read()[1]
    assert first.shape == (H, 2 * W, 3), "the camera's frame and the drawn one side by side"
    assert _near(first, 600, 330, (60, 60, 60), tol=10), (
        "on the left as the camera saw it, the mask edge not drawn"
    )
    assert _near(first, W + 600, 330, manipulation.runtime.REPLAY_COLOURS[0], tol=60), (
        "on the right drawn as replayed"
    )
    assert client.get("/api/pregrasp/replay/video.mp4", params={"trial": 1}).status_code == 404


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_an_act_replay_exports_outside_the_gui_too(overlay, tmp_path):
    act = _act(tmp_path)
    video = overlay.render(act)
    assert video == act / "replay.mp4" and _frame_count(video) == 1


def test_act_timeline_samples_a_fixed_clock_without_waiting_for_inference(tmp_path):
    class Camera:
        def read_newer_color_and_depth(self, _after):
            # A slower asynchronous camera may repeat an image across timeline
            # ticks; each row still belongs to the fixed main clock.
            frame = int(time.perf_counter() * 12)
            return (
                np.full((24, 32, 3), frame % 255, np.uint8),
                np.full((24, 32), 400 + frame % 10, np.uint16),
                frame / 12,
            )

    timeline = ActTimeline(tmp_path / "camera", Camera(), 30)
    timeline.start()
    time.sleep(0.24)
    summary = timeline.close()
    rows = json.loads((tmp_path / "camera" / "timeline.json").read_text())["rows"]
    assert 5 <= len(rows) <= 12
    assert [r["slot"] for r in rows] == list(range(len(rows)))
    assert np.allclose(np.diff([r["t_mono"] for r in rows]), 1 / 30)
    assert 1 < summary["saved"] < len(rows), "camera frames are asynchronous to the main clock"
    assert not summary["errors"]
    assert len(list((tmp_path / "camera" / "rgb").glob("*.jpg"))) == summary["saved"]


def test_timeline_draws_the_arm_at_the_camera_publication_time(tmp_path, monkeypatch):
    act = _act(tmp_path, groups=False)
    meta = json.loads((act / "act.json").read_text())
    meta["frames"][0]["t_result"] = 10.0
    early, late = np.eye(4), np.eye(4)
    late[0, 3] = 0.04
    np.savez(
        act / "arm.npz",
        t=np.array([10.0, 10.04]),
        t_mono=np.array([0.03, 0.07]),
        tip_obs=np.stack([early, late]),
    )
    tips = []
    monkeypatch.setattr(
        manipulation.rendering, "_draw_tool", lambda _img, _bc, _intr, tip, _label: tips.append(tip)
    )
    row = {
        "slot": 0,
        "t_mono": 0.05,
        "t_wall": 10.02,
        "camera_mono": 0.04,
        "image": None,
        "status": "held_camera_frame",
    }
    manipulation.replay._timeline_pair(act, meta, row, {})
    assert len(tips) == 1
    assert tips[0][0, 3] == pytest.approx(0.01), "the arm follows the image timestamp, not the later slot"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_act_export_uses_each_fixed_timeline_camera_frame(overlay, client, tmp_path, monkeypatch):
    act = _act(tmp_path)
    camera = act / "camera" / "rgb"
    camera.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        cv2.imwrite(str(camera / f"{i:06d}.jpg"), np.full((H, W, 3), 20 + 20 * i, np.uint8))
    (act / "camera" / "timeline.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "saved": 6,
                "rows": [
                    {
                        "slot": i,
                        "t_mono": i / 30,
                        "t_wall": 10 + i / 30,
                        "image": i,
                        "camera_mono": i / 30,
                        "status": "captured",
                    }
                    for i in range(6)
                ],
            }
        )
    )
    monkeypatch.setattr(manipulation.runtime, "_trials", [{"run": str(act)}])
    listing = client.get("/api/pregrasp/replay", params={"trial": 0})
    assert listing.status_code == 200
    assert listing.json()["n"] == 6 and listing.json()["fps"] == 30
    assert listing.json()["frames"][-1]["t"] == pytest.approx(5 / 30, abs=0.001)
    frame = client.get("/api/pregrasp/replay/frame.jpg", params={"trial": 0, "i": 5, "pair": 1})
    assert frame.status_code == 200
    pair = cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_COLOR)
    assert abs(int(pair[100, 100, 0]) - 120) <= 3
    video = overlay.render(act)
    assert _frame_count(video) == 6, "the video follows the fixed clock, not the single P2P frame"
    cap = cv2.VideoCapture(str(video))
    first, last = None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        first = frame if first is None else first
        last = frame
    assert first is not None and last is not None
    assert int(last[100, 100, 0]) - int(first[100, 100, 0]) > 80, "real camera motion survives the export"


@pytest.mark.parametrize("selected_at, expected", [(9.0, True), (11.0, False)])
def test_replay_labels_prediction_only_after_it_was_selected(tmp_path, monkeypatch, selected_at, expected):
    act = _act(tmp_path, held=False, groups=False)
    meta = json.loads((act / "act.json").read_text())
    meta["hold_selection"] = {
        "at": selected_at,
        "source": "grasp prediction",
        "evidence": "predicted",
    }
    labels = []
    monkeypatch.setattr(manipulation.replay, "_replay_text", lambda _img, text, *_args: labels.append(text))
    manipulation.replay._replay_draw(act, meta, 0, {})
    assert (
        any("grasp prediction" in s and "no post-lift visual verification" in s for s in labels) is expected
    )
