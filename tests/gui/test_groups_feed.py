"""The point groups in the act's own flow and camera view: the view's picture painted on the act's camera frames
(groups_scene.drawing and paint_groups), the server's feed of it (each object's poses from its designation on), an
object designated as soon as the act's tracker finds it, and the camera view drawing the act's pose of an object no
view places, carried by the groups.

The view is the real one's server (benchmarks/group_live.py, ``MjpegView``) with its loop played here, as in
tests/gui/test_act_with_groups.py."""

from __future__ import annotations

import asyncio
import collections
import time

import numpy as np

from lerobot.gui.api import manipulation
from lerobot.showservo import groups_scene
from lerobot.showservo.groups import GroupTracker
from lerobot.showservo.wireframe import Wireframe
from tests.gui.test_act_with_groups import _free_port, _ViewLoop, group_live  # noqa: F401  (a fixture)
from tests.gui.test_manipulation_motion import INTR

K = np.array([[INTR["fx"], 0.0, INTR["cx"]], [0.0, INTR["fy"], INTR["cy"]], [0.0, 0.0, 1.0]])


def _pose(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


def test_the_groups_picture_is_painted_on_another_camera_frame_as_the_view_draws_it():
    """A tray of points, still, and a block of points lifted off it: the view's picture (drawing) carries every
    standing track with its group and the block's surface as a label map, and paint_groups puts them on a frame of
    the same camera in the groups' colours: the world white and untinted, the block tinted in its own colour."""
    rng = np.random.default_rng(3)
    tray = np.column_stack([rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.12, 0.12, 200), np.full(200, 0.45)])
    block = np.column_stack([rng.uniform(0.04, 0.08, 40), rng.uniform(0.0, 0.03, 40), np.full(40, 0.42)])
    t = GroupTracker()
    for k in range(60):  # the block lifts from frame 20, 1.5 mm a frame towards the camera
        lift = np.array([0.0, 0.0, -0.0015 * max(k - 20, 0)])
        xyz = np.vstack([tray, block + lift])
        t.update(xyz + rng.normal(0.0, 0.0015, xyz.shape), np.ones(240, bool))
    block_group = int(np.bincount(t.group_of[200:][t.group_of[200:] >= 0]).argmax())
    world = groups_scene.base_group(t.group_of)
    assert block_group != world, "the lifted block is a group of its own"
    uv = groups_scene.project(xyz, K)
    surfaces = np.full((480, 848), -1, dtype=np.int32)
    u0, v0 = uv[200:].min(axis=0).astype(int)
    u1, v1 = uv[200:].max(axis=0).astype(int)
    surfaces[v0 : v1 + 1, u0 : u1 + 1] = block_group  # the block's surface, as group_surfaces would paint it

    d = groups_scene.drawing(K, t, xyz, np.ones(240, bool), {}, surfaces, world)
    assert len(d["points"]) == 240 and {p[2] for p in d["points"]} == {world, block_group}
    painted = groups_scene.paint_groups(np.zeros((480, 848, 3), np.uint8), d)
    colour = np.array(groups_scene.group_colour(block_group, world), dtype=int)
    inside = painted[(v0 + v1) // 2, (u0 + u1) // 2].astype(int)
    assert np.abs(inside - colour // 2).max() <= 2, (inside, colour)  # half tint over a black frame
    u, v = (int(x) for x in uv[0])  # a tray track: white, on an untinted world
    assert (painted[v, u] == 255).all()
    assert (painted[5, 5] == 0).all(), "nothing painted where no group is"


def test_the_groups_picture_carries_each_groups_wireframe_and_the_view_paints_it_by_depth():
    """The tracks drawn as dots read as clusters of small round marks that blink and jitter; given a wireframe every
    frame, the picture carries each group's links as row pairs into its points (one group's tracks only), each row's
    depth and the colour scale, and paint_groups draws the links coloured by depth, with the scale at the right edge,
    instead of a dot per track."""
    rng = np.random.default_rng(3)
    tray = np.column_stack([rng.uniform(-0.2, 0.2, 400), rng.uniform(-0.12, 0.12, 400), np.full(400, 0.45)])
    block = np.column_stack([rng.uniform(0.04, 0.08, 80), rng.uniform(0.0, 0.03, 80), np.full(80, 0.42)])
    t, wire = GroupTracker(), Wireframe()
    for k in range(60):  # the block lifts from frame 20, 1.5 mm a frame towards the camera
        lift = np.array([0.0, 0.0, -0.0015 * max(k - 20, 0)])
        xyz = np.vstack([tray, block + lift]) + rng.normal(0.0, 0.0015, (480, 3))
        t.update(xyz, np.ones(480, bool))
        d = groups_scene.drawing(
            K, t, xyz, np.ones(480, bool), {}, None, groups_scene.base_group(t.group_of), wire=wire
        )
    links, pts = np.asarray(d["links"]), np.asarray(d["points"])
    assert len(links) > 300 and (pts[links[:, 0], 2] == pts[links[:, 1], 2]).all(), "links within a group"
    depth = np.asarray(d["depth"])
    assert len(depth) == len(pts) and abs(np.median(depth[:400]) - 450) < 5, "each row's depth, mm"
    near, far = d["depth_range"]
    assert 380 < near < far < 470, d["depth_range"]
    painted = groups_scene.paint_groups(np.zeros((480, 848, 3), np.uint8), d)
    mid = ((pts[links[:, 0], :2] + pts[links[:, 1], :2]) / 2).round().astype(int)
    assert (painted[mid[:, 1], mid[:, 0]].max(axis=1) > 40).mean() > 0.9, "the links are drawn"
    assert not (painted[:, :-90] == 255).all(axis=2).any(), "no white dot: the world's tracks were white dots"
    assert painted[240, 848 - 13].max() > 100, "the depth scale at the right edge"


def test_the_groups_picture_says_what_carries_each_object():
    """After an act whose point groups carried the cube short of a pushed tray, nothing on record said what had carried
    it. The picture an act keeps says, per object, the group carrying it, whether its own points placed it, and which
    of the picture's points are its own tracks; and per group, its members and the fit its motion came from."""
    rng = np.random.default_rng(3)
    tray = np.column_stack([rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.12, 0.12, 200), np.full(200, 0.45)])
    block = np.column_stack([rng.uniform(0.04, 0.08, 40), rng.uniform(0.0, 0.03, 40), np.full(40, 0.42)])
    t = GroupTracker()
    t.add_object("block", np.arange(200, 240), _pose(0.06, 0.015, 0.42))
    for k in range(60):  # the block lifts from frame 20, 1.5 mm a frame towards the camera
        lift = np.array([0.0, 0.0, -0.0015 * max(k - 20, 0)])
        xyz = np.vstack([tray, block + lift])
        t.update(xyz + rng.normal(0.0, 0.0015, xyz.shape), np.ones(240, bool))
    world = groups_scene.base_group(t.group_of)
    corners = np.array([[0.04, 0.0, 0.42], [0.08, 0.0, 0.42], [0.08, 0.03, 0.42], [0.04, 0.03, 0.42]])
    objects = {"block": (corners, corners.mean(axis=0), None)}
    d = groups_scene.drawing(K, t, xyz, np.ones(240, bool), objects, None, world)
    o = d["objects"]["block"]
    assert o["group"] == t.objects["block"].group != world, "carried by its own group, not the world's"
    assert sorted(o["own_points"]) == list(range(200, 240)), "its own tracks are the picture's rows 200..239"
    assert o["n_seen"] == 40 and isinstance(o["own"], bool) and isinstance(o["why"], str)
    g = d["groups"][str(o["group"])]
    assert g["members"] == int((t.group_of == o["group"]).sum()) and 0 < g["n_fit"] <= g["members"]
    assert d["groups"][str(world)]["members"] >= 190 and g["rms_mm"] >= 0.0


def test_the_feed_takes_each_objects_poses_from_its_designation_on_and_the_newest_picture(
    tmp_path,
    monkeypatch,
    group_live,  # noqa: F811
):
    """Designated through the feed, an object's poses count from the frame after its designation; the camera view
    paints the newest picture while it is fresh and acts use the point groups, and leaves the frame alone otherwise."""
    port = _free_port()
    monkeypatch.setattr(manipulation.runtime, "GROUPS_VIEW", ("127.0.0.1", port))
    monkeypatch.setattr(
        manipulation.runtime._state, "groups", manipulation.runtime._GroupsView()
    )  # a view started elsewhere
    monkeypatch.setattr(manipulation.runtime._state, "groups_feed", manipulation.runtime._GroupsFeed())
    monkeypatch.setattr(manipulation.runtime._state, "groups_with_acts", True)
    view = group_live.MjpegView(port, tmp_path / "recordings")
    picture = {  # a world track, a track in no group, and an object's outline
        "points": [[100, 100, 0, 1], [300, 300, -1, 1]],
        "surfaces": None,
        "scale": 4,
        "base": 0,
        "quiet": [],
        "objects": {
            "cube": {"outline": [[200, 50], [260, 50], [260, 110]], "colour": [230, 25, 75], "own": False}
        },
    }
    view.add_poses(
        time.time(), {"cube": _pose(0.0, 0.0, 0.5).ravel().tolist()}, picture
    )  # an earlier designation
    loop = _ViewLoop(view, lambda: _pose(0.1, 0.0, 0.43))

    async def run():
        why, since = await manipulation.groups._designate_in_groups({"cube": [425, 250]}, None, lambda: False)
        assert why == "", why
        for _ in range(100):
            await asyncio.sleep(0.02)
            if len(manipulation.runtime._state.groups_feed.frames.get("cube", ())) >= 3:
                break
        return since

    try:
        since = asyncio.run(run())
        feed = manipulation.runtime._state.groups_feed
        frames = list(feed.frames["cube"])
        assert len(frames) >= 3 and all(t > since for t, _ in frames), "only frames after the designation"
        assert all(np.allclose(m, _pose(0.1, 0.0, 0.43)) for _, m in frames), (
            "none of the earlier designation's"
        )
        with manipulation.runtime._state.lock:
            feed.drawing = {**picture, "stamp": time.time()}
        painted = manipulation.groups._paint_groups(np.zeros((480, 848, 3), np.uint8))
        assert (painted[100, 100] == 255).all(), "a fresh picture is painted"
        # Cleaner than the debug view's: no track in no group, no outline (the act draws its own pose of each object).
        debug = groups_scene.paint_groups(np.zeros((480, 848, 3), np.uint8), feed.drawing)
        assert debug[300, 300].any() and debug[50, 230].any(), "the debug view draws both"
        assert not painted[300, 300].any() and not painted[50, 230].any(), "the act's view neither"
        with manipulation.runtime._state.lock:
            feed.drawing = {**picture, "stamp": time.time() - 2 * manipulation.runtime.GROUPS_DRAW_MAX_AGE_S}
        assert not manipulation.groups._paint_groups(np.zeros((480, 848, 3), np.uint8)).any(), (
            "a stale one is not"
        )
        with manipulation.runtime._state.lock:
            feed.drawing = {**picture, "stamp": time.time()}
        monkeypatch.setattr(manipulation.runtime._state, "groups_with_acts", False)
        assert not manipulation.groups._paint_groups(np.zeros((480, 848, 3), np.uint8)).any(), (
            "nor any with the groups off"
        )
    finally:
        loop.close()


def test_an_object_the_tracker_finds_is_designated_in_the_point_groups(tmp_path, monkeypatch, group_live):  # noqa: F811
    """The place object's locate puts it in the point groups as soon as it is stored, at the deepest pixel of its
    mask, so the camera view shows it carried before any act; while an act runs, the act does that itself."""
    port = _free_port()
    monkeypatch.setattr(manipulation.runtime, "GROUPS_VIEW", ("127.0.0.1", port))
    monkeypatch.setattr(manipulation.runtime._state, "groups", manipulation.runtime._GroupsView())
    monkeypatch.setattr(manipulation.runtime._state, "groups_feed", manipulation.runtime._GroupsFeed())
    monkeypatch.setattr(manipulation.runtime._state, "groups_with_acts", True)
    monkeypatch.setattr(manipulation.runtime._state, "located", {})
    monkeypatch.setattr(manipulation.runtime._state, "targets", {})
    monkeypatch.setattr(manipulation.runtime._state, "act", manipulation.runtime._Act())
    view = group_live.MjpegView(port, tmp_path / "recordings")
    loop = _ViewLoop(view, lambda: _pose(0.1, 0.0, 0.43))
    mask = np.zeros((480, 848), bool)
    mask[230:270, 400:460] = True

    async def run(acting: bool):
        manipulation.runtime._state.act.on = acting
        manipulation.acquisition._store_located(
            "cube", {"object": "cube", "ok": True, "delta": np.eye(4), "mask": mask}
        )
        await asyncio.gather(*manipulation.runtime._groups_follows)

    try:
        asyncio.run(run(acting=True))
        assert "cube" not in view.designated, "an act designates its objects itself"
        asyncio.run(run(acting=False))
        assert view.designated.get("cube", {}).get("ok"), view.designated
        assert manipulation.runtime._state.groups_feed.reason == ""
        asked = view.received
        assert asked == 1, "one request, for the find"
    finally:
        loop.close()


def test_the_camera_view_draws_the_acts_pose_of_an_object_no_view_places_carried_by_the_groups(monkeypatch):
    """A frame the tracker does not trust (the wrist over the gamepad) drew nothing of the gamepad; the view now draws
    where the act holds it: its last placing view, moved on by the point groups, here 20 mm since that view."""
    rgb = np.full((480, 848, 3), 128, np.uint8)  # plain grey: any yellow is the drawn cloud
    depth = np.full((480, 848), 0.45, np.float32)
    xyz = np.array([[x, y, 0.43] for x in np.linspace(-0.03, 0.03, 7) for y in np.linspace(-0.02, 0.02, 5)])
    teach = manipulation.runtime._Teach(
        at="t", box=(0, 0, 0, 0), rgb=rgb, depth_m=depth, intr=INTR,
        keypoints={"mode": "features", "concept": "gamepad", "n_points": len(xyz), "xyz": xyz},
    )  # fmt: skip
    t_view = time.time() - 1.0
    feed = manipulation.runtime._GroupsFeed()
    feed.frames["gamepad"] = collections.deque(
        [(t_view, _pose(0.0, 0.0, 0.43)), (time.time(), _pose(0.02, 0.0, 0.43))]
    )
    monkeypatch.setattr(manipulation.runtime._state, "groups_feed", feed)
    monkeypatch.setattr(manipulation.runtime._state, "groups_with_acts", True)
    monkeypatch.setattr(manipulation.runtime._state, "demo", None)
    monkeypatch.setattr(
        manipulation.runtime._state,
        "test",
        manipulation.runtime._Test(at="t", rgb=rgb, result={"ok": True, "delta_cam": np.eye(4)}),
    )
    monkeypatch.setattr(manipulation.runtime._state.track, "history", [(t_view, True, np.eye(4))])

    def yellow_at(img, pts):
        uv = groups_scene.project(pts, K).astype(int)
        hits = [img[v, u] for u, v in uv if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]]
        return np.mean(
            [(b < 80) and (g > 160) and (r > 200) for b, g, r in hits]
        )  # the cloud's (0, 220, 255)

    drawn: list[np.ndarray] = []  # the frame as drawn, before the JPEG blurs one-pixel dots
    monkeypatch.setattr(manipulation.camera, "_jpeg", lambda bgr: drawn.append(bgr.copy()) or b"")
    manipulation.rendering._render_live(rgb, {}, None, None, teach, {"state": "untrusted"})
    img = drawn[0]
    moved = xyz + [0.02, 0.0, 0.0]
    assert yellow_at(img, moved[::3]) > 0.5, "drawn where the groups carried it"
    assert yellow_at(img, xyz[::3]) < yellow_at(img, moved[::3]), "not where it was last seen"


def test_the_act_tells_the_view_where_the_arm_has_the_object_it_holds(tmp_path, monkeypatch, group_live):  # noqa: F811
    """From the grip until the place lets go, the point groups cannot see the gamepad in the fingers; the act sends
    where the arm has it, the fingertip by the joints times how it sat in the fingers when they closed, and the view
    takes the newest word on it, a release included."""
    from lerobot.gui.api import jog

    port = _free_port()
    monkeypatch.setattr(manipulation.runtime, "GROUPS_VIEW", ("127.0.0.1", port))
    monkeypatch.setattr(manipulation.runtime, "GROUPS_HELD_S", 0.01)
    view = group_live.MjpegView(port, tmp_path / "recordings")
    t_bc = _pose(0.1, -0.2, 0.5)  # camera to base
    tip = {"pose": _pose(0.2, 0.0, 0.1)}
    monkeypatch.setattr(jog, "current_tip_and_anchor", lambda: (tip["pose"], np.eye(4), {}))
    tip_obj = _pose(0.0, 0.0, -0.02)  # the gamepad 20 mm below the fingertip

    async def run():
        task = asyncio.create_task(manipulation.groups._feed_held("gamepad", tip_obj, t_bc))
        await asyncio.sleep(0.1)
        first = view.take_held()
        tip["pose"] = _pose(0.25, 0.05, 0.15)  # the arm carries it
        await asyncio.sleep(0.1)
        second = view.take_held()
        task.cancel()
        await manipulation.groups._groups_call("POST", "/objects/held", {"gamepad": None})
        return first, second, view.take_held()

    try:
        first, second, released = asyncio.run(run())
    finally:
        view.server.shutdown()
        view.server.server_close()
    to_cam = np.linalg.inv(t_bc)
    assert np.allclose(first["gamepad"], to_cam @ _pose(0.2, 0.0, 0.1) @ tip_obj, atol=1e-6)
    assert np.allclose(second["gamepad"], to_cam @ _pose(0.25, 0.05, 0.15) @ tip_obj, atol=1e-6)
    assert released == {"gamepad": None}


def test_only_the_demos_objects_found_are_designated_not_a_teach_by_name(tmp_path, monkeypatch, group_live):  # noqa: F811
    """A demo's load teaches its concept by name (object_1), and SAM 3's mask for that covered half the camera view
    on 2026-10-10: designated, it drew an outline across the whole tray. Only a teach against one of the demo's
    objects, the ones the act follows, is designated."""
    port = _free_port()
    monkeypatch.setattr(manipulation.runtime, "GROUPS_VIEW", ("127.0.0.1", port))
    monkeypatch.setattr(manipulation.runtime._state, "groups", manipulation.runtime._GroupsView())
    monkeypatch.setattr(manipulation.runtime._state, "groups_feed", manipulation.runtime._GroupsFeed())
    monkeypatch.setattr(manipulation.runtime._state, "groups_with_acts", True)
    monkeypatch.setattr(manipulation.runtime._state, "act", manipulation.runtime._Act())
    monkeypatch.setattr(manipulation.runtime._state, "demo", None)
    view = group_live.MjpegView(port, tmp_path / "recordings")
    loop = _ViewLoop(view, lambda: _pose(0.1, 0.0, 0.43))
    mask = np.zeros((480, 848), bool)
    mask[280:360, 180:290] = True

    def teach(concept: str, ref_object: str | None):
        job = manipulation.runtime._Job(
            id=f"teach-{concept}", kind="teach", concept=concept, rgb=np.zeros((480, 848, 3), np.uint8),
            depth_m=np.full((480, 848), 0.45, np.float32), intr=dict(INTR), created=time.time(),
        )  # fmt: skip
        job.extra = {"ref_object": ref_object} if ref_object else {}
        job.result = {
            "ok": True, "mask": mask, "uv": np.zeros((5, 2)), "xyz": np.zeros((5, 3)), "n_points": 5,
            "radius_mm": 30.0, "shape_class": "rod", "yaw_observable": False, "ref_ok": bool(ref_object),
        }  # fmt: skip
        manipulation.runtime._state.teach_job = job.id
        manipulation.acquisition._apply_teach_result(job)

    async def run():
        teach("object_1", None)
        teach("gamepad", "gamepad")
        await asyncio.gather(*manipulation.runtime._groups_follows)

    try:
        asyncio.run(run())
        assert set(view.designated) == {"gamepad"}, view.designated
    finally:
        loop.close()
        with manipulation.runtime._state.lock:
            manipulation.runtime._state.teach = manipulation.runtime._state.test = None
