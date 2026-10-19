"""A recording's perception (lerobot.showservo.perception): the point groups run with the live view's settings, their
dump made a cache the reader gives back as written, and the engine's estimator applied to it against the recording's
own depth; a change of the pipeline is a new cache."""

from __future__ import annotations

import io
import json
import pathlib
import sys

import cv2
import numpy as np
import pytest

from lerobot.showservo import groups_scene, perception
from lerobot.showservo.groups import GroupTracker
from lerobot.showservo.recordings import open_recording

H, W, FPS = 480, 848, 30.0
FX = FY = 604.2
CX, CY = 419.2, 250.6
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
T0 = 1_792_000_000.0
TABLE_M, TOP_M = 0.45, 0.42  # the table, and the top of a 40 x 30 mm block lying on it
BLOCK = (slice(230, 270), slice(400, 450))  # where the block is in the image


@pytest.fixture(scope="module")
def group_live():
    bench = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path.insert(0, str(bench))
    try:
        import group_live

        yield group_live
    finally:
        sys.path.remove(str(bench))


def _pixel_to_xyz(u, v, z):
    z = np.full_like(u, z, dtype=float)
    return np.stack([(u - CX) * z / FX, (v - CY) * z / FY, z], axis=-1)


def _demo(root, n):
    """A demo of ``n`` frames: the block where BLOCK is, still in every depth frame; its mask its designation."""
    cam = root / "recording"
    (cam / "rgb").mkdir(parents=True)
    (cam / "depth").mkdir()
    depth = np.full((H, W), TABLE_M * 1000, np.uint16)
    depth[BLOCK] = TOP_M * 1000
    rgb = np.full((H, W, 3), 60, np.uint8)
    for k in range(n):
        cv2.imwrite(str(cam / "rgb" / f"{k:06d}.jpg"), rgb)
        cv2.imwrite(str(cam / "depth" / f"{k:06d}.png"), depth)
    np.savetxt(cam / "times.txt", T0 + np.arange(n) / FPS, fmt="%.6f")
    np.savetxt(cam / "cam_K.txt", K)
    q = np.zeros((n, 7))
    np.savez_compressed(
        root / "showservo_demo.npz",
        t=np.arange(n) / FPS,
        t0=T0,
        q_obs=q,
        q_cmd=q,
        tips=np.tile(np.eye(4), (n, 1, 1)),
    )
    mask = np.zeros((H, W), bool)
    mask[BLOCK] = True
    np.savez_compressed(
        root / "objects.npz", names=json.dumps([["block", 0, [425, 250], None]]), o0_mask=mask
    )
    return mask


def _groups_run(group_live, n, moved_from):
    """The point groups over the demo: a table and the block's top, the block's points sliding along x at 2 mm a
    frame from frame ``moved_from`` (the depth frames never show it move). Its dump frames and pictures, as the script
    makes them."""
    rng = np.random.default_rng(4)
    uu, vv = rng.uniform(0, W, 300), rng.uniform(0, H, 300)
    keep = ~((vv >= 230) & (vv < 270) & (uu >= 400) & (uu < 450))
    table = _pixel_to_xyz(uu[keep][:200], vv[keep][:200], TABLE_M)
    bu, bv = rng.uniform(400, 450, 30), rng.uniform(230, 270, 30)
    block = _pixel_to_xyz(bu, bv, TOP_M)
    t = GroupTracker()
    centre = block.mean(axis=0)
    pose = np.eye(4)
    pose[:3, 3] = centre
    t.add_object("block", np.arange(200, 230), pose)
    corners = block[[np.argmin(bu), np.argmax(bu), np.argmin(bv), np.argmax(bv)]]
    objects = {"block": (corners, centre, None)}
    frames, pictures = [], []
    for k in range(n):
        shift = np.array([0.002 * max(k - moved_from, 0), 0.0, 0.0])
        xyz = np.vstack([table, block + shift]) + rng.normal(0.0, 0.0015, (230, 3))
        t.update(xyz, np.ones(230, bool))
        frames.append(group_live.groups_frame(T0 + k / FPS, t, groups_scene_matrix()))
        pictures.append(groups_scene.drawing(K, t, xyz, np.ones(230, bool), objects))
    return t, frames, pictures


def groups_scene_matrix():
    from lerobot.showservo.groups import _matrix

    return _matrix


def test_the_pipeline_runs_the_groups_with_the_live_views_settings(group_live, monkeypatch, tmp_path):
    """The live view the GUI starts and the pipeline's run of the same script track alike: both command lines, read
    by the script's own parser, ask for the same TAPIR and grouping settings; the pipeline's over the staged frames,
    dumping, with each object's click."""
    from lerobot.gui.api.manipulation import groups as gui_groups, runtime

    captured = {}

    class Proc:
        stdout = io.StringIO("")

        def poll(self):
            return None

    def popen(cmd, **_kw):
        captured["cmd"] = cmd
        return Proc()

    monkeypatch.setattr(gui_groups.subprocess, "Popen", popen)
    monkeypatch.setattr(runtime._state, "groups", runtime._GroupsView())
    gui_groups._start_groups("http://127.0.0.1:9102")
    live = group_live.parser().parse_args(captured["cmd"][2:])
    cmd = perception.group_command(
        "python", tmp_path / "stage", tmp_path / "tracks.npz", {"dowel": (10, 20), "ring": (30, 40)}
    )
    offline = group_live.parser().parse_args(cmd[2:])
    for setting in ("pips", "resize", "own_share", "borrowed_points", "every", "start", "end"):
        assert getattr(offline, setting) == getattr(live, setting), setting
    assert live.live and not offline.live
    assert (offline.recording, offline.dump) == (str(tmp_path / "stage"), str(tmp_path / "tracks.npz"))
    assert offline.object == ["dowel=10,20", "ring=30,40"]


def test_the_cache_keeps_what_the_groups_dumped_and_the_estimators_view_of_each_frame(group_live, tmp_path):
    """The groups' dump (write_groups_dump) made a cache (finish) and read back (load): every frame's pose, whether
    the object was moving to another group, its designation mask and picture as the script wrote them; and each frame's
    row, the engine's estimator on the groups' pose against the recording's own depth: taken while the block lies
    where the depth has it, refused once the groups have it slid aside where the camera sees the table, and not
    confident in a frame the groups say it is moving to another group."""
    n, moved_from = 60, 32
    mask = _demo(tmp_path / "demo", n)
    rec = open_recording(tmp_path / "demo")
    tracker, frames, pictures = _groups_run(group_live, n, moved_from)
    frames[20] = (*frames[20][:4], ["block"])  # the groups say it is moving to another group in frame 20
    d = perception.cache_dir(rec)
    d.mkdir(parents=True)
    group_live.write_groups_dump(
        d / "tracks.npz", frames, pictures, {"block": mask}, {"block": tracker.objects["block"].tracks}
    )
    p = perception.finish(rec, d)
    assert p.version == perception.version() and p.names == ["block"] and len(p.rows) == n
    np.testing.assert_allclose(p.poses["block"], [f[3]["block"] for f in frames])
    assert p.moving["block"].tolist() == ["block" in f[4] for f in frames] and p.moving["block"][20]
    np.testing.assert_array_equal(p.masks["block"], mask)
    assert p.pictures[7]["objects"] == pictures[7]["objects"] and p.pictures[7]["base"] == pictures[7]["base"]
    np.testing.assert_array_equal(p.picture(7)["groups_points"], np.asarray(pictures[7]["points"]))
    rows = [r["objects"]["block"] for r in p.rows]
    for k, row in enumerate(rows):
        np.testing.assert_allclose(row["candidates"][0]["motion"], p.motion("block", k), atol=1e-5)
    assert not rows[20]["taken"] and rows[20]["why"] == "point groups: moving to another group"
    assert all(rows[k]["taken"] for k in range(31, moved_from)), [
        rows[k]["why"] for k in range(31, moved_from)
    ]
    assert p.motion("block", n - 1)[0, 3] > 0.03, "the groups followed the slide"
    assert not any(rows[k]["taken"] for k in range(n - 5, n)), [rows[k]["why"] for k in range(n - 5, n)]
    assert all(rows[k]["seen_through"] > 0.5 for k in range(n - 5, n))


def test_a_change_of_the_pipeline_is_a_new_cache(tmp_path, monkeypatch):
    """The version is a hash of the pipeline's sources: one changed is another version, and a recording processed by
    the earlier one has no perception by this one."""
    _demo(tmp_path / "demo", 3)
    rec = open_recording(tmp_path / "demo")
    source = tmp_path / "rule.py"
    source.write_text("TOL = 1\n")
    monkeypatch.setattr(perception, "SOURCES", (source,))
    before = perception.version()
    (perception.cache_dir(rec)).mkdir(parents=True)
    (perception.cache_dir(rec) / perception.RECORD).write_text(json.dumps({"version": before, "frames": []}))
    source.write_text("TOL = 2\n")
    assert perception.version() != before and perception.cache_dir(rec).name != before
    assert perception.load(rec) is None
