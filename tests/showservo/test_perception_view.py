"""A recording drawn with its perception (lerobot.showservo.perception_view): the groups as the act's views paint
them, each object green where the estimator took its pose and red where it refused it; and an act's demo time, by
which a frame of it is laid beside the demo's."""

from __future__ import annotations

import numpy as np

from lerobot.showservo import groups_scene, perception, perception_view
from lerobot.showservo.recordings import open_recording

from .test_perception import K, _demo, _groups_run, group_live  # noqa: F401  (a fixture)


def _processed(group_live, tmp_path):  # noqa: F811
    n, moved_from = 60, 32
    mask = _demo(tmp_path / "demo", n)
    rec = open_recording(tmp_path / "demo")
    tracker, frames, pictures = _groups_run(group_live, n, moved_from)
    d = perception.cache_dir(rec)
    d.mkdir(parents=True)
    group_live.write_groups_dump(
        d / "tracks.npz", frames, pictures, {"block": mask}, {"block": tracker.objects["block"].tracks}
    )
    return rec, perception.finish(rec, d)


def _uv(points, k):
    uv = (points @ k.T)[:, :2] / points[:, 2:3]
    return np.round(uv).astype(int)


def test_a_frame_shows_where_the_groups_put_each_object_and_whether_the_estimator_took_it(
    group_live,  # noqa: F811
    tmp_path,
):
    """Where the block lies as the depth has it, its pose taken: its surface drawn green there. Slid aside by the
    groups where the camera sees the table, refused: red, where the groups put it. Under it, the groups' picture
    painted as the act's view paints it, pixel for pixel where nothing else is drawn."""
    rec, p = _processed(group_live, tmp_path)
    shapes = perception_view.surfaces(rec, p)
    taken = next(k for k, r in enumerate(p.rows) if k > 31 and r["objects"]["block"]["taken"])
    refused = len(p.rows) - 1
    assert not p.rows[refused]["objects"]["block"]["taken"]
    for k, colour in ((taken, perception_view.TAKEN), (refused, perception_view.REFUSED)):
        img = perception_view.render(rec, p, k, shapes=shapes)
        m = p.motion("block", k)
        u, v = _uv(shapes["block"] @ m[:3, :3].T + m[:3, 3], K)[::3][len(shapes["block"]) // 6]
        assert np.abs(img[v, u].astype(int) - colour).max() <= 30, (k, img[v, u], colour)
    plain = perception_view.render(rec, p, 5, shapes={})  # no object drawn: only the picture and the words
    painted = groups_scene.paint_act_view(
        __import__("cv2").cvtColor(rec.color(rec.frame_at(float(p.t[5]))), __import__("cv2").COLOR_RGB2BGR),
        groups_scene.recorded_picture(p.pictures[5], p.picture(5)),
    )
    assert np.array_equal(plain[100:400, 200:700], painted[100:400, 200:700])


def test_an_acts_demo_time_is_the_one_it_recorded_or_its_legs_place_in_the_demo():
    """An act that recorded each target's demo time is placed by it. One that did not, by its legs: the leg toward a
    key frame runs from the key frame before to that one, its targets spread over it in order; going back says
    nothing. Before its first and after its last target, no demo time."""
    with_td = [
        {"t": 100.0, "step": "leg: key frame 1", "td": 2.0},
        {"t": 101.0, "step": "leg: key frame 1", "td": 3.0},
        {"t": 102.0, "step": "transition: key frame 1", "td": 1.0},
    ]
    clock = perception_view.act_clock(with_td, [])
    assert clock(100.5) == 2.5 and clock(99.0) is None and clock(103.0) is None
    keypoints = [
        {"t": 4.0, "kind": "key", "frame": "dowel", "move": "line"},
        {"t": 6.0, "kind": "key", "frame": "dowel", "move": "line"},
    ]
    labels = [m["label"] for m in perception_view.core.key_frames(keypoints)]
    targets = [
        {"t": 10.0, "step": f"leg: {labels[0]}"},
        {"t": 11.0, "step": f"leg: {labels[1]}"},
        {"t": 11.5, "step": f"leg: {labels[1]}"},
        {"t": 12.0, "step": f"leg: {labels[1]}"},
        {"t": 12.5, "step": f"going back: {labels[1]}"},
    ]
    clock = perception_view.act_clock(targets, keypoints)
    assert clock(11.0) == 4.0 and clock(12.0) == 6.0 and clock(11.5) == 5.0, (
        "the second leg, key frame 1 to 2"
    )
    times = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
    assert perception_view.act_time_at(clock, times, 5.0) == 11.5
    assert perception_view.act_time_at(clock, times, 7.0) is None
