# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The GUI's side of the act engine (gui/api/manipulation/engine_act.py): what the tracker's frames put into the
engine's scene log, and how an act asks for the engine and its injected faults."""

from __future__ import annotations

import time
import types

import numpy as np
import pytest

from lerobot.gui.api.manipulation import engine_act, execution, runtime
from lerobot.showservo.engine.perturb import OpenGripper, Perturbations
from lerobot.showservo.engine.scene import SceneLog
from tests.gui.test_act_with_groups import group_live  # noqa: F401  (a fixture)


def _job(stamp):
    return types.SimpleNamespace(camera_stamp_mono=stamp, created=None if stamp is None else 1000.0 + stamp)


def _target(anchor):
    return types.SimpleNamespace(anchor=anchor, last={}, hold_views=[])


def test_every_object_the_trackers_are_confident_in_is_measured_every_frame(monkeypatch):
    """The feed passes each object's tracker measurement on every frame, by the trackers' own verdict only: a frame
    the key-frame runner's acceptance rules turned away (``result`` None) still measures every object; a lost or
    failed one is not measured; the frame's time is kept either way (act_execution.md R4)."""
    scene = SceneLog()
    monkeypatch.setattr(engine_act.planning, "_delta_base", lambda d, delta, t_bc: np.asarray(delta) + 1.0)
    monkeypatch.setattr(engine_act.planning, "_pose_frame", lambda d, name: 1)
    monkeypatch.setattr(engine_act.holds, "_demo_control_pose", lambda d, name, f: (np.eye(4), ""))
    anchor = np.eye(4)
    anchor[:3, 3] = (0.01, 0.0, 0.0)
    monkeypatch.setattr(
        runtime._state,
        "targets",
        {"ring": _target(anchor), "cup": _target(np.eye(4))},
    )
    feed = engine_act._Feed(scene, types.SimpleNamespace(), np.eye(4), "dowel", ["ring", "cup"])
    r = {
        "ok": True,
        "delta": np.eye(4),
        "others": [{"name": "cup", "ok": True, "lost": True}, {"name": "ring", "ok": True, "lost": False}],
        "other_delta_0": np.eye(4),
        "other_delta_1": np.eye(4),
    }
    feed(_job(12.5), r, None)
    feed(_job(12.6), {**r, "ok": False}, None)  # Point2Pose found no fit of the dowel in this one
    assert scene.frames() == [12.5, 12.6]
    assert [m.t for m in scene.measurements("dowel")] == [12.5]
    np.testing.assert_allclose(scene.measurements("dowel")[0].motion, np.eye(4) + 1.0)
    assert [m.t for m in scene.measurements("ring")] == [12.5, 12.6]
    np.testing.assert_allclose(scene.measurements("ring")[0].motion[:3, 3], (0.01, 0.0, 0.0))
    assert scene.measurements("cup") == []


def test_a_frame_without_a_camera_stamp_is_not_fed():
    scene = SceneLog()
    feed = engine_act._Feed(scene, None, np.eye(4), "dowel", [])
    feed(_job(None), {"ok": True, "delta": np.eye(4)}, None)
    assert scene.frames() == [] and scene.measurements("dowel") == []


def test_an_act_asks_for_the_engine_and_its_faults():
    body = execution.ActBody(
        runner="engine", perturb=[{"kind": "open_gripper", "leg": "pre-place 1", "s": 0.4}]
    )
    assert body.runner == "engine"
    p = Perturbations.from_spec(body.perturb)
    assert isinstance(p.items[0], OpenGripper) and p.items[0].leg == "pre-place 1"
    assert execution.ActBody().runner == "key_frames"
    with pytest.raises(ValueError):
        execution.ActBody(runner="planner")
    with pytest.raises(ValueError):
        Perturbations.from_spec([{"kind": "teleport"}])


@pytest.mark.parametrize(
    ("connected", "refused"),
    [
        ("so107_follower_gravity", None),
        ("bi_so107_follower", "only a so107_follower_gravity arm"),
        (None, "connect"),
    ],
)
def test_the_engine_drives_only_the_gravity_compensated_arm(monkeypatch, connected, refused):
    """The engine commands joints from the calibrated kinematics without watching the fingertip: a plain follower
    settles up to 33 mm short under load, so an engine act on one is refused before it starts."""
    from lerobot.gui.api import jog

    monkeypatch.setattr(jog, "connected_robot_type", lambda: connected)
    reason = engine_act.arm_refusal()
    assert (reason is None) if refused is None else (refused in reason)


def test_the_fingertip_now_is_the_latest_encoder_read(monkeypatch):
    """The engine asks for the fingertip "now" (the grasp's hold is measured from it). The jog's history never holds a
    read after now, so a lookup that wants two reads around the time found none, and the hold was taken from the
    commanded pose: on the rig the fingertip stood 6.3-7.5 mm from it at the grasp and the dowel hovered 5 mm off the
    ring (act 20261018_111834). The latest read answers for now, up to two ticks old; a past time is interpolated."""
    from lerobot.gui.api import jog

    class _Kin:
        def forward_kinematics(self, q):
            out = np.eye(4)
            out[:3, 3] = np.asarray(q[:3], dtype=float)
            return out

    names = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
    j = jog._Jog()
    j.pose_history.append((10.0, dict.fromkeys(names, 0.0)))
    j.pose_history.append((10.0 + 1.0 / jog.HZ, dict.fromkeys(names, 3.0)))
    monkeypatch.setattr(jog, "_jog", j)
    scene = engine_act._ArmScene(None, _Kin(), names)
    np.testing.assert_allclose(scene.tip_at(10.0 + 0.5 / jog.HZ)[:3, 3], 1.5)  # between two reads
    np.testing.assert_allclose(scene.tip_at(10.0 + 1.5 / jog.HZ)[:3, 3], 3.0)  # now: the latest read
    assert scene.tip_at(10.0 + 4.0 / jog.HZ) is None  # no read for that long: not the arm now


def test_only_the_engine_acts_a_key_frame_held_where_its_leg_begins():
    from lerobot.showservo import motion as core

    marks = [
        {"t": 1.0, "kind": "key", "frame": "dowel", "move": "line", "grip": "first"},
        {"t": 2.0, "kind": "key", "frame": core.START_FRAME, "move": "follow", "grip": "first"},
    ]
    assert "act engine only" in execution._runner_problem("key_frames", marks)
    assert execution._runner_problem("engine", marks) == ""
    assert execution._runner_problem("key_frames", marks[:1]) == ""


def test_an_object_with_a_declared_turn_symmetry_is_centred_on_its_axis():
    """The world measures a symmetric object at a point its symmetry's turns keep still: a point on the axis measured
    from its demo surface, not the visible surface's mean, which a turn about the axis would move."""
    rng = np.random.default_rng(0)
    along = rng.uniform(-0.05, 0.05, 2000)
    angle = rng.uniform(0.0, np.pi, 2000)  # the half a camera above sees
    half_cylinder = np.c_[along, 0.01 * np.cos(angle), 0.01 * np.sin(angle)] + np.array([0.1, -0.2, 0.01])
    spec = {"kind": "continuous", "axis": "long_axis", "end_swap": True}
    centres, symmetries = engine_act.object_geometry(
        {"long": {"symmetry": spec}, "plain": {}}, {"long": half_cylinder, "plain": half_cylinder}
    )
    np.testing.assert_allclose(centres["long"][1:], [-0.2, 0.01], atol=0.001)
    np.testing.assert_allclose(centres["plain"], half_cylinder.mean(axis=0))
    assert symmetries["long"][0]["kind"] == "continuous" and symmetries["long"][0].get("end_swap")
    assert abs(abs(symmetries["long"][1] @ [1.0, 0.0, 0.0]) - 1.0) < 1e-3
    assert "plain" not in symmetries


def test_the_feed_measures_only_what_its_estimator_takes(monkeypatch):
    """Which tracker output is a measurement is the estimator's (lerobot.showservo.estimation): the feed hands it every
    object's candidate with its tracker's verdict, adds what it takes, and counts what it refuses for the act's
    record."""
    from lerobot.showservo.estimation import Estimate

    monkeypatch.setattr(engine_act.planning, "_delta_base", lambda d, delta, t_bc: np.asarray(delta))
    monkeypatch.setattr(engine_act.planning, "_pose_frame", lambda d, name: 1)
    monkeypatch.setattr(engine_act.holds, "_demo_control_pose", lambda d, name, f: (np.eye(4), ""))
    monkeypatch.setattr(
        runtime._state,
        "targets",
        {"ring": _target(np.eye(4)), "cup": _target(np.eye(4))},
    )
    seen = []

    class OnlyTheDowel:
        def estimate(self, t, candidates, frame=None):
            seen.append({n: [(c.source, c.ok, c.why) for c in cs] for n, cs in candidates.items()})
            return {
                n: Estimate(cs[0].motion, "taken") if n == "dowel" else Estimate(None, "not this one")
                for n, cs in candidates.items()
            }

    scene = SceneLog()
    feed = engine_act._Feed(
        scene, types.SimpleNamespace(), np.eye(4), "dowel", ["ring", "cup"], OnlyTheDowel()
    )
    r = {
        "ok": True,
        "delta": np.eye(4),
        "others": [{"name": "cup", "ok": True, "lost": True}, {"name": "ring", "ok": True, "lost": False}],
        "other_delta_0": np.eye(4),
        "other_delta_1": np.eye(4),
    }
    feed(_job(3.0), r, None)
    assert seen == [
        {
            "dowel": [("point2pose", True, ""), ("old runner checks", False, "not certified")],
            "ring": [("point2pose", True, ""), ("old runner checks", False, "not certified")],
            "cup": [("point2pose", False, "lost"), ("old runner checks", False, "not certified")],
        }
    ]
    assert [m.t for m in scene.measurements("dowel")] == [3.0]
    assert scene.measurements("ring") == [] and scene.measurements("cup") == []
    assert feed.refused == {"ring: not this one": 1, "cup: not this one": 1}
    # The act's record of the frame: each object's candidates and what the estimator made of them.
    (row,) = feed.log
    assert row["t"] == 3.0
    dowel, cup = row["objects"]["dowel"], row["objects"]["cup"]
    assert (
        dowel["taken"]
        and dowel["why"] == "taken"
        and cup
        == {
            "candidates": [
                {"source": "point2pose", "ok": False, "why": "lost", "motion": np.eye(4).tolist()},
                {
                    "source": "old runner checks",
                    "ok": False,
                    "why": "not certified",
                    "motion": np.eye(4).tolist(),
                },
            ],
            "taken": False,
            "why": "not this one",
            "seen_through": None,
        }
    )


def test_the_old_runners_checks_are_a_candidate_beside_point2pose(monkeypatch):
    """The key-frame runner's checks run on every frame before the feed (lerobot.gui.api.manipulation.tracking): the
    session object's view they certified comes as the feed's ``result``, another object's as its track's newest view
    of this frame. Each is a candidate of its own, so an estimator may take them instead of Point2Pose's verdict."""
    from lerobot.showservo.estimation import old_runner_checks

    monkeypatch.setattr(engine_act.planning, "_delta_base", lambda d, delta, t_bc: np.asarray(delta))
    monkeypatch.setattr(engine_act.planning, "_pose_frame", lambda d, name: 1)
    monkeypatch.setattr(engine_act.holds, "_demo_control_pose", lambda d, name, f: (np.eye(4), ""))
    certified_ring = np.eye(4)
    certified_ring[:3, 3] = (0.0, 0.02, 0.0)
    ring = _target(np.eye(4))
    ring.last, ring.hold_views = {"state": "tracking"}, [(1003.0, certified_ring)]
    cup = _target(np.eye(4))
    cup.last = {"state": "untrusted", "reason": "only 56% of its points are seen"}
    monkeypatch.setattr(runtime._state, "targets", {"ring": ring, "cup": cup})
    certified_dowel = np.eye(4)
    certified_dowel[:3, 3] = (0.01, 0.0, 0.0)
    scene = SceneLog()
    feed = engine_act._Feed(
        scene, types.SimpleNamespace(), np.eye(4), "dowel", ["ring", "cup"], old_runner_checks()
    )
    r = {
        "ok": True,
        "delta": np.eye(4),
        "others": [{"name": "cup", "ok": True, "lost": False}, {"name": "ring", "ok": True, "lost": False}],
        "other_delta_0": np.eye(4),
        "other_delta_1": np.eye(4),
    }
    feed(_job(3.0), r, {"delta_cam": certified_dowel})
    np.testing.assert_allclose(scene.measurements("dowel")[0].motion, certified_dowel)
    np.testing.assert_allclose(scene.measurements("ring")[0].motion, certified_ring)
    assert scene.measurements("cup") == []
    assert feed.refused == {"cup: old runner checks: only 56% of its points are seen": 1}
    feed(
        _job(4.0), r, None
    )  # the runner refused this view of the dowel; its view of the ring is not of this frame
    assert len(scene.measurements("dowel")) == 1 and len(scene.measurements("ring")) == 1


def test_the_point_groups_pose_in_the_same_camera_frame_is_a_candidate(monkeypatch):
    """With the objects designated in the point groups, each frame's candidates include the groups' pose of each in
    the same camera frame (stamped by the wall clock there, by the perf counter here): its motion since designation,
    applied to where the act found it. Between two of the groups' frames it is interpolated; a frame the groups have
    not reached yet has none, rather than their last pose stamped with a newer time."""
    import collections

    from lerobot.showservo.estimation import GROUPS, SourceVerdict

    monkeypatch.setattr(engine_act.planning, "_delta_base", lambda d, delta, t_bc: np.asarray(delta))
    monkeypatch.setattr(runtime._state, "targets", {})
    designated = np.eye(4)
    designated[:3, 3] = (0.10, 0.0, 0.40)
    moved = designated.copy()
    moved[0, 3] += 0.020
    monkeypatch.setattr(
        runtime._state.groups_feed,
        "frames",
        {"dowel": collections.deque([(1000.0, designated), (1000.1, moved)])},
    )
    start = np.eye(4)
    start[:3, 3] = (0.30, 0.0, 0.0)
    scene = SceneLog()
    feed = engine_act._Feed(
        scene,
        types.SimpleNamespace(),
        np.eye(4),
        "dowel",
        [],
        SourceVerdict(GROUPS),
        groups_from=1000.0,
        starts={"dowel": start},
    )
    feed.wall_offset = lambda: 1000.0
    r = {"ok": True, "delta": np.eye(4)}
    feed(_job(0.05), r, None)  # halfway between the groups' two frames
    feed(_job(0.2), r, None)  # after their newest
    assert [m.t for m in scene.measurements("dowel")] == [0.05]
    np.testing.assert_allclose(scene.measurements("dowel")[0].motion[:3, 3], (0.31, 0.0, 0.0), atol=1e-9)
    assert feed.refused == {"dowel: no point groups candidate": 1}


def test_an_object_moving_to_another_group_has_no_confident_groups_pose(monkeypatch):
    """While most of an object's seen points are in a group it has not moved to yet, the groups' pose of it is still
    the old group's: lifted, it reads as lying where it lay, and a check of the grasp took that for an answer (act
    20261018_231143 replayed, 'the other outcome' 0.4 mm away). Such a frame's groups pose is not confident."""
    import collections

    from lerobot.showservo.estimation import GROUPS, SourceVerdict

    monkeypatch.setattr(engine_act.planning, "_delta_base", lambda d, delta, t_bc: np.asarray(delta))
    monkeypatch.setattr(runtime._state, "targets", {})
    pose = np.eye(4)
    monkeypatch.setattr(
        runtime._state.groups_feed,
        "frames",
        {"dowel": collections.deque([(1000.0, pose), (1000.1, pose), (1000.2, pose)])},
    )
    monkeypatch.setattr(
        runtime._state.groups_feed,
        "evidence",
        {"dowel": collections.deque([(1000.0, {}), (1000.1, {"moving": True}), (1000.2, {})])},
    )
    scene = SceneLog()
    feed = engine_act._Feed(
        scene,
        types.SimpleNamespace(),
        np.eye(4),
        "dowel",
        [],
        SourceVerdict(GROUPS),
        groups_from=1000.0,
        starts={"dowel": np.eye(4)},
    )
    feed.wall_offset = lambda: 1000.0
    r = {"ok": True, "delta": np.eye(4)}
    feed(_job(0.05), r, None)  # between a frame in transition and one before it
    feed(_job(0.2), r, None)  # the frame after
    assert [m.t for m in scene.measurements("dowel")] == [0.2]
    assert feed.refused == {"dowel: point groups: moving to another group": 1}


def _engine_view(tmp_path, monkeypatch, group_live):  # noqa: F811
    """The real point groups view's server on a free port, its live loop played (tests/gui/test_act_with_groups.py)."""
    from tests.gui.test_act_with_groups import _free_port, _ViewLoop

    port = _free_port()
    monkeypatch.setattr(runtime, "GROUPS_VIEW", ("127.0.0.1", port))
    monkeypatch.setattr(runtime._state, "groups", runtime._GroupsView())  # a view started elsewhere
    monkeypatch.setattr(runtime._state, "groups_feed", runtime._GroupsFeed())
    monkeypatch.setattr(runtime._state, "groups_with_acts", True)
    view = group_live.MjpegView(port, tmp_path / "recordings")
    return view, _ViewLoop(view, lambda: np.eye(4))


def _masked(x0, y0):
    mask = np.zeros((480, 848), bool)
    mask[y0 : y0 + 40, x0 : x0 + 60] = True
    return mask


@pytest.mark.parametrize("ring_found", [True, False])
def test_an_engine_act_designates_its_objects_in_the_point_groups_where_it_found_them(
    tmp_path,
    monkeypatch,
    group_live,  # noqa: F811
    ring_found,
):
    """Before the arm moves, the session object is designated where its find saw it and each other object where its
    locate did; the act gets the time from which the groups' frames have them. An object with no place to click is
    not guessed: the act is told to click it."""
    import asyncio

    view, loop = _engine_view(tmp_path, monkeypatch, group_live)
    located = {"ring": {"mask": _masked(300, 280)}} if ring_found else {}
    monkeypatch.setattr(engine_act.acquisition, "_located", lambda demo, name: located.get(name))
    teach = types.SimpleNamespace(
        keypoints={"mask": _masked(100, 260)}, rgb=np.zeros((480, 848, 3), np.uint8)
    )
    test = types.SimpleNamespace(result={"live_mask": _masked(120, 270)})
    try:
        before = time.time()
        why, since = asyncio.run(
            engine_act._designate(
                types.SimpleNamespace(), teach, test, "dowel", ["ring"], None, lambda: False
            )
        )
    finally:
        loop.close()
    if ring_found:
        assert why == "" and since is not None and since >= before, why
        assert set(view.designated) == {"dowel", "ring"}
    else:
        assert why == "where ring is, for the point groups: click it in the camera view" and since is None
        assert view.designated == {}


def test_an_engine_act_stops_when_the_point_groups_view_draws_nothing(monkeypatch):
    """A view that draws no frame for GROUPS_STALE_S holds every object it carries still, whatever happens to it."""
    monkeypatch.setattr(runtime, "GROUPS_STALE_S", 0.3)
    monkeypatch.setattr(runtime._state, "groups_feed", runtime._GroupsFeed())
    now = time.monotonic()
    runtime._state.groups_feed.fresh = now
    assert engine_act._groups_stopped(now - 5.0) == "", "a fresh frame"
    runtime._state.groups_feed.fresh = 0.0
    assert engine_act._groups_stopped(now) == "", "designated just now: its first frame may still come"
    runtime._state.groups_feed.fresh = now - 1.0
    assert engine_act._groups_stopped(now - 1.0) == "the point groups view drew no frame for 0.3 s"


def test_an_engine_act_takes_the_demos_grips_from_its_perception_or_says_why_not(
    tmp_path,
    monkeypatch,
    group_live,  # noqa: F811
):
    """At an engine act's start, how the demo's objects sat in its gripper comes from the demo's perception by the
    pipeline as it is now: every pose it took, on the demo's clock and in the base frame, goes to the same rule an act
    keeps its own hold by. A demo not saved, or not processed by this pipeline, is said, and the act holds its objects
    as they lay at the grasp."""
    from lerobot.showservo import perception
    from lerobot.showservo.engine import demo_beliefs
    from lerobot.showservo.recordings import open_recording
    from tests.showservo.test_perception import T0, _demo, _groups_run

    timeline = types.SimpleNamespace(demo="the demo", events=["the events"])
    t_bc = np.eye(4)
    t_bc[:3, 3] = (0.1, 0.2, 0.3)
    assert engine_act._demo_grips(None, timeline, t_bc, {}, {})[1] == {
        "used": False,
        "why": "the demo is not saved",
    }
    mask = _demo(tmp_path / "demo", 40)
    used = engine_act._demo_grips(str(tmp_path / "demo"), timeline, t_bc, {}, {})[1]
    assert not used["used"] and perception.version() in used["why"]
    rec = open_recording(tmp_path / "demo")
    tracker, frames, pictures = _groups_run(group_live, 40, 100)
    d = perception.cache_dir(rec)
    d.mkdir(parents=True)
    group_live.write_groups_dump(
        d / "tracks.npz", frames, pictures, {"block": mask}, {"block": tracker.objects["block"].tracks}
    )
    p = perception.finish(rec, d)
    seen = {}

    def grips(demo, events, views, centres, symmetry=None):
        seen.update(demo=demo, events=events, views=views)
        return {"block": "its grips"}, {"block": {"views_held": 0}}

    monkeypatch.setattr(demo_beliefs, "grips", grips)
    out, info = engine_act._demo_grips(str(tmp_path / "demo"), timeline, t_bc, {"block": np.zeros(3)}, {})
    assert out == {"block": "its grips"} and info == {
        "used": True,
        "version": p.version,
        "objects": {"block": {"views_held": 0}},
    }
    assert seen["demo"] == "the demo" and seen["events"] == ["the events"]
    taken = [(r["t"], r["objects"]["block"]) for r in p.rows if r["objects"]["block"]["taken"]]
    assert len(seen["views"]["block"]) == len(taken) > 0
    (t, row), (td, motion) = taken[-1], seen["views"]["block"][-1]
    assert td == pytest.approx(t - T0), "on the demo's clock"
    cam = np.asarray(row["candidates"][0]["motion"])
    np.testing.assert_allclose(motion, t_bc @ cam @ np.linalg.inv(t_bc), err_msg="in the base frame")
