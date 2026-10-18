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

import types

import numpy as np
import pytest

from lerobot.gui.api.manipulation import engine_act, execution, runtime
from lerobot.showservo.engine.perturb import OpenGripper, Perturbations
from lerobot.showservo.engine.scene import SceneLog


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
