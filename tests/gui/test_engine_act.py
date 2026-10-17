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


def _job(stamp, h=480, w=848):
    return types.SimpleNamespace(camera_stamp_mono=stamp, depth_m=np.full((h, w), 0.43))


def test_a_tracked_frame_feeds_tracks_depth_and_the_views_that_placed_its_objects(monkeypatch):
    scene = SceneLog()
    demo = types.SimpleNamespace()
    monkeypatch.setattr(engine_act.planning, "_delta_base", lambda d, delta, t_bc: np.asarray(delta) + 1.0)
    monkeypatch.setattr(engine_act.planning, "_object_motion", lambda d, name, t_bc: (np.eye(4) * 2.0, ""))
    feed = engine_act._Feed(scene, demo, np.eye(4), "dowel", ["ring", "cup"])
    r = {
        "track_idx": np.array([3, 4, 5]),
        "track_uv": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "track_vis": np.array([True, False, True]),
        "others": [{"name": "cup"}, {"name": "ring"}],
        "other_track_idx_1": np.array([7, 8]),
        "other_track_uv_1": np.array([[9.0, 9.0], [8.0, 8.0]]),
        "other_track_vis_1": np.array([True, True]),
    }
    monkeypatch.setitem(runtime._state.located, "ring", {"stamp": 12.5})
    monkeypatch.setitem(
        runtime._state.located, "cup", {"stamp": 11.0}
    )  # placed by an earlier frame, not this one
    feed(_job(12.5), r, {"delta_cam": np.eye(4)})
    assert [d.t for d in scene.depths()] == [12.5]
    dowel = scene.tracks("dowel")[0]
    assert dowel.ids.tolist() == [3, 4, 5] and dowel.visible.tolist() == [True, False, True]
    assert scene.tracks("ring")[0].ids.tolist() == [7, 8]
    assert scene.tracks("cup") == []  # its arrays were not in the frame
    np.testing.assert_allclose(scene.views("dowel")[0].motion, np.eye(4) + 1.0)
    assert len(scene.views("ring")) == 1 and scene.views("cup") == []


def test_a_frame_without_a_camera_stamp_is_not_fed():
    scene = SceneLog()
    feed = engine_act._Feed(scene, None, np.eye(4), "dowel", [])
    feed(_job(None), {"track_uv": np.zeros((2, 2))}, None)
    assert scene.depths() == [] and scene.tracks("dowel") == []


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
