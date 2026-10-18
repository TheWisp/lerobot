# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The world (src/lerobot/showservo/docs/act_execution.md, "The loops, and who writes what"): each object's pose from
its views, and its noise measured on them. Views come about ten a second, as the rig's."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.showservo import symmetry as sym
from lerobot.showservo.engine.scene import Measurement, SceneLog
from lerobot.showservo.engine.world import World

SIGMA_M = 0.0015  # each view's scatter, per axis


def _world():
    scene = SceneLog(None, keep_s=120.0)
    return scene, World(scene, {"block": np.eye(4)}, {"block": np.array([0.0, -0.2, 0.0])})


def _view(scene, world, t, rng, x_m=0.0):
    m = np.eye(4)
    m[:3, 3] = rng.normal(0.0, SIGMA_M, 3) + np.array([x_m, 0.0, 0.0])
    scene.add_frame(t)
    scene.add_measurement("block", Measurement(t=t, motion=m))
    world.update()


def test_a_still_objects_noise_is_how_far_its_views_land_from_the_ones_before():
    rng = np.random.default_rng(0)
    scene, world = _world()
    for k in range(300):
        _view(scene, world, k / 10, rng)
    # The 95th percentile of a 3D normal's radius is 2.8 sigma, 4.2 mm; the estimate's own scatter adds a little.
    assert 0.0035 < world.noise("block")[0] < 0.0055, world.noise("block")
    assert not world.moves


def test_a_push_is_a_move_and_not_noise():
    """2026-10-18 on the rig (acts 20261018_113106 and 20261018_135723, act_execution.md E28): the fingers coming down
    pushed the object 14 and 11 mm; the views showing the push counted toward its noise before the move test ran on
    them, so the threshold rose just past the push (6.6 mm noise, 13.3 threshold; 5.7 and 11.3) and no move was seen.
    Here the same push, as the rig's views showed it: a move, and the noise as it was."""
    rng = np.random.default_rng(1)
    scene, world = _world()
    for k in range(60):
        _view(scene, world, k / 10, rng)
    before = world.noise("block")[0]
    for k, x_mm in enumerate((2.0, 5.4, 10.8, 11.4, 12.5, 13.0, 13.5, 13.5, 13.5, 13.5)):
        _view(scene, world, 6.0 + k / 10, rng, x_m=x_mm / 1000)
    assert "block" in world.moves
    assert world.noise("block")[0] <= before + SIGMA_M, (before, world.noise("block"))


# ── symmetry: of the poses that look the same, the one nearest the demo's ──

AXIS, POINT = np.array([0.6, 0.8, 0.0]), np.array([0.05, -0.2, 0.01])


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))


@pytest.mark.parametrize(
    ("spec", "turn", "flip", "left_deg"),
    [
        ({"kind": "continuous", "axis": "long_axis"}, 73.0, False, 0.0),
        ({"kind": "continuous", "axis": "long_axis", "end_swap": True}, 73.0, True, 0.0),
        ({"kind": "discrete", "axis": "surface_normal", "order": 4}, 100.0, False, 10.0),
        ({"kind": "none"}, 30.0, False, 30.0),
    ],
)
def test_the_equivalent_pose_kept_is_the_one_nearest_the_reference(spec, turn, flip, left_deg):
    shift = np.eye(4)
    shift[:3, 3] = (0.01, 0.02, 0.0)
    motion = shift @ sym.turn_about(AXIS, POINT, turn)
    if flip:
        motion = motion @ sym.turn_about(np.cross(AXIS, [0.0, 0.0, 1.0]), POINT, 180.0)
    out = sym.nearest_equivalent(motion, spec, AXIS, POINT)
    assert _angle_deg(out, np.eye(4)) == pytest.approx(left_deg, abs=1e-6)
    np.testing.assert_allclose(
        out[:3, :3] @ POINT + out[:3, 3], motion[:3, :3] @ POINT + motion[:3, 3], atol=1e-9
    )


def test_views_turned_about_a_declared_axis_move_nothing_in_the_world():
    rng = np.random.default_rng(2)
    scene = SceneLog(None, keep_s=120.0)
    world = World(
        scene,
        {"block": np.eye(4)},
        {"block": POINT},
        symmetry={"block": ({"kind": "continuous", "axis": "long_axis"}, AXIS)},
    )
    for k in range(60):
        m = sym.turn_about(AXIS, POINT, rng.uniform(-90.0, 90.0))
        m[:3, 3] += rng.normal(0.0, SIGMA_M, 3)
        scene.add_frame(k / 10)
        scene.add_measurement("block", Measurement(t=k / 10, motion=m))
        world.update()
    assert _angle_deg(world.motion("block"), np.eye(4)) < 1.0
    assert world.noise("block")[1] < 1.0  # no turn counted as noise
    assert not world.moves
