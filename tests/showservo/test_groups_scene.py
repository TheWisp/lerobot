"""The groups for the eye: the world wears no colour, a group moving differently does, and nothing blinks."""

from __future__ import annotations

import numpy as np

from lerobot.showservo import groups_scene as scene


def test_the_world_keeps_its_own_colours_and_another_group_is_tinted():
    img = np.full((20, 20, 3), 100, np.uint8)
    surfaces = np.full((20, 20), -1, np.int32)
    surfaces[:10] = 0  # the world
    surfaces[10:] = 1  # a body moving differently
    out = scene.paint_surfaces(img, surfaces, base=0)
    assert (out[:10] == 100).all()
    assert (out[12:18, 2:18] != 100).any()
    assert (
        scene.paint_surfaces(img, np.where(surfaces == 1, 0, surfaces), base=0) == 100
    ).all()  # one group: no tint


def test_the_base_is_the_group_with_the_most_tracks_and_wears_white():
    group_of = np.array([1, 1, 1, 0, 0, -1])
    assert scene.base_group(group_of) == 1
    assert scene.group_colour(1, 1) == (255, 255, 255)
    assert scene.group_colour(0, 1) != (255, 255, 255)
    assert scene.group_colour(-1, 1) == (160, 160, 160)
    assert scene.base_group(np.array([-1, -1])) is None


def test_a_label_held_one_frame_does_not_show_and_one_held_two_does():
    memory = scene.SurfaceMemory()
    world = np.zeros((4, 4), np.int32)
    flipped = world.copy()
    flipped[1, 1] = 1
    memory.update(world)
    memory.update(world)
    assert (memory.update(flipped) == world).all()  # a blink
    assert memory.update(flipped)[1, 1] == 1  # held: shown
    hole = world.copy()
    hole[2, 2] = -1
    memory.update(world)
    assert memory.update(hole)[2, 2] == 0  # a depth hole that opens for one frame does not open on the screen


def test_a_groups_paint_spreads_from_its_tracks_over_its_own_surface_and_stops_at_a_step():
    """Two flat surfaces 30 mm apart, side by side. A moving group's tracks on the left one paint the left surface
    around them, never the right one across the step, and never farther than the reach; the world paints nothing."""
    depth = np.full((120, 200), 0.50, np.float32)
    depth[:, 100:] = 0.53  # a step: another body
    uv = np.array([[30.0, 60.0], [50.0, 60.0], [150.0, 60.0]], np.float32)
    seen = np.array([True, True, True])
    group_of = np.array(
        [1, 1, 0]
    )  # two tracks of a body moving differently on the left; the world's on the right
    paint = scene.group_surfaces(depth, uv, seen, group_of, base=0, reach_px=20)
    assert paint[60, 30] == 1 and paint[60, 50] == 1 and paint[60, 40] == 1  # between the two tracks
    assert (paint[:, 100:] == -1).all()  # not across the step, and the world's own track paints nothing
    assert paint[60, 2] == -1 and paint[5, 30] == -1  # farther than the reach
    assert (scene.group_surfaces(depth, uv, seen, np.array([0, 0, 0]), base=0) == -1).all()
