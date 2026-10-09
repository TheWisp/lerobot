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
