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


def _fake_tracker(histories, members):
    """Groups with a motion history each (Rigid3, newest last) and the members' anchors; the fake the World reads."""
    from collections import deque
    from types import SimpleNamespace

    groups = {}
    group_of, anchors = [], []
    for gid, hist in histories.items():
        groups[gid] = SimpleNamespace(id=gid, history=deque(hist), motion=hist[-1])
        for p in members[gid]:
            group_of.append(gid)
            anchors.append(p)
    return SimpleNamespace(groups=groups, group_of=np.array(group_of), anchor=np.array(anchors, dtype=float))


def test_the_world_is_the_stillest_group_not_the_biggest_and_a_tie_keeps_it():
    from lerobot.showservo.pose import Rigid3

    still = [Rigid3.identity()] * 20
    slide = [Rigid3(np.eye(3), np.array([0.004 * i, 0.0, 0.0])) for i in range(20)]  # 4 mm a frame
    big = [[0.1 * i, 0.0, 0.5] for i in range(30)]  # the tray: thirty tracks
    small = [[0.0, 0.1 * i, 0.5] for i in range(8)]  # the desk: eight
    world = scene.World(window=15, settle=5)
    assert world.update(_fake_tracker({0: still, 1: still}, {0: big, 1: small})) == 0  # a tie: the bigger one
    assert world.update(_fake_tracker({0: still, 1: still}, {0: big, 1: small})) == 0  # and it keeps it
    assert (
        world.update(_fake_tracker({0: slide, 1: still}, {0: big, 1: small})) == 1
    )  # the tray slides: not the world
    assert (
        world.update(_fake_tracker({0: still, 1: still}, {0: big, 1: small})) == 1
    )  # at rest again, the desk keeps it
    young = _fake_tracker({0: slide, 1: still[:3]}, {0: big, 1: small})
    world = scene.World(window=15, settle=5)
    world.update(_fake_tracker({0: still, 1: still}, {0: big, 1: small}))
    assert world.update(young) == 0  # a group three frames old is not yet trusted to be still
    assert world.update(_fake_tracker({}, {})) is None


def test_a_group_too_young_to_judge_is_quiet_until_it_settles():
    """A split's first frames: the newborn group's motion is not known yet, so it is drawn as the world, white,
    rather than flashing a colour that may be the wrong way round once the stiller side is known."""
    from lerobot.showservo.pose import Rigid3

    still = [Rigid3.identity()] * 20
    big = [[0.1 * i, 0.0, 0.5] for i in range(30)]
    small = [[0.0, 0.1 * i, 0.5] for i in range(8)]
    world = scene.World(window=15, settle=5)
    world.update(_fake_tracker({0: still}, {0: big}))
    assert world.update(_fake_tracker({0: still, 1: still[:3]}, {0: big, 1: small})) == 0
    assert world.quiet == {1}
    assert scene.group_colour(1, 0, world.quiet) == (255, 255, 255)
    img = np.full((20, 20, 3), 100, np.uint8)
    surfaces = np.where(np.arange(20)[:, None] < 10, 0, 1).astype(np.int32)
    assert (
        scene.paint_surfaces(img, surfaces, base=0, quiet=world.quiet) == 100
    ).all()  # nothing painted yet
    assert world.update(_fake_tracker({0: still, 1: still[:8]}, {0: big, 1: small})) == 0
    assert world.quiet == set()
