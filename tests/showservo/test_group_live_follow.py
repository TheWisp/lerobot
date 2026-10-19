"""The point groups view's following (benchmarks/group_live.py): new points around an object wherever it is now, so
what moves with it (the fingers holding it) is tracked and can join its group; and the per-frame record of the groups
its --dump keeps for offline analysis."""

from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np
import pytest

from lerobot.showservo import groups as groups_mod
from lerobot.showservo.groups import GroupTracker
from lerobot.showservo.groups_scene import stable_points

H, W = 480, 848  # the rig camera's colour frame


@pytest.fixture(scope="module")
def group_live():
    bench = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path.insert(0, str(bench))
    try:
        import group_live

        yield group_live
    finally:
        sys.path.remove(str(bench))


def _scene():
    """A textured frame with depth everywhere, and an object's 10 own points seen at (400, 240)."""
    rng = np.random.default_rng(5)
    gray = cv2.GaussianBlur(rng.uniform(0, 255, (H, W)).astype(np.float32), (0, 0), 2).astype(np.uint8)
    pts = np.ones((H, W, 3))
    own_uv = np.array([400.0, 240.0]) + rng.uniform(-8, 8, (10, 2))
    return gray, pts, own_uv


def test_following_adds_points_around_an_object_until_it_has_as_many_as_it_borrows(group_live):
    """Around where the object's own points are seen, corners up to ``n_ring``, all within RING_PX[1] of them, none on a
    track already there, on any surface with depth: points hidden under the fingers do not count as standing around
    it. With ``n_ring`` seen around it nothing is added, which bounds the following; nor while none of its own is
    seen."""
    gray, pts, own_uv = _scene()
    n_ring, own = 40, np.arange(10)
    reach = group_live.RING_PX[1]
    rng = np.random.default_rng(6)
    hidden_uv = np.array([400.0, 240.0]) + rng.uniform(-60, 60, (n_ring, 2))  # under the fingers
    uv = np.vstack([own_uv, hidden_uv])
    seen = np.r_[np.ones(10, bool), np.zeros(n_ring, bool)]
    owned = np.r_[np.ones(10, bool), np.zeros(n_ring, bool)]
    fresh, near = group_live.follow_points(stable_points, gray, pts, uv, seen, owned, own, n_ring)
    at = np.median(own_uv, axis=0)
    assert near == 0 and 0 < len(fresh) <= n_ring, (near, len(fresh))
    assert (np.linalg.norm(fresh - at, axis=1) <= reach).all(), "around it"
    assert np.linalg.norm(fresh[:, None] - own_uv[None], axis=2).min() > 6, "not on a track already there"

    around = at + rng.uniform(-reach / 2, reach / 2, (n_ring + 5, 2))  # more than it borrows, seen around it
    uv = np.vstack([own_uv, around])
    seen, owned = np.ones(15 + n_ring, bool), np.r_[np.ones(10, bool), np.zeros(n_ring + 5, bool)]
    fresh, near = group_live.follow_points(stable_points, gray, pts, uv, seen, owned, own, n_ring)
    assert near == n_ring + 5 and len(fresh) == 0

    seen = np.r_[np.zeros(10, bool), np.ones(n_ring + 5, bool)]  # none of its own seen
    fresh, near = group_live.follow_points(stable_points, gray, pts, uv, seen, owned, own, n_ring)
    assert len(fresh) == 0 and near == 0


def test_following_adds_points_only_where_the_camera_measured_depth(group_live):
    gray, pts, own_uv = _scene()
    pts[:, :400] = np.nan  # no depth left of the object
    fresh, _ = group_live.follow_points(
        stable_points, gray, pts, own_uv, np.ones(10, bool), np.ones(10, bool), np.arange(10), 40
    )
    assert len(fresh) and (fresh[:, 0] >= 400).all()


def test_the_dump_keeps_each_frames_groups_as_the_tracker_has_them(group_live):
    """A tray and a cube on it, the cube lifted off: each dumped frame keeps every track's group, each group's motion,
    each object's pose and which objects are moving to another group, as the tracker has them that frame."""
    rng = np.random.default_rng(3)
    tray = np.column_stack([rng.uniform(-0.2, 0.2, 200), rng.uniform(-0.12, 0.12, 200), np.full(200, 0.45)])
    cube = np.column_stack([rng.uniform(0.04, 0.08, 30), rng.uniform(0.0, 0.03, 30), np.full(30, 0.42)])
    t = GroupTracker()
    pose = np.eye(4)
    pose[:3, 3] = (0.06, 0.015, 0.42)
    t.add_object("cube", np.arange(200, 230), pose)
    for k in range(50):
        lift = np.array([0.0, 0.0, -0.0015 * max(k - 20, 0)])
        xyz = np.vstack([tray, cube + lift])
        t.update(xyz + rng.normal(0.0, 0.0015, xyz.shape), np.ones(230, bool))
    stamp, group_of, motions, poses, moving = group_live.groups_frame(12.5, t, groups_mod._matrix)
    assert stamp == 12.5 and np.array_equal(group_of, t.group_of) and group_of is not t.group_of
    assert set(motions) == set(t.groups) and len(t.groups) == 2, "the tray's group and the cube's"
    for gid, g in t.groups.items():
        np.testing.assert_allclose(motions[gid], groups_mod._matrix(g.motion))
    np.testing.assert_allclose(poses["cube"], t.pose("cube"))
    assert moving == sorted(o.name for o in t.objects.values() if o.pending is not None)
