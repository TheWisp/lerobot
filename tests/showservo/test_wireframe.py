"""Each point group's wireframe, kept from frame to frame (lerobot.showservo.wireframe), on a scene at the rig's size:
the rig's camera, a tray of 1600 tracks 0.45 m away and a box of 200 tracks standing 30 mm proud of it, a group of its
own; every frame the tracks jitter 0.4 px in the image and 1.5 mm in depth, as the rig's do."""

from __future__ import annotations

import numpy as np

from lerobot.showservo import wireframe
from lerobot.showservo.wireframe import Wireframe

FX = FY = 600.0
CX, CY = 424.0, 240.0


def _scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(xyz (N, 3) m, group (N,)): the tray, group 0, and the box on it, group 1."""
    tray = np.column_stack(
        [rng.uniform(-0.25, 0.25, 1600), rng.uniform(-0.15, 0.15, 1600), np.full(1600, 0.45)]
    )
    box = np.column_stack([rng.uniform(0.02, 0.08, 200), rng.uniform(-0.03, 0.03, 200), np.full(200, 0.42)])
    return np.vstack([tray, box]), np.r_[np.zeros(1600, int), np.ones(200, int)]


def _seen(xyz: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One frame's view of ``xyz``: (uv px, xyz m) with the tracks' jitter."""
    uv = np.column_stack([FX * xyz[:, 0] / xyz[:, 2] + CX, FY * xyz[:, 1] / xyz[:, 2] + CY])
    uv = uv + rng.normal(0.0, 0.4, uv.shape)
    z = xyz[:, 2] + rng.normal(0.0, 0.0015, len(xyz))
    return uv, np.column_stack([(uv[:, 0] - CX) / FX * z, (uv[:, 1] - CY) / FY * z, z])


def _pairs(tracks: np.ndarray, links: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted(map(int, p))) for p in tracks[links]} if len(links) else set()


def _changed(a: set, b: set) -> float:
    return len(a ^ b) / max(1, len(a | b))


def _crossings(uv: np.ndarray, links: np.ndarray) -> int:
    """How many pairs of links cross inside both (sharing an end does not count), every pair tried."""
    p1, p2 = uv[links[:, 0]], uv[links[:, 1]]

    def turn(o, a, b):
        return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (
            b[..., 0] - o[..., 0]
        )

    n = 0
    for s in range(0, len(links), 200):
        a1, a2, ia = p1[s : s + 200, None], p2[s : s + 200, None], links[s : s + 200, None, :]
        apart = ~(ia[..., :, None] == links[None, :, None, :]).any(axis=(2, 3))
        hit = (turn(a1, a2, p1[None]) * turn(a1, a2, p2[None]) < 0) & (
            turn(p1[None], p2[None], a1) * turn(p1[None], p2[None], a2) < 0
        )
        n += int((hit & apart).sum())
    return n // 2


def test_a_still_scene_keeps_its_wireframe_where_one_rebuilt_every_frame_flickers():
    """The fix for links that flipped every frame: after the first looks, a still scene's drawn links stay the same
    from frame to frame (under 1 % change), where the same wireframe built afresh every frame changes several times
    as much, from nothing but the jitter."""
    rng = np.random.default_rng(1)
    xyz, group = _scene(rng)
    tracks = np.arange(len(xyz))
    kept, prev_kept, prev_fresh, ch_kept, ch_fresh = Wireframe(), None, None, [], []
    for f in range(30):
        uv, seen = _seen(xyz, rng)
        now_kept = _pairs(tracks, kept.update(tracks, uv, seen, group).links)
        now_fresh = _pairs(tracks, Wireframe().update(tracks, uv, seen, group).links)
        if f >= 10:
            ch_kept.append(_changed(now_kept, prev_kept))
            ch_fresh.append(_changed(now_fresh, prev_fresh))
        prev_kept, prev_fresh = now_kept, now_fresh
    assert np.median(ch_kept) < 0.01, np.median(ch_kept)
    assert np.median(ch_fresh) > 4 * max(np.median(ch_kept), 0.005), (np.median(ch_fresh), np.median(ch_kept))


def test_no_link_joins_two_groups_or_reaches_past_its_range():
    """A link holds within a group: neither the box's tracks nor those of a sheet lying flat on the tray (a group of
    its own at the tray's depth, which only the groups tell apart) link to the tray's, and no link is longer in 3D
    than a kept one may be (DROP of the group's spacings). Within a group hardly any two links cross: a new one is
    tried against those kept, and a kept one that jitter folds across another is dropped on the next look."""
    rng = np.random.default_rng(2)
    xyz, group = _scene(rng)
    under = (xyz[:, 0] > -0.15) & (xyz[:, 0] < -0.08) & (np.abs(xyz[:, 1]) < 0.03) & (group == 0)
    xyz, group = xyz[~under], group[~under]  # the sheet covers the tray there
    sheet = np.column_stack(
        [rng.uniform(-0.15, -0.08, 150), rng.uniform(-0.03, 0.03, 150), np.full(150, 0.45)]
    )
    xyz, group = np.vstack([xyz, sheet]), np.r_[group, np.full(150, 2)]
    tracks = np.arange(len(xyz))
    wire = Wireframe()
    for f in range(4 * wireframe.ADD_EVERY + 1):
        uv, seen = _seen(xyz, rng)
        mesh = wire.update(tracks, uv, seen, group)
        a, b = mesh.links[:, 0], mesh.links[:, 1]
        assert len(mesh.links) > 2000, "the scene is meshed"
        assert (group[a] == group[b]).all(), "no link from the box or the sheet to the tray"
        reach = wireframe.DROP * np.array([mesh.spacing_mm[int(g)] for g in group[a]])
        assert (np.linalg.norm(seen[a] - seen[b], axis=1) * 1000.0 <= reach).all()
        crossed = sum(_crossings(uv, mesh.links[group[a] == g]) for g in np.unique(group))
        assert crossed <= 0.005 * len(mesh.links), (f, crossed)


def test_a_scene_farther_off_and_sparser_is_meshed_as_fully():
    """No millimetres assumed: the same scene three times as far from the camera and three times as wide (its tracks
    three times as far apart, as on YCBInEOAT's mustard0, where a 20 mm range left almost no link) gets as many links
    as at the rig's distance."""
    rng = np.random.default_rng(7)
    xyz, group = _scene(rng)
    tracks = np.arange(len(xyz))
    counts = []
    for scale in (1.0, 3.0):
        wire = Wireframe()
        for _ in range(4):
            uv, seen = _seen(xyz * scale, rng)
            mesh = wire.update(tracks, uv, seen, group)
        counts.append(len(mesh.links))
    assert counts[1] > 0.9 * counts[0], counts


def test_neighbouring_triangles_are_drawn_as_quads():
    """Most faces are quads: pairs of triangles drawn without the side they share."""
    rng = np.random.default_rng(3)
    xyz, group = _scene(rng)
    tracks = np.arange(len(xyz))
    wire = Wireframe()
    for _ in range(6):
        uv, seen = _seen(xyz, rng)
        mesh = wire.update(tracks, uv, seen, group)
    assert mesh.quads / (mesh.quads + mesh.triangles) > 0.5, (mesh.quads, mesh.triangles)


def test_a_new_track_is_linked_once_it_has_qualified_and_a_track_that_leaves_takes_its_links():
    """A track that appears is linked after PENDING looks (one every ADD_EVERY frames), not on its first frame; one
    that is no longer placed loses its links on that frame."""
    rng = np.random.default_rng(4)
    xyz, group = _scene(rng)
    newcomer = len(xyz) - 1  # in the middle of the tray, after the others have settled
    xyz[newcomer] = [0.0, 0.0, 0.45]
    group[newcomer] = 0
    tracks = np.arange(len(xyz))
    wire = Wireframe()
    for _ in range(8):  # the others settle without it
        uv, seen = _seen(xyz, rng)
        wire.update(tracks[:-1], uv[:-1], seen[:-1], group[:-1])
    linked_at = None
    for f in range(12):
        uv, seen = _seen(xyz, rng)
        mesh = wire.update(tracks, uv, seen, group)
        if (mesh.links == newcomer).any():
            linked_at = f
            break
    assert linked_at is not None and linked_at >= wireframe.ADD_EVERY * (wireframe.PENDING - 1), linked_at
    leaver = 100
    keep = tracks != leaver
    uv, seen = _seen(xyz, rng)
    mesh = wire.update(tracks[keep], uv[keep], seen[keep], group[keep])
    assert not (tracks[keep][mesh.links] == leaver).any(), "a track not placed keeps no link"


def test_a_groups_pieces_are_joined_only_when_asked_once_each_and_the_join_stays():
    """Two patches of one group 60 mm apart: no link between them by default; with joins, exactly one, longer than a
    gap and so a strip, and the same pair of tracks frame after frame."""
    rng = np.random.default_rng(5)
    a = np.column_stack([rng.uniform(-0.05, -0.02, 150), rng.uniform(-0.02, 0.02, 150), np.full(150, 0.45)])
    b = np.column_stack([rng.uniform(0.04, 0.07, 150), rng.uniform(-0.02, 0.02, 150), np.full(150, 0.45)])
    xyz, group = np.vstack([a, b]), np.zeros(300, int)
    tracks = np.arange(300)
    plain, joined, joins = Wireframe(), Wireframe(joins=True), []
    for _ in range(8):
        uv, seen = _seen(xyz, rng)
        m = plain.update(tracks, uv, seen, group)
        assert not len(m.joins) and not ((m.links < 150).any(axis=1) & (m.links >= 150).any(axis=1)).any()
        j = joined.update(tracks, uv, seen, group)
        assert len(j.joins) == 1, j.joins
        joins.append(_pairs(tracks, j.joins))
        p, q = j.joins[0]
        assert np.linalg.norm(seen[p] - seen[q]) * 1000.0 > wireframe.REMOTE * j.spacing_mm[0]
    assert all(s == joins[-1] for s in joins[2:]), "the join stays on the same tracks"


def test_the_depth_scale_follows_the_scene_slowly():
    """The colour scale starts at the scene's depths and, when the scene comes 100 mm closer, moves towards it a
    little each frame, so colours do not jump."""
    rng = np.random.default_rng(6)
    xyz, group = _scene(rng)
    tracks = np.arange(len(xyz))
    wire = Wireframe()
    uv, seen = _seen(xyz, rng)
    near0, far0 = wire.update(tracks, uv, seen, group).depth_range
    assert 410 < near0 < 425 and 445 < far0 < 455, (near0, far0)
    closer = xyz * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, 1.0]) * (xyz[:, 2:] - 0.1)
    uv, seen = _seen(closer, rng)
    near1, far1 = wire.update(tracks, uv, seen, group).depth_range
    assert far0 - 100 * 2 * wireframe.RANGE_SMOOTH < far1 < far0, (far0, far1)  # one step of the average
