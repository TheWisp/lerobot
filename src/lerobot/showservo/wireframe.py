# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Each point group drawn as a wireframe of quads between its tracks, kept from frame to frame.

A track links to its neighbours in its own group (the group's Delaunay triangulation in the image), a link only as
long in 3D as ADD times the group's own spacing (its triangles' median side, which a scene's track density and its
distance set: no millimetres assumed), so none crosses a depth step or another group; neighbouring triangles merge into
quads by dropping the side they share. Rebuilt from scratch every frame, the links flip with the tracks' jitter, the
depth's noise around the cut and tracks blinking in and out (on an act of 2026-10-11, a fifth of the still world's
links changed every frame); so the wireframe keeps what it drew: a link stays while both its tracks are there in the
same group and within DROP times its spacing, a new one is drawn once it has qualified PENDING times (looked for every
ADD_EVERY frames) and does not cross one kept, a triangle stays while its three sides do, a quad while both its
triangles do, and each track's depth is smoothed. Optionally (``joins``) each group's pieces are joined into one by
the shortest links between them, a join kept until one clearly shorter appears.

It runs every frame of the live view, so its cost is held to a few ms at the rig's ~2000 tracks: the Delaunay is
OpenCV's (about 1 ms there), what is kept is checked in a few array operations, and Python loops only over what
changed. Pure numpy and OpenCV, so the tracker's own environment can load it by path."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# A group's spacing: the median 3D length of its Delaunay triangles' sides, smoothed over looks. Each range is a multiple
# of it: on the rig (act 20261011_080249) the still world's spacing is 9.6 mm, so a new link reaches 20 mm, a kept one
# goes at 26 mm and a join past 40 mm is a strip, the ranges chosen there by eye.
ADD = 2.1  # a new link: no longer than this many spacings in 3D
DROP = 2.7  # a kept link: dropped once longer than this for MISSES frames, so depth noise does not flip it
SPACING_SMOOTH = 0.3  # each look's spacing is this much of the kept one
MISSES = 3
PENDING = 2  # a new link is drawn once it has qualified this many looks in a row
ADD_EVERY = 3  # new links are looked for every this many frames: the Delaunay and crossing tests cost most
MERGE_MIN = 0.7  # two triangles make a quad when their shared side is this share of their longest other one
DEPTH_SMOOTH = 0.3  # a track's depth: an exponential average, this much of each new reading
RANGE_SMOOTH = 0.05  # the depth colour scale follows the scene's 2nd to 98th percentile this slowly
RANGE_PERCENTILES = (2.0, 98.0)
STICKY = 0.8  # a join already drawn counts at this share of its length: replaced only by one clearly shorter
REMOTE = 4.2  # a join longer than this many spacings reaches a remote piece, drawn as a strip; a shorter one a link
CELL_PX = 48  # kept links are filed by their middle in cells this big, for a new link's crossing test
# A kept link no longer a Delaunay side, found crossing nothing, is tried again after this many looks.
CLEAN_LOOKS = 5
FORGET_FRAMES = 30  # a track not placed this many frames loses its smoothed depth


@dataclass
class Mesh:
    """One frame's wireframe, as indices into that frame's inputs: ``links`` (L, 2) the sides drawn (quads', lone
    triangles', and joins no longer than REMOTE spacings), ``joins`` (J, 2) the longer links joining each group's
    pieces into one (none unless asked for), drawn as strips; ``depth_mm`` (N,) each input's smoothed depth, 0 where
    it has none;
    the depth colour scale ``depth_range`` (near, far) in mm; ``spacing_mm`` each group's spacing; ``quads`` and
    ``triangles`` count the faces."""

    links: np.ndarray
    joins: np.ndarray
    depth_mm: np.ndarray
    depth_range: tuple[float, float]
    spacing_mm: dict[int, float]
    quads: int
    triangles: int


def _key2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One int64 per unordered pair of non-negative ints below 2**31."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    return (np.minimum(a, b) << 32) | np.maximum(a, b)


def _split2(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return k >> 32, k & 0xFFFFFFFF


def _key3(t: np.ndarray) -> np.ndarray:
    """One int64 per unordered triple of ints below 2**21 (a frame's input indices)."""
    s = np.sort(np.asarray(t, np.int64).reshape(-1, 3), axis=1)
    return (s[:, 0] << 42) | (s[:, 1] << 21) | s[:, 2]


def _member(table: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Whether each of ``query`` is in ``table``, which is sorted: one binary search, no sort (np.isin sorts both)."""
    if not len(table):
        return np.zeros(np.shape(query), bool)
    i = np.clip(np.searchsorted(table, query), 0, len(table) - 1)
    return table[i] == query


def _turn(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The cross product of o->a and o->b, over the last axis."""
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (
        b[..., 0] - o[..., 0]
    )


def _coord_keys(xy: np.ndarray) -> np.ndarray:
    """One uint64 per float32 point, its exact bits: how a corner OpenCV hands back is found among the points."""
    bits = np.ascontiguousarray(xy, dtype=np.float32).view(np.uint32).astype(np.uint64).reshape(-1, 2)
    return (bits[:, 0] << np.uint64(32)) | bits[:, 1]


def triangulate(uv: np.ndarray) -> np.ndarray:
    """The Delaunay triangles of points ``uv`` (N, 2) px, as (T, 3) indices; a point on the same spot as an earlier
    one is left out."""
    if len(uv) < 3:
        return np.zeros((0, 3), np.int64)
    lo, hi = np.floor(uv.min(axis=0)) - 8, np.ceil(uv.max(axis=0)) + 8
    sub = cv2.Subdiv2D((int(lo[0]), int(lo[1]), int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1))
    pts = np.ascontiguousarray(uv, dtype=np.float32)
    keys, first = np.unique(_coord_keys(pts), return_index=True)
    sub.insert(pts[first])
    corners = _coord_keys(sub.getTriangleList().astype(np.float32).reshape(-1, 2)).reshape(-1, 3)
    j = np.clip(np.searchsorted(keys, corners), 0, len(keys) - 1)
    mine = (keys[j] == corners).all(axis=1)  # not a triangle on the subdivision's outer corners
    return first[j[mine]].astype(np.int64)


class _Pieces:
    """Union-find over ints."""

    def __init__(self, keys) -> None:
        self.parent = {k: k for k in keys}

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def join(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[ra] = rb
        return True


class Wireframe:
    """The wireframe of every group, kept across the frames :meth:`update` is given, in order. What it keeps is by
    track number; each frame works on that frame's input indices."""

    def __init__(self, joins: bool = False) -> None:
        self.joins_on = joins
        self._ids = np.zeros(0, np.int64)  # tracks with a smoothed depth, sorted
        self._z = np.zeros(0)  # their smoothed depth, mm
        self._at = np.zeros(0, np.int64)  # the frame each was last placed
        self._links = np.zeros((0, 2), np.int64)  # kept links, by track
        self._miss = np.zeros(0, np.int64)  # frames each has been out of range
        self._pend = np.zeros(0, np.int64)  # candidate links by track (_key2), sorted
        self._pend_n = np.zeros(0, np.int64)  # frames in a row each has qualified
        self._blocked = np.zeros(0, np.int64)  # candidates found crossing a kept link (by track), sorted
        self._blocker = np.zeros(0, np.int64)  # the kept link each crossed (by track)
        self._clean = np.zeros(0, np.int64)  # stale kept links found crossing nothing (by track), sorted
        self._clean_at = np.zeros(0, np.int64)  # the look each was found so
        self._looks = 0
        self._faces = np.zeros((0, 3), np.int64)  # kept triangles, by track
        self._face_sign = np.zeros(0, np.int64)  # each one's turning sense when it was made
        self._quads = np.zeros((0, 2, 3), np.int64)  # kept quads: their two triangles, by track
        self._joins: set[tuple[int, int]] = set()  # by track
        self._range: tuple[float, float] | None = None
        self._spacing: dict[int, float] = {}  # group -> its spacing, mm
        self._lut = np.zeros(0, np.int64)  # track number -> this frame's index, -1 for none
        self._lut_set = np.zeros(0, np.int64)  # the numbers set in it
        self._frames = 0

    def update(self, tracks: np.ndarray, uv: np.ndarray, xyz: np.ndarray, group: np.ndarray) -> Mesh:
        """This frame's wireframe from its placed, grouped tracks: ``tracks`` (N,) their numbers, distinct, the same for
        a track all its life, dense from 0 (the tracker's own list indices: a lookup table that long is kept); ``uv``
        (N, 2) px; ``xyz`` (N, 3) camera frame, m; ``group`` (N,) their groups."""
        tracks = np.asarray(tracks, np.int64).reshape(-1)
        uv = np.asarray(uv, float).reshape(-1, 2)
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        group = np.asarray(group, np.int64).reshape(-1)
        n = len(tracks)
        z = xyz[:, 2] * 1000.0
        ok = np.isfinite(z) & (z > 0) & np.isfinite(uv).all(axis=1)
        depth = self._smooth(tracks, z, ok)
        pos = np.where(
            ok[:, None], xyz * 1000.0 * (depth / np.where(ok, z, 1.0))[:, None], np.nan
        )  # mm, same ray
        if ok.any():
            lo, hi = (float(v) for v in np.percentile(depth[ok], RANGE_PERCENTILES))
            a = RANGE_SMOOTH
            self._range = (
                (lo, max(hi, lo + 1.0))
                if self._range is None
                else ((1 - a) * self._range[0] + a * lo, (1 - a) * self._range[1] + a * hi)
            )
        # What is kept is by track number, everything below by this frame's index: a table from one to the other, kept
        # and grown, with only last frame's entries cleared, makes each lookup one gather.
        top = int(tracks.max()) + 1 if n else 0
        if top > len(self._lut):
            self._lut = np.concatenate(
                [self._lut, np.full(max(top, 2 * len(self._lut)) - len(self._lut), -1, np.int64)]
            )
        self._lut[self._lut_set] = -1
        self._lut[tracks] = np.arange(n)
        self._lut_set = tracks

        def index(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            inside = ids < len(self._lut)
            i = np.where(inside, self._lut[np.where(inside, ids, 0)], -1)
            here = i >= 0
            here[here] &= ok[i[here]]
            return np.maximum(i, 0), here

        links = self._kept_links(index, pos, group)
        links = self._unfold(index, links, uv)
        if self._frames % ADD_EVERY == 0:  # new links and triangles: looked for every ADD_EVERY frames
            tri = self._triangles(uv, ok, group)
            links = self._uncross(tri, links, tracks, uv)
            self._looks += 1
            links = self._new_links(tri, links, tracks, pos, uv, group)
        else:
            tri = np.zeros((0, 3), np.int64)
        faces, signs = self._kept_faces(index, links, tri, uv)
        quads = self._kept_quads(index, faces, uv, pos)
        lk = _key2(links[:, 0], links[:, 1])
        diag = np.sort(self._diagonals(quads))
        drawn = links[~_member(diag, lk)] if len(diag) else links
        joins = np.zeros((0, 2), np.int64)
        if self.joins_on:
            joins = self._join(links, tracks, pos, group, ok)
            short = (
                np.linalg.norm(pos[joins[:, 0]] - pos[joins[:, 1]], axis=1)
                <= REMOTE * self._spacings(group[joins[:, 0]])
                if len(joins)
                else []
            )
            if len(joins):
                drawn = np.vstack([drawn, joins[short]])
                joins = joins[~short]
        # what is kept, by track
        self._links = tracks[links]
        self._faces, self._face_sign = tracks[faces], signs
        self._quads = tracks[quads] if len(quads) else np.zeros((0, 2, 3), np.int64)
        self._joins = {(int(tracks[a]), int(tracks[b])) for a, b in joins} if self.joins_on else set()
        self._frames += 1
        return Mesh(
            links=drawn.astype(int),
            joins=joins.astype(int),
            depth_mm=np.where(ok, depth, 0.0),
            depth_range=self._range or (0.0, 1.0),
            spacing_mm=dict(self._spacing),
            quads=len(quads),
            triangles=len(faces) - 2 * len(quads),
        )

    def _smooth(self, tracks: np.ndarray, z: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """Each track's depth smoothed over frames; tracks not placed for FORGET_FRAMES are forgotten."""
        i = np.clip(np.searchsorted(self._ids, tracks), 0, max(len(self._ids) - 1, 0))
        known = (self._ids[i] == tracks) if len(self._ids) else np.zeros(len(tracks), bool)
        depth = np.where(
            known & ok, (1 - DEPTH_SMOOTH) * (self._z[i] if len(self._ids) else 0.0) + DEPTH_SMOOTH * z, z
        )
        upd = known & ok
        self._z[i[upd]] = depth[upd]
        self._at[i[upd]] = self._frames
        new = ok & ~known
        ids = np.concatenate([self._ids, tracks[new]])
        zs = np.concatenate([self._z, depth[new]])
        at = np.concatenate([self._at, np.full(int(new.sum()), self._frames)])
        live = at > self._frames - FORGET_FRAMES
        o = np.argsort(ids[live], kind="stable")
        self._ids, self._z, self._at = ids[live][o], zs[live][o], at[live][o]
        return np.where(ok, depth, 0.0)

    def _update_spacing(self, side_group: np.ndarray, length: np.ndarray) -> None:
        """Each group's spacing from this look's Delaunay sides (``length`` mm, ``side_group`` their group)."""
        order = np.argsort(side_group, kind="stable")
        groups, starts = np.unique(side_group[order], return_index=True)
        for g, part in zip(groups.tolist(), np.split(length[order], starts[1:]), strict=True):
            m = float(np.median(part))
            old = self._spacing.get(g)
            self._spacing[g] = m if old is None else (1 - SPACING_SMOOTH) * old + SPACING_SMOOTH * m

    def _spacings(self, group: np.ndarray) -> np.ndarray:
        """Each entry's group's spacing, mm; infinite for a group not looked at yet (it has no links)."""
        if not len(group):
            return np.zeros(0)
        groups, inverse = np.unique(group, return_inverse=True)
        return np.array([self._spacing.get(int(g), np.inf) for g in groups])[inverse]

    def _kept_links(self, index, pos: np.ndarray, group: np.ndarray) -> np.ndarray:
        """The kept links still standing, as index pairs: both tracks here, in the same group, within DROP of its
        spacings (out of it fewer than MISSES frames)."""
        if not len(self._links):
            return np.zeros((0, 2), np.int64)
        a, ha = index(self._links[:, 0])
        b, hb = index(self._links[:, 1])
        here = ha & hb
        here[here] &= group[a[here]] == group[b[here]]
        long = np.zeros(len(a), bool)
        long[here] = np.linalg.norm(pos[a[here]] - pos[b[here]], axis=1) > DROP * self._spacings(
            group[a[here]]
        )
        miss = np.where(long, self._miss + 1, 0)
        keep = here & (miss < MISSES)
        self._miss = miss[keep]
        return np.stack([a[keep], b[keep]], axis=1)

    @staticmethod
    def _triangles(uv: np.ndarray, ok: np.ndarray, group: np.ndarray) -> np.ndarray:
        """This frame's Delaunay triangles of each group's tracks, as index triples."""
        out = []
        for g in np.unique(group[ok]):
            members = np.flatnonzero(ok & (group == g))
            if len(members) >= 3:
                out.append(members[triangulate(uv[members])])
        return np.vstack(out) if out else np.zeros((0, 3), np.int64)

    def _new_links(self, tri, links, tracks, pos, uv, group) -> np.ndarray:
        """``links`` and the new ones: this frame's Delaunay sides within ADD of their group's spacings (the spacings
        updated from these sides first), not kept yet, once they have been so PENDING looks in a row (on the first
        frame at once) and cross no kept link."""
        if len(tri):
            keys = np.unique(
                np.concatenate(
                    [_key2(tri[:, 0], tri[:, 1]), _key2(tri[:, 1], tri[:, 2]), _key2(tri[:, 0], tri[:, 2])]
                )
            )
            sides = np.stack(_split2(keys), axis=1)
            length = np.linalg.norm(pos[sides[:, 0]] - pos[sides[:, 1]], axis=1)
            self._update_spacing(group[sides[:, 0]], length)
            fresh = ~_member(np.sort(_key2(links[:, 0], links[:, 1])), keys)
            sides = sides[fresh & (length <= ADD * self._spacings(group[sides[:, 0]]))]
        else:
            sides = np.zeros((0, 2), np.int64)
        by_track = _key2(tracks[sides[:, 0]], tracks[sides[:, 1]]) if len(sides) else np.zeros(0, np.int64)
        i = np.clip(np.searchsorted(self._pend, by_track), 0, max(len(self._pend) - 1, 0))
        was = (self._pend[i] == by_track) if len(self._pend) else np.zeros(len(by_track), bool)
        count = np.where(was, self._pend_n[i] + 1 if len(self._pend) else 1, 1)
        ready = (count >= PENDING) | (self._frames == 0)
        # One found crossing a kept link stays out while that link is kept, without being tested again.
        kept_by_track = (
            np.sort(_key2(tracks[links[:, 0]], tracks[links[:, 1]])) if len(links) else np.zeros(0, np.int64)
        )
        j = np.clip(np.searchsorted(self._blocked, by_track), 0, max(len(self._blocked) - 1, 0))
        still = (
            (self._blocked[j] == by_track) & _member(kept_by_track, self._blocker[j])
            if len(self._blocked)
            else np.zeros(len(by_track), bool)
        )
        test = ready & ~still
        cand = sides[test]
        hit, blocker = (
            self._crossing(cand, links, uv)
            if len(cand) and len(links)
            else (np.zeros(len(cand), bool), np.zeros(len(cand), np.int64))
        )
        accepted = np.zeros(len(sides), bool)
        accepted[np.flatnonzero(test)[~hit]] = True
        blocked_now = np.concatenate([by_track[still], by_track[test][hit]])
        blockers = np.concatenate(
            [
                self._blocker[j][still] if len(self._blocked) else np.zeros(0, np.int64),
                _key2(tracks[links[blocker[hit], 0]], tracks[links[blocker[hit], 1]]),
            ]
        )
        o = np.argsort(blocked_now, kind="stable")
        self._blocked, self._blocker = blocked_now[o], blockers[o]
        o = np.argsort(by_track[~accepted], kind="stable")
        self._pend, self._pend_n = by_track[~accepted][o], count[~accepted][o]
        self._miss = np.concatenate([self._miss, np.zeros(int(accepted.sum()), np.int64)])
        return np.vstack([links, sides[accepted]]) if accepted.any() else links

    @staticmethod
    def _crossing(cand: np.ndarray, links: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each candidate link, whether it crosses a kept one inside both (sharing an end does not count), and the
        index of one it crosses. Kept links are looked up by the cells around each candidate's middle: any crossing
        one lies within a link's length of it."""

        def cell_key(xy: np.ndarray) -> np.ndarray:
            c = np.floor(xy / CELL_PX).astype(np.int64)
            return c[..., 0] * 100003 + c[..., 1]

        ck = cell_key((uv[links[:, 0]] + uv[links[:, 1]]) / 2)
        o = np.argsort(ck, kind="stable")
        ck = ck[o]
        around = (
            cell_key((uv[cand[:, 0]] + uv[cand[:, 1]]) / 2)[:, None]
            + (np.array([dx * 100003 + dy for dx in (-1, 0, 1) for dy in (-1, 0, 1)])[None, :])
        )  # (C, 9): the cells around each candidate's middle
        lo, hi = np.searchsorted(ck, around), np.searchsorted(ck, around, side="right")
        count = (hi - lo).ravel()
        if not count.sum():
            return np.zeros(len(cand), bool), np.zeros(len(cand), np.int64)
        # every (candidate, kept link) pair the cells give, in one go
        which = np.repeat(np.repeat(np.arange(len(cand)), 9), count)
        starts = np.repeat(lo.ravel(), count)
        steps = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
        mi = o[starts + steps]
        m = links[mi]
        c = cand[which]
        apart = (m[:, 0] != c[:, 0]) & (m[:, 0] != c[:, 1]) & (m[:, 1] != c[:, 0]) & (m[:, 1] != c[:, 1])
        p1, p2, q1, q2 = uv[c[:, 0]], uv[c[:, 1]], uv[m[:, 0]], uv[m[:, 1]]
        hit = (
            apart & (_turn(p1, p2, q1) * _turn(p1, p2, q2) < 0) & (_turn(q1, q2, p1) * _turn(q1, q2, p2) < 0)
        )
        first = np.full(len(cand), -1, np.int64)
        first[which[hit][::-1]] = mi[hit][::-1]  # the first crossing link found for each candidate
        return first >= 0, np.maximum(first, 0)

    def _unfold(self, index, links: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """``links`` without the longest side of each kept triangle that has turned over: a corner jittered across the
        side opposite it (in a sliver), so that side now crosses another link; the Delaunay proposes the right one
        again. ``_miss`` follows."""
        if not len(self._faces) or not len(links):
            return links
        f, here = index(self._faces)
        whole = here.all(axis=1)
        f = f[whole]
        now = np.sign(_turn(uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]])).astype(np.int64)
        over = f[(now != self._face_sign[whole]) & (now != 0)]
        if not len(over):
            return links
        sides = np.stack([over[:, [0, 1]], over[:, [1, 2]], over[:, [0, 2]]], axis=1)  # (K, 3, 2)
        longest = sides[
            np.arange(len(over)), np.linalg.norm(uv[sides[..., 0]] - uv[sides[..., 1]], axis=2).argmax(axis=1)
        ]
        stay = ~_member(np.sort(_key2(longest[:, 0], longest[:, 1])), _key2(links[:, 0], links[:, 1]))
        self._miss = self._miss[stay]
        return links[stay]

    def _uncross(self, tri: np.ndarray, links: np.ndarray, tracks: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """``links`` without those that cross another: two sides of this frame's Delaunay never cross, so only a kept
        link that is no longer one of them is tried, and dropped where it crosses; one found clean is tried again only
        after CLEAN_LOOKS looks (a new link never crosses it: each is tried on the way in, so only motion can make it
        cross). ``_miss`` follows."""
        if not len(links) or not len(tri):
            return links
        sides = np.sort(
            np.concatenate(
                [_key2(tri[:, 0], tri[:, 1]), _key2(tri[:, 1], tri[:, 2]), _key2(tri[:, 0], tri[:, 2])]
            )
        )
        stale = np.flatnonzero(~_member(sides, _key2(links[:, 0], links[:, 1])))
        by_track = _key2(tracks[links[stale, 0]], tracks[links[stale, 1]])
        j = np.clip(np.searchsorted(self._clean, by_track), 0, max(len(self._clean) - 1, 0))
        fresh = (
            (self._clean[j] == by_track) & (self._looks - self._clean_at[j] < CLEAN_LOOKS)
            if len(self._clean)
            else np.zeros(len(stale), bool)
        )
        test = stale[~fresh]
        hit, _ = self._crossing(links[test], links, uv) if len(test) else (np.zeros(0, bool), None)
        clean = np.concatenate([by_track[fresh], by_track[~fresh][~hit]])
        at = np.concatenate(
            [
                self._clean_at[j[fresh]] if len(self._clean) else np.zeros(0, np.int64),
                np.full(int((~hit).sum()), self._looks),
            ]
        )
        o = np.argsort(clean, kind="stable")
        self._clean, self._clean_at = clean[o], at[o]
        stay = np.ones(len(links), bool)
        stay[test[hit]] = False
        self._miss = self._miss[stay]
        return links[stay]

    def _kept_faces(
        self, index, links: np.ndarray, tri: np.ndarray, uv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Triangles, as index triples, with each one's turning sense when it was made: the kept ones whose three sides
        still stand, and this frame's Delaunay triangles whose three sides are links and that are not kept already."""
        lk = np.sort(_key2(links[:, 0], links[:, 1]))

        def sides_linked(f: np.ndarray) -> np.ndarray:
            if not len(f) or not len(lk):
                return np.zeros(len(f), bool)
            return (
                _member(lk, _key2(f[:, 0], f[:, 1]))
                & _member(lk, _key2(f[:, 1], f[:, 2]))
                & _member(lk, _key2(f[:, 0], f[:, 2]))
            )

        kept, kept_sign = np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
        if len(self._faces):
            f, here = index(self._faces)
            whole = here.all(axis=1)
            f, sign = f[whole], self._face_sign[whole]
            stand = sides_linked(f)
            kept, kept_sign = f[stand], sign[stand]
        new = tri[sides_linked(tri)] if len(tri) else tri
        if len(kept) and len(new):
            new = new[~_member(np.sort(_key3(kept)), _key3(new))]
        new_sign = (
            np.sign(_turn(uv[new[:, 0]], uv[new[:, 1]], uv[new[:, 2]])).astype(np.int64)
            if len(new)
            else kept_sign[:0]
        )
        return (
            (np.vstack([kept, new]), np.concatenate([kept_sign, new_sign])) if len(new) else (kept, kept_sign)
        )

    def _kept_quads(self, index, faces: np.ndarray, uv: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Quads, as (Q, 2, 3) index triples of their two triangles: the kept ones whose triangles both stand (not
        judged convex again: jitter would flip a flat one in and out), and new ones among the triangles left, the one
        with the fewest possible partners first, each with its partner whose shared side is longest against their
        other sides, where that side is at least MERGE_MIN of the longest other one and the quad is convex."""
        fk = _key3(faces) if len(faces) else np.zeros(0, np.int64)
        fk_sorted = np.sort(fk)
        kept = np.zeros((0, 2, 3), np.int64)
        if len(self._quads):
            qi, qh = index(self._quads)
            qi = qi[qh.reshape(len(qh), -1).all(axis=1)]
            if len(qi) and len(fk):
                stand = _member(fk_sorted, _key3(qi[:, 0])) & _member(fk_sorted, _key3(qi[:, 1]))
                kept = qi[stand]
        used = (
            _member(np.sort(np.concatenate([_key3(kept[:, 0]), _key3(kept[:, 1])])), fk)
            if len(kept)
            else np.zeros(len(fk), bool)
        )
        free = np.flatnonzero(~used)
        if len(free) < 2:
            return kept
        # sides of the free triangles shared by exactly two of them: the candidate quads
        f = faces[free]
        sides = np.concatenate([_key2(f[:, 0], f[:, 1]), _key2(f[:, 1], f[:, 2]), _key2(f[:, 0], f[:, 2])])
        owner = np.tile(np.arange(len(free)), 3)
        o = np.argsort(sides, kind="stable")
        s, w = sides[o], owner[o]
        pair = np.flatnonzero(s[1:] == s[:-1])
        if not len(pair):
            return kept
        t1, t2, shared = w[pair], w[pair + 1], s[pair]
        a, b = _split2(shared)
        c1 = f[t1].sum(axis=1) - a - b  # each triangle's corner off the shared side
        c2 = f[t2].sum(axis=1) - a - b
        ab = np.linalg.norm(pos[a] - pos[b], axis=1)
        others = np.stack(
            [
                np.linalg.norm(pos[a] - pos[c1], axis=1),
                np.linalg.norm(pos[b] - pos[c1], axis=1),
                np.linalg.norm(pos[a] - pos[c2], axis=1),
                np.linalg.norm(pos[b] - pos[c2], axis=1),
            ],
            axis=1,
        )
        longest = others.max(axis=1)
        score = np.where(longest > 0, ab / np.where(longest > 0, longest, 1.0), 0.0)
        quad = np.stack([uv[a], uv[c1], uv[b], uv[c2]], axis=1)
        turns = np.stack(
            [_turn(quad[:, k], quad[:, (k + 1) % 4], quad[:, (k + 2) % 4]) for k in range(4)], axis=1
        )
        good = (score >= MERGE_MIN) & ((turns > 0).all(axis=1) | (turns < 0).all(axis=1))
        t1, t2, score = t1[good], t2[good], score[good]
        if not len(t1):
            return kept
        partners = np.bincount(np.concatenate([t1, t2]), minlength=len(free))
        best = np.zeros(len(free))
        np.maximum.at(best, t1, score)
        np.maximum.at(best, t2, score)
        # each candidate in the order of its first triangle: fewest partners, then best score; a greedy matching
        lead = np.where(
            partners[t1] < partners[t2], t1, np.where(partners[t2] < partners[t1], t2, np.minimum(t1, t2))
        )
        rank = np.lexsort((-score, -best[lead], partners[lead]))
        taken = np.zeros(len(free), bool)
        new = []
        for k in rank.tolist():
            x, y = int(t1[k]), int(t2[k])
            if not taken[x] and not taken[y]:
                taken[x] = taken[y] = True
                new.append((free[x], free[y]))
        if not new:
            return kept
        nq = np.stack([faces[[p for p, _ in new]], faces[[q for _, q in new]]], axis=1)
        return np.vstack([kept, nq]) if len(kept) else nq

    @staticmethod
    def _diagonals(quads: np.ndarray) -> np.ndarray:
        """The shared side of each quad's two triangles, as link keys."""
        if not len(quads):
            return np.zeros(0, np.int64)
        t1, t2 = quads[:, 0], quads[:, 1]
        shared = t1[(t1[:, :, None] == t2[:, None, :]).any(axis=2)].reshape(-1, 2)  # two corners in common
        return _key2(shared[:, 0], shared[:, 1])

    def _join(self, links, tracks, pos, group, ok) -> np.ndarray:
        """Per group, the minimum spanning tree over its pieces: between every two pieces their closest pair of tracks
        in 3D, a join already drawn counted at STICKY of its length. As index pairs."""
        by_track = {int(t): i for i, t in enumerate(tracks.tolist())}
        out = []
        for g in np.unique(group[ok]):
            members = np.flatnonzero(ok & (group == g)).tolist()
            pieces = _Pieces(members)
            mset = set(members)
            for a, b in links.tolist():
                if a in mset and b in mset:
                    pieces.join(a, b)
            parts: dict[int, list[int]] = {}
            for m in members:
                parts.setdefault(pieces.find(m), []).append(m)
            if len(parts) < 2:
                continue
            cands = []
            for ta, tb in self._joins:
                a, b = by_track.get(ta), by_track.get(tb)
                if a in mset and b in mset and pieces.find(a) != pieces.find(b):
                    cands.append((STICKY * float(np.linalg.norm(pos[a] - pos[b])), a, b))
            reps = list(parts)
            arr = {r: pos[parts[r]] for r in reps}
            for x, rx in enumerate(reps):
                for ry in reps[x + 1 :]:
                    d = np.linalg.norm(arr[rx][:, None, :] - arr[ry][None, :, :], axis=2)
                    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
                    cands.append((float(d[i, j]), parts[rx][i], parts[ry][j]))
            out += [(min(a, b), max(a, b)) for _w, a, b in sorted(cands) if pieces.join(a, b)]
        return np.array(out, np.int64).reshape(-1, 2)
