# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Rigid groups over tracked 3D points: which points move together, and an object's pose from its group while its
own points are hidden.

Two loops. **Membership:** every tracked point belongs to one group or to none. A group has one rigid motion per
frame, fitted over its visible members (:func:`pose.ransac_fit_rigid`) from where each was when it joined, in the
group's frame. A member whose residual stays above ``leave_m`` for ``leave_frames`` frames leaves. A point of no group
joins the group whose motion explains its last ``join_frames`` frames (carried back into that group's frame, it held
still), or founds a new group with the other free points that moved rigidly with it (fission). Two groups whose
relative motion has held still for ``merge_frames`` frames merge (fusion). **Pose:** an object is a set of tracks and
a frame; its pose is its group's motion composed with its anchor in the group's frame, re-anchored when its tracks
change group. While its own points are hidden, the group's other members carry it: points borrowed from whatever it
rests on, for as long as it rests on it.

Nothing is assumed static. The camera's own frame is just where the first group was born; the tray is a group like
any other and moves like one. All geometry is 3D in the camera frame, so a turn out of the image plane is a rigid
motion and not a disagreement.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.showservo.pose import Rigid3, RigidFit, ransac_fit_rigid


@dataclass
class Group:
    """One rigid body's motion, camera <- its own frame (the camera frame at its birth)."""

    id: int
    motion: Rigid3
    born: int
    history: deque = field(default_factory=lambda: deque(maxlen=120))  # motion per frame, newest last
    supported: bool = True  # fitted this frame on enough visible members, or held from the last frame
    rms: float = 0.0
    n_fit: int = 0


@dataclass
class TrackedObject:
    """A named set of tracks with a frame. ``pose`` is camera <- object, the newest estimate."""

    name: str
    tracks: np.ndarray
    pose: np.ndarray
    group: int | None = None
    anchor: np.ndarray | None = None  # pose in its group's frame
    n_seen: int = 0
    n_grouped: int = 0
    model: dict = field(
        default_factory=dict
    )  # track -> its position in the object's frame, from its first sighting
    carried: np.ndarray | None = (
        None  # this frame's pose from the group alone, before its own points had a say
    )
    own_ok: bool = False  # its own points placed it this frame


class GroupTracker:
    """Pre: every ``update`` brings every track so far (indices are identities; the arrays only grow), as 3D points in
    the camera frame with a ``seen`` flag (tracked, visible, with depth). Post: ``groups`` and each track's group;
    each object's pose."""

    def __init__(
        self,
        *,
        leave_m: float = 0.010,
        join_m: float = 0.006,
        leave_frames: int = 3,
        join_frames: int = 3,
        min_group: int = 8,
        merge_frames: int = 60,
        merge_m: float = 0.004,
        merge_deg: float = 5.0,
        step_deg: float = 45.0,
        step_m: float = 0.15,
        min_own: int = 6,
    ) -> None:
        self.leave_m, self.join_m = leave_m, join_m
        self.leave_frames, self.join_frames = leave_frames, join_frames
        self.min_group, self.min_own = min_group, min_own
        self.merge_frames, self.merge_m, self.merge_deg = merge_frames, merge_m, merge_deg
        self.step_deg, self.step_m = (
            step_deg,
            step_m,
        )  # a group's motion within one frame, for the fit's prior
        self.frame = -1
        self.group_of = np.zeros(0, dtype=int)
        self.anchor = np.zeros((0, 3))
        self.strikes = np.zeros(0, dtype=int)  # frames in a row a member sat outside its group's fit
        self.unexplained = np.zeros(
            0, dtype=int
        )  # frames in a row a free point was seen and no group explained it
        # A point that left a group may not rejoin it at once: a slow slide (2 mm a frame) is within the join test's
        # noise over a short window, while its distance from where it joined is what made it leave.
        self.banned = np.full(0, -1, dtype=int)
        self.cooldown = np.zeros(0, dtype=int)
        self.groups: dict[int, Group] = {}
        self.objects: dict[str, TrackedObject] = {}
        self.positions: deque = deque(maxlen=join_frames + 1)  # (xyz, seen) of the last frames, newest last
        self._next_group = 0

    # ── the two loops ──

    def update(self, xyz: np.ndarray, seen: np.ndarray) -> None:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        seen = np.asarray(seen, dtype=bool).reshape(-1) & np.isfinite(xyz).all(axis=1)
        self._grow(len(xyz))
        self.frame += 1
        self.positions.append((xyz, seen))
        self.cooldown = np.maximum(self.cooldown - 1, 0)
        self._fit_groups(xyz, seen)
        self._leave()
        self._join(xyz, seen)
        self._fission(xyz, seen)
        self._fusion()
        self._place_objects(xyz, seen)

    def _grow(self, n: int) -> None:
        k = n - len(self.group_of)
        if k > 0:
            self.group_of = np.concatenate([self.group_of, np.full(k, -1)])
            self.anchor = np.concatenate([self.anchor, np.full((k, 3), np.nan)])
            self.strikes = np.concatenate([self.strikes, np.zeros(k, dtype=int)])
            self.unexplained = np.concatenate([self.unexplained, np.zeros(k, dtype=int)])
            self.banned = np.concatenate([self.banned, np.full(k, -1, dtype=int)])
            self.cooldown = np.concatenate([self.cooldown, np.zeros(k, dtype=int)])

    def _fit_groups(self, xyz: np.ndarray, seen: np.ndarray) -> None:
        for g in self.groups.values():
            members = np.flatnonzero(self.group_of == g.id)
            use = seen[members]
            fit: RigidFit | None = None
            if use.sum() >= 4:
                fit = ransac_fit_rigid(
                    self.anchor[members],
                    xyz[members],
                    valid=use,
                    inlier_m=self.leave_m,
                    prior=g.motion,
                    prior_rot_deg=self.step_deg,
                    prior_trans_m=self.step_m,
                )
            if fit is not None and fit.ok:
                g.motion, g.supported, g.rms, g.n_fit = fit.transform, True, fit.rms, fit.n_inliers
                res = np.linalg.norm(g.motion.apply(self.anchor[members]) - xyz[members], axis=1)
                self.strikes[members[use & (res > self.leave_m)]] += 1
                self.strikes[members[use & (res <= self.leave_m)]] = 0
            else:
                g.supported = False  # too few seen: its motion is held, its members stay
            g.history.append(g.motion)

    def _leave(self) -> None:
        leaving = np.flatnonzero(self.strikes >= self.leave_frames)
        self.banned[leaving] = self.group_of[leaving]
        self.cooldown[leaving] = 2 * (self.leave_frames + self.join_frames)
        self.group_of[leaving] = -1
        self.anchor[leaving] = np.nan
        self.strikes[leaving] = 0

    def _window(self, i: int) -> np.ndarray | None:
        """A free point's positions over the window, oldest first, or None when it was not seen throughout."""
        if len(self.positions) < self.positions.maxlen:
            return None
        pts = []
        for xyz, seen in self.positions:
            if i >= len(seen) or not seen[i]:
                return None
            pts.append(xyz[i])
        return np.asarray(pts)

    def _join(self, xyz: np.ndarray, seen: np.ndarray) -> None:
        # A group younger than the window is judged over the frames it has (two at least).
        candidates = [g for g in self.groups.values() if len(g.history) >= 2]
        inverses = {
            g.id: [m.inverse() for m in list(g.history)[-self.positions.maxlen :]] for g in candidates
        }
        members = {g.id: np.flatnonzero(self.group_of == g.id) for g in candidates}
        for i in np.flatnonzero((self.group_of == -1) & seen):
            track = self._window(int(i))
            if track is None:
                continue
            explained = []
            for g in candidates:
                if self.banned[i] == g.id and self.cooldown[i] > 0:
                    continue
                invs = inverses[g.id]
                carried = np.asarray(
                    [inv.apply(p[None])[0] for inv, p in zip(invs, track[-len(invs) :], strict=True)]
                )
                # RMS about the mean: the noise of a still point, not its worst sample
                if (
                    float(np.sqrt((np.linalg.norm(carried - carried.mean(axis=0), axis=1) ** 2).mean()))
                    <= self.join_m
                ):
                    explained.append(g)
            if not explained:
                self.unexplained[i] += 1
                continue
            # Groups that move alike explain it alike (they will merge): the one with a member nearest to it takes it.
            best = min(explained, key=lambda g: self._nearest(members[g.id], xyz, int(i)))
            self.group_of[i] = best.id
            self.anchor[i] = best.motion.inverse().apply(xyz[i][None])[0]
            self.unexplained[i] = 0

    @staticmethod
    def _nearest(members: np.ndarray, xyz: np.ndarray, i: int) -> float:
        if len(members) == 0:
            return np.inf
        return float(np.nanmin(np.linalg.norm(xyz[members] - xyz[i], axis=1)))

    def _fission(self, xyz: np.ndarray, seen: np.ndarray) -> None:
        if len(self.positions) < self.positions.maxlen:
            return
        xyz0, seen0 = self.positions[0]
        # Only points no group has explained for the whole window may found one: a still point that noise kept out
        # of its group for a frame is not a new body.
        free = np.flatnonzero((self.group_of == -1) & seen & (self.unexplained >= self.join_frames))
        free = free[free < len(seen0)]
        free = free[seen0[free]]
        if len(free) < self.min_group:
            return
        fit = ransac_fit_rigid(
            xyz0[free], xyz[free], inlier_m=self.join_m, min_points=max(4, self.min_group), iters=64
        )
        if not fit.ok or fit.n_inliers < self.min_group:
            return
        members = free[fit.inliers]
        g = Group(id=self._next_group, motion=Rigid3.identity(), born=self.frame)
        g.history.append(g.motion)
        g.n_fit = len(members)
        self._next_group += 1
        self.groups[g.id] = g
        self.group_of[members] = g.id
        self.anchor[members] = xyz[members]  # the group's frame is the camera's at its birth
        self.unexplained[members] = 0

    def _fusion(self) -> None:
        ids = sorted(self.groups)
        for a_id in ids:
            for b_id in ids:
                if b_id <= a_id or a_id not in self.groups or b_id not in self.groups:
                    continue
                a, b = self.groups[a_id], self.groups[b_id]
                if min(len(a.history), len(b.history)) < self.merge_frames:
                    continue
                rels = [
                    ma.inverse().compose(mb)
                    for ma, mb in zip(
                        list(a.history)[-self.merge_frames :],
                        list(b.history)[-self.merge_frames :],
                        strict=True,
                    )
                ]
                last = rels[-1]
                # Judged where b's points are: a small body's fit turns within its noise, which moves nothing there.
                members = np.flatnonzero(self.group_of == b.id)
                centre = (
                    self.anchor[members].mean(axis=0, keepdims=True) if len(members) else np.zeros((1, 3))
                )
                still = all(
                    np.linalg.norm(r.apply(centre) - last.apply(centre)) <= self.merge_m
                    and np.degrees(r.compose(last.inverse()).angle) <= self.merge_deg
                    for r in rels
                )
                if still:
                    self._merge(a, b)

    def _merge(self, a: Group, b: Group) -> None:
        """``b`` joins ``a``: its members re-anchored in ``a``'s frame, its objects with them."""
        members = np.flatnonzero(self.group_of == b.id)
        carried = b.motion.apply(self.anchor[members]) if len(members) else np.zeros((0, 3))
        self.anchor[members] = a.motion.inverse().apply(carried) if len(members) else self.anchor[members]
        self.group_of[members] = a.id
        for obj in self.objects.values():
            if obj.group == b.id:
                obj.group, obj.anchor = a.id, _matrix(a.motion.inverse().compose(_rigid(obj.pose)))
        del self.groups[b.id]

    # ── objects ──

    def add_object(self, name: str, tracks: np.ndarray, pose: np.ndarray) -> None:
        """Pre: ``tracks`` are track indices, ``pose`` camera <- object (4x4) now."""
        self.objects[name] = TrackedObject(
            name=name, tracks=np.asarray(tracks, dtype=int), pose=np.asarray(pose, dtype=np.float64)
        )

    def _place_objects(self, xyz: np.ndarray, seen: np.ndarray) -> None:
        """Each object: carried by its group (the one most of its tracks are in), then placed by its own points when
        at least ``min_own`` of them are seen in that group, which re-anchors it there. Its model is each track's
        position in the object's frame at its first sighting, so a track added later still places it."""
        for obj in self.objects.values():
            tracks = obj.tracks[obj.tracks < len(self.group_of)]
            groups = self.group_of[tracks]
            obj.n_seen = int(seen[tracks].sum()) if len(tracks) else 0
            obj.n_grouped = int((groups >= 0).sum())
            obj.own_ok = False
            if obj.n_grouped == 0:
                obj.carried = obj.pose.copy()
                continue
            g_id = int(np.bincount(groups[groups >= 0]).argmax())
            g = self.groups[g_id]
            if obj.group != g_id or obj.anchor is None:
                obj.group = g_id
                obj.anchor = _matrix(g.motion.inverse().compose(_rigid(obj.pose)))
            obj.carried = _matrix(g.motion.compose(_rigid(obj.anchor)))
            obj.pose = obj.carried.copy()
            own = tracks[(groups == g_id) & seen[tracks]]
            known = np.asarray([t for t in own if int(t) in obj.model], dtype=int)
            if len(known) >= self.min_own:
                fit = ransac_fit_rigid(
                    np.asarray([obj.model[int(t)] for t in known]),
                    xyz[known],
                    inlier_m=self.leave_m,
                    prior=_rigid(obj.carried),
                    prior_rot_deg=self.step_deg,
                    prior_trans_m=self.step_m,
                )
                if fit.ok and fit.n_inliers >= self.min_own:
                    obj.pose = _matrix(fit.transform)
                    obj.anchor = _matrix(g.motion.inverse().compose(fit.transform))
                    obj.own_ok = True
            inv = _rigid(obj.pose).inverse()
            for t in own:
                obj.model.setdefault(int(t), inv.apply(xyz[t][None])[0])

    def pose(self, name: str) -> np.ndarray:
        return self.objects[name].pose.copy()

    def state(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "groups": {
                g.id: {
                    "members": int((self.group_of == g.id).sum()),
                    "supported": g.supported,
                    "rms_mm": g.rms * 1000.0,
                    "n_fit": g.n_fit,
                    "born": g.born,
                    "motion": _matrix(g.motion).tolist(),
                }
                for g in self.groups.values()
            },
            "free": int((self.group_of == -1).sum()),
            "objects": {
                o.name: {
                    "group": o.group,
                    "n_seen": o.n_seen,
                    "n_grouped": o.n_grouped,
                    "own_ok": o.own_ok,
                    "pose": o.pose.tolist(),
                    "carried": None if o.carried is None else o.carried.tolist(),
                }
                for o in self.objects.values()
            },
        }


def _rigid(m: np.ndarray) -> Rigid3:
    m = np.asarray(m, dtype=np.float64)
    return Rigid3(m[:3, :3].copy(), m[:3, 3].copy())


def _matrix(r: Rigid3) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r.rot, r.trans
    return m
