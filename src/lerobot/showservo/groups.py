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

from lerobot.showservo.pose import Rigid3, RigidFit, fit_rigid, ransac_fit_rigid


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
        min_group: int = 16,
        merge_frames: int = 15,
        merge_m: float = 0.004,
        step_deg: float = 45.0,
        step_m: float = 0.15,
        min_own: int = 6,
        max_leaves: int = 3,
        full_tenure: int = 30,
        retire_unexplained: int = 150,
        retire_unseen: int = 150,
        split_trigger_m: float = 0.004,
        split_inlier_m: float = 0.004,
        rest_alpha: float = 0.1,
        split_speed_share: float = 0.5,
    ) -> None:
        self.leave_m, self.join_m = leave_m, join_m
        self.leave_frames, self.join_frames = leave_frames, join_frames
        self.min_group, self.min_own = min_group, min_own
        # A point's record: it earns weight in its group's fit over its first full_tenure frames of holding its place
        # (ORB-SLAM keeps a map point by how often it is found where predicted), and after max_leaves departures it
        # is retired for good: a corner that keeps slipping off its surface is no reference.
        self.max_leaves, self.full_tenure = max_leaves, full_tenure
        self.retire_unexplained = (
            retire_unexplained  # a point no group has explained for this long is no reference
        )
        self.merge_frames, self.merge_m = merge_frames, merge_m
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
        self.tenure = np.zeros(0, dtype=int)  # frames in a row a member has held its place in its group
        self.leaves = np.zeros(0, dtype=int)  # how many groups it has been struck out of
        self.retired = np.zeros(0, dtype=bool)
        # Frames in a row a track has not been seen (under a hand, a sheet of paper, the arm). A field track
        # hidden for retire_unseen frames is retired: it is probably gone for good. An object's own track
        # (owned) never is: the object is placed by its own points again the moment they show.
        self.retire_unseen = retire_unseen
        # A body leaving its group is caught as a body (wake at once): split_trigger_m is how far a member must
        # sit from its group's fit to be a candidate, split_inlier_m how closely the candidates must share one
        # rigid motion. 0 turns it off, leaving the point-by-point leave and fission.
        self.split_trigger_m, self.split_inlier_m = split_trigger_m, split_inlier_m
        # Each member's rest: where it has sat in its group's frame lately (a running mean, rest_alpha a frame).
        # How far it is from its rest is how far it has moved since it last held still: a slow move adds up, an
        # anchor's old error does not count.
        self.rest_alpha = rest_alpha
        # Tracking and depth err more on a surface that moves fast (blur, lag): a member's offset counts towards a
        # split only beyond this share of how far its group moved it over the last frames.
        self.split_speed_share = split_speed_share
        self.rest = np.zeros((0, 3))
        self.rest_group = np.zeros(0, dtype=int)
        self.split_log: list[
            tuple
        ] = []  # (frame, from, into, points, median offset m, trigger m), each split
        self.unseen = np.zeros(0, dtype=int)
        self.owned = np.zeros(0, dtype=bool)
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
        self.unseen = np.where(seen, 0, self.unseen + 1)
        self.cooldown = np.maximum(self.cooldown - 1, 0)
        self._fit_groups(xyz, seen)
        self._split(xyz, seen)
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
            self.tenure = np.concatenate([self.tenure, np.zeros(k, dtype=int)])
            self.leaves = np.concatenate([self.leaves, np.zeros(k, dtype=int)])
            self.retired = np.concatenate([self.retired, np.zeros(k, dtype=bool)])
            self.unseen = np.concatenate([self.unseen, np.zeros(k, dtype=int)])
            self.owned = np.concatenate([self.owned, np.zeros(k, dtype=bool)])
            self.rest = np.concatenate([self.rest, np.full((k, 3), np.nan)])
            self.rest_group = np.concatenate([self.rest_group, np.full(k, -1, dtype=int)])

    def _weights(self, members: np.ndarray) -> np.ndarray:
        """A member's say in its group's fit: the established members, those that have held their place for
        full_tenure frames, define the group's frame; newcomers vote on consensus and join the refit but nominate
        nothing, so a young crowd that moves together leaves rather than taking the group with it (SLAM's map
        outlives its pending points). A group with no established member yet is fitted by headcount."""
        return (self.tenure[members] >= self.full_tenure).astype(float)

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
                    hypo_weights=self._weights(members),
                    prior=g.motion,
                    prior_rot_deg=self.step_deg,
                    prior_trans_m=self.step_m,
                )
            if fit is not None and fit.ok:
                g.motion, g.supported, g.rms, g.n_fit = fit.transform, True, fit.rms, fit.n_inliers
                res = np.linalg.norm(g.motion.apply(self.anchor[members]) - xyz[members], axis=1)
                self.strikes[members[use & (res > self.leave_m)]] += 1
                self.strikes[members[use & (res <= self.leave_m)]] = 0
                self.tenure[members[use & (res <= self.leave_m)]] += 1
            else:
                g.supported = False  # too few seen: its motion is held, its members stay
            g.history.append(g.motion)

    def _split(self, xyz: np.ndarray, seen: np.ndarray) -> None:
        """A body leaving its group, decided as a body and on its motion, as a physics engine wakes a body: each
        member's offset from its rest in the group's frame (how far it moved since it last held still); when at
        least min_group members moved more than split_trigger_m, share one rigid motion among themselves within
        split_inlier_m, and one motion over every seen member would not explain them, every member that motion
        explains better than the group's leaves at once, into the group already moving that way or into a new
        one. Point by point, a member leaves only once its own offset has cleared leave_m for leave_frames; the
        body's motion is weighed on all its points at once. The refit tells a body from an error of the group's
        own fit, which the far members would show too, and which one motion over all of them absorbs."""
        groups = list(self.groups.values())
        for g in groups:
            members = np.flatnonzero((self.group_of == g.id) & seen)
            if not len(members):
                continue
            now = g.motion.inverse().apply(xyz[members])  # where each is, in the group's frame
            fresh = self.rest_group[members] != g.id
            self.rest[members[fresh]] = now[fresh]
            self.rest_group[members[fresh]] = g.id
            rest = self.rest[members]
            if self.split_trigger_m > 0 and len(members) >= 2 * self.min_group and g.supported:
                self._split_group(g, members, rest, now, xyz)
            still_in = self.group_of[members] == g.id  # the rest follows the members that stayed
            self.rest[members[still_in]] += self.rest_alpha * (now[still_in] - rest[still_in])

    def _split_group(self, g: Group, members, rest, now, xyz) -> None:
        moved = np.linalg.norm(now - rest, axis=1)
        # Thresholds scale with the group's own noise this frame (its members' median offset, which its still
        # majority sets): a member still in its group sits beyond three of that rarely, a moving body's at once.
        noise = float(np.median(moved))
        trigger = max(self.split_trigger_m, 3.0 * noise)
        inlier = max(self.split_inlier_m, 2.0 * noise)
        pick = moved > trigger
        speed = np.zeros(len(members))
        w = len(self.positions) - 1
        if self.split_speed_share > 0 and w > 0 and len(g.history) > w:
            carried = g.history[-1 - w].apply(rest)  # where the group put each member w frames ago
            speed = np.linalg.norm(g.motion.apply(rest) - carried, axis=1)
            pick &= moved > self.split_speed_share * speed
        if pick.sum() < self.min_group or pick.sum() > len(members) // 2:
            return  # too few to be a body, or the group's own fit has not caught up with its majority
        fit = ransac_fit_rigid(rest[pick], now[pick], inlier_m=inlier, min_points=self.min_group, iters=64)
        if not fit.ok or fit.n_inliers < self.min_group:
            return
        in_body = np.flatnonzero(pick)[fit.inliers]
        close = moved < 2 * self.leave_m  # the refit leaves out gross outliers (slipped tracks)
        one, _ = fit_rigid(rest[close], now[close])
        if (np.linalg.norm(one.apply(rest[in_body]) - now[in_body], axis=1) < inlier).mean() >= 0.5:
            return  # one motion explains them: an error of the group's fit, not a body of its own
        # Still moving, not displaced once: over each of the last two frames the body's offset from its rest grew
        # by more than half a millimetre. A track that jumped (a hand brushing past, a glitch) holds its new offset,
        # a body in motion keeps adding to it.
        if len(self.positions) >= 3 and len(g.history) >= 3:
            ids = members[in_body]
            offsets = []
            for k in (2, 1):
                xk, sk = self.positions[-1 - k]
                ok = ids < len(sk)
                ok[ok] = sk[ids[ok]]
                if ok.sum() < self.min_group:
                    return
                at = g.history[-1 - k].inverse().apply(xk[ids[ok]])
                offsets.append(float(np.median(np.linalg.norm(at - rest[in_body][ok], axis=1))))
            offsets.append(float(np.median(moved[in_body])))
            if not (offsets[1] > offsets[0] + 0.0005 and offsets[2] > offsets[1] + 0.0005):
                return
        # The whole body, not only the members past the trigger: all that its motion explains better than the
        # group's (a turn moves the near ones less, and they would split off a frame later on their own).
        under_body = np.linalg.norm(fit.transform.apply(rest) - now, axis=1)
        body = members[(under_body < inlier) & (under_body < moved)]
        if len(body) < self.min_group:
            return
        # A group already moving that way takes them: carried into its frame over the frames it has (one, if it
        # was born a frame ago from the same body), they held still.
        target = None
        for h in self.groups.values():
            k = min(len(self.positions) - 1, len(h.history) - 1)
            if h.id == g.id or k < 1:
                continue
            xyz0, seen0 = self.positions[-1 - k]
            known = body[body < len(seen0)]
            known = known[seen0[known]]
            if len(known) < self.min_group:
                continue
            drift = np.linalg.norm(
                h.motion.inverse().apply(xyz[known]) - h.history[-1 - k].inverse().apply(xyz0[known]), axis=1
            )
            if float(np.median(drift)) < inlier:
                target = h
                break
        if target is None:
            target = Group(id=self._next_group, motion=Rigid3.identity(), born=self.frame)
            target.history.append(target.motion)
            target.n_fit = len(body)
            self._next_group += 1
            self.groups[target.id] = target
        self.split_log.append(
            (
                self.frame,
                g.id,
                target.id,
                len(body),
                float(np.median(moved[np.isin(members, body)])),
                trigger,
                float(np.median(speed[np.isin(members, body)])),
            )
        )
        self.group_of[body] = target.id
        self.anchor[body] = target.motion.inverse().apply(xyz[body])
        self.rest[body] = self.anchor[body]
        self.rest_group[body] = target.id
        self.strikes[body] = 0
        self.unexplained[body] = 0

    def _leave(self) -> None:
        leaving = np.flatnonzero(self.strikes >= self.leave_frames)
        self.banned[leaving] = self.group_of[leaving]
        self.cooldown[leaving] = 2 * (self.leave_frames + self.join_frames)
        self.group_of[leaving] = -1
        self.anchor[leaving] = np.nan
        self.strikes[leaving] = 0
        self.tenure[leaving] = 0
        self.leaves[leaving] += 1
        self.retired[leaving[self.leaves[leaving] >= self.max_leaves]] = True
        self.retired[self.unexplained >= self.retire_unexplained] = True
        gone = (self.unseen >= self.retire_unseen) & ~self.owned & ~self.retired
        self.retired[gone] = True
        self.group_of[gone] = -1
        self.anchor[gone] = np.nan

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
        for i in np.flatnonzero((self.group_of == -1) & seen & ~self.retired):
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
        free = np.flatnonzero(
            (self.group_of == -1) & seen & ~self.retired & (self.unexplained >= self.join_frames)
        )
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
        # Strays of a group they were struck out of (its fit lagged while it moved, a rim's depth jittered) come to
        # rest explained by that group but banned from it for a while: they are not a new body. A new group must
        # move differently from the group most of its founders came from; strays rejoin it when the ban ends.
        came = self.banned[members]
        came = came[came >= 0]
        if len(came):
            source = self.groups.get(int(np.bincount(came).argmax()))
            if source is not None and len(source.history) >= len(self.positions):
                hist = list(source.history)
                then, now = hist[-len(self.positions)], hist[-1]
                predicted = now.apply(then.inverse().apply(xyz0[members]))
                if float(np.median(np.linalg.norm(predicted - xyz[members], axis=1))) <= self.join_m:
                    return
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
                # Judged at every one of b's points, not at a centre: a turn moves nothing at the centre and much
                # at the rim, and a point that would sit outside leave_m after the merge would only be struck out
                # of it. Over the window, the two moved alike (median within merge_m) at all of them (none beyond
                # leave_m).
                members = np.flatnonzero(self.group_of == b.id)
                pts = self.anchor[members][np.isfinite(self.anchor[members]).all(axis=1)]
                if not len(pts):
                    continue
                off = np.stack([np.linalg.norm(r.apply(pts) - last.apply(pts), axis=1) for r in rels])
                if float(np.median(off)) <= self.merge_m and float(off.max()) <= self.leave_m:
                    self._merge(a, b)

    def _merge(self, a: Group, b: Group) -> None:
        """``b`` joins ``a``: its members re-anchored in ``a``'s frame where they are now (seen) or where ``b``
        puts them (hidden), its objects with them."""
        members = np.flatnonzero(self.group_of == b.id)
        carried = b.motion.apply(self.anchor[members]) if len(members) else np.zeros((0, 3))
        if self.positions and len(members):
            xyz, seen = self.positions[-1]
            known = members < len(seen)
            use = np.zeros(len(members), bool)
            use[known] = seen[members[known]]
            carried[use] = xyz[members[use]]
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
        if len(tracks):
            self._grow(int(np.max(tracks)) + 1)
            self.owned[np.asarray(tracks, dtype=int)] = True

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
            "retired": int(self.retired.sum()),
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
