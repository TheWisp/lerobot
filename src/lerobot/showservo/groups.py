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
from time import perf_counter
from typing import Any

import numpy as np

from lerobot.showservo.placement import view_places
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
    why: str = ""  # why its own points did not place it this frame, when they were fitted and turned away
    held: bool = (
        False  # in the gripper: its pose is set from outside (the arm's joints), see GroupTracker.hold
    )
    joined: int = -1  # the frame it came to rest in its group
    pending: int | None = (
        None  # another group most of its seen points are in, and for how many frames in a row
    )
    pending_frames: int = 0
    left: dict = field(default_factory=dict)  # group -> its anchor there when it left, for a return


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
        own_share: float = 0.97,
        point_noise_m: float = 0.003,
        place_tol_m: float = 0.003,
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
        # An object's own points place it, over what its group carries, only under the act's one rule
        # (lerobot.showservo.placement): own_share of its modelled points seen in its group, and the points fitted
        # pinning its middle to place_tol_m for point_noise_m of noise on each. With the wrist over all but a corner
        # of a gamepad, 16 own points placed it 23 degrees off while it lay still (docs/proofs/act-loop).
        self.own_share, self.point_noise_m, self.place_tol_m = own_share, point_noise_m, place_tol_m
        self.rest = np.zeros((0, 3))
        self.rest_group = np.zeros(0, dtype=int)
        # Where each member last held still, in its group's frame (the rest's group): its place the last frame it sat
        # within its group's noise of its rest. The rest is a running mean and follows a point that has started to
        # move; this does not, so a new body's motion since it last held still is measured from here.
        self.still = np.zeros((0, 3))
        self.still_frame = np.zeros(0, dtype=int)  # the frame it last held still in its group
        self.departed = np.zeros(0, dtype=int)  # the frame it last held still in the group it last left
        self.split_log: list[
            tuple
        ] = []  # (frame, from, into, points, median offset, trigger, group motion), a split
        self.split_why: list | None = None  # set to a list to record why each candidate split was declined
        self.unseen = np.zeros(0, dtype=int)
        self.owned = np.zeros(0, dtype=bool)
        self.groups: dict[int, Group] = {}
        self.objects: dict[str, TrackedObject] = {}
        self.positions: deque = deque(maxlen=join_frames + 1)  # (xyz, seen) of the last frames, newest last
        self._next_group = 0
        self.timing_ms: dict[str, float] = {}

    # ── the two loops ──

    def update(self, xyz: np.ndarray, seen: np.ndarray) -> None:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        seen = np.asarray(seen, dtype=bool).reshape(-1) & np.isfinite(xyz).all(axis=1)
        self._grow(len(xyz))
        self.frame += 1
        self.positions.append((xyz, seen))
        self.unseen = np.where(seen, 0, self.unseen + 1)
        self.cooldown = np.maximum(self.cooldown - 1, 0)
        stages = (
            ("fit", lambda: self._fit_groups(xyz, seen)),
            ("split", lambda: self._split(xyz, seen)),
            ("leave", self._leave),
            ("join", lambda: self._join(xyz, seen)),
            ("fission", lambda: self._fission(xyz, seen)),
            ("fusion", self._fusion),
            ("objects", lambda: self._place_objects(xyz, seen)),
        )

        timing = {}
        for name, run in stages:
            start = perf_counter()
            run()
            timing[name] = round((perf_counter() - start) * 1000, 2)
        self.timing_ms = timing

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
            self.still = np.concatenate([self.still, np.full((k, 3), np.nan)])
            self.still_frame = np.concatenate([self.still_frame, np.full(k, -1, dtype=int)])
            self.departed = np.concatenate([self.departed, np.full(k, -1, dtype=int)])

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
                    fast_prior=True,
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
        explains better than the group's beyond its measured noise leaves at once, into the group already moving that way or into a new
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
            moved = np.linalg.norm(now - rest, axis=1)
            calm = fresh | (moved <= max(self.split_trigger_m, 3.0 * float(np.median(moved))))
            self.still[members[calm]] = rest[calm]  # frozen from the frame a point clears its noise
            self.still_frame[members[calm & ~fresh]] = self.frame
            if self.split_trigger_m > 0 and len(members) >= 2 * self.min_group and g.supported:
                self._split_group(g, members, rest, now, xyz, seen)
            still_in = self.group_of[members] == g.id  # the rest follows the members that stayed
            self.rest[members[still_in]] += self.rest_alpha * (now[still_in] - rest[still_in])

    def _split_group(self, g: Group, members, rest, now, xyz, seen) -> None:
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
            self._why(g, "candidates", int(pick.sum()), len(members))
            return  # too few to be a body, or the group's own fit has not caught up with its majority
        fit = ransac_fit_rigid(rest[pick], now[pick], inlier_m=inlier, min_points=self.min_group, iters=64)
        if not fit.ok or fit.n_inliers < self.min_group:
            self._why(g, "no shared motion", int(pick.sum()), int(fit.n_inliers) if fit.ok else 0)
            return
        in_body = np.flatnonzero(pick)[fit.inliers]
        close = moved < 2 * self.leave_m  # the refit leaves out gross outliers (slipped tracks)
        one, _ = fit_rigid(rest[close], now[close])
        if (np.linalg.norm(one.apply(rest[in_body]) - now[in_body], axis=1) < inlier).mean() >= 0.5:
            self._why(g, "fit error")
            return  # one motion explains them: an error of the group's fit, not a body of its own
        # The SAME points must form a rigid body throughout the recent frames. Increasing offset magnitudes
        # alone are not shared motion: independent depth errors can grow in unrelated directions, then happen
        # to agree this frame. RANSAC on just that last frame selects such a noisy tail in a dense point field.
        # Keep only witnesses supported in every frame before checking that their offset is still growing.
        if len(self.positions) < 3 or len(g.history) < 3:
            return
        ids = members[in_body]
        witnesses = self._rigid_history(ids, inlier, frames=self.positions.maxlen)
        coherent = np.isin(ids, witnesses)
        if coherent.sum() < self.min_group:
            self._why(g, "incoherent history", int(coherent.sum()), len(ids))
            return
        in_body = in_body[coherent]
        ids = members[in_body]
        try:
            offsets = []
            for k in (2, 1):
                xk, _sk = self.positions[-1 - k]
                at = g.history[-1 - k].inverse().apply(xk[ids])
                motion, _ = fit_rigid(rest[in_body], at)
                offsets.append(
                    float(np.median(np.linalg.norm(motion.apply(rest[in_body]) - rest[in_body], axis=1)))
                )
            motion, _ = fit_rigid(rest[in_body], now[in_body])
            offsets.append(
                float(np.median(np.linalg.norm(motion.apply(rest[in_body]) - rest[in_body], axis=1)))
            )
        except AssertionError:
            self._why(g, "degenerate history")
            return  # the intersection can contain only copies of one point even when each larger fit was valid
        if not (offsets[1] > offsets[0] + 0.0005 and offsets[2] > offsets[1] + 0.0005):
            self._why(g, "not growing", *offsets)
            return
        # Refit only the witnesses that survived the history check. Grow the body only where it improves the
        # residual by more than the group's measured noise: an arbitrarily tiny improvement otherwise annexes
        # unrelated stationary points, spreading the new group's tint across the view.
        under_body = np.linalg.norm(motion.apply(rest) - now, axis=1)
        body = members[(under_body < inlier) & (under_body + noise < moved)]
        if len(body) < self.min_group:
            self._why(g, "body too small", len(body))
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
        # The body's motion since it last held still, in the group's frame (its split was decided on the rest's, which
        # follows it once it moves) and in the camera's.
        back, _ = fit_rigid(self.still[ids], now[in_body])
        since_still = g.motion.compose(back).compose(g.motion.inverse())
        if target is None:  # born with it, as :meth:`_since_still` for fission
            target = Group(id=self._next_group, motion=since_still, born=self.frame)
            target.history.append(target.motion)
            target.n_fit = len(body)
            self._next_group += 1
            self.groups[target.id] = target
        # Hidden members go with what they were among: each follows the side most of its nearest seen members took,
        # judged where they all last sat in the group's frame, as an object under a sheet goes with what it rests on.
        # Left behind, a hidden point would ride with whatever its old group turns out to be once the split is over.
        # Objects' own tracks have no vote: the ones still seen on a covered object may have slid onto what covers it,
        # and would take the rest of it along with the cover.
        hidden = np.flatnonzero((self.group_of == g.id) & ~seen[: len(self.group_of)])
        voters = ~self.owned[members]
        if len(hidden):
            where = np.where(
                (self.rest_group[hidden] == g.id)[:, None], self.rest[hidden], self.anchor[hidden]
            )
            known = np.isfinite(where).all(axis=1)
            hidden, where = hidden[known], where[known]
            in_body = np.isin(members, body)[voters]
            k = min(8, int(voters.sum()))
            if len(hidden) and k:
                dist = np.linalg.norm(where[:, None] - rest[voters][None], axis=2)
                nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
                goes = in_body[nearest].mean(axis=1) > 0.5
                if goes.any():
                    follow = hidden[goes]
                    xyz = xyz.copy()
                    xyz[follow] = g.motion.apply(
                        motion.apply(where[goes])
                    )  # where the body's motion puts them
                    body = np.concatenate([body, follow])
                    # An object hidden among them moved with the body too, since before the split was seen: its pose
                    # takes the body's motion in the group's frame, and it rests in the body's group from now on.
                    for obj in self.objects.values():
                        if (
                            obj.group == g.id
                            and obj.anchor is not None
                            and np.isin(obj.tracks, follow).mean() > 0.5
                        ):
                            obj.pose = _matrix(g.motion.compose(back).compose(_rigid(obj.anchor)))
                            obj.left[obj.group] = obj.anchor.copy()
                            obj.group, obj.joined = target.id, self.frame
                            obj.anchor = _matrix(target.motion.inverse().compose(_rigid(obj.pose)))
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
        self.departed[body] = self.still_frame[body]
        self.group_of[body] = target.id
        self.anchor[body] = target.motion.inverse().apply(xyz[body])
        self.rest[body] = self.anchor[body]
        self.still[body] = self.anchor[body]
        self.rest_group[body] = target.id
        self.strikes[body] = 0
        self.unexplained[body] = 0

    def _why(self, g: Group, reason: str, *numbers) -> None:
        if self.split_why is not None:
            self.split_why.append((self.frame, g.id, reason, *numbers))

    def _leave(self) -> None:
        leaving = np.flatnonzero(self.strikes >= self.leave_frames)
        self.departed[leaving] = self.still_frame[leaving]
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

    def _rigid_history(self, members: np.ndarray, inlier_m: float, frames: int) -> np.ndarray:
        """The same visible tracks must support a rigid fit in every frame, not only at the endpoints.

        Use actual observations as the reference, not the running rest: a noisy track's biased rest can make
        independent errors look like repeated displacement. Each frame nominates a fit; only their common
        inliers can witness the new body. No assumption about direction, constant speed or a static camera.
        """
        history = list(self.positions)[-frames:]
        if len(history) < frames:
            return members[:0]
        members = members[members < len(history[0][1])]
        for _xyz, seen in history:
            members = members[seen[members]]
        if len(members) < self.min_group:
            return members[:0]
        first = history[0][0][members]
        coherent = np.ones(len(members), dtype=bool)
        for xyz, _seen in history[1:]:
            fit = ransac_fit_rigid(
                first, xyz[members], inlier_m=inlier_m, min_points=self.min_group, iters=64
            )
            if not fit.ok:
                return members[:0]
            coherent &= fit.inliers
        return members[coherent]

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
        members = self._rigid_history(free[fit.inliers], self.join_m, frames=len(self.positions))
        if len(members) < self.min_group:
            return
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
        g = Group(id=self._next_group, motion=self._since_still(members, xyz), born=self.frame)
        g.history.append(g.motion)
        g.n_fit = len(members)
        self._next_group += 1
        self.groups[g.id] = g
        self.group_of[members] = g.id
        self.anchor[members] = g.motion.inverse().apply(xyz[members])
        self.unexplained[members] = 0

    def _since_still(self, members: np.ndarray, xyz: np.ndarray) -> Rigid3:
        """The motion a new group's founders made since they last held still in the groups they left: from where each
        would be had it stayed (where it last held still there, carried by that group's motion now) to where it is. A group starts with it,
        not at rest, so the motion that it took to tell the body apart is not lost to what moves with it: its frame is
        where its founders last sat. Identity when too few founders have a rest to measure it on."""
        came = self.rest_group[members]
        known = np.asarray([c in self.groups for c in came], dtype=bool) & np.isfinite(
            self.still[members]
        ).all(axis=1)
        if known.sum() < self.min_group:
            return Rigid3.identity()
        ids = members[known]
        stayed = np.stack(
            [
                self.groups[int(c)].motion.apply(self.still[i][None])[0]
                for i, c in zip(ids, came[known], strict=True)
            ]
        )
        fit = ransac_fit_rigid(stayed, xyz[ids], inlier_m=self.leave_m, min_points=self.min_group, iters=64)
        return fit.transform if fit.ok and fit.n_inliers >= self.min_group else Rigid3.identity()

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
                obj.joined = self.frame
        del self.groups[b.id]

    # ── objects ──

    def add_object(self, name: str, tracks: np.ndarray, pose: np.ndarray) -> None:
        """Pre: ``tracks`` are track indices, ``pose`` camera <- object (4x4) now. An object designated again keeps
        the new tracks only: its earlier own tracks retire and leave their group, as a track lost for good does
        (:meth:`_leave`), since an owned track never retires otherwise and every act designates its objects again."""
        old = self.objects.get(name)
        if old is not None:
            prev = old.tracks[(old.tracks < len(self.owned)) & ~np.isin(old.tracks, tracks)]
            self.owned[prev] = False
            self.retired[prev] = True
            self.group_of[prev] = -1
            self.anchor[prev] = np.nan
        self.objects[name] = TrackedObject(
            name=name, tracks=np.asarray(tracks, dtype=int), pose=np.asarray(pose, dtype=np.float64)
        )
        if len(tracks):
            self._grow(int(np.max(tracks)) + 1)
            self.owned[np.asarray(tracks, dtype=int)] = True

    def retire_unowned(self, tracks: np.ndarray) -> None:
        """Remove obsolete borrowed tracks when an object's surrounding field is reseeded.

        Their old corners are still visible, so the normal unseen/failed-fit retirement
        would retain them indefinitely and make every later TAPIR step more expensive.
        """
        ids = np.asarray(tracks, dtype=int)
        ids = ids[(ids >= 0) & (ids < len(self.group_of))]
        ids = ids[~self.owned[ids]]
        self.retired[ids] = True
        self.group_of[ids] = -1
        self.anchor[ids] = np.nan
        self.rest_group[ids] = -1

    def _place_objects(self, xyz: np.ndarray, seen: np.ndarray) -> None:
        """Each object: carried by the group it moves with, and placed by its own points where they pin it.

        Which group: the one most of its seen points are in. Another group takes it once most of its seen points have
        been in that group for join_frames frames in a row (the wait a point serves before it joins a group), or at
        once on a view of its own points there that places it; it moves there as :meth:`_move_object` says. With none
        of its points seen it stays where it is. A few of its points that slide onto what passes over it take it along
        until its own points, seen again where it lies, place it back: points alone cannot tell them from the few a
        gripper leaves in view of the object it holds. A body splitting off with it hidden among its points takes it
        along (:meth:`_split_group`), and a merge re-points it (:meth:`_merge`).

        Placed: its own seen points in that group, fitted to its model, re-place it when their inliers pin its middle
        (:func:`view_places`, ``own_share`` of its modelled points seen) and put it beyond the place tolerance from
        where its group carries it. Its model is each track's position in the object's frame at its first sighting,
        so a track added later still places it."""
        for obj in self.objects.values():
            tracks = obj.tracks[obj.tracks < len(self.group_of)]
            groups = self.group_of[tracks]
            obj.n_seen = int(seen[tracks].sum()) if len(tracks) else 0
            obj.n_grouped = int((groups >= 0).sum())
            obj.own_ok, obj.why = False, ""
            if obj.held:  # where the arm has it, set by hold()
                obj.carried, obj.why = obj.pose.copy(), "held"
                continue
            if obj.n_grouped == 0:
                obj.carried = obj.pose.copy()
                continue
            if (
                obj.group not in self.groups or obj.anchor is None
            ):  # designated: in the group its tracks are in
                obj.group, obj.joined = int(np.bincount(groups[groups >= 0]).argmax()), self.frame
                obj.anchor = _matrix(self.groups[obj.group].motion.inverse().compose(_rigid(obj.pose)))
            visible = groups[seen[tracks] & (groups >= 0)]
            g_id = int(np.bincount(visible).argmax()) if len(visible) else obj.group
            if g_id == obj.group:
                obj.pending, obj.pending_frames = None, 0
            else:
                obj.pending_frames = obj.pending_frames + 1 if obj.pending == g_id else 1
                obj.pending = g_id
                if obj.pending_frames >= self.join_frames:
                    self._move_object(obj, g_id)
            # Back in a group it was in before: its place there then, or where it was carried to since, whichever more
            # of its points seen there sit at. A few of its points that slid onto what passed over it carry it off
            # and come to rest beside it; its own, seen again, outnumber them where it lay.
            old = obj.left.get(obj.group)
            if old is not None:
                motion = self.groups[obj.group].motion
                then = self._support(obj, motion.compose(_rigid(old)), obj.group)
                if then > self._support(obj, motion.compose(_rigid(obj.anchor)), obj.group):
                    obj.left[obj.group], obj.anchor = obj.anchor, old
            obj.carried = _matrix(self.groups[obj.group].motion.compose(_rigid(obj.anchor)))
            obj.pose = obj.carried.copy()
            g = self.groups[g_id]
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
                    fast_prior=True,
                )
                if fit.ok and fit.n_inliers >= self.min_own:
                    modelled = [int(t) for t in tracks if int(t) in obj.model]
                    centre = np.mean([obj.model[t] for t in modelled], axis=0)
                    middle = obj.carried[:3, :3] @ centre + obj.carried[:3, 3]
                    places, obj.why = view_places(
                        len(known) / max(1, len(modelled)),
                        xyz[known][fit.inliers],
                        middle,
                        self.own_share,
                        self.point_noise_m,
                        self.place_tol_m,
                    )
                    if places:
                        obj.own_ok = True
                        # Still with its group until proven otherwise: its own points move it only when they put it
                        # beyond the place tolerance from where the group carries it. Each fit of a small object's
                        # points re-placed it every frame, a 36 mm cube 1-5 degrees from one frame to the next while it
                        # lay still; its group is fitted on hundreds of points.
                        pts = np.asarray([obj.model[t] for t in modelled])
                        apart = np.linalg.norm(
                            fit.transform.apply(pts) - _rigid(obj.carried).apply(pts), axis=1
                        )
                        if g_id != obj.group or float(np.sqrt(np.mean(apart**2))) > self.place_tol_m:
                            obj.pose = _matrix(fit.transform)
                            if g_id != obj.group:
                                obj.left[obj.group] = obj.anchor.copy()
                                obj.joined, obj.pending, obj.pending_frames = self.frame, None, 0
                            obj.group, obj.anchor = g_id, _matrix(g.motion.inverse().compose(fit.transform))
            inv = _rigid(obj.pose).inverse()
            for t in own:
                obj.model.setdefault(int(t), inv.apply(xyz[t][None])[0])

    def _move_object(self, obj: TrackedObject, b_id: int) -> None:
        """``obj`` moves with group ``b_id`` from now, from where it was when its points there last held still in the
        group they left (the median of their frames): what that group has done since, it did too; the frames it took
        to tell it apart do not leave it behind. A group born after that frame started with the motion its founders
        made since they last held still (:meth:`_since_still`), its frame where they sat then; the object sat where its
        old group had it at that group's birth. Without those frames it moves from where its old group carries it now.
        It keeps its place in the group it leaves, for a return (:meth:`_place_objects`)."""
        a, b = self.groups[obj.group], self.groups[b_id]
        xyz, seen = self.positions[-1]
        tracks = obj.tracks[obj.tracks < len(seen)]
        there = tracks[seen[tracks] & (self.group_of[tracks] == b_id) & (self.departed[tracks] >= 0)]
        since = int(np.median(self.departed[there])) if len(there) else self.frame
        obj.left[a.id] = obj.anchor.copy()
        then_a, then_b = self._motion_at(a, since), self._motion_at(b, max(since, b.born))
        if since >= b.born and then_a is not None and then_b is not None:
            obj.anchor = _matrix(then_b.inverse().compose(then_a).compose(_rigid(obj.anchor)))
        elif self._motion_at(a, b.born) is not None:
            obj.anchor = _matrix(self._motion_at(a, b.born).compose(_rigid(obj.anchor)))
        else:
            obj.anchor = _matrix(b.motion.inverse().compose(a.motion).compose(_rigid(obj.anchor)))
        obj.group, obj.joined, obj.pending, obj.pending_frames = b_id, self.frame, None, 0

    def _motion_at(self, g: Group, frame: int) -> Rigid3 | None:
        """``g``'s motion at ``frame``, from its history; None outside it."""
        back = self.frame - frame
        return g.history[-1 - back] if 0 <= back < len(g.history) else None

    def _support(self, obj: TrackedObject, pose: Rigid3, g_id: int) -> int:
        """How many of ``obj``'s points seen in group ``g_id`` this frame (with a place in its model) are within leave_m
        of where ``pose`` puts them."""
        xyz, seen = self.positions[-1]
        tracks = obj.tracks[obj.tracks < len(seen)]
        there = [
            int(t) for t in tracks[seen[tracks] & (self.group_of[tracks] == g_id)] if int(t) in obj.model
        ]
        if not there:
            return 0
        off = np.linalg.norm(pose.apply(np.asarray([obj.model[t] for t in there])) - xyz[there], axis=1)
        return int((off <= self.leave_m).sum())

    def hold(self, name: str, pose: np.ndarray) -> None:
        """``name`` is in the gripper, at ``pose`` (camera <- object) by the arm's joints: its pose until released. The
        fingers hide it and its hidden tracks stay with what it lay on, so no group carries it out of sight; the arm
        knows where it is."""
        obj = self.objects[name]
        obj.held, obj.pose = True, np.asarray(pose, dtype=np.float64).copy()

    def release(self, name: str) -> None:
        """``name`` left the gripper where hold() last put it, and rests in the world (the biggest group) from there,
        until its own points show it elsewhere."""
        obj = self.objects[name]
        if not obj.held:
            return
        obj.held = False
        grouped = self.group_of[self.group_of >= 0]
        if len(grouped) and self.groups:
            world = self.groups[int(np.bincount(grouped).argmax())]
            obj.group, obj.anchor = world.id, _matrix(world.motion.inverse().compose(_rigid(obj.pose)))
            obj.joined = self.frame
        else:
            obj.group, obj.anchor = None, None

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
