"""Scene-wide rigid motion groups from timestamped 3D point identities.

Each frame: fit visible members, separate inconsistent measurements, discover
new groups, merge sustained co-movement, then expose measured/predicted points.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from scipy.spatial import cKDTree

from .registration import DegenerateRegistrationError, rigid_fit
from .surface_observation import move


@dataclass(frozen=True)
class GroupConfig:
    min_inliers: int = 12
    min_motion_points: int = 4
    fit_threshold_m: float = 0.005
    support_confirmation_s: float = 0.8
    split_tolerance_m: float = 0.003
    split_confirmation_s: float = 0.2
    merge_tolerance_m: float = 0.003
    merge_confirmation_s: float = 0.8
    max_hidden_s: float = 30.0
    max_points: int = 4096
    max_groups: int = 32
    motion_hypotheses: int = 48

    def __post_init__(self):
        for name, minimum in (
            ("min_inliers", 3),
            ("min_motion_points", 3),
            ("max_points", 3),
            ("max_groups", 1),
            ("motion_hypotheses", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.max_points < self.min_inliers:
            raise ValueError("max_points must accommodate min_inliers")
        if self.min_motion_points > self.min_inliers:
            raise ValueError("min_motion_points cannot exceed new-group min_inliers")
        for name in ("fit_threshold_m", "split_tolerance_m", "merge_tolerance_m", "max_hidden_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "support_confirmation_s",
            "split_confirmation_s",
            "merge_confirmation_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


def _readonly(value, dtype=None):
    value = np.array(value, dtype=dtype, copy=True)
    value.flags.writeable = False
    return value


@dataclass(frozen=True)
class GroupView:
    id: int
    ids: np.ndarray
    local_xyz: np.ndarray
    base_from_group: np.ndarray
    measured: np.ndarray
    base_xyz: np.ndarray
    confirmed: bool
    supported: bool
    residual_m: float
    previous_measured: np.ndarray
    current_measured: np.ndarray
    from_time_s: float | None
    time_s: float


@dataclass(frozen=True)
class GroupFrame:
    frame_id: int
    time_s: float
    groups: Mapping[int, GroupView]
    point_count: int


@dataclass
class _Group:
    id: int
    anchors: dict
    pose: np.ndarray
    since: float
    confirmed: bool = False
    supported: bool = False
    residual: float = float("inf")
    last_supported: float = 0.0
    inliers: set[int] = field(default_factory=set)


def fit_point_motion(source, target, config, *, min_points=None, threshold_m=None, robust_tolerance_m=None):
    """One bounded consensus solve over known pairs, with batched SVD hypotheses."""
    minimum = config.min_inliers if min_points is None else min_points
    n = len(source)
    if n < minimum:
        return None
    threshold = config.fit_threshold_m if threshold_m is None else threshold_m
    if robust_tolerance_m is not None:
        threshold = min(threshold, robust_tolerance_m)
    try:
        pose = rigid_fit(source, target)
    except DegenerateRegistrationError:
        return None
    residual = np.linalg.norm(move(source, pose) - target, axis=1)
    keep = residual <= threshold
    need_hypotheses = not keep.all()
    if robust_tolerance_m is not None:
        need_hypotheses |= np.count_nonzero(residual > 0.7 * robust_tolerance_m) >= minimum
    if need_hypotheses:
        # Nearby triplets can discover a small moving body among a large
        # background. Scoring and SVD are vectorized; the work is capped.
        count = min(n, config.motion_hypotheses)
        centers = np.linspace(0, n - 1, count, dtype=int)
        neighbors = cKDTree(source).query(source[centers], k=min(8, n))[1]
        triples = np.stack((neighbors[:, 0], neighbors[:, 1], neighbors[:, -1]), axis=1)
        a, b = source[triples], target[triples]
        ac, bc = a.mean(axis=1), b.mean(axis=1)
        aa, bb = a - ac[:, None], b - bc[:, None]
        u, singular, vt = np.linalg.svd(np.einsum("hni,hnj->hij", aa, bb))
        correction = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
        correction[:, 2, 2] = np.linalg.det(vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1))
        rotations = vt.transpose(0, 2, 1) @ correction @ u.transpose(0, 2, 1)
        shifts = bc - np.einsum("hij,hj->hi", rotations, ac)
        errors = np.linalg.norm(
            np.einsum("hij,nj->hni", rotations, source) + shifts[:, None] - target, axis=2
        )
        valid = singular[:, 1] > 1e-12
        if robust_tolerance_m is None:
            votes = (errors <= threshold).sum(axis=1)
            votes[~valid] = 0
            best = int(votes.argmax())
            if votes[best] > keep.sum():
                keep = errors[best] <= threshold
        else:
            # Truncated error lets a coherent majority outvote a small body
            # even when a compromise fit places both within the split limit.
            limit_sq = robust_tolerance_m**2
            scores = np.minimum(errors**2, limit_sq).sum(axis=1)
            scores[~valid] = np.inf
            best = int(scores.argmin())
            baseline = np.minimum(residual**2, limit_sq).sum()
            if scores[best] < baseline:
                keep = errors[best] <= threshold
    if keep.sum() < minimum:
        return None
    try:
        pose = rigid_fit(source[keep], target[keep])
    except DegenerateRegistrationError:
        return None
    residual = np.linalg.norm(move(source, pose) - target, axis=1)
    keep = residual <= threshold
    return (pose, keep, residual) if keep.sum() >= minimum else None


class SceneMotionGroups:
    """Bounded scene state. Point IDs are keys, never unbounded array offsets."""

    def __init__(self, config=GroupConfig()):
        self.config = config
        self._groups = {}
        self._seen = {}
        self._previous = {}
        self._merge_since = {}
        self._departing = {}
        self._joining = {}
        self._banned_from = {}
        self._free_since = {}
        self._history = deque()
        self._next_id = 0
        self._time = None
        self.frame = GroupFrame(-1, -1.0, MappingProxyType({}), 0)

    def _new(self, anchors, pose, time, *, confirmed=False):
        group = _Group(
            self._next_id,
            anchors,
            pose,
            time,
            confirmed or self.config.support_confirmation_s == 0,
        )
        self._next_id += 1
        group.supported = True
        group.last_supported = time
        group.inliers = set(anchors)
        self._groups[group.id] = group
        return group

    def _join_candidates(self, free, current, previous_poses, time):
        """Only measured, persistent co-motion may attach a new point to a group."""
        if not free or not self._groups:
            return set()
        eligible = set()
        group_points = {}
        for group in self._groups.values():
            if not group.supported or group.id not in previous_poses:
                continue
            visible = [current[i] for i in group.inliers if i in current]
            if visible:
                group_points[group.id] = cKDTree(np.asarray(visible))
        for i in free:
            banned_gid, banned_until = self._banned_from.get(i, (None, -float("inf")))
            pending = self._joining.get(i)
            if pending is not None:
                gid, anchor, since = pending
                group = self._groups.get(gid)
                if group is not None and group.supported and not (gid == banned_gid and time < banned_until):
                    error = np.linalg.norm(move(anchor[None], group.pose)[0] - current[i])
                    if error <= self.config.merge_tolerance_m:
                        eligible.add(i)
                        if time - since + 1e-9 >= self.config.merge_confirmation_s:
                            group.anchors[i] = anchor
                            group.inliers.add(i)
                            del self._joining[i]
                            self._free_since.pop(i, None)
                        continue
                del self._joining[i]
            if i not in self._previous:
                continue
            choices = []
            for gid, tree in group_points.items():
                if gid == banned_gid and time < banned_until:
                    continue
                group = self._groups[gid]
                delta = group.pose @ np.linalg.inv(previous_poses[gid])
                predicted = move(self._previous[i][None], delta)[0]
                error = np.linalg.norm(predicted - current[i])
                if error <= self.config.merge_tolerance_m:
                    choices.append((float(tree.query(current[i])[0]), gid))
            if choices:
                _, gid = min(choices)
                group = self._groups[gid]
                anchor = move(current[i][None], np.linalg.inv(group.pose))[0]
                self._joining[i] = (gid, anchor, time)
                self._free_since.pop(i, None)
                eligible.add(i)
        return eligible

    def _birth_from_history(self, free, current, time):
        """Found a body only from the same measured IDs moving rigidly over the split window."""
        if not free:
            return
        earliest = self._history[0][1] if self._history else self._previous
        ready = sorted(
            i
            for i in free
            if i in earliest
            and i in self._previous
            and (
                not self._groups
                or time - self._free_since.get(i, time) + 1e-9 >= self.config.split_confirmation_s
            )
        )
        while len(ready) >= self.config.min_inliers:
            if len(self._groups) >= self.config.max_groups:
                dormant = [g for g in self._groups.values() if not g.supported]
                if not dormant:
                    break
                expired = min(dormant, key=lambda g: (g.confirmed, g.last_supported, g.id))
                del self._groups[expired.id]
            source = np.array([earliest[i] for i in ready])
            target = np.array([current[i] for i in ready])
            fitted = fit_point_motion(source, target, self.config)
            if fitted is None:
                break
            pose, keep, residual = fitted
            selected = np.asarray(ready)[keep]
            # A single lucky pair is insufficient: the same points must also
            # agree with an intermediate, independently measured frame.
            if self._groups and len(self._history) > 2:
                midpoint = self._history[len(self._history) // 2][1]
                common = [int(i) for i in selected if int(i) in midpoint]
                if len(common) < self.config.min_inliers:
                    break
                mid_fit = fit_point_motion(
                    np.array([earliest[i] for i in common]),
                    np.array([midpoint[i] for i in common]),
                    self.config,
                )
                if mid_fit is None:
                    break
                selected = np.asarray(common)[mid_fit[1]]
                if len(selected) < self.config.min_inliers:
                    break
                pose = rigid_fit(
                    np.array([earliest[int(i)] for i in selected]),
                    np.array([current[int(i)] for i in selected]),
                )
            anchors = {int(i): earliest[int(i)].copy() for i in selected}
            group = self._new(anchors, pose, time)
            group.residual = float(np.sqrt(np.mean(residual[keep] ** 2)))
            for i in selected:
                self._free_since.pop(int(i), None)
            ready = [i for i in ready if i not in group.anchors]

    def update(self, scene):
        time = float(scene.source_time_s)
        if self._time is not None and time <= self._time:
            raise ValueError("group measurements must advance in source time")
        ids = np.asarray(scene.ids)
        xyz = np.asarray(scene.base_xyz, float)
        if len(np.unique(ids)) != len(ids) or not np.isfinite(xyz).all():
            raise ValueError("group input requires unique IDs and finite measured points")
        # Bound the whole state, including current measurements and hidden IDs.
        order = np.argsort(ids, kind="stable")[: self.config.max_points]
        current = {int(ids[i]): xyz[i].copy() for i in order}
        previous_poses = {gid: group.pose.copy() for gid, group in self._groups.items()}
        # Keep only the short, measured birth window; hidden predictions never
        # enter this history.
        self._history.append((time, current))
        while len(self._history) > 2 and time - self._history[1][0] >= self.config.split_confirmation_s:
            self._history.popleft()
        self._seen.update(dict.fromkeys(current, time))
        survivors = sorted(self._seen, key=lambda i: (-self._seen[i], i))[: self.config.max_points]
        self._seen = {i: self._seen[i] for i in survivors if time - self._seen[i] <= self.config.max_hidden_s}
        for group in list(self._groups.values()):
            group.anchors = {i: p for i, p in group.anchors.items() if i in self._seen}
            if not group.anchors:
                del self._groups[group.id]
                continue
            visible = sorted(set(group.anchors) & current.keys())
            group.supported = False
            group.inliers = set()
            if len(visible) < self.config.min_motion_points:
                if (
                    not group.confirmed
                    and time - group.last_supported + 1e-9 >= self.config.split_confirmation_s
                ):
                    del self._groups[group.id]
                continue
            source = np.array([group.anchors[i] for i in visible])
            target = np.array([current[i] for i in visible])
            fitted = fit_point_motion(
                source,
                target,
                self.config,
                min_points=self.config.min_motion_points,
                robust_tolerance_m=self.config.split_tolerance_m,
            )
            if fitted is None:
                if (
                    not group.confirmed
                    and time - group.last_supported + 1e-9 >= self.config.split_confirmation_s
                ):
                    del self._groups[group.id]
                continue
            pose, keep, residual = fitted
            group.pose, group.supported = pose, True
            group.last_supported = time
            group.residual = float(np.sqrt(np.mean(residual[keep] ** 2)))
            group.confirmed |= time - group.since + 1e-9 >= self.config.support_confirmation_s
            for i, error in zip(visible, residual, strict=True):
                if error <= self.config.split_tolerance_m:
                    self._departing.pop(i, None)
                    group.inliers.add(i)
                else:
                    since = self._departing.setdefault(i, time)
                    if time - since + 1e-9 >= self.config.split_confirmation_s:
                        del group.anchors[i]
                        del self._departing[i]
                        self._free_since[i] = since
                        self._banned_from[i] = (group.id, time + self.config.merge_confirmation_s)

        # Existing groups need enough members to measure motion; only birth
        # requires the larger new-group consensus above.
        self._groups = {
            gid: g for gid, g in self._groups.items() if len(g.anchors) >= self.config.min_motion_points
        }
        # New IDs first get a chance to join a nearby, measured group. A point
        # found free for one frame does not immediately found a new body.
        owned = {i for group in self._groups.values() for i in group.anchors}
        free = sorted((current.keys() & self._previous.keys()) - owned)
        joining = self._join_candidates(free, current, previous_poses, time)
        unexplained = [i for i in free if i not in joining]
        for i in unexplained:
            self._free_since.setdefault(i, time)
        self._birth_from_history(unexplained, current, time)
        self._joining = {i: candidate for i, candidate in self._joining.items() if i in current}
        self._banned_from = {
            i: entry for i, entry in self._banned_from.items() if i in self._seen and time < entry[1]
        }
        self._free_since = {i: since for i, since in self._free_since.items() if i in current}
        self._departing = {i: since for i, since in self._departing.items() if i in self._seen}

        # Compare relative motion at the joining body's actual points. Testing
        # its noisy rotation across the entire distant scene invents motion.
        merge_next = {}
        keys = sorted(self._groups)
        for ai, aid in enumerate(keys):
            if aid not in self._groups:
                continue
            a = self._groups[aid]
            if not a.supported:
                continue
            for bid in keys[ai + 1 :]:
                if bid not in self._groups:
                    continue
                b = self._groups[bid]
                if not b.supported:
                    continue
                relative = np.linalg.inv(a.pose) @ b.pose
                baseline, last, since = self._merge_since.get(
                    (aid, bid), (relative.copy(), relative.copy(), time)
                )
                bp = np.array([b.anchors[i] for i in b.inliers])
                now = move(bp, relative)
                error = max(
                    float(np.quantile(np.linalg.norm(now - move(bp, baseline), axis=1), 0.9)),
                    float(np.quantile(np.linalg.norm(now - move(bp, last), axis=1), 0.9)),
                )
                if error > self.config.merge_tolerance_m:
                    baseline, since = relative.copy(), time
                if (
                    (aid, bid) in self._merge_since
                    and error <= self.config.merge_tolerance_m
                    and time - since + 1e-9 >= self.config.merge_confirmation_s
                ):
                    moved = move(np.array(list(b.anchors.values())), relative)
                    a.anchors.update(dict(zip(b.anchors, moved, strict=True)))
                    a.inliers.update(b.inliers)
                    a.confirmed |= b.confirmed
                    del self._groups[bid]
                else:
                    merge_next[(aid, bid)] = (baseline, relative.copy(), since)
        self._merge_since = merge_next
        views = {}
        for gid, group in self._groups.items():
            member_ids = np.array(sorted(group.anchors), dtype=np.int64)
            if not len(member_ids):
                continue
            local = np.array([group.anchors[int(i)] for i in member_ids])
            measured = np.array([int(i) in current and int(i) in group.inliers for i in member_ids])
            positions = move(local, group.pose)
            positions[measured] = np.array([current[int(i)] for i in member_ids[measured]]).reshape(-1, 3)
            pairs = [int(i) for i in member_ids[measured] if int(i) in self._previous]
            views[gid] = GroupView(
                gid,
                _readonly(member_ids),
                _readonly(local),
                _readonly(group.pose),
                _readonly(measured),
                _readonly(positions),
                group.confirmed,
                group.supported,
                group.residual,
                _readonly(np.array([self._previous[i] for i in pairs]).reshape(-1, 3)),
                _readonly(np.array([current[i] for i in pairs]).reshape(-1, 3)),
                self._time,
                time,
            )
        self._previous, self._time = current, time
        self.frame = GroupFrame(scene.frame_id, time, MappingProxyType(views), len(self._seen))
        return self.frame
