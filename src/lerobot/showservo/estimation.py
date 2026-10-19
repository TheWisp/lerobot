# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Which tracker output is a measurement (src/lerobot/showservo/docs/act_execution.md, "The loops, and who writes
what").

The trackers report each object's pose every camera frame, each with its own verdict; the world takes measurements.
This layer sits between them: for each object in a frame, a pose to take, or none and why. Neither the world nor the
engine sees a tracker. An estimator is anything with :meth:`PoseEstimator.estimate`; a better one — another model,
agreement between sources, a check against depth — replaces the one in use without touching the world or the engine.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One tracker's pose of one object in one camera frame: its motion since the demo (base frame), and the tracker's
    own verdict on it."""

    source: str
    motion: np.ndarray
    ok: bool
    why: str = ""


@dataclasses.dataclass(frozen=True)
class Frame:
    """What the camera measured in the frame the candidates are of: its depth image (metres, 0 where it has none),
    aligned to the image the trackers saw; its intrinsics; and where the camera is (base to camera)."""

    depth_m: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    cam_from_base: np.ndarray


@dataclasses.dataclass(frozen=True)
class Estimate:
    """An object's pose in one frame, to take as a measurement; None when there is none to take. ``why`` says which;
    ``seen_through``, when the camera was checked against it, the share of its surface the camera saw past."""

    motion: np.ndarray | None
    why: str
    seen_through: float | None = None


class PoseEstimator(Protocol):
    def estimate(
        self, t: float, candidates: dict[str, list[Candidate]], frame: Frame | None = None
    ) -> dict[str, Estimate]:
        """Each object's estimate from the candidates of the camera frame captured at ``t``, and what that frame
        measured when it is given."""
        ...


POINT2POSE = "point2pose"  # the session tracker's fit, by its own verdict (found, the object not lost)
OLD_RUNNER = "old runner checks"  # the views the key-frame runner's checks certify, surface-corrected where they correct
GROUPS = "point groups"  # the point groups' pose: carried by the group the object moves with, placed by its own points


class SourceVerdict:
    """One source's own verdict: an object's pose is that source's candidate when the source is confident in it."""

    def __init__(self, source: str):
        self.source = source

    def estimate(
        self, t: float, candidates: dict[str, list[Candidate]], frame: Frame | None = None
    ) -> dict[str, Estimate]:
        out = {}
        for name, found in candidates.items():
            c = next((c for c in found if c.source == self.source), None)
            if c is None:
                out[name] = Estimate(None, f"no {self.source} candidate")
            elif c.ok:
                out[name] = Estimate(c.motion, f"{c.source}: confident")
            else:
                out[name] = Estimate(None, f"{c.source}: {c.why or 'not confident'}")
        return out


def tracker_verdict() -> SourceVerdict:
    """The rule the engine started with: Point2Pose's own verdict. It passes views every other check rejects
    (act_execution.md E30)."""
    return SourceVerdict(POINT2POSE)


def old_runner_checks() -> SourceVerdict:
    """The key-frame runner's checks: enough of the object's tracks seen with a depth that agrees, its fit pinning its
    middle, its surface agreeing (lerobot.gui.api.manipulation.tracking). Not the runner's carrying between views by
    the point groups, which needs to know what the gripper holds."""
    return SourceVerdict(OLD_RUNNER)


REST_S = 1.0  # a choice: each object's first second of views, which its tolerances are measured on (as the world's)
MIN_REST_VIEWS = 3  # the fewest views a tolerance is measured on


class SeenThrough:
    """A source's confident candidate, unless the camera saw through it (act_execution.md E33): for each point of the
    object's surface at the candidate's pose, the camera's ray to it met something farther than the point by more than
    the object's tolerance, so the surface the pose puts there is not there. A point the ray met something nearer on
    is hidden, not contradicted. A candidate is refused when more of its surface is seen through than the object's
    cut-off. Both are measured on the object's own views in its first REST_S, before anything has touched it: the
    tolerance twice the 95th percentile of how far the measured depth lay from its points, the cut-off twice the
    largest share then seen through, and at least two points' worth. Until they are measured its views pass.
    ``surfaces`` are each object's surface points at its demo pose, base frame."""

    def __init__(self, source: str, surfaces: dict[str, np.ndarray]):
        self.source = source
        self.surfaces = {n: np.asarray(p, dtype=float).reshape(-1, 3) for n, p in surfaces.items()}
        self.rest: dict[str, list[np.ndarray]] = {}  # each object's gaps per view, in its first REST_S
        self.since: dict[str, float] = {}
        self.limits: dict[str, tuple[float, float]] = {}  # each object's tolerance (m) and cut-off (share)

    def gaps(self, name: str, motion: np.ndarray, frame: Frame) -> np.ndarray:
        """For each of ``name``'s surface points at ``motion`` the camera has depth for: the measured depth on its ray
        less the point's (metres)."""
        m = frame.cam_from_base @ np.asarray(motion, dtype=float)
        p = self.surfaces[name] @ m[:3, :3].T + m[:3, 3]
        p = p[p[:, 2] > 0]
        u = np.round(frame.fx * p[:, 0] / p[:, 2] + frame.cx).astype(int)
        v = np.round(frame.fy * p[:, 1] / p[:, 2] + frame.cy).astype(int)
        h, w = frame.depth_m.shape
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        measured = np.zeros(len(p))
        measured[ok] = frame.depth_m[v[ok], u[ok]]
        ok &= measured > 0
        return measured[ok] - p[ok, 2]

    def estimate(
        self, t: float, candidates: dict[str, list[Candidate]], frame: Frame | None = None
    ) -> dict[str, Estimate]:
        out = SourceVerdict(self.source).estimate(t, candidates)
        for name, e in out.items():
            if e.motion is None or frame is None or name not in self.surfaces:
                continue
            gaps = self.gaps(name, e.motion, frame)
            if not len(gaps):
                continue
            if name not in self.limits:
                self.since.setdefault(name, t)
                self.rest.setdefault(name, []).append(gaps)
                if t - self.since[name] >= REST_S and len(self.rest[name]) >= MIN_REST_VIEWS:
                    views = self.rest.pop(name)
                    tol = 2.0 * float(np.percentile(np.abs(np.concatenate(views)), 95))
                    cut = max(
                        2.0 * max(float(np.mean(g > tol)) for g in views), 2.0 / min(len(g) for g in views)
                    )
                    self.limits[name] = (tol, cut)
                out[name] = Estimate(e.motion, f"{e.why}; its tolerances are being measured")
                continue
            tol, cut = self.limits[name]
            share = float(np.mean(gaps > tol))
            if share > cut:
                out[name] = Estimate(
                    None, "the camera sees through more of its surface than its cut-off", share
                )
            else:
                out[name] = Estimate(e.motion, f"{e.why}; {share:.0%} of its surface seen through", share)
        return out
