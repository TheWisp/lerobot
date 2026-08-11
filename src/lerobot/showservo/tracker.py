# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Frame-rate point tracking: the geometry half of "semantics once, geometry always".

The interface is three methods so the learned tier (CoTracker3-online, TAPNext) can
replace KLT without touching the servo. v0 default is KLT: ~2 ms on CPU, no GPU
contention with the binder, and honest about loss — a point that fails the
forward-backward check is reported invalid rather than silently dragged along, which
is what lets the grouping layer evict it instead of fitting to a lie.

Tracking here is deliberately IDENTITY-FREE: points carry indices, not meanings.
Which index belongs to which team, and what a team's motion means, is decided one
layer up in :mod:`lerobot.showservo.grouping`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _import_cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError("KLT tracking needs opencv-python (`uv sync --extra all`)") from e
    return cv2


@dataclass
class TrackState:
    """Where every tracked point is now, and whether to believe it.

    Post: ``uv`` is (N, 2) and ``valid`` is (N,) bool with the same N for the whole
    life of a tracker instance — indices are stable, so a caller may hold onto them.
    An invalid point's ``uv`` is its last believed position, kept only so a re-bind
    has somewhere to look; it must never be fed to a fit.
    """

    uv: np.ndarray
    valid: np.ndarray

    def __post_init__(self):
        self.uv = np.asarray(self.uv, dtype=np.float64).reshape(-1, 2)
        self.valid = np.asarray(self.valid, dtype=bool).reshape(-1)
        assert len(self.uv) == len(self.valid)

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


class KLTTracker:
    """Pyramidal Lucas-Kanade with a forward-backward consistency gate.

    Pre: every frame passed to :meth:`step` has the same shape and dtype as the one
    given to :meth:`init`. Post: :meth:`step` returns a :class:`TrackState` whose
    length equals the number of points added so far.

    The forward-backward check is the whole reason this is trustworthy: KLT's own
    status flag stays 1 through occlusions, sliding the point onto whatever texture
    now occupies the window. Re-tracking backwards and demanding the round trip land
    within ``fb_threshold`` px catches exactly that case.
    """

    def __init__(
        self,
        *,
        win_size: int = 21,
        max_level: int = 3,
        fb_threshold: float = 1.0,
        max_iter: int = 30,
        epsilon: float = 0.01,
    ):
        cv2 = _import_cv2()
        assert fb_threshold > 0
        self._cv2 = cv2
        self._fb_threshold = float(fb_threshold)
        self._lk = {
            "winSize": (win_size, win_size),
            "maxLevel": max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, epsilon),
        }
        self._prev_gray: np.ndarray | None = None
        self._uv: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._valid: np.ndarray = np.zeros((0,), dtype=bool)

    # --- lifecycle ---------------------------------------------------------------

    def init(self, frame: np.ndarray, points: np.ndarray) -> TrackState:
        """Seed the tracker. Pre: ``points`` is (N, 2) pixel coords inside the frame."""
        gray = self._gray(frame)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        assert len(pts) >= 1, "nothing to track"
        assert _in_bounds(pts, gray.shape).all(), "seed points must lie inside the frame"
        self._prev_gray = gray
        self._uv = pts.astype(np.float32)
        self._valid = np.ones(len(pts), dtype=bool)
        return self.state

    def add(self, points: np.ndarray) -> np.ndarray:
        """Adopt new points mid-track. Post: returns their indices, (M,) int.

        Used by replenishment. Indices already handed out never shift.
        """
        assert self._prev_gray is not None, "add() before init()"
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        assert _in_bounds(pts, self._prev_gray.shape).all()
        start = len(self._uv)
        self._uv = np.concatenate([self._uv, pts.astype(np.float32)])
        self._valid = np.concatenate([self._valid, np.ones(len(pts), dtype=bool)])
        return np.arange(start, start + len(pts))

    def drop(self, indices) -> None:
        """Invalidate points permanently (grouping's eviction verdict)."""
        self._valid[np.asarray(indices, dtype=int)] = False

    @property
    def state(self) -> TrackState:
        return TrackState(self._uv.astype(np.float64), self._valid.copy())

    # --- the loop ----------------------------------------------------------------

    def step(self, frame: np.ndarray) -> TrackState:
        """Advance one frame. Pre: :meth:`init` was called. Post: dead stays dead."""
        assert self._prev_gray is not None, "step() before init()"
        gray = self._gray(frame)
        live = np.flatnonzero(self._valid)
        if len(live) == 0:
            self._prev_gray = gray
            return self.state

        cv2 = self._cv2
        p0 = self._uv[live].reshape(-1, 1, 2)
        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **self._lk)
        p0r, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, p1, None, **self._lk)

        fb_err = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        ok = (
            (st_fwd.ravel() == 1)
            & (st_bwd.ravel() == 1)
            & (fb_err < self._fb_threshold)
            & _in_bounds(p1.reshape(-1, 2), gray.shape)
        )

        moved = p1.reshape(-1, 2)
        self._uv[live[ok]] = moved[ok]  # a lost point keeps its last believed position
        self._valid[live[~ok]] = False
        self._prev_gray = gray
        return self.state

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        assert frame.ndim in (2, 3), f"expected an image, got shape {frame.shape}"
        if frame.ndim == 2:
            gray = frame
        else:
            assert frame.shape[2] == 3, "expected RGB"
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return np.ascontiguousarray(gray, dtype=np.uint8)


def shi_tomasi_points(
    frame: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    max_points: int = 16,
    quality: float = 0.01,
    min_distance: float = 8.0,
    exclude: np.ndarray | None = None,
    exclude_radius: float = 8.0,
) -> np.ndarray:
    """Corner candidates for replenishment. Post: (M, 2) float64, M <= ``max_points``.

    Pre: ``mask``, when given, is a bool array shaped like the frame's first two axes
    and confines candidates to one team's region — replenishing from the whole frame
    would rent the background into the team and violate invariant 3.
    """
    cv2 = _import_cv2()
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    cv_mask = None
    if mask is not None:
        assert mask.shape == gray.shape, "mask must match the frame"
        cv_mask = (np.asarray(mask) > 0).astype(np.uint8) * 255

    found = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(max_points) + (0 if exclude is None else len(exclude)),
        qualityLevel=float(quality),
        minDistance=float(min_distance),
        mask=cv_mask,
    )
    if found is None:
        return np.zeros((0, 2), dtype=np.float64)
    pts = found.reshape(-1, 2).astype(np.float64)

    if exclude is not None and len(exclude):
        keep = (
            np.linalg.norm(pts[:, None, :] - np.asarray(exclude, float)[None, :, :], axis=2) > exclude_radius
        ).all(axis=1)
        pts = pts[keep]
    return pts[:max_points]


def _in_bounds(pts: np.ndarray, shape) -> np.ndarray:
    h, w = shape[0], shape[1]
    return (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
