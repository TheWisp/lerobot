# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Whether a view of an object places it: the act's one rule (src/lerobot/showservo/docs/act_loop.md), shared by
the act's tracker and the point groups. Numpy alone, so the point groups' process can load it by path."""

from __future__ import annotations

import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def placement_error(points: np.ndarray, middle: np.ndarray) -> float:
    """How far a least-squares rigid fit to ``points`` (N, 3) can be off at ``middle`` (3,), per unit of independent
    noise on each point: the fit's covariance for unit noise, carried to ``middle``. A number without units; points too
    few or too nearly in a line to fix a rotation give infinity."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        return float("inf")
    c = pts.mean(axis=0)
    info = np.zeros((6, 6))
    for q in pts - c:
        j = np.hstack([-_skew(q), np.eye(3)])
        info += j.T @ j
    if np.linalg.cond(info) > 1e12:
        return float("inf")
    a = np.hstack([-_skew(np.asarray(middle, dtype=float) - c), np.eye(3)])
    return float(np.sqrt(np.trace(a @ np.linalg.inv(info) @ a.T)))


def view_places(
    share_seen: float,
    points: np.ndarray | None,
    middle: np.ndarray | None,
    need_share: float,
    noise_m: float,
    tol_m: float,
) -> tuple[bool, str]:
    """Does a view place its object, so that its pose may replace the one held? ``(ok, reason)``.

    Enough of the object is seen: ``share_seen`` of its tracked points, at least ``need_share`` (covered in part, its
    points drift onto what covers it while the fit still looks right). And the points seen (``points``, (N, 3)) pin
    the pose: :func:`placement_error` at ``middle``, times ``noise_m``, within ``tol_m``. A tracker that reports no fit
    points (``points`` None) is judged by the share alone.
    """
    if share_seen < need_share:
        return False, f"only {share_seen:.0%} of its points are seen; a view counts from {need_share:.0%}"
    if points is None or middle is None:
        return True, ""
    err_m = placement_error(points, middle) * noise_m
    if err_m > tol_m:
        return (
            False,
            f"the points seen place its middle only to {err_m * 1000:.1f} mm; a view counts within {tol_m * 1000:.0f} mm",
        )
    return True, ""
