"""Rotational equivalences inferred from an RGB-D surface, without a CAD model.

An object's *appearance* symmetry and a task's allowed grasp/landing freedom are
different declarations. This module only describes an axis and tests the visible
surface for an approximately round, elongated cross-section; callers decide which
equivalences an interaction actually permits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

AXES = ("surface_normal", "long_axis")
KINDS = ("none", "discrete", "continuous")


def normalise(spec: dict[str, Any] | int | None) -> dict[str, Any]:
    """A saved symmetry declaration. Old integer orders mean discrete turns about the visible face normal.

    ``end_swap`` is a second, independent equivalence for a long object whose
    ends are interchangeable. It reverses the measured long-axis direction;
    rolling about that axis does not do so. The flag describes appearance, not
    permission to change the demo's grasp contact.
    """
    if not isinstance(spec, dict):
        order = max(1, int(spec or 1))
        return {"kind": "discrete" if order > 1 else "none", "axis": "surface_normal", "order": order}
    kind = str(spec.get("kind", "none"))
    axis = str(spec.get("axis", "surface_normal"))
    order = int(spec.get("order", 1))
    end_swap = spec.get("end_swap", False)
    if kind not in KINDS or axis not in AXES or order < 1 or order > 12 or not isinstance(end_swap, bool):
        raise ValueError("symmetry needs a valid kind, observed axis and order from 1 to 12")
    if end_swap and axis != "long_axis":
        raise ValueError("end interchangeability needs an observed long axis")
    if kind == "discrete" and order == 1:
        kind = "none"
    out = {"kind": kind, "axis": axis, "order": order if kind == "discrete" else 1}
    if end_swap:
        out["end_swap"] = True
    return out


def observed_axis(points: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """(unit direction, point on axis) from the object's visible depth points.

    The point on a long axis fits circles in central cross-sections. A surface
    centroid lies above the centreline of a half-visible cylinder and would
    translate a rotated grasp. This fit is an estimate, not a full object model.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    if len(points) < 30 or not np.isfinite(points).all() or axis not in AXES:
        raise ValueError("an observed symmetry axis needs 30 finite surface points and a known axis")
    middle = np.median(points, axis=0)
    centered = points - middle
    _u, _s, basis = np.linalg.svd(centered, full_matrices=False)
    if axis == "surface_normal":
        from lerobot.showservo.pose import main_plane_normal

        return main_plane_normal(points), points.mean(axis=0)
    long = basis[0]
    local = centered @ basis.T
    axial = local[:, 0]
    # End caps distort a cylinder fit; use its middle half, split into slices
    # so a short visible edge cannot dominate a single global least-squares fit.
    low, high = np.percentile(axial, [25, 75])
    cuts = np.linspace(low, high, 5)
    width = max((high - low) / 4, 1e-6)
    centres = []
    for cut in cuts:
        cross = local[np.abs(axial - cut) <= width, 1:]
        if len(cross) < 20:
            continue
        matrix = np.column_stack((2 * cross, np.ones(len(cross))))
        a, b, _r2 = np.linalg.lstsq(matrix, np.sum(cross**2, axis=1), rcond=None)[0]
        centres.append((a, b))
    if not centres:
        raise ValueError("the visible object has no central cross-section")
    offset = np.median(centres, axis=0)
    return long, middle + offset @ basis[1:]


def round_long_axis(points: np.ndarray, *, depth_noise_m: float = 0.0015) -> dict[str, Any]:
    """Evidence for rotating a grasp around a long rigid object's observed axis.

    Each central slice must fit a circle to within sensor noise or 15% of its
    radius, and the visible length must span three diameters. This deliberately
    rejects an uncertain shape instead of declaring a box/cable cylindrical.
    The tolerance has a physical meaning (depth noise), not an object name.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    if len(points) < 100 or not np.isfinite(points).all():
        return {"ok": False, "reason": "too few finite depth points"}
    middle = np.median(points, axis=0)
    _u, _s, basis = np.linalg.svd(points - middle, full_matrices=False)
    local = (points - middle) @ basis.T
    lo, hi = np.percentile(local[:, 0], [5, 95])
    length = float(hi - lo)
    slices = []
    for cut in np.linspace(lo + 0.2 * length, hi - 0.2 * length, 5):
        cross = local[np.abs(local[:, 0] - cut) < 0.08 * length, 1:]
        if len(cross) < 30:
            continue
        matrix = np.column_stack((2 * cross, np.ones(len(cross))))
        a, b, k = np.linalg.lstsq(matrix, np.sum(cross**2, axis=1), rcond=None)[0]
        radius = float(np.sqrt(max(0, k + a * a + b * b)))
        residual = float(np.median(np.abs(np.linalg.norm(cross - [a, b], axis=1) - radius)))
        slices.append((radius, residual))
    if len(slices) < 3:
        return {"ok": False, "reason": "too few observed cross-sections"}
    radii, errors = np.asarray(slices).T
    radius, error = float(np.median(radii)), float(np.median(errors))
    stable = float(np.median(np.abs(radii - radius))) <= max(depth_noise_m, 0.15 * radius)
    ok = (
        radius > depth_noise_m
        and length >= 6 * radius
        and error <= max(depth_noise_m, 0.15 * radius)
        and stable
    )
    direction, point = observed_axis(points, "long_axis")
    return {
        "ok": bool(ok),
        "reason": "" if ok else "the visible cross-sections do not support a long round axis",
        "axis": direction,
        "point": point,
        "length_m": length,
        "radius_m": radius,
        "median_radial_error_m": error,
    }


def turn_about(axis: np.ndarray, point: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rigid transform rotating around a measured line in the current frame."""
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("a rotation axis must have nonzero finite length")
    axis = axis / norm
    point = np.asarray(point, float)
    a = np.deg2rad(angle_deg)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(a) * cross + (1 - np.cos(a)) * cross @ cross
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = point - rot @ point
    return out
