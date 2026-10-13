"""Constrain an observed planar patch, without a table, category or full-pose prior.

This is a local measurement correction. A matching plane observes its normal and
normal offset, not twist around that normal or translation along the plane.
Those components remain the proposal's. Missing or ambiguous patches must not
be presented as a successful verification.
"""

from __future__ import annotations

import numpy as np

from . import pose_verification
from .pose_verification import MASK_EROSION, _pose

# Work limits: a majority plane needs at most 52 independent triples for a
# 99.9% chance of an all-inlier sample. The fixed seed makes replay reproducible.
PLANE_TRIALS = 64
MAX_POINTS = 1600
POLICY_VERSION = "rgbd-plane-v1"


def _sample(points):
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) > MAX_POINTS:
        p = p[np.linspace(0, len(p) - 1, MAX_POINTS, dtype=int)]
    return p


def depth_points(depth, mask, intr):
    """Interior measured pixels only; aligned depth uses the colour intrinsics."""
    import cv2

    depth, mask = np.asarray(depth, float), np.asarray(mask, bool)
    inside = cv2.erode(mask.astype(np.uint8), np.ones((MASK_EROSION, MASK_EROSION), np.uint8)) > 0
    y, x = np.nonzero(inside & np.isfinite(depth) & (depth > 0))
    z = depth[y, x]
    return _sample(np.column_stack(((x - intr["cx"]) * z / intr["fx"], (y - intr["cy"]) * z / intr["fy"], z)))


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))


def _svd_plane(points):
    centre = points.mean(axis=0)
    _, values, vt = np.linalg.svd(points - centre, full_matrices=False)
    return centre, vt[-1], values


def _plane(points, noise_m, report=None):
    """A robust majority plane, with disjoint fitting/validation samples."""
    report = {} if report is None else report

    def unavailable(reason):
        report.update(state="inconclusive", reason=reason)
        return None

    p = _sample(points)
    report["points"] = len(p)
    if len(p) < 12:
        return unavailable("too few surface samples for disjoint fit and validation")
    # Fit on alternating samples; the remaining samples are never used to pick
    # the normal. Both sets span the patch instead of isolating a narrow edge.
    train, test = p[::2], p[1::2]
    tri = train[np.random.default_rng(17).integers(len(train), size=(PLANE_TRIALS, 3))]
    ns = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    sizes = np.linalg.norm(ns, axis=1)
    good = sizes > np.finfo(float).eps
    if not good.any():
        return unavailable("surface samples are collinear")
    ns = ns[good] / sizes[good, None]
    residuals = abs(train @ ns.T - (tri[good, 0] * ns).sum(axis=1))
    best = int(np.argmin(np.median(residuals, axis=0)))
    # The correspondence budget covers reference AND current measurements.
    # Split it equally instead of spending the whole allowance on each plane.
    plane_noise = noise_m / 2
    keep = residuals[:, best] <= plane_noise
    for _ in range(5):
        if keep.sum() < 6:
            return unavailable("too few plane inliers")
        centre, normal, singular = _svd_plane(train[keep])
        signed = (train - centre) @ normal
        # A Gaussian-consistent MAD rejects depth discontinuities; the existing
        # point-noise allowance caps the fit's ability to explain nonplanarity.
        sigma = 1.4826 * np.median(abs(signed - np.median(signed)))
        threshold = min(plane_noise, max(3 * sigma, 1e-9))
        updated = abs(signed) <= threshold
        if np.array_equal(keep, updated):
            break
        keep = updated
    if keep.sum() <= len(train) / 2:
        return unavailable("no majority reference patch in fitting samples")
    centre, normal, singular = _svd_plane(train[keep])
    distances = abs((test - centre) @ normal)
    validation = distances <= threshold
    if validation.sum() <= len(test) / 2 or validation.sum() < 6:
        return unavailable("plane does not explain the validation samples")
    other_centre, other_normal, other_singular = _svd_plane(test[validation])
    # A line does not determine a plane normal. Retain a physical span relative
    # to the supplied point-noise allowance, not a pixel- or object-size rule.
    span = min(singular[1] / np.sqrt(keep.sum()), other_singular[1] / np.sqrt(validation.sum()))
    if span <= noise_m:
        return unavailable("surface span is too small relative to point noise")
    # A curved cap can fit within a plane's residual threshold yet its normal
    # follows the viewing direction rather than a rigid material face (a small
    # sphere is a counterexample). Check predictable bending on held-out points.
    _, _, axes = np.linalg.svd(train[keep] - centre, full_matrices=False)

    def quadratic(points):
        xyz = (points - centre) @ axes.T
        u, v = (xyz[:, :2] / span).T
        return np.column_stack((np.ones(len(u)), u, v, u * u, u * v, v * v)), xyz[:, 2]

    design, height = quadratic(train[keep])
    coefficients, _, rank, _ = np.linalg.lstsq(design, height, rcond=None)
    valid_design, valid_height = quadratic(test[validation])
    quad_rms = float(np.sqrt(np.mean((valid_design @ coefficients - valid_height) ** 2)))
    bend = float(np.ptp(valid_design[:, 3:] @ coefficients[3:]))
    report.update(bend_m=bend, quadratic_validation_rms_m=quad_rms)
    # The bend must fit the per-surface budget plus the measured model error at
    # both ends. This is a consistency budget, not a calibrated confidence bound.
    if rank < 6 or bend > plane_noise + 2 * quad_rms:
        return unavailable("predictable curvature exceeds the plane measurement budget")
    report.update(state="supported", reason="majority plane passes split and curvature checks")
    return {
        "centre": centre,
        "normal": normal,
        "points": p,
        "validation_centre": other_centre,
        "validation_normal": other_normal,
        "inliers": p[abs((p - centre) @ normal) <= threshold],
        "support": float((keep.sum() + validation.sum()) / len(p)),
        "split_angle_deg": _angle(normal, other_normal),
        "split_offset_m": abs(float((other_centre - centre) @ normal)),
        "rms_m": float(np.sqrt(np.mean(distances[validation] ** 2))),
        "span_m": float(span),
        "bend_m": bend,
        "quadratic_validation_rms_m": quad_rms,
    }


def constrain(pose, reference, depth, mask, intr, *, noise_m, tol_m, tol_deg):
    """Return a measured correction or an explicit reason it cannot be certified.

    Only nearby corresponding majority patches can change the proposal. Validation
    uses samples excluded from plane fitting. It is a consistency check with sensor
    data, not independent ground truth or proof of all six pose components.
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    raw = _pose(pose)
    result = {"state": "inconclusive", "pose": None, "observed_dof": 0, "policy": POLICY_VERSION}
    if raw is None or reference is None or depth is None or mask is None or intr is None:
        return {**result, "reason": "pose or surface observation missing"}
    if np.asarray(depth).ndim != 2 or np.shape(mask) != np.shape(depth):
        return {**result, "reason": "depth and mask shapes differ"}
    ref_info, live_info = {}, {}
    ref = _plane(reference, noise_m, ref_info)
    live = _plane(depth_points(depth, mask, intr), noise_m, live_info)
    result["evidence"] = {"reference": ref_info, "live": live_info}
    result.update(reference_planar=ref is not None, live_planar=live is not None)
    if ref is None or live is None:
        why = ref_info["reason"] if ref is None else live_info["reason"]
        return {**result, "reason": why}
    evidence = {
        name: {
            k: plane[k]
            for k in (
                "support",
                "split_angle_deg",
                "split_offset_m",
                "rms_m",
                "span_m",
                "bend_m",
                "quadratic_validation_rms_m",
            )
        }
        for name, plane in [("reference", ref), ("live", live)]
    }
    for name in evidence:
        result["evidence"][name].update(evidence[name])
    if (
        ref["split_angle_deg"] + live["split_angle_deg"] > tol_deg
        or ref["split_offset_m"] + live["split_offset_m"] > tol_m
    ):
        return {**result, "reason": "surface split fits disagree beyond placement tolerance"}
    n0 = raw[:3, :3] @ ref["normal"]
    n1 = live["normal"] * (1 if live["normal"] @ n0 >= 0 else -1)
    axis = np.cross(n0, n1)
    turn = float(np.arctan2(np.linalg.norm(axis), n0 @ n1))
    correction = Rotation.from_rotvec(axis * turn / max(np.linalg.norm(axis), 1e-15)).as_matrix()
    # Rotate about the proposal's reference-patch centre. Keep its tangential
    # location, then translate ONLY along the measured normal onto the plane.
    centre = raw[:3, :3] @ ref["centre"] + raw[:3, 3]
    offset = float((live["centre"] - centre) @ n1)
    updated = raw.copy()
    updated[:3, :3] = correction @ raw[:3, :3]
    updated[:3, 3] = centre + offset * n1 - updated[:3, :3] @ ref["centre"]
    old = ref["inliers"] @ raw[:3, :3].T + raw[:3, 3]
    moved = ref["inliers"] @ updated[:3, :3].T + updated[:3, 3]
    radius = 2 * noise_m  # same local-association radius as the existing refit
    before = cKDTree(live["inliers"]).query(old)[0]
    after = cKDTree(live["inliers"]).query(moved)[0]
    result.update(
        rotation_deg=float(np.degrees(turn)),
        normal_shift_mm=offset * 1000,
        validation_normal_before_deg=_angle(
            raw[:3, :3] @ ref["validation_normal"], live["validation_normal"]
        ),
        validation_normal_after_deg=_angle(
            updated[:3, :3] @ ref["validation_normal"], live["validation_normal"]
        ),
        raw_plane_rms_mm=float(np.sqrt(np.mean(((old - live["centre"]) @ n1) ** 2)) * 1000),
        corrected_plane_rms_mm=float(np.sqrt(np.mean(((moved - live["centre"]) @ n1) ** 2)) * 1000),
        local_support_before=float(np.mean(before <= radius)),
        local_support_after=float(np.mean(after <= radius)),
    )
    if np.mean(before <= radius) <= 0.5 or np.mean(after <= radius) <= 0.5 or abs(offset) > radius:
        return {**result, "reason": "the planes do not have enough nearby common surface"}
    if np.mean(np.minimum(after, radius) ** 2) > np.mean(np.minimum(before, radius) ** 2) + noise_m**2:
        return {**result, "reason": "plane alignment worsens the measured point support"}
    return {
        **result,
        "state": "corrected" if turn > 1e-8 or abs(offset) > 1e-9 else "consistent",
        "pose": updated.tolist(),
        "observed_dof": 3,
        "reason": "measured plane constrains tilt and normal offset; tangent motion remains the proposal",
    }


def resolve(pose, reference, depth, mask, intr, *, noise_m, tol_m, tol_deg, surface=None):
    """Select the measured correction, retain agreement, or withhold the proposal.

    Non-planar surfaces keep the existing pose if the general surface check agrees.
    A contradictory or unavailable check cannot be bypassed through hold history.
    Neither branch certifies unobservable motion components or sensor calibration.
    """
    raw = _pose(pose)
    constraint = constrain(pose, reference, depth, mask, intr, noise_m=noise_m, tol_m=tol_m, tol_deg=tol_deg)
    if constraint["pose"] is not None:
        return constraint
    if surface is None:
        surface = pose_verification.surface(
            pose, reference, depth, mask, intr, noise_m=noise_m, tol_m=tol_m, tol_deg=tol_deg
        )
    if raw is not None and constraint.get("reference_planar") is False and surface["state"] == "consistent":
        return {
            **constraint,
            "state": "unchanged",
            "pose": raw.tolist(),
            "reason": "general surface check agrees; no supported plane correction",
            "surface_state": surface["state"],
        }
    return {
        **constraint,
        "state": "rejected",
        "pose": None,
        "reason": f"surface not verified ({surface['state']} point fit): {constraint['reason']}",
        "surface_state": surface["state"],
    }
