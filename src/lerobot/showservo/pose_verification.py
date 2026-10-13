# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Raw pose comparisons; control arbitration lives in ``surface_constraint``.

Groups compare motion since a fixed common observation, not absolute registration.
RGB-D supplies an alternative local surface fit, not ground truth: symmetry, shared
depth errors and partial views can make either estimate wrong. Missing evidence is
explicit. All distances compare the SAME object centre, not transform translations
whose origin may be far outside the object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .placement import placement_error
from .pose import Rigid3, fit_surface

# Work bounds, not confidence thresholds. The find uses the same 5-pixel erosion.
SURFACE_POINTS = 400
SURFACE_ITERS = 12
MASK_EROSION = 5


def _points(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 3))
    pts = np.asarray(value, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > SURFACE_POINTS:
        pts = pts[np.linspace(0, len(pts) - 1, SURFACE_POINTS, dtype=int)]
    return pts


def _pose(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    p = np.asarray(value, dtype=float)
    if p.shape != (4, 4) or not np.isfinite(p).all():
        return None
    if (
        not np.allclose(p[3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-3)
        or np.linalg.det(p[:3, :3]) < 0
    ):
        return None
    return p


def difference(a: np.ndarray, b: np.ndarray, centre: np.ndarray, tol_m: float, tol_deg: float) -> dict:
    """Two transforms of the same reference object, with the planner's tolerances."""
    angle = float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3] @ b[:3, :3].T) - 1) / 2, -1, 1))))
    distance = float(np.linalg.norm(a[:3, :3] @ centre + a[:3, 3] - b[:3, :3] @ centre - b[:3, 3]))
    return {
        "state": "disagreement" if distance > tol_m or angle > tol_deg else "consistent",
        "centre_mm": distance * 1000,
        "rotation_deg": angle,
        "limit_mm": tol_m * 1000,
        "limit_deg": tol_deg,
    }


@dataclass
class GroupReference:
    epoch: Any = None
    p2p: np.ndarray | None = None
    group: np.ndarray | None = None
    stamp: float | None = None

    def compare(self, pose, group, centre, stamp, epoch, *, tol_m, tol_deg) -> dict:
        """Retain the first shared frame for this session/designation. Do NOT rebase
        on accepted P2P updates: doing so hides accumulated drift."""
        pose, group = _pose(pose), _pose(group)
        if epoch != self.epoch:
            self.epoch, self.p2p, self.group, self.stamp = epoch, None, None, None
        if pose is None or group is None:
            return {"state": "unavailable", "reason": "no synchronized pair of poses"}
        if self.p2p is None:
            self.p2p, self.group, self.stamp = pose.copy(), group.copy(), float(stamp)
            return {
                "state": "baseline",
                "reference_t": self.stamp,
                "reference_p2p": self.p2p.tolist(),
                "reference_group": self.group.tolist(),
            }
        expected = group @ np.linalg.inv(self.group) @ self.p2p
        return {
            **difference(pose, expected, centre, tol_m, tol_deg),
            "reference_t": self.stamp,
            "predicted_pose": expected.tolist(),
            "group_pose": group.tolist(),
        }


def surface(pose, reference, depth, mask, intr, *, noise_m, tol_m, tol_deg) -> dict:
    """Compare P2P to a bounded local fit of reference geometry to current RGB-D.

    Use only reference points projecting into the current object mask with valid
    depth, excluding foreground occlusion beyond twice the existing point-noise
    allowance. Do not invent a surface normal or constrain motion to a table.
    A fit with inadequate support is inconclusive; a different lower-cost fit is
    a diagnostic disagreement, NOT a declaration that the alternative is correct.
    """
    import cv2
    from scipy.spatial import cKDTree

    pose, ref = _pose(pose), _points(reference)
    if pose is None or len(ref) < 6 or depth is None or mask is None or intr is None:
        return {"state": "unavailable", "reason": "pose, reference surface, mask or depth missing"}
    depth, mask = np.asarray(depth, dtype=float), np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        return {"state": "unavailable", "reason": "mask and depth shapes differ"}
    inside = cv2.erode(mask.astype(np.uint8), np.ones((MASK_EROSION, MASK_EROSION), np.uint8)) > 0
    valid = inside & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    z = depth[v, u]
    live = _points(np.column_stack(((u - intr["cx"]) * z / intr["fx"], (v - intr["cy"]) * z / intr["fy"], z)))
    predicted = ref @ pose[:3, :3].T + pose[:3, 3]
    front = np.isfinite(predicted).all(axis=1) & (predicted[:, 2] > 0)
    ids = np.flatnonzero(front)
    projected = predicted[ids]
    uv = np.round(
        np.column_stack(
            (
                intr["fx"] * projected[:, 0] / projected[:, 2] + intr["cx"],
                intr["fy"] * projected[:, 1] / projected[:, 2] + intr["cy"],
            )
        )
    ).astype(int)
    h, w = depth.shape
    bounds = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids, uv = ids[bounds], uv[bounds]
    observed = depth[uv[:, 1], uv[:, 0]]
    same_object = valid[uv[:, 1], uv[:, 0]]
    near_m = 2 * noise_m
    occluded = same_object & (observed < predicted[ids, 2] - near_m)
    visible = ids[same_object & ~occluded]
    evidence = {
        "reference_points": len(ref),
        "visible_reference_points": len(visible),
        "occluded_reference_points": int(occluded.sum()),
        "live_points": len(live),
        "noise_mm": noise_m * 1000,
        "fit_radius_mm": near_m * 1000,
    }
    if len(visible) < 6 or len(live) < 6:
        return {"state": "inconclusive", "reason": "too little overlapping depth", **evidence}
    centre = ref.mean(axis=0)
    ref = ref[visible]
    moved = ref @ pose[:3, :3].T + pose[:3, 3]
    tree = cKDTree(live)
    raw_dist, _ = tree.query(moved)
    raw_cost = float(np.sqrt(np.mean(np.minimum(raw_dist, near_m) ** 2)))
    try:
        fit, cost = fit_surface(
            ref, live, Rigid3(pose[:3, :3], pose[:3, 3]), near_m=near_m, turns=1, iters=SURFACE_ITERS
        )
    except (AssertionError, np.linalg.LinAlgError):
        return {"state": "inconclusive", "reason": "surface fit is degenerate", **evidence}
    alternative = np.eye(4)
    alternative[:3, :3], alternative[:3, 3] = fit.rot, fit.trans
    fitted = ref @ fit.rot.T + fit.trans
    dist, ix = tree.query(fitted)
    supported = dist <= noise_m
    centre_error = placement_error(live[ix[supported]], fit.rot @ centre + fit.trans) * noise_m
    result = {
        **difference(pose, alternative, centre, tol_m, tol_deg),
        **evidence,
        "raw_rms_mm": raw_cost * 1000,
        "refit_rms_mm": cost * 1000,
        "supported_points": int(supported.sum()),
        "refit_pose": alternative.tolist(),
        "centre_uncertainty_mm": float(centre_error * 1000) if np.isfinite(centre_error) else None,
    }
    # A strict majority prevents a small accidental patch from explaining the
    # whole visible reference. Noise and reach bounds are the planner's existing ones.
    if supported.sum() <= len(ref) / 2 or cost > noise_m or centre_error > tol_m:
        result.update(state="inconclusive", reason="alternative surface fit has insufficient support")
    elif result["state"] == "disagreement" and cost >= raw_cost - 1e-9:
        result.update(state="inconclusive", reason="different pose does not improve the surface fit")
    return result


def lines(checks: list[dict]) -> list[tuple[str, bool]]:
    """Same readout for live and recorded frames; absent old diagnostics stay absent."""
    out = []
    for check in checks:
        for key, label in (("groups", "groups"), ("surface", "RGB-D refit")):
            item = check.get(key) or {}
            state = item.get("state", "unavailable")
            conflict = state == "disagreement"
            text = f"{check['object']} P2P vs {label}: {'DISAGREEMENT' if conflict else state}"
            if "rotation_deg" in item:
                text += f" {item['rotation_deg']:.1f} deg / {item['centre_mm']:.1f} mm"
            if conflict:
                decision = check.get("constraint") or {}
                if check.get("accepted") and decision.get("state") in ("corrected", "consistent"):
                    text += "; surface-constrained pose used"
                else:
                    text += f"; P2P {'accepted' if check.get('accepted') else 'rejected'}"
            out.append((text, conflict))
        decision = check.get("constraint")
        if decision is not None:
            state = decision["state"]
            text = f"{check['object']} control: {state}"
            if state in ("corrected", "consistent"):
                text += f" angle {decision.get('rotation_deg', 0):.1f} deg; {decision.get('observed_dof', 0)} observed DOF"
                if decision.get("unobserved_rotation"):
                    text += "; axial roll unobserved"
            elif state == "rejected":
                text += f"; {decision.get('reason', 'surface unavailable')}"
            if not check.get("accepted") and state != "rejected":
                text += "; withheld by visibility checks"
            out.append((text, state == "rejected"))
    return out
