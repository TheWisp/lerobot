# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Camera-only reconstruction: object grasp geometry from a mask + metric depth,
and the table plane via RANSAC. Everything is computed in the camera frame, then
lifted to the robot base frame with ``T_base_camera``.

No motor motion and no learned 3D prior here — the metric RealSense gives the
visible surface directly, which is all a top-down gripper grasp needs.
"""

from __future__ import annotations

import numpy as np

from .calib import depth_to_pointcloud, transform_points


def fit_plane_ransac(
    pts: np.ndarray, threshold: float = 0.006, iters: int = 300, seed: int = 0
) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane fit. Plane is ``normal . x + d = 0`` with unit ``normal``.

    Args:
        pts: ``(N,3)`` points.
        threshold: inlier distance (meters).

    Returns ``(normal, d, inlier_mask)``. ``normal`` is oriented toward the
    camera origin (so it points "up" off a table the camera looks down on).
    """
    assert pts.shape[0] >= 3, f"need >=3 points for a plane, got {pts.shape[0]}"
    rng = np.random.default_rng(seed)
    n = pts.shape[0]
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -normal @ p0
        dist = np.abs(pts @ normal + d)
        inliers = dist < threshold
        c = int(inliers.sum())
        if c > best_count:
            best_count, best_inliers = c, inliers
    # Refit on inliers (least squares) for a stable normal.
    inl = pts[best_inliers]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid)
    normal = vt[2]
    d = -normal @ centroid
    # Orient toward the camera origin (0,0,0): points "up" off the table.
    if (normal @ (-centroid)) < 0:
        normal, d = -normal, -d
    return normal, float(d), best_inliers


def analyze_object(depth_m: np.ndarray, mask: np.ndarray, intr: dict[str, float]) -> dict | None:
    """Object grasp geometry from a mask + metric depth, in the camera frame.

    Returns ``None`` if the mask has too few valid depth pixels. Otherwise a dict
    with ``centroid`` (mean XYZ), ``top`` (closest-to-camera point = grasp
    candidate), ``axes`` (3x3 PCA, rows = principal directions), ``n_points``.
    """
    pts, _ = depth_to_pointcloud(depth_m, intr, mask=mask)
    if pts.shape[0] < 30:
        return None
    # Drop depth outliers (flyaway pixels at mask edges) via per-axis MAD.
    med = np.median(pts, axis=0)
    keep = np.all(np.abs(pts - med) < 5 * (np.median(np.abs(pts - med), axis=0) + 1e-6), axis=1)
    pts = pts[keep]
    centroid = pts.mean(axis=0)
    top = pts[np.argmin(pts[:, 2])]  # min depth = nearest the top-down camera
    _, _, vt = np.linalg.svd(pts - centroid)
    return {"centroid": centroid, "top": top, "axes": vt, "n_points": int(pts.shape[0])}


def reconstruct_frame(
    depth_m: np.ndarray,
    intr: dict[str, float],
    T_base_camera: np.ndarray,  # noqa: N803
    object_mask: np.ndarray,
    plane_stride: int = 4,
) -> dict:
    """Reconstruct object + table geometry for one frame, in the base frame.

    ``plane_stride`` subsamples the full-frame cloud before RANSAC for speed.
    Returns base-frame ``object_centroid``, ``object_top``, ``table_normal``,
    ``table_point``, plus the raw camera-frame object dict under ``object_cam``.
    """
    obj = analyze_object(depth_m, object_mask, intr)
    if obj is None:
        raise ValueError("object mask has too few valid depth pixels")

    # Plane support: full-res unprojection (so uv matches intrinsics), object
    # pixels excluded, then subsample the POINTS for RANSAC speed.
    scene_pts, _ = depth_to_pointcloud(depth_m, intr, mask=~object_mask.astype(bool))
    scene_pts = scene_pts[:: max(1, plane_stride * plane_stride)]
    normal_cam, d_cam, _ = fit_plane_ransac(scene_pts)
    # A representative on-plane point: project the object centroid onto the table.
    table_pt_cam = obj["centroid"] - (normal_cam @ obj["centroid"] + d_cam) * normal_cam

    to_base = lambda p: transform_points(T_base_camera, p[None])[0]  # noqa: E731
    object_centroid_b = to_base(obj["centroid"])
    object_top_b = to_base(obj["top"])
    table_pt_b = to_base(table_pt_cam)
    # Normal is a direction: rotate only.
    table_normal_b = T_base_camera[:3, :3] @ normal_cam
    table_normal_b = table_normal_b / np.linalg.norm(table_normal_b)

    return {
        "object_centroid": object_centroid_b,
        "object_top": object_top_b,
        "table_normal": table_normal_b,
        "table_point": table_pt_b,
        "object_height": float(np.dot(object_top_b - table_pt_b, table_normal_b)),
        "object_cam": obj,
        "table_cam": {"normal": normal_cam, "d": d_cam},
    }
