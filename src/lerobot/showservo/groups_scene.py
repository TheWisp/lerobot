# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The scene around the point groups: 3D from depth, the tray's plane, the object under a click, the points to
track around it, and the drawing of groups and objects on a frame. Pure numpy and OpenCV, so the tracker's own
environment can load it by path."""

from __future__ import annotations

import cv2
import numpy as np

PALETTE = [
    (60, 180, 75),
    (230, 25, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
]


def points_3d(depth_m: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Camera-frame xyz (m) of every pixel, HxWx3; NaN where there is no depth."""
    h, w = depth_m.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth_m.astype(np.float64)
    z[z <= 0] = np.nan
    return np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1)


def tray_height(pts: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Height of every pixel above the tray's plane (:func:`tray_plane`)."""
    return heights(pts, tray_plane(pts, region))


def heights(pts: np.ndarray, plane: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Height of every pixel above ``plane`` (centre, unit normal toward the camera)."""
    centre, n = plane
    return (pts - centre) @ n


def tray_plane(pts: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The plane fitted (RANSAC) to the region's points, (centre, unit normal): the tray, where most of the region
    is tray. The normal points toward the camera, so heights above the tray are positive."""
    p = pts[region & np.isfinite(pts).all(axis=2)]
    rng = np.random.default_rng(0)
    best = (0, None)
    for _ in range(200):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        count = int((np.abs((p - a) @ n) < 0.004).sum())
        if count > best[0]:
            best = (count, (a, n))
    a, n = best[1]
    inl = p[np.abs((p - a) @ n) < 0.004]
    centre = inl.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((inl - centre).T))
    n = vecs[:, 0]
    if n[2] > 0:
        n = -n
    return centre, n


def object_mask(height: np.ndarray, click: tuple[int, int], band_m: float = 0.012) -> np.ndarray:
    """The raised body under the click, at the click's height: a stacked object parts from what it stands on."""
    u, v = click
    raised = (height > 0.004) & np.isfinite(height)
    _, lab = cv2.connectedComponents(raised.astype(np.uint8))
    body = lab == lab[v, u]
    h0 = np.nanmedian(height[max(v - 3, 0) : v + 4, max(u - 3, 0) : u + 4])
    mask = body & (np.abs(height - h0) < band_m)
    _, lab = cv2.connectedComponents(mask.astype(np.uint8))
    return lab == lab[v, u]


def trackable_points(gray: np.ndarray, where: np.ndarray, n: int, spacing: int) -> np.ndarray:
    """Up to ``n`` points to track inside ``where``: corners first (what a tracker holds on to), then a grid where
    corners are sparse. (N, 2) as (x, y)."""
    where8 = where.astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=n, qualityLevel=0.01, minDistance=spacing, mask=where8)
    pts = corners.reshape(-1, 2) if corners is not None else np.zeros((0, 2), np.float32)
    if len(pts) < n:
        h, w = where.shape
        ys, xs = np.mgrid[spacing // 2 : h : spacing, spacing // 2 : w : spacing]
        grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        grid = grid[where[grid[:, 1].astype(int), grid[:, 0].astype(int)]]
        if len(pts):
            d = np.linalg.norm(grid[:, None] - pts[None], axis=2).min(axis=1)
            grid = grid[d >= spacing]
        pts = np.vstack([pts, grid[: n - len(pts)]])
    return pts[:n]


def stable_points(
    gray: np.ndarray,
    where: np.ndarray,
    n: int,
    cell_px: int,
    near: np.ndarray | None = None,
    spacing: int = 8,
) -> np.ndarray:
    """Up to ``n`` corners inside ``where`` that a tracker can hold, balanced over the image: each ``cell_px`` cell
    gives up its best corners in turn until the budget is spent (ORB-SLAM's per-cell extraction), so coverage
    surrounds a target instead of piling up where texture is densest. No grid fallback: a smooth surface offers
    nothing a tracker can hold, and a point put there only drifts. With ``near`` (x, y), the cells closest to it are
    served first. (N, 2) as (x, y)."""
    where8 = where.astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=4 * n, qualityLevel=0.005, minDistance=spacing, mask=where8
    )
    if corners is None:
        return np.zeros((0, 2), np.float32)
    pts = corners.reshape(-1, 2).astype(np.float32)  # strongest first
    cells: dict[tuple[int, int], list[np.ndarray]] = {}
    for p in pts:
        cells.setdefault((int(p[0] // cell_px), int(p[1] // cell_px)), []).append(p)
    order = list(cells)
    if near is not None:
        order.sort(key=lambda c: np.hypot((c[0] + 0.5) * cell_px - near[0], (c[1] + 0.5) * cell_px - near[1]))
    chosen: list[np.ndarray] = []
    while len(chosen) < n and any(cells[c] for c in order):
        for c in order:
            if cells[c]:
                chosen.append(cells[c].pop(0))
                if len(chosen) >= n:
                    break
    return np.asarray(chosen, dtype=np.float32).reshape(-1, 2)


def around(mask: np.ndarray, inner_px: int, outer_px: int) -> np.ndarray:
    """The ring around a mask, ``inner_px`` clear of it and ``outer_px`` out."""
    m = mask.astype(np.uint8)
    out = cv2.dilate(m, np.ones((2 * outer_px + 1, 2 * outer_px + 1), np.uint8)) > 0
    inn = cv2.dilate(m, np.ones((2 * inner_px + 1, 2 * inner_px + 1), np.uint8)) > 0
    return out & ~inn


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack(
            [k[0, 0] * xyz[:, 0] / xyz[:, 2] + k[0, 2], k[1, 1] * xyz[:, 1] / xyz[:, 2] + k[1, 2]], axis=1
        )


def contour_3d(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """The object's outline as 3D points (camera frame), for drawing it where a pose puts it."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)
    p = pts[c[:, 1], c[:, 0]]
    fill = np.nanmedian(pts[mask], axis=0)
    return np.where(np.isfinite(p), p, fill)


def lookup_3d(
    uv: np.ndarray, visible: np.ndarray, pts: np.ndarray, window: int = 2, edge_m: float = 0.012
) -> tuple[np.ndarray, np.ndarray]:
    """Each track's 3D point from the depth around it: the median over a (2*window+1) square, and nothing where
    that square's depth spreads more than ``edge_m`` (a corner on a depth edge, the rim's lip, reads a mixed pixel).
    (xyz, seen). Pre: ``uv`` (N, 2) as (x, y)."""
    h, w = pts.shape[:2]
    n = len(uv)
    xyz = np.full((n, 3), np.nan)
    seen = np.zeros(n, bool)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & np.asarray(visible, bool)
    for i in np.flatnonzero(inside):
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        patch = pts[max(v - window, 0) : v + window + 1, max(u - window, 0) : u + window + 1].reshape(-1, 3)
        patch = patch[np.isfinite(patch).all(axis=1)]
        if len(patch) < 3:
            continue
        z = patch[:, 2]
        if np.percentile(z, 90) - np.percentile(z, 10) > edge_m:
            continue
        xyz[i] = np.median(patch, axis=0)
        seen[i] = True
    return xyz, seen


def draw(rgb, k, tracker, xyz, seen, objects, header: str, footer: str = "") -> np.ndarray:
    """Tracks by group (filled when seen, hollow where the group predicts them when not; grey when free) and each
    object's outline where its group puts it (solid), with its own-points fit as a cross when it has one.
    ``objects``: name -> (outline_3d, centre0, p2p_outline_uv or None)."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for t in range(len(tracker.group_of)):
        g = tracker.group_of[t]
        colour = (160, 160, 160) if g < 0 else PALETTE[g % len(PALETTE)][::-1]
        if seen[t]:
            uv = project(xyz[t], k)[0]
            cv2.circle(img, (int(uv[0]), int(uv[1])), 3, colour, -1)
        elif g >= 0 and g in tracker.groups:
            uv = project(tracker.groups[g].motion.apply(tracker.anchor[t][None])[0], k)[0]
            if np.isfinite(uv).all():
                cv2.circle(img, (int(uv[0]), int(uv[1])), 3, colour, 1)
    for i, (name, (outline, centre0, other_uv)) in enumerate(objects.items()):
        obj = tracker.objects[name]
        colour = PALETTE[i][::-1]
        pose = obj.pose
        uv = project((outline - centre0) @ pose[:3, :3].T + pose[:3, 3], k)
        if np.isfinite(uv).all():
            cv2.polylines(img, [uv.astype(np.int32).reshape(-1, 1, 2)], True, colour, 2)
        if other_uv is not None and np.isfinite(other_uv).all():
            cv2.polylines(img, [other_uv.astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 255), 1)
        c = project(pose[:3, 3], k)[0]
        label = f"{name}: seen {obj.n_seen}/{len(obj.tracks)}, group {obj.group}"
        if obj.own_ok and obj.carried is not None:
            label += f", own points {np.linalg.norm(obj.carried[:3, 3] - pose[:3, 3]) * 1000:.1f} mm from the group"
        else:
            label += ", carried by the group"
        if np.isfinite(c).all():
            cv2.putText(
                img,
                label,
                (max(int(c[0]) - 60, 4), max(int(c[1]) - 14 - 16 * i, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                2,
            )
    cv2.putText(img, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if footer:
        cv2.putText(img, footer, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img
