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

import base64
import warnings
from collections import deque

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
    All tracks at once. (xyz, seen). Pre: ``uv`` (N, 2) as (x, y)."""
    h, w = pts.shape[:2]
    n = len(uv)
    xyz = np.full((n, 3), np.nan)
    seen = np.zeros(n, bool)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & np.asarray(visible, bool)
    idx = np.flatnonzero(inside)
    if not len(idx):
        return xyz, seen
    u = np.rint(uv[idx, 0]).astype(int)
    v = np.rint(uv[idx, 1]).astype(int)
    d = np.arange(-window, window + 1)
    vv = (v[:, None, None] + d[None, :, None]).repeat(len(d), axis=2)  # (N, rows, cols)
    uu = (u[:, None, None] + d[None, None, :]).repeat(len(d), axis=1)
    off = (vv < 0) | (vv >= h) | (uu < 0) | (uu >= w)
    patch = pts[np.clip(vv, 0, h - 1), np.clip(uu, 0, w - 1)].reshape(len(idx), -1, 3)
    patch[off.reshape(len(idx), -1)] = np.nan
    finite = np.isfinite(patch).all(axis=2)
    patch[~finite] = np.nan
    enough = finite.sum(axis=1) >= 3
    # Most windows have depth everywhere: those take the plain percentile and median, far quicker than the NaN-aware
    # ones the rest need.
    full = finite.all(axis=1)
    lo, hi = np.empty(len(idx)), np.empty(len(idx))
    med = np.empty((len(idx), 3))
    if full.any():
        lo[full], hi[full] = np.percentile(patch[full][:, :, 2], [10, 90], axis=1)
        med[full] = np.median(patch[full], axis=1)
    if (~full).any():
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows: too few points, dropped below
            lo[~full], hi[~full] = np.nanpercentile(patch[~full][:, :, 2], [10, 90], axis=1)
            med[~full] = np.nanmedian(patch[~full], axis=1)
    ok = enough & ~(hi - lo > edge_m)
    xyz[idx[ok]] = med[ok]
    seen[idx[ok]] = True
    return xyz, seen


def outline_mask(
    shape, outline_3d: np.ndarray, centre0: np.ndarray, pose: np.ndarray, k: np.ndarray, grow_px: int = 10
):
    """Where an object is now in the image: its outline from its first frame, moved by its pose, filled and grown
    by ``grow_px``. HxW bool."""
    mask = np.zeros(shape, np.uint8)
    uv = project((outline_3d - centre0) @ pose[:3, :3].T + pose[:3, 3], k)
    if len(uv) >= 3 and np.isfinite(uv).all():
        cv2.fillPoly(mask, [uv.astype(np.int32).reshape(-1, 1, 2)], 1)
        mask = cv2.dilate(mask, np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8))
    return mask.astype(bool)


def group_surfaces(
    depth_m: np.ndarray,
    uv: np.ndarray,
    seen: np.ndarray,
    group_of: np.ndarray,
    base: int | None,
    scale: int = 2,
    edge_m: float = 0.008,
    reach_px: int = 28,
    quiet=(),
) -> np.ndarray:
    """Which surface moves with which group, for the eye, painted only where it is measured: from each seen track
    of a group other than the base, its label spreads over the smooth surface it sits on (no crossing of a depth
    step of more than ``edge_m``, a hole, or another body) out to ``reach_px``. Dense tracks paint a body whole;
    a lone track paints a disc; nothing far from a track is painted, and the base group, the world, never is. The
    tracks stay the truth; this shows what they sit on. HxW int, -1 where nothing is painted. Computed at
    1/``scale`` resolution; a few 3x3 dilations, well under a millisecond each."""
    d = depth_m[::scale, ::scale].astype(np.float32)
    valid = d > 0
    k3 = np.ones((3, 3), np.uint8)
    step = cv2.dilate(np.where(valid, d, 0.0), k3) - cv2.erode(np.where(valid, d, 1e3), k3)
    body = (valid & (step <= edge_m)).astype(np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    h, w = body.shape
    out = np.full((h, w), -1, dtype=np.int32)
    tracks = np.flatnonzero(
        np.asarray(seen, bool)
        & (group_of >= 0)
        & (group_of != (-1 if base is None else base))
        & ~np.isin(group_of, list(quiet))
    )
    if len(tracks):
        u = np.clip((uv[tracks, 0] / scale).astype(int), 0, w - 1)
        v = np.clip((uv[tracks, 1] / scale).astype(int), 0, h - 1)
        for g in np.unique(group_of[tracks]):
            on = group_of[tracks] == g
            seed = np.zeros((h, w), np.uint8)
            seed[v[on], u[on]] = 1
            seed &= body
            for i in range(max(1, reach_px // scale)):  # geodesic: only over the smooth surface
                grown = (
                    cv2.dilate(seed, k3 if i % 2 else cross) & body
                )  # alternating kernels: round, not square
                if np.array_equal(grown, seed):
                    break
                seed = grown
            out[seed.astype(bool)] = int(g)
    full = np.full(depth_m.shape, -1, dtype=np.int32)
    big = out.repeat(scale, axis=0).repeat(scale, axis=1)
    hh, ww = min(full.shape[0], big.shape[0]), min(full.shape[1], big.shape[1])
    full[:hh, :ww] = big[:hh, :ww]
    return full


# The groups for the eye: the base group, the one with the most tracks, is the world and wears no colour; a group
# that moves differently lights up in one of these, none of them a green the tray and the objects wear.
GROUP_COLOURS = [
    (240, 50, 230),
    (70, 240, 240),
    (255, 225, 25),
    (245, 130, 48),
    (145, 30, 180),
    (230, 25, 75),
]


def base_group(group_of: np.ndarray) -> int | None:
    """The group with the most tracks: the world, as far as the eye is concerned."""
    grouped = group_of[group_of >= 0]
    return int(np.bincount(grouped).argmax()) if len(grouped) else None


def group_colour(g: int, base: int | None, quiet=()) -> tuple[int, int, int]:
    """BGR for a track or surface of group ``g``: grey for none, white for the base group and for a group still
    settling (``quiet``: too young for its motion to be known), a GROUP_COLOURS hue otherwise."""
    if g < 0:
        return (160, 160, 160)
    if g == base or g in quiet:
        return (255, 255, 255)
    return GROUP_COLOURS[g % len(GROUP_COLOURS)][::-1]


class World:
    """Which group is the world, for the eye: the one whose members moved least over the last ``window`` frames,
    not the biggest (a tray carrying most of the tracks is the thing that moves, not the world). The choice sticks:
    another group takes it only once its members, over a history of more than ``settle`` frames, have moved less
    than half as far, so two still groups never trade places on noise."""

    def __init__(self, window: int = 15, settle: int = 5, floor_m: float = 0.003):
        self.window, self.settle, self.floor_m = window, settle, floor_m
        self.current: int | None = None
        self.quiet: set[int] = set()  # groups too young to judge: drawn as the world until they settle

    def _moved(self, tracker, g) -> float:
        """How far the group's members moved over the window: the median of them, which a turn moves too (half of a
        body's points sit outside its middle) and which the fit's jitter on a few far points does not."""
        hist = list(g.history)
        members = np.flatnonzero(tracker.group_of == g.id)
        if len(hist) < 2 or not len(members):
            return 0.0
        then = hist[max(0, len(hist) - 1 - self.window)].apply(tracker.anchor[members])
        now = hist[-1].apply(tracker.anchor[members])
        return float(np.nanmedian(np.linalg.norm(now - then, axis=1)))

    def update(self, tracker) -> int | None:
        groups = dict(tracker.groups)
        if not groups:
            self.current, self.quiet = None, set()
            return None
        moved = {gid: self._moved(tracker, g) for gid, g in groups.items()}
        if self.current not in groups:  # the first frame, or the world merged into another group
            settled = [gid for gid, g in groups.items() if len(g.history) > self.settle] or list(groups)
            self.current = min(settled, key=lambda gid: (moved[gid], -int((tracker.group_of == gid).sum())))
        for gid, g in groups.items():
            if (
                gid != self.current
                and len(g.history) > self.settle
                and moved[self.current] > self.floor_m  # a world still within noise is not up for grabs
                and moved[gid] < 0.5 * moved[self.current]
            ):
                self.current = gid
        self.quiet = {
            gid for gid, g in groups.items() if gid != self.current and len(g.history) <= self.settle
        }
        return self.current


class SurfaceMemory:
    """The surfaces' labels over the last three frames, a pixel showing the group that held it in two of them:
    a piece that loses its tracks for a frame, or a depth hole that opens and closes, does not blink."""

    def __init__(self):
        self.maps: deque = deque(maxlen=3)

    def update(self, surfaces: np.ndarray) -> np.ndarray:
        self.maps.append(surfaces)
        if len(self.maps) < 3:
            return surfaces
        a, b, c = self.maps
        return np.where(a == b, a, np.where(b == c, b, np.where(a == c, a, -1)))


def paint_surfaces(
    img: np.ndarray, surfaces: np.ndarray, base: int | None = None, alpha: float = 0.5, quiet=()
) -> np.ndarray:
    """The surfaces of every group but the base in the group's colour, see-through, with a solid rim; the base
    group's surfaces, the world, stay as the camera saw them, so a tint means "moves differently"."""
    overlay = img.copy()
    k3 = np.ones((3, 3), np.uint8)
    rims = []
    for g in np.unique(surfaces):
        if g < 0 or g == base or g in quiet:
            continue
        colour = group_colour(int(g), base)
        m = surfaces == g
        overlay[m] = colour
        rims.append((m & ~cv2.erode(m.astype(np.uint8), k3).astype(bool), colour))
    if not rims:
        return img
    img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)
    for rim, colour in rims:
        img[rim] = colour
    return img


def draw(
    rgb, k, tracker, xyz, seen, objects, header: str, footer: str = "", surfaces=None, base=None, quiet=()
) -> np.ndarray:
    """Tracks by group: filled where seen, filled where the group puts them for a frame or two unseen, hollow
    there once hidden three frames (under a hand, a sheet); grey when free. White for the world (``base``, the
    stillest group, :class:`World`; the biggest when not given) and coloured for a group moving differently,
    the surfaces they sit on in that colour when ``surfaces`` (:func:`group_surfaces`) is given, and each
    object's outline where its group puts it (solid), with its own-points fit as a cross when it has one.
    ``objects``: name -> (outline_3d, centre0, p2p_outline_uv or None)."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    if base is None:
        base = base_group(tracker.group_of)
    if surfaces is not None:
        img = paint_surfaces(img, surfaces, base, quiet=quiet)
    for t in range(len(tracker.group_of)):
        g = tracker.group_of[t]
        if tracker.retired[t]:
            continue  # a corner that kept slipping: no longer a reference, no longer drawn
        colour = group_colour(g, base, quiet)
        if seen[t]:
            uv = project(xyz[t], k)[0]
            cv2.circle(img, (int(uv[0]), int(uv[1])), 3, colour, -1)
        elif g >= 0 and g in tracker.groups:  # where its group puts it, a shade smaller: not seen this frame
            uv = project(tracker.groups[g].motion.apply(tracker.anchor[t][None])[0], k)[0]
            if np.isfinite(uv).all():
                hidden = (
                    tracker.unseen[t] >= 3
                )  # three frames: the tracker's visibility blinks, a cover does not
                cv2.circle(img, (int(uv[0]), int(uv[1])), 3, colour, 1 if hidden else -1)
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
            label += f", own points {np.linalg.norm(obj.carried[:3, 3] - pose[:3, 3]) * 1000:.0f} mm from the group"
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
    x = 8  # the groups by size, each in its colour, so the legend is the picture's own key
    for g in sorted(tracker.groups, key=lambda g: -int((tracker.group_of == g).sum())):
        label = f"g{g}: {int((tracker.group_of == g).sum())} pts" + (
            "  = the world, no tint" if g == base else ("  settling" if g in quiet else "")
        )
        cv2.putText(img, label, (x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, group_colour(g, base, quiet), 2)
        x += cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0] + 18
    if footer:
        cv2.putText(img, footer, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


SURFACE_NONE = 65535  # a pixel no group's surface holds, in drawing()'s label map


def drawing(k, tracker, xyz, seen, objects, surfaces=None, base=None, quiet=(), scale: int = 4) -> dict:
    """The groups as a camera view outside this process draws them (:func:`paint_groups`), in pixels, JSON-ready:
    every standing track where it is seen, or where its group puts it (hollow once hidden three frames), with its
    group; the surfaces that move differently (:func:`group_surfaces`) as a label map at 1/``scale``, a 16-bit PNG;
    each object's outline where its pose puts it, in its :data:`PALETTE` colour, whether its own points placed it this
    frame (and why not), the group carrying it and which of ``points`` are its own tracks; and each group's members,
    the visible members its motion was fitted on, and that fit's residual: what an act's record keeps to tell what
    carried an object. The same picture :func:`draw` paints, without its words."""
    if base is None:
        base = base_group(tracker.group_of)
    points, listed = [], {}  # listed: track -> its row in points
    for t in range(len(tracker.group_of)):
        if tracker.retired[t]:
            continue
        g = int(tracker.group_of[t])
        if seen[t]:
            uv, filled = project(xyz[t], k)[0], True
        elif g >= 0 and g in tracker.groups:
            uv = project(tracker.groups[g].motion.apply(tracker.anchor[t][None])[0], k)[0]
            filled = bool(tracker.unseen[t] < 3)
        else:
            continue
        if np.isfinite(uv).all():
            listed[t] = len(points)
            points.append([int(uv[0]), int(uv[1]), g, int(filled)])
    labels = None
    if surfaces is not None:
        small = np.asarray(surfaces)[::scale, ::scale]
        ok, png = cv2.imencode(".png", np.where(small < 0, SURFACE_NONE, small).astype(np.uint16))
        labels = base64.b64encode(png.tobytes()).decode() if ok else None
    outlines = {}
    for i, (name, (outline, centre0, _other)) in enumerate(objects.items()):
        obj = tracker.objects[name]
        uv = project((outline - centre0) @ obj.pose[:3, :3].T + obj.pose[:3, 3], k)
        outlines[name] = {
            "outline": uv.astype(int).tolist() if len(uv) and np.isfinite(uv).all() else [],
            "colour": list(PALETTE[i % len(PALETTE)]),
            "own": bool(obj.own_ok),
            "why": obj.why,
            "group": obj.group,
            "n_seen": int(obj.n_seen),
            "n_grouped": int(obj.n_grouped),
            "own_points": [listed[int(t)] for t in obj.tracks if int(t) in listed],
        }
    members = np.bincount(tracker.group_of[tracker.group_of >= 0]) if (tracker.group_of >= 0).any() else []
    groups = {
        str(g.id): {
            "members": int(members[g.id]) if g.id < len(members) else 0,
            "n_fit": int(g.n_fit),
            "rms_mm": round(float(g.rms) * 1000.0, 2),
            "supported": bool(g.supported),
        }
        for g in tracker.groups.values()
    }
    return {
        "points": points,
        "surfaces": labels,
        "scale": scale,
        "base": base,
        "quiet": sorted(int(g) for g in quiet),
        "objects": outlines,
        "groups": groups,
    }


def paint_groups(
    img: np.ndarray,
    d: dict,
    alpha: float = 0.5,
    radius: int = 3,
    world_radius: int | None = None,
    free: bool = True,
    outlines: bool = True,
) -> np.ndarray:
    """:func:`drawing`'s picture on a BGR frame of the same camera: the surfaces that move differently tinted, the
    tracks in their groups' colours (white for the world, hollow where hidden) as dots of ``radius`` (the world's of
    ``world_radius`` when given), with ``free`` the tracks in no group (grey), and with ``outlines`` each object's
    outline."""
    base, quiet = d.get("base"), set(d.get("quiet") or ())
    if d.get("surfaces"):
        small = cv2.imdecode(np.frombuffer(base64.b64decode(d["surfaces"]), np.uint8), cv2.IMREAD_UNCHANGED)
        if small is not None:
            labels = small.astype(np.int32)
            labels[labels == SURFACE_NONE] = -1
            scale = int(d.get("scale") or 1)
            labels = labels.repeat(scale, axis=0).repeat(scale, axis=1)[: img.shape[0], : img.shape[1]]
            full = np.full(img.shape[:2], -1, dtype=np.int32)
            full[: labels.shape[0], : labels.shape[1]] = labels
            img = paint_surfaces(img, full, base, alpha, quiet=quiet)
    for u, v, g, filled in d.get("points") or ():
        if g < 0 and not free:
            continue
        r = world_radius if world_radius is not None and (g == base or g in quiet) else radius
        cv2.circle(img, (int(u), int(v)), r, group_colour(int(g), base, quiet), -1 if filled else 1)
    for o in (d.get("objects") or {}).values() if outlines else ():
        if len(o.get("outline") or ()) >= 2:
            pts = np.asarray(o["outline"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, tuple(o["colour"])[::-1], 2)
    return img
