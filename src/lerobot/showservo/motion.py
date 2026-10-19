#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure object registration, rigid-motion composition and key-frame planning.

These calculations have no GUI, camera or arm ownership. The manipulation API
supplies measured observations and enforces acceptance before commanding motion.
Legacy demo mark formats are normalized into the same key-frame representation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lerobot.showservo import placement


def keypoints_in_box(
    rgb: np.ndarray, depth_m: np.ndarray, intr: dict[str, float], box: tuple[int, int, int, int]
) -> dict[str, Any]:
    """SIFT keypoints inside ``box`` (x0, y0, x1, y1), with camera-frame 3D where depth exists.

    Post: ``uv`` (n, 2), ``desc`` (n, 128), ``xyz`` (n, 3), ``valid`` (n,) bool for
    points with usable depth. Raises ``ValueError`` when fewer than 8 have depth.
    """
    from lerobot.showservo.binder import sift_keypoints
    from lerobot.showservo.pose import CameraIntrinsics, sample_depth

    x0, y0, x1, y1 = (int(v) for v in box)
    assert x1 > x0 and y1 > y0, "box must have area"
    mask = np.zeros(depth_m.shape, dtype=np.uint8)
    mask[max(0, y0) : y1, max(0, x0) : x1] = 1
    uv, desc = sift_keypoints(rgb, mask)
    z, valid = sample_depth(depth_m, uv)
    if int(valid.sum()) < 8:
        raise ValueError(
            f"only {int(valid.sum())} keypoints with depth inside the box; pick a textured object"
        )
    cam = CameraIntrinsics(fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"])
    xyz = np.zeros((len(uv), 3))
    xyz[valid] = cam.deproject(uv[valid], z[valid])
    return {"uv": uv, "desc": desc, "xyz": xyz, "valid": valid}


def register(
    teach: dict[str, Any], rgb: np.ndarray, depth_m: np.ndarray, intr: dict[str, float]
) -> dict[str, Any]:
    """The object's rigid motion, camera frame, from the taught keypoints to a new frame.

    Post: ``ok`` with the 4x4 ``delta_cam`` (teach -> now) and the evidence: match
    counts, 2D inliers, 3D inliers, rms in metres, the similarity scale (a rigid
    object keeps its size; a scale off 1 flags a depth fault). ``ok`` False carries a
    ``reason`` and no transform.
    """
    from lerobot.fewshot.registration import mutual_matches
    from lerobot.showservo.binder import sift_keypoints
    from lerobot.showservo.grouping import fit_team
    from lerobot.showservo.pose import CameraIntrinsics, ransac_fit_rigid, sample_depth

    live_uv, live_desc = sift_keypoints(rgb, None)
    if len(live_uv) < 8:
        return {"ok": False, "reason": "too few features in the new frame"}
    ia, ib = mutual_matches(teach["desc"], live_desc, ratio=0.9)
    if len(ia) < 6:
        return {"ok": False, "reason": f"only {len(ia)} mutual matches", "n_matches": int(len(ia))}
    fit2 = fit_team(teach["uv"][ia], live_uv[ib], inlier_px=6.0)
    if not fit2.ok:
        return {"ok": False, "reason": "no 2D consensus among the matches", "n_matches": int(len(ia))}
    inl = fit2.inliers
    src = teach["xyz"][ia][inl]
    z, live_valid = sample_depth(depth_m, live_uv[ib][inl])
    cam = CameraIntrinsics(fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"])
    dst = np.zeros_like(src)
    dst[live_valid] = cam.deproject(live_uv[ib][inl][live_valid], z[live_valid])
    valid = teach["valid"][ia][inl] & live_valid
    fit3 = ransac_fit_rigid(src, dst, valid, inlier_m=0.006)
    if not fit3.ok:
        return {
            "ok": False,
            "reason": "no 3D consensus (depth holes or the object is not rigid in view)",
            "n_matches": int(len(ia)),
            "n_inliers_2d": int(inl.sum()),
        }
    delta = np.eye(4)
    delta[:3, :3], delta[:3, 3] = fit3.transform.rot, fit3.transform.trans
    inl3 = fit3.inliers
    return {
        "ok": True,
        "delta_cam": delta,
        "n_matches": int(len(ia)),
        "n_inliers_2d": int(inl.sum()),
        "n_inliers_3d": int(inl3.sum()),
        "rms_m": float(fit3.rms),
        "scale": float(fit3.scale),
        "teach_uv": teach["uv"][ia][inl][inl3],
        "live_uv": live_uv[ib][inl][inl3],
    }


def transport_pose(t_base_cam: np.ndarray, delta_cam: np.ndarray, pose_teach: np.ndarray) -> np.ndarray:
    """Carry a base-frame pose by an object motion measured in the camera frame.

    ``delta_base = T_bc @ delta_cam @ T_bc^-1``; post: ``delta_base @ pose_teach``.
    """
    t_bc = np.asarray(t_base_cam, dtype=float)
    delta_base = t_bc @ np.asarray(delta_cam, dtype=float) @ np.linalg.inv(t_bc)
    return delta_base @ np.asarray(pose_teach, dtype=float)


def motion_summary(delta: np.ndarray) -> dict[str, float]:
    """Translation in mm and rotation in degrees of a 4x4 motion, for a readout."""
    from scipy.spatial.transform import Rotation

    d = np.asarray(delta, dtype=float)
    return {
        "translation_mm": float(np.linalg.norm(d[:3, 3]) * 1000.0),
        "rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec()))),
    }


def turn_and_lean(delta: np.ndarray, up: np.ndarray) -> dict[str, float]:
    """A rigid motion's turn about ``up`` and how far it tips ``up`` over, in degrees.

    For a base-frame motion with ``up`` = +z this is what the operator can check
    by eye before Go: the gripper turns ``turn_deg`` about vertical and leans
    ``lean_deg``. A motion whose axis is vertical has zero lean.
    """
    r = np.asarray(delta, dtype=float)[:3, :3]
    n = np.asarray(up, dtype=float)
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    v = r @ e1
    v -= np.dot(v, n) * n
    turn = float(np.degrees(np.arctan2(np.dot(v, e2), np.dot(v, e1))))
    lean = float(np.degrees(np.arccos(np.clip(np.dot(r @ n, n), -1.0, 1.0))))
    return {"turn_deg": turn, "lean_deg": lean}


# ── shape mode: textureless objects, found by what rises above the table ─────

MIN_TEXTURE_POINTS = 30  # below this the SIFT points are noise that will not re-match; shape is safer
ABOVE_TABLE_M = 0.004


def table_plane(
    depth_m: np.ndarray, intr: dict[str, float], box: tuple[int, int, int, int], margin: int = 40
):
    """The table as a plane fitted to the depth in a ring just outside ``box``: ``(unit normal, point)``."""
    from lerobot.showservo.pose import CameraIntrinsics

    h, w = depth_m.shape
    x0, y0, x1, y1 = box
    ring = np.zeros((h, w), dtype=bool)
    ring[max(0, y0 - margin) : min(h, y1 + margin), max(0, x0 - margin) : min(w, x1 + margin)] = True
    ring[max(0, y0) : y1, max(0, x0) : x1] = False
    vs, us = np.nonzero(ring & (depth_m > 0))
    if len(us) < 200:
        raise ValueError("not enough depth around the box to fit the table")
    cam = CameraIntrinsics(fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"])
    pts = cam.deproject(np.stack([us, vs], axis=1), depth_m[vs, us])
    c = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    if n[2] > 0:
        n = -n  # facing the camera
    # One robust pass against clutter in the ring.
    dist = (pts - c) @ n
    keep = np.abs(dist) < 3.0 * max(float(np.median(np.abs(dist))), 1e-4)
    if keep.sum() >= 100:
        c = pts[keep].mean(axis=0)
        _u, _s, vt = np.linalg.svd(pts[keep] - c, full_matrices=False)
        n = vt[2]
        if n[2] > 0:
            n = -n
    return n, c


def above_table(
    depth_m: np.ndarray, intr: dict[str, float], plane, region: np.ndarray | None = None
) -> np.ndarray:
    """Boolean image of pixels whose depth point stands more than :data:`ABOVE_TABLE_M` above the plane."""
    from lerobot.showservo.pose import CameraIntrinsics

    n, c = plane
    h, w = depth_m.shape
    vs, us = np.nonzero((depth_m > 0) & (region if region is not None else np.ones((h, w), dtype=bool)))
    cam = CameraIntrinsics(fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"])
    pts = cam.deproject(np.stack([us, vs], axis=1), depth_m[vs, us])
    # The normal faces the camera, so "above the table" is toward the camera: positive along n.
    height = (pts - c) @ n
    out = np.zeros((h, w), dtype=bool)
    out[vs[height > ABOVE_TABLE_M], us[height > ABOVE_TABLE_M]] = True
    return out


FOOTPRINT_MM = 2.0  # raster cell for the footprint yaw search
YAW_STEP_DEG = 5.0
ROUND_IOU_SPREAD = 0.12  # a footprint whose IoU barely changes with yaw has no measurable turn


def plane_basis(plane) -> tuple[np.ndarray, np.ndarray]:
    """Right-handed in-plane axes ``(e1, e2)`` with ``e2 = n x e1``, fixed by the plane alone."""
    n, _c = plane
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def footprint(points: np.ndarray, plane, centre: np.ndarray) -> np.ndarray:
    """In-plane 2D coordinates of ``points`` about ``centre``, in the plane's basis."""
    e1, e2 = plane_basis(plane)
    d = points - centre
    return np.stack([d @ e1, d @ e2], axis=1)


def _raster(xy: np.ndarray, half_m: float, cell_m: float) -> np.ndarray:
    n = int(np.ceil(2 * half_m / cell_m)) + 1
    ij = np.floor((xy + half_m) / cell_m).astype(int)
    ok = (ij >= 0).all(axis=1) & (ij < n).all(axis=1)
    img = np.zeros((n, n), dtype=bool)
    img[ij[ok, 1], ij[ok, 0]] = True
    return img


def footprint_yaw(
    taught_xy: np.ndarray, found_xy: np.ndarray, prefer_deg: float | None = None
) -> dict[str, Any]:
    """The turn about the table normal that best overlays the taught footprint on the found one.

    Scans the full circle and refines to a degree. Among near-equal peaks (a
    rectangle has two, a square four) it takes the one nearest ``prefer_deg``,
    the previous answer when tracking, so a square sitting halfway between two
    of its own symmetries does not flip between them frame to frame; without a
    preference, the smallest turn. Post: ``yaw_deg`` and ``symmetric`` (True
    when the overlap barely depends on the turn, i.e. a round footprint, in
    which case ``yaw_deg`` is 0).
    """
    half = float(max(np.abs(taught_xy).max(), np.abs(found_xy).max())) * 1.2 + FOOTPRINT_MM / 1000
    cell = FOOTPRINT_MM / 1000
    target = _raster(found_xy, half, cell)

    def iou(theta_deg: float) -> float:
        t = np.radians(theta_deg)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        img = _raster(taught_xy @ rot.T, half, cell)
        inter, union = np.logical_and(img, target).sum(), np.logical_or(img, target).sum()
        return float(inter / max(union, 1))

    angles = np.arange(-180.0, 180.0, YAW_STEP_DEG)
    scores = np.array([iou(a) for a in angles])
    spread = float(scores.max() - scores.min())
    if spread < ROUND_IOU_SPREAD:
        return {"yaw_deg": 0.0, "symmetric": True, "iou": float(scores.max()), "iou_spread": spread}
    peak = scores.max()
    candidates = angles[scores >= peak - 0.02]
    anchor = 0.0 if prefer_deg is None else float(prefer_deg)
    turns = (candidates - anchor + 180.0) % 360.0 - 180.0
    coarse = float(candidates[np.argmin(np.abs(turns))])
    fine = np.arange(coarse - YAW_STEP_DEG, coarse + YAW_STEP_DEG + 0.5, 1.0)
    fine_scores = np.array([iou(a) for a in fine])
    best = float(fine[np.argmax(fine_scores)])
    return {"yaw_deg": best, "symmetric": False, "iou": float(fine_scores.max()), "iou_spread": spread}


def colour_model(rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any] | None:
    """Hue-saturation histogram of the object's pixels, or None when its colour does not single it out."""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = mask.astype(np.uint8)
    if int(m.sum()) < 30:
        return None
    hist = cv2.calcHist([hsv], [0, 1], m, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    back = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    own = float((back[mask] > 50).mean())  # the gate must keep most of the object itself
    if own < 0.7:
        return None
    return {"hist": hist, "own_pass": own}


def colour_gate(rgb: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Pixels whose colour matches the taught object (a boolean image), lightly cleaned."""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    back = cv2.calcBackProject([hsv], [0, 1], model["hist"], [0, 180, 0, 256], 1)
    gate = (back > 50).astype(np.uint8)
    gate = cv2.morphologyEx(gate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return gate.astype(bool)


def shape_stats(depth_m: np.ndarray, intr: dict[str, float], plane, mask: np.ndarray) -> dict[str, Any]:
    """Centroid, height and in-plane principal axis of the object cloud under ``mask``."""
    from lerobot.showservo.pose import CameraIntrinsics

    n, c = plane
    vs, us = np.nonzero(mask & (depth_m > 0))
    if len(us) < 30:
        raise ValueError("object cloud too small")
    cam = CameraIntrinsics(fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"])
    pts = cam.deproject(np.stack([us, vs], axis=1), depth_m[vs, us])
    height = (pts - c) @ n
    centroid = pts.mean(axis=0)
    # Principal axis of the footprint, in the table plane.
    flat = pts - np.outer((pts - c) @ n, n)
    fc = flat.mean(axis=0)
    _u, s, vt = np.linalg.svd(flat - fc, full_matrices=False)
    axis = vt[0] - np.dot(vt[0], n) * n
    axis /= max(np.linalg.norm(axis), 1e-9)
    return {
        "centroid": centroid,
        "axis": axis,
        "elongation": float(s[0] / max(s[1], 1e-9)),
        "height_m": float(np.percentile(height, 90)),
        "n_points": int(len(us)),
        "mask": mask,
        "footprint": footprint(pts, plane, centroid),
    }


def shape_teach_mask(
    depth_m: np.ndarray, intr: dict[str, float], mask: np.ndarray, rgb: np.ndarray | None = None
) -> dict[str, Any]:
    """:func:`shape_teach` for a designated mask instead of a drawn box: the table from a ring around it."""
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        raise ValueError("empty mask")
    import cv2

    box = (int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1)
    plane = table_plane(depth_m, intr, box)
    # The object is the above-table blob the designation overlaps most, not the designation
    # itself: the find later takes whole blobs, and a blob clipped to the mask has a
    # different centre from the same blob found whole.
    standing = above_table(depth_m, intr, plane)
    n_lab, labels, _stats, _cent = cv2.connectedComponentsWithStats(standing.astype(np.uint8), connectivity=8)
    best, best_overlap = None, 0
    for lab in range(1, n_lab):
        overlap = int(np.count_nonzero(mask & (labels == lab)))
        if overlap > best_overlap:
            best, best_overlap = lab, overlap
    obj = standing & mask if best is None else labels == best
    try:
        stats = shape_stats(depth_m, intr, plane, obj)
    except ValueError as e:
        raise ValueError("nothing stands above the table inside the mask") from e
    colour = colour_model(rgb, obj) if rgb is not None else None
    return {"mode": "shape", "plane": plane, "colour": colour, **stats}


def shape_teach(
    depth_m: np.ndarray, intr: dict[str, float], box: tuple[int, int, int, int], rgb: np.ndarray | None = None
) -> dict[str, Any]:
    """The object as what stands above the table inside the box, with its colour when that singles it out.

    Raises ``ValueError`` when nothing stands above the table there.
    """
    x0, y0, x1, y1 = box
    plane = table_plane(depth_m, intr, box)
    region = np.zeros(depth_m.shape, dtype=bool)
    region[max(0, y0) : y1, max(0, x0) : x1] = True
    mask = above_table(depth_m, intr, plane, region)
    try:
        stats = shape_stats(depth_m, intr, plane, mask)
    except ValueError as e:
        raise ValueError(
            "nothing stands above the table inside the box (a flat or dark object gives no depth)"
        ) from e
    colour = colour_model(rgb, mask) if rgb is not None else None
    return {"mode": "shape", "plane": plane, "colour": colour, **stats}


def shape_register(
    teach: dict[str, Any], depth_m: np.ndarray, intr: dict[str, float], rgb: np.ndarray | None = None
) -> dict[str, Any]:
    """Find the taught shape anywhere on the table and return its rigid motion in the camera frame.

    Every blob above the table is a candidate, gated by the taught colour when
    the colour singled the object out at teach time (which also splits it from a
    touching neighbour of another colour); the one whose point count and height
    best match wins. Translation is the centroid shift; the turn about the table
    normal comes from overlaying the taught footprint on the found one over the
    full circle, and is reported as absent when the overlap does not depend on it.
    """
    import cv2

    plane = teach["plane"]
    mask = above_table(depth_m, intr, plane)
    colour_used = False
    if teach.get("colour") is not None and rgb is not None:
        mask &= colour_gate(rgb, teach["colour"])
        colour_used = True
    n_lab, labels, stats, _cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    best, best_score = None, float("inf")
    for lab in range(1, n_lab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 0.3 * teach["n_points"]:
            continue
        try:
            st = shape_stats(depth_m, intr, plane, labels == lab)
        except ValueError:
            continue
        score = (
            abs(np.log(st["n_points"] / teach["n_points"])) + abs(st["height_m"] - teach["height_m"]) / 0.01
        )
        if score < best_score:
            best, best_score = st, score
    if best is None:
        return {"ok": False, "reason": "nothing above the table resembles the taught object", "mode": "shape"}
    if best_score > 1.5:
        return {
            "ok": False,
            "reason": f"best blob differs too much (score {best_score:.2f}): {best['n_points']} vs {teach['n_points']} points, "
            f"height {best['height_m'] * 1000:.0f} vs {teach['height_m'] * 1000:.0f} mm",
            "mode": "shape",
        }
    n, _c = plane
    yaw = footprint_yaw(teach["footprint"], best["footprint"])
    rot = np.eye(3)
    if not yaw["symmetric"]:
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_rotvec(n * np.radians(yaw["yaw_deg"])).as_matrix()
    delta = np.eye(4)
    delta[:3, :3] = rot
    delta[:3, 3] = best["centroid"] - rot @ teach["centroid"]
    return {
        "ok": True,
        "mode": "shape",
        "delta_cam": delta,
        "n_points": best["n_points"],
        "n_points_teach": teach["n_points"],
        "height_mm": best["height_m"] * 1000.0,
        "score": float(best_score),
        "symmetric": bool(yaw["symmetric"]),
        "yaw_deg": float(yaw["yaw_deg"]),
        "footprint_iou": yaw["iou"],
        "colour_used": colour_used,
        "live_mask": best["mask"],
    }


# ── the table prior: objects on the table turn about its normal ──────────────


def snap_to_table_yaw(
    delta_cam: np.ndarray, normal_cam: np.ndarray, centroid_cam: np.ndarray
) -> dict[str, Any]:
    """Replace a fitted rigid motion by the turn about the table normal that moves the object the same way.

    A thin object's cloud constrains its in-plane turn well and the tilt of the
    turn's axis badly, so a 6-DoF fit can carry a large turn about a wrongly
    tilted axis. Post: ``delta`` is a rotation about ``normal_cam`` by ``yaw_deg``
    plus a translation chosen so the object's centroid lands where the original
    fit put it; ``tilt_deg`` is the angle between the original axis and the normal
    (0 when the original was already a pure yaw), discarded.
    """
    from scipy.spatial.transform import Rotation

    d = np.asarray(delta_cam, dtype=float)
    n = np.asarray(normal_cam, dtype=float)
    n = n / np.linalg.norm(n)
    c = np.asarray(centroid_cam, dtype=float)
    r = d[:3, :3]
    # The yaw that best matches r on the plane: rotate an in-plane vector and measure its turn.
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    v = r @ e1
    v -= np.dot(v, n) * n
    yaw = float(np.arctan2(np.dot(v, e2), np.dot(v, e1)))
    rotvec = Rotation.from_matrix(r).as_rotvec()
    ang = float(np.linalg.norm(rotvec))
    tilt = 0.0 if ang < 1e-6 else float(np.degrees(np.arccos(np.clip(abs(np.dot(rotvec / ang, n)), -1, 1))))
    r_yaw = Rotation.from_rotvec(n * yaw).as_matrix()
    c_new = r @ c + d[:3, 3]
    out = np.eye(4)
    out[:3, :3] = r_yaw
    out[:3, 3] = c_new - r_yaw @ c
    return {
        "delta": out,
        "yaw_deg": float(np.degrees(yaw)),
        "tilt_deg": tilt,
        "rotation_deg": float(np.degrees(ang)),
    }


# The face must hold this share of the designated cloud (a cube from above shows its sides too),
# with this many depth points (fewer and its normal is no better than the fit's), and this many
# times the points of the next-largest plane (one face, not two competing).
FACE_PLANARITY_MIN = 0.35
FACE_MIN_POINTS = 100
FACE_DOMINANCE_MIN = 2.0
# A certified fit on a sliver of the card is not a find: the bench's certificate wants a handful of
# inliers, which a wrong match set among hundreds of points can supply by chance.
FIND_MIN_INLIERS = 20
FIND_MIN_INLIER_SHARE = 0.05
# A find of the demo's view that matches less than this share of its points has, on the rig, been off by more
# turn than the grasp tolerates; the share falls as the object lies further from its angle in the demo.
FIND_STRONG_SHARE = 0.15
# The live tracker's algorithms, as the worker names them. Point2Pose (SAM2 masks carried from the
# teach, BootsTAPIR point tracks, cluster RANSAC refined against its TSDF) is the tracker: measured
# against ground truth on nine YCBInEOAT videos it averages 85.1 ADD-S AUC to PatchFit's 80.5 and
# holds objects turned inside a hand. PatchFit, the tracker built here (SAM3 by name, DINO patches
# matched frame by frame, a rigid fit, a growing card), stays as a comparison in three modes: SAM3 and
# DINO on every frame; DINO matched in a window with SAM3 only to acquire; KLT on the matches. Depth
# only is the fourth comparison. p2p_dense is Point2Pose with the registration done by the whole
# visible depth surface against its TSDF (experimental; benchmarked before trusted).
TRACK_ALGOS = ("refind", "dino", "klt", "depth", "p2p", "p2p_dense")
P2P_ALGOS = ("p2p", "p2p_dense")  # the Point2Pose modes: published configuration, and the dense register


def transport_trajectory(
    t_base_cam: np.ndarray, delta_cam: np.ndarray, poses_teach: np.ndarray
) -> np.ndarray:
    """Carry a whole base-frame path (N, 4, 4) by an object motion measured in the camera frame."""
    t_bc = np.asarray(t_base_cam, dtype=float)
    delta_base = t_bc @ np.asarray(delta_cam, dtype=float) @ np.linalg.inv(t_bc)
    return np.einsum("ij,njk->nik", delta_base, np.asarray(poses_teach, dtype=float))


def find_trusted(n_inliers: int, n_card: int) -> tuple[bool, str]:
    """Is a certified fit with ``n_inliers`` of a ``n_card``-point card a find worth acting on? (ok, reason)."""
    need = max(FIND_MIN_INLIERS, int(FIND_MIN_INLIER_SHARE * n_card))
    if n_inliers >= need:
        return True, ""
    return (
        False,
        f"only {n_inliers} of the card's {n_card} points agree (need {need}); the find is not trusted",
    )


def find_strength(n_inliers: int | None, n_card: int | None) -> tuple[bool | None, float | None]:
    """Is a find of the demo's view strong enough to act on? ``(strong, share of the card's points matched)``;
    ``(None, None)`` when the card's size is unknown, as for a find made before it was reported."""
    if not n_card or n_inliers is None:
        return None, None
    share = n_inliers / n_card
    return share >= FIND_STRONG_SHARE, share


def face_usable(face: dict[str, Any] | None) -> bool:
    """Can this face carry the turn's axis? One plane must hold enough of the cloud and clearly dominate."""
    if not face:
        return False
    return (
        face["planarity"] >= FACE_PLANARITY_MIN
        and face.get("n_plane", face["n"]) >= FACE_MIN_POINTS
        and face.get("dominance", float("inf")) >= FACE_DOMINANCE_MIN
    )


def compose_with_face(
    delta_fit: np.ndarray, n_teach: np.ndarray, n_find: np.ndarray, centroid_teach: np.ndarray
) -> dict[str, Any]:
    """The object's motion with its axis from the dense face normals and its turn from the feature fit.

    ``R = R_yaw(n_find) @ R_align`` where ``R_align`` is the smallest rotation
    taking the taught face normal onto the found one, and the yaw about
    ``n_find`` is whatever the fit's rotation did to an in-plane direction. The
    translation keeps the fit's centroid displacement. Post: ``delta``,
    ``yaw_deg`` (about the found face normal), ``face_tilt_deg`` (how much the
    face itself tilted between teach and find), and ``fit_axis_tilt_deg`` (how
    far the raw fit's axis was from the found normal, for the record).
    """
    from scipy.spatial.transform import Rotation

    d = np.asarray(delta_fit, dtype=float)
    a = np.asarray(n_teach, dtype=float)
    a /= np.linalg.norm(a)
    b = np.asarray(n_find, dtype=float)
    b /= np.linalg.norm(b)
    c = np.asarray(centroid_teach, dtype=float)
    axis = np.cross(a, b)
    s_ = float(np.linalg.norm(axis))
    ang = float(np.arctan2(s_, float(np.dot(a, b))))
    r_align = np.eye(3) if s_ < 1e-9 else Rotation.from_rotvec(axis / s_ * ang).as_matrix()
    seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e = seed - np.dot(seed, a) * a
    e /= np.linalg.norm(e)
    v_fit = d[:3, :3] @ e
    v_fit -= np.dot(v_fit, b) * b
    v_al = r_align @ e
    v_al -= np.dot(v_al, b) * b
    yaw = float(np.arctan2(np.dot(np.cross(v_al, v_fit), b), np.dot(v_al, v_fit)))
    r = Rotation.from_rotvec(b * yaw).as_matrix() @ r_align
    c_new = d[:3, :3] @ c + d[:3, 3]
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = c_new - r @ c
    rv = Rotation.from_matrix(d[:3, :3]).as_rotvec()
    fit_ang = float(np.linalg.norm(rv))
    fit_tilt = (
        0.0 if fit_ang < 1e-6 else float(np.degrees(np.arccos(np.clip(abs(np.dot(rv / fit_ang, b)), -1, 1))))
    )
    return {
        "delta": out,
        "yaw_deg": float(np.degrees(yaw)),
        "face_tilt_deg": float(np.degrees(ang)),
        "fit_axis_tilt_deg": fit_tilt,
    }


# ── the act: straight lines to the pre-grasp points, then the grasp replayed 1:1 with the object ──
KEYPOINT_KINDS = ("pregrasp", "grasp_end", "preplace", "place_end", "pose", "waypoint")
GRASP_KINDS = ("pregrasp", "grasp_end")  # follow the object picked, which is then held
PLACE_KINDS = ("preplace", "place_end")  # follow the object it is placed onto
# "pose" is no motion: it sets the frame an object's demo pose is read from, where the operator sees it clearly.
# A "waypoint" is reached in a straight line, as a pre-grasp or a pre-place is, at the pose its "frame" holds it in:
# "world", the demo's own fingertip pose in the arm's frame, moved with no object. The demo's arm stood there, so the
# servos reach it wherever the objects lie. For now world waypoints come after the place end: the lift away after the
# release, which the place itself replays moved with the object placed onto.
WORLD_FRAME = "world"
# A key frame held where its leg begins: the demo's motion since the key frame before it, applied from wherever the arm
# began the leg (C0 . D(t0)^-1 . D(t)). A lift that turns as the demo turned, from wherever the grasp happened. Only the
# act engine replays it.
START_FRAME = "start"
ACT_REACH_TOL_M = (
    0.003  # a planned pose solved to within this is reached: under the hand-eye calibration's own error
)
ACT_REACH_TOL_DEG = 3.0
ACT_SOLVE_TOL_M = 0.0005  # the solve keeps going to this; the reach tolerance above only judges the result
ACT_SOLVE_TOL_DEG = 0.5
ACT_IK_CALLS = 40  # the IK moves a bounded step per call; the first solve from far away needs several
# A tracker view moves an object's pose only when it places the object: enough of it seen, and the points seen
# pinning the pose (src/lerobot/showservo/docs/act_loop.md). The second half: the rigid fit's error at the object's
# middle for this much noise on each point in 3D, within the reach tolerance. With the wrist over all but a corner of
# the gamepad, 16 fit points bunched there put it 30 mm and 22 degrees off while every one of them fitted; over five
# recorded acts, 0.78 or less per mm of this noise held for every view within 4.9 mm and 3.4 degrees, 1.20 or more for
# all but two views 6.3-29.8 mm off (docs/proofs/act-loop). The noise is estimated from fully seen views of a still
# gamepad, whose targets scattered about 1 mm at 0.35 per mm; measured properly, it moves the line.
POINT_NOISE_M = 0.003


def view_places(
    share_seen: float,
    points: np.ndarray | None,
    middle: np.ndarray | None,
    need_share: float,
    noise_m: float = POINT_NOISE_M,
    tol_m: float = ACT_REACH_TOL_M,
) -> tuple[bool, str]:
    """Does a tracker view place its object, so that its pose may replace the one held? ``(ok, reason)``: the act's
    one rule (:func:`lerobot.showservo.placement.view_places`) with the act's own noise and reach tolerance."""
    return placement.view_places(share_seen, points, middle, need_share, noise_m, tol_m)


placement_error = placement.placement_error


def carried_motion(
    frames: list[tuple[float, np.ndarray]],
    since: float,
    symmetry_line: tuple[np.ndarray, np.ndarray] | None = None,
    end_swap: bool = False,
) -> np.ndarray:
    """How the point groups moved an object from frame time ``since`` to their newest frame, camera frame (4x4).

    ``frames`` are their pose of the object frame by frame, ``(frame time, camera <- object)``, oldest first: its own
    points' where they place it, otherwise where its group carries it. The motion is the newest pose times the
    inverse of the pose at ``since``, interpolated between the frames around it. A view that placed the object at
    ``since``, with motion V from the demo's view, puts it at ``carried_motion(frames, since) @ V`` now: it moved with
    what it rests on while no view placed it (src/lerobot/showservo/docs/act_loop.md). Before their first frame, the
    first stands; without frames, no motion.
    """
    if not frames:
        return np.eye(4)
    start, end = _groups_pose_at(frames, since), np.asarray(frames[-1][1], dtype=float)
    if symmetry_line is None:
        return end @ np.linalg.inv(start)
    # The group fit may spin an appearance-symmetric object around its axis
    # while its visible surface and centreline stay still. Keep the measured
    # centreline motion and the shortest rotation that aligns its direction;
    # the unobserved axial roll cannot retarget a grasp.
    from scipy.spatial.transform import Rotation

    reference = np.asarray(frames[0][1], dtype=float)
    axis, point = (np.array(v, dtype=float, copy=True) for v in symmetry_line)
    axis /= np.linalg.norm(axis)
    local_axis = reference[:3, :3].T @ axis
    local_point = reference[:3, :3].T @ (point - reference[:3, 3])
    a = start[:3, :3] @ local_axis
    b = end[:3, :3] @ local_axis
    if end_swap and np.dot(a, b) < 0:
        b = -b
    cross, dot = np.cross(a, b), float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-9:
        if dot > 0:
            rotate = np.eye(3)
        else:
            # A reversal without end-swap permission is observable. Its
            # shortest turn is not unique; choose a stable orthogonal axis.
            basis = np.eye(3)[int(np.argmin(np.abs(a)))]
            normal = np.cross(a, basis)
            rotate = Rotation.from_rotvec(np.pi * normal / np.linalg.norm(normal)).as_matrix()
    else:
        rotate = Rotation.from_rotvec(
            cross / np.linalg.norm(cross) * np.arctan2(np.linalg.norm(cross), dot)
        ).as_matrix()
    p0 = start[:3, :3] @ local_point + start[:3, 3]
    p1 = end[:3, :3] @ local_point + end[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rotate
    out[:3, 3] = p1 - rotate @ p0
    return out


def carried_between(frames: list[tuple[float, np.ndarray]], since: float, until: float) -> np.ndarray:
    """How the point groups moved an object from the frame read at ``since`` to the one read at ``until``, camera frame:
    their pose at ``until`` times the inverse of their pose at ``since``, each as in :func:`carried_motion`."""
    if not frames:
        return np.eye(4)
    return _groups_pose_at(frames, until) @ np.linalg.inv(_groups_pose_at(frames, since))


def _groups_pose_at(frames: list[tuple[float, np.ndarray]], t: float) -> np.ndarray:
    """The point groups' pose of an object at frame time ``t``: interpolated between the frames around it, the first or
    the last outside them. Pre: ``frames`` is not empty, oldest first."""
    stamps = np.array([s for s, _ in frames], dtype=float)
    k = int(np.searchsorted(stamps, t))
    if k == 0:
        return np.asarray(frames[0][1], dtype=float)
    if k == len(frames):
        return np.asarray(frames[-1][1], dtype=float)
    (t0, a), (t1, b) = frames[k - 1], frames[k]
    return interp_rigid(a, b, (t - t0) / max(t1 - t0, 1e-9))


# The depth check: a tracked point counts as seen only while the depth under it puts it within this of where the object's
# motion takes it from where it was when a view last placed the object. Beside a nearer object the depth camera reads
# that object, or nothing, under some of the points the colour tracker still sees (src/lerobot/showservo/docs/act_loop.md,
# O13); a real move moves them all alike.
DEPTH_AGREE_M = 0.02
DEPTH_FIT_MIN = (
    6  # tracks with a place and a reading needed to fit the object's motion; fewer, it is taken as carried
)
DEPTH_FIT_TRIES = 64  # three-track samples tried for the motion most tracks agree on
# A track with no reading under it counts as unseen only with a reading this close (pixels) more than the tolerance
# nearer than where it is expected: something in front of it. Otherwise the depth camera just has no reading there (a
# face at a grazing angle, a dark patch) and the track is left out of the count.
DEPTH_NEAR_PX = 8


def depth_seen(
    idx: np.ndarray,
    uv: np.ndarray,
    vis: np.ndarray,
    depth_m: np.ndarray,
    intr: dict[str, float],
    expected: dict[int, np.ndarray],
    carry: np.ndarray,
    tol_m: float,
) -> tuple[int, int, dict[int, np.ndarray], np.ndarray]:
    """How many of an object's tracked points are seen with a believable depth, out of how many, and where those are.

    ``idx`` (N,) are the tracks' numbers, ``uv`` (N, 2) their pixels and ``vis`` (N,) whether the colour tracker sees
    them; ``depth_m`` is the frame's depth in metres and ``intr`` its pinhole (fx, fy, cx, cy). ``expected`` holds where
    each track was when a view last placed the object (camera frame, by number) and ``carry`` how the object moved
    since as far as is known (4x4, camera frame; the identity when nothing says). The object's motion is the rigid
    motion most of the tracks agree on, within ``tol_m``, from where ``carry`` takes their expected places to where
    their readings put them now (RANSAC over three-track samples, then refitted on those that agree), and ``carry``
    itself when fewer than DEPTH_FIT_MIN tracks have both. A track counts when it is seen, has a reading under it, and
    that reading puts it within ``tol_m`` of where the motion takes its expected place; a track with no expected place
    counts when seen with a reading. A track seen with no reading under it is left out of the count unless a reading
    within DEPTH_NEAR_PX pixels lies more than ``tol_m`` nearer than its expected place: something in front of it. Post:
    ``(count, judged, now, move)``: ``judged`` the tracks the count is out of (all, less those left out), ``now`` the
    counted tracks' positions in this frame by number, ``move`` the object's motion since the expected places (4x4,
    ``carry`` included)."""
    from lerobot.showservo.pose import fit_rigid

    idx = np.asarray(idx).astype(int).reshape(-1)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    vis = np.asarray(vis).astype(bool).reshape(-1)
    assert len(idx) == len(uv) == len(vis), "a number, a pixel and a visibility per track"
    h, w = depth_m.shape[:2]
    z = np.asarray(depth_m, dtype=float)[
        np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1), np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    ]
    pts = np.stack(
        [(uv[:, 0] - intr["cx"]) * z / intr["fx"], (uv[:, 1] - intr["cy"]) * z / intr["fy"], z], axis=1
    )
    carry = np.asarray(carry, dtype=float)
    read = [q for q in range(len(idx)) if vis[q] and np.isfinite(z[q]) and z[q] > 0.0]
    known = [q for q in read if int(idx[q]) in expected]
    move = carry
    if len(known) >= DEPTH_FIT_MIN:
        src = np.array([carry[:3, :3] @ expected[int(idx[q])] + carry[:3, 3] for q in known])
        dst = pts[known]
        rng = np.random.default_rng(0)  # the same frame judged the same way
        best = None
        for _ in range(DEPTH_FIT_TRIES):
            pick = rng.choice(len(known), 3, replace=False)
            try:
                fit, _scale = fit_rigid(src[pick], dst[pick])
            except AssertionError:  # three tracks nearly in a line fix no turn
                continue
            agree = np.linalg.norm(src @ fit.rot.T + fit.trans - dst, axis=1) <= tol_m
            if best is None or agree.sum() > best.sum():
                best = agree
        if best is not None and best.sum() >= 3:
            try:
                fit, _scale = fit_rigid(src[best], dst[best])
                step = np.eye(4)
                step[:3, :3], step[:3, 3] = fit.rot, fit.trans
                move = step @ carry
            except AssertionError:
                pass
    now: dict[int, np.ndarray] = {}
    for q in read:
        e = expected.get(int(idx[q]))
        if e is None or float(np.linalg.norm(pts[q] - (move[:3, :3] @ e + move[:3, 3]))) <= tol_m:
            now[int(idx[q])] = pts[q]
    left_out = 0
    dm = np.asarray(depth_m, dtype=float)
    for q in range(len(idx)):
        if not vis[q] or (np.isfinite(z[q]) and z[q] > 0.0):
            continue
        e = expected.get(int(idx[q]))
        if e is None:
            left_out += 1
            continue
        u, v, r = int(round(uv[q, 0])), int(round(uv[q, 1])), DEPTH_NEAR_PX
        win = dm[max(0, v - r) : v + r + 1, max(0, u - r) : u + r + 1]
        if not ((win > 0.0) & (win < (move[:3, :3] @ e + move[:3, 3])[2] - tol_m)).any():
            left_out += 1
    return len(now), len(idx) - left_out, now, move


def depth_expect(
    expected: dict[int, np.ndarray], now: dict[int, np.ndarray], carry: np.ndarray
) -> dict[int, np.ndarray]:
    """Where an object's tracks are expected after a view placed it: the tracks it counted (``now``, from
    :func:`depth_seen`) where it saw them, the others where ``carry`` (its ``move``) takes their expected places."""
    carry = np.asarray(carry, dtype=float)
    out = {k: carry[:3, :3] @ e + carry[:3, 3] for k, e in expected.items()}
    out.update(now)
    return out


def keypoints_problem(keypoints: list[dict[str, Any]], t_start: float, t_end: float) -> str:
    """Why these marks cannot be saved, or '' when they can.

    An empty list clears the marks. Otherwise: at least one pre-grasp, at most one
    grasp end and it comes after the last pre-grasp, every time inside the demo. A place
    comes after the grasp end: at least one pre-place, at most one place end and it comes
    after the last pre-place. The pre-grasps and the grasp end follow one object, the one
    picked; the pre-places and the place end follow another, the one it goes onto, and a
    place names both. A pose mark names its object, once, and for an object the marks
    follow comes no later than that object's last pre-grasp or pre-place, before the arm
    can have moved it; pose marks alone are a list of their own. A waypoint is held in the
    world frame and comes after the place end.

    Marks of kind KEY_KIND are key frames as they are, held to :func:`key_frames_problem` instead; a list does not
    mix them with the kinds above, and an object's pose mark comes no later than the arm first follows the demo in its
    frame.
    """
    if not keypoints:
        return ""
    if any(k.get("kind") == KEY_KIND for k in keypoints):
        return _key_marks_problem(keypoints, t_start, t_end)
    for k in keypoints:
        if k.get("kind") not in KEYPOINT_KINDS:
            return f"a mark is one of {KEYPOINT_KINDS}"
        tk = k.get("t")
        if not isinstance(tk, (int, float)) or not (t_start <= float(tk) <= t_end):
            return f"a mark's time must lie within the demo ({t_start:.1f} to {t_end:.1f} s)"
        if k["kind"] == "waypoint" and k.get("frame") != WORLD_FRAME:
            return f"a waypoint is held in the {WORLD_FRAME} frame"
    waypoints = [float(k["t"]) for k in keypoints if k["kind"] == "waypoint"]
    if len(set(waypoints)) < len(waypoints):
        return "there is a waypoint at that moment already"
    poses = [k for k in keypoints if k["kind"] == "pose"]
    if any(not k.get("object") for k in poses):
        return "a pose mark names its object"
    if len({k["object"] for k in poses}) < len(poses):
        return "an object's pose is read at one frame"
    if len(poses) == len(keypoints):
        return ""
    pre = [float(k["t"]) for k in keypoints if k["kind"] == "pregrasp"]
    ends = [float(k["t"]) for k in keypoints if k["kind"] == "grasp_end"]
    if not pre:
        return "mark at least one pre-grasp"
    if len(ends) > 1:
        return "the grasp has one end"
    if ends and ends[0] <= max(pre):
        return "the grasp ends after the last pre-grasp"
    pre_place = [float(k["t"]) for k in keypoints if k["kind"] == "preplace"]
    place_ends = [float(k["t"]) for k in keypoints if k["kind"] == "place_end"]
    if pre_place or place_ends:
        if not ends:
            return "set the grasp end first: the place comes after the grasp"
        if min(pre_place + place_ends) <= ends[0]:
            return "the place comes after the grasp end"
        if not pre_place:
            return "mark at least one pre-place"
        if len(place_ends) > 1:
            return "the place has one end"
        if place_ends and place_ends[0] <= max(pre_place):
            return "the place ends after the last pre-place"
    if waypoints and not place_ends:
        return "set the place end first: a waypoint comes after the place"
    if waypoints and min(waypoints) <= place_ends[0]:
        return "a waypoint comes after the place end"
    picked = {k.get("object") or "" for k in keypoints if k["kind"] in GRASP_KINDS}
    onto = {k.get("object") or "" for k in keypoints if k["kind"] in PLACE_KINDS}
    if len(picked) > 1:
        return "the pre-grasps and the grasp end follow one object"
    if len(onto) > 1:
        return "the pre-places and the place end follow one object"
    if onto:
        held, target = next(iter(picked)), next(iter(onto))
        if not held or not target:
            return (
                "a place needs both objects clicked on the recording: the one picked and the one it goes onto"
            )
        if held == target:
            return "the place goes onto another object than the one picked"
    picked_obj = next(iter(picked), "")
    onto_obj = next(iter(onto), "")
    for k in poses:
        if k["object"] == picked_obj and float(k["t"]) > max(pre):
            return f"{k['object']}'s pose is read no later than its last pre-grasp, before the arm can have moved it"
        if k["object"] == onto_obj and float(k["t"]) > max(pre_place):
            return f"{k['object']}'s pose is read no later than its last pre-place, before the arm can have moved it"
    return ""


# Closing on something stops the gripper short of its command; closing on nothing reaches it. Over the 59 recorded acts
# that closed it, every closing on nothing stopped -0.06 to 0.48 units short (9 marked missed, 3 whose frames show the
# cube left on the table), and every closing on the object 0.89 to 10.56 short. The line sits halfway between: the
# gamepad of the pick-and-place demo, squeezed lightly, was held 0.89 to 1.09 short.
GRASP_HELD_SHORT = 0.7


# The gripper's reading has stopped when it moves less than this over GRIP_STILL_S.
GRIP_STILL_UNITS = 0.3
GRIP_STILL_S = 0.1


def firm_grip(
    t: np.ndarray, cmd: np.ndarray, obs: np.ndarray, i_from: int, i_to: int, closing: float
) -> int | None:
    """The sample the grip became firm, or None: the first sample in ``[i_from, i_to]`` after the closing command has
    begun (moved more than :data:`GRASP_HELD_SHORT` toward closing from its value at ``i_from``) where the reading has
    stopped, within :data:`GRIP_STILL_UNITS` over :data:`GRIP_STILL_S`, more than :data:`GRASP_HELD_SHORT` short of
    the command, as an object between the fingers stops it. ``closing`` is +1 when closing raises the reading.

    Separate from the grasp's end mark, which says where the replayed motion ends and usually follows a lift: on the
    stacking demo of 2026-10-07 the grip was firm at 8.33 s, the lift began at 8.83 s and the mark sat at 9.16 s.
    """
    t, cmd, obs = (np.asarray(x, dtype=float) for x in (t, cmd, obs))
    for i in range(i_from, i_to + 1):
        if closing * (cmd[i] - cmd[i_from]) <= GRASP_HELD_SHORT:
            continue
        j = min(int(np.searchsorted(t, t[i] + GRIP_STILL_S)), len(t) - 1)
        window = obs[i : j + 1]
        if j > i and np.ptp(window) < GRIP_STILL_UNITS and closing * (cmd[i] - obs[i]) > GRASP_HELD_SHORT:
            return i
    return None


# ── key frames: the act in the demo's own terms ──
# A key frame is a moment of the demo (``t``), the frame the demo's fingertip pose then is held in, and how the arm gets
# there. The frame is an object designated on the recording, so the pose moves with that object's motion since the
# demo; WORLD_FRAME, the arm's own frame, moved with nothing; or "", the object taught before the demo on demos recorded
# that way. "line": a straight line from where the arm stands. "follow": the demo's own path from the key frame before,
# sample for sample, on the demo's clock and with its gripper command; it is held in the frame of the key frame before
# it, where the arm stands as it begins. On a line the gripper goes to the demo's command at the key frame: "first" with
# the arm standing before the line, "along" with the arm. "lift": straight up by ``up_mm`` from where the key frame
# before it left the arm, turned as it was and the gripper as it was, in no object's frame (WORLD_FRAME); its moment
# only orders it. It takes what the arm holds clear of the scene before a line sweeps it across: the line from a pick
# to the pre-place over the target knocked the cube aside on the rig (2026-10-12) when the pick lay near it.
#
# What the arm's own contact decides is the one exception (:func:`demo_holding`): from the firm grip in a stretch the
# arm follows until the command opens past the fingers, the demo held something. A key frame made then is reached
# holding it: the gripper keeps its grip on a line, and every pose is corrected on the gripper's side for how the
# object sits in the gripper now against the demo, so that the held object, not the gripper, goes where the demo put it.
MOVES = ("line", "follow", "lift")
GRIPS = ("first", "along")
CONSTRAINTS = ("pose", "orientation", "best_effort")
KEY_KIND = (
    "key"  # a mark that is a key frame as it is: t, frame, move, grip, constraint (and up_mm on a lift)
)
LIFT_MAX_MM = 150.0  # a lift is a clearance over the scene, not a way to send the arm elsewhere
# The operator's marks as key frames: (move, grip, label). Pre-grasps and pre-places are lines in their object's frame;
# a grasp end or a place end follows the demo in it; a waypoint is a line in the world frame, the gripper moving along.
_MARK_FRAMES = {
    "pregrasp": ("line", "first", "pre-grasp {n}"),
    "grasp_end": ("follow", "first", "grasp"),
    "preplace": ("line", "first", "pre-place {n}"),
    "place_end": ("follow", "first", "place"),
    "waypoint": ("line", "along", "waypoint {n}"),
}


def key_frames(marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The key frames the operator's marks make, in time order: ``t``, ``frame``, ``move``, ``grip`` and ``label``,
    what the act calls it (each kind numbered from 1). A "key" mark is a key frame as it is, labelled "key frame n";
    the other kinds translate one to one (_MARK_FRAMES). A mark naming no object is in the frame of the object taught
    before the demo (""), a waypoint in the one it names. Pose marks set the frame an object's demo pose is read from
    and make none. Pre: ``keypoints_problem(marks, ...) == ''``."""
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for k in sorted(marks, key=lambda k: float(k["t"])):
        if k["kind"] == "pose":
            continue
        if k["kind"] == KEY_KIND and k["move"] == "lift":
            seen["lift"] = seen.get("lift", 0) + 1
            out.append(
                {
                    "t": float(k["t"]),
                    "frame": WORLD_FRAME,
                    "move": "lift",
                    "grip": "first",
                    "up_mm": float(k.get("up_mm") or 0.0),
                    "constraint": k.get("constraint", "best_effort"),
                    "label": f"lift {seen['lift']}",
                }
            )
            continue
        if k["kind"] == KEY_KIND:
            seen[KEY_KIND] = seen.get(KEY_KIND, 0) + 1
            out.append(
                {
                    "t": float(k["t"]),
                    "frame": k.get("frame") or "",
                    "move": k["move"],
                    "grip": k.get("grip", "first"),
                    "constraint": k.get("constraint", "pose"),
                    "label": f"key frame {seen[KEY_KIND]}",
                }
            )
            continue
        move, grip, label = _MARK_FRAMES[k["kind"]]
        seen[k["kind"]] = seen.get(k["kind"], 0) + 1
        frame = k.get("frame") if k["kind"] == "waypoint" else k.get("object")
        out.append(
            {
                "t": float(k["t"]),
                "frame": frame or "",
                "move": move,
                "grip": grip,
                "constraint": k.get("constraint", "pose"),
                "label": label.format(n=seen[k["kind"]]),
            }
        )
    return out


def key_frames_problem(frames: list[dict[str, Any]], t_start: float, t_end: float) -> str:
    """Why the arm cannot act these key frames, or '' when it can. Each has a moment inside the demo, no two the same,
    a move of MOVES and a grip of GRIPS. The first is a line, as a follow replays the demo from the key frame before
    it, and a follow is held in that key frame's frame, where the arm stands as it begins. Pre: in time order."""
    for n, k in enumerate(frames):
        if k.get("move") not in MOVES:
            return f"a key frame's move is one of {MOVES}"
        if k.get("grip") not in GRIPS:
            return f"a key frame's gripper is one of {GRIPS}"
        if k.get("constraint", "pose") not in CONSTRAINTS:
            return f"a key frame's constraint is one of {CONSTRAINTS}"
        if k["move"] == "follow" and k.get("constraint") == "best_effort":
            return "a follow replays the demo path: use a pose or orientation constraint"
        tk = k.get("t")
        if not isinstance(tk, (int, float)) or not (t_start <= float(tk) <= t_end):
            return f"a key frame's time must lie within the demo ({t_start:.1f} to {t_end:.1f} s)"
        if n and float(tk) <= float(frames[n - 1]["t"]):
            return "there is a key frame at that moment already"
        if k["move"] == "follow" and n == 0:
            return "the first key frame is reached in a straight line: a follow replays the demo from the one before"
        if k["move"] == "lift" and n == 0:
            return "a lift rises from where the key frame before it left the arm: add one before it"
        if k["move"] == "lift" and not (0.0 < float(k.get("up_mm") or 0.0) <= LIFT_MAX_MM):
            return f"a lift rises by more than 0 and at most {LIFT_MAX_MM:g} mm"
        if k["move"] == "follow" and frames[n - 1]["move"] == "lift":
            return (
                "a follow replays the demo from the key frame before it, and a lift leaves the arm off the demo's "
                "path: put a straight line between them"
            )
        if k["move"] == "follow" and k.get("frame") not in (START_FRAME, frames[n - 1].get("frame")):
            return (
                "a follow is held in the frame of the key frame before it, where the arm stands as it begins"
            )
    return ""


def _key_marks_problem(marks: list[dict[str, Any]], t_start: float, t_end: float) -> str:
    """:func:`keypoints_problem` for a list of key frames as they are (KEY_KIND) and pose marks."""
    if any(k.get("kind") not in (KEY_KIND, "pose") for k in marks):
        return "the key frames are marked one way: as key frames, or as pre-grasps, grasp ends and places"
    for k in marks:
        tk = k.get("t")
        if not isinstance(tk, (int, float)) or not (t_start <= float(tk) <= t_end):
            return f"a mark's time must lie within the demo ({t_start:.1f} to {t_end:.1f} s)"
        if k["kind"] == KEY_KIND and not isinstance(k.get("frame"), str):
            return "a key frame names its frame: an object clicked on the recording, or the world"
        if k["kind"] == KEY_KIND and k.get("move") not in MOVES:
            return f"a key frame's move is one of {MOVES}"
        if k["kind"] == KEY_KIND and k.get("constraint", "pose") not in CONSTRAINTS:
            return f"a key frame's constraint is one of {CONSTRAINTS}"
    poses = [k for k in marks if k["kind"] == "pose"]
    if any(not k.get("object") for k in poses):
        return "a pose mark names its object"
    if len({k["object"] for k in poses}) < len(poses):
        return "an object's pose is read at one frame"
    frames = key_frames(marks)
    problem = key_frames_problem(frames, t_start, t_end)
    if problem:
        return problem
    for k in poses:
        follows = [f["t"] for f in frames if f["frame"] == k["object"] and f["move"] == "follow"]
        if follows:
            begins = max(f["t"] for f in frames if f["t"] < min(follows))
            if float(k["t"]) > begins:
                return (
                    f"{k['object']}'s pose is read no later than the arm first follows the demo in its frame, "
                    "before the arm can have moved it"
                )
    return ""


def demo_holdings(
    t: np.ndarray, cmd: np.ndarray, obs: np.ndarray, frames: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every recorded grasp/release interval, independent of object names or task.

    A firm grip starts an attachment to the object named by that recorded segment.
    It ends when the opening command passes the observed fingers, not when an
    interaction segment ends. A later recorded closing can attach another object
    (or regrasp the same one). Grasp and release transitions need their own follow
    segments so the segment's reference identifies the object being grasped.
    """
    t, cmd, obs = (np.asarray(x, dtype=float) for x in (t, cmd, obs))
    out: list[dict[str, Any]] = []
    available = 0

    def index(tk: float) -> int:
        return int(np.argmin(np.abs(t - tk)))

    for n, k in enumerate(frames):
        if k["move"] != "follow":
            continue
        i0, i1 = max(available, index(frames[n - 1]["t"])), index(k["t"])
        if i0 >= i1:
            continue
        closing = float(np.sign(cmd[i1] - cmd[i0])) or 1.0
        i = firm_grip(t, cmd, obs, i0, i1, closing)
        if i is None:
            continue
        j = i
        while j + 1 < len(t) and closing * (cmd[j + 1] - obs[j + 1]) > 0.0:
            j += 1
        out.append(
            {
                "grip": i,
                "release": j + 1 if j + 1 < len(t) else None,
                "closing": closing,
                "frame": k["frame"],
                "begins": n,
            }
        )
        available = j + 1
    return out


def holding_at(
    holdings: list[dict[str, Any]], t: np.ndarray, frames: list[dict[str, Any]], n: int
) -> dict[str, Any] | None:
    """The attachment active at a key frame, including later grasps."""
    return next((h for h in holdings if held_at(h, t, frames, n)), None)


def interaction_phases(
    holdings: list[dict[str, Any]], t: np.ndarray, frames: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Contiguous key-frame ranges, each containing at most one grasp interval.

    A boundary follows the actual release key frame; intermediate tool-use
    segments keep their attachment. The final range includes the empty retreat.
    """
    start, out = 0, []
    for h, following in zip(holdings[:-1], holdings[1:], strict=True):
        release_t = float(t[h["release"]])
        stop = next(n + 1 for n, k in enumerate(frames) if k["t"] >= release_t)
        if stop > following["begins"]:
            raise ValueError("mark the release and the next grasp in separate follow segments")
        out.append((start, stop))
        start = stop
    out.append((start, len(frames)))
    return out


def held_at(holding: dict[str, Any] | None, t: np.ndarray, frames: list[dict[str, Any]], n: int) -> bool:
    """Was the demo holding something as key frame ``n`` begins: at its own moment for a line, at the key frame
    before for a follow, between the grip and the release (:func:`demo_holding`)."""
    if holding is None:
        return False
    tk = frames[n - 1]["t"] if frames[n]["move"] == "follow" else frames[n]["t"]
    i = int(np.argmin(np.abs(np.asarray(t, dtype=float) - tk)))
    return holding["grip"] <= i and (holding["release"] is None or i < holding["release"])


HOLD_VIEWS = 5
HOLD_MIN_VIEWS = 3
HOLD_AGREE_M = 0.005
HOLD_AGREE_DEG = 5.0


def average_hold(holds: list[np.ndarray], centre: np.ndarray) -> dict[str, Any] | None:
    """One hold from a strict majority of views, or None without at least :data:`HOLD_MIN_VIEWS` in that majority.

    Each hold is ``tip^-1 . motion``: the object's motion from its view in the demo, as
    seen from the gripper. ``centre`` (base frame) is the object's centre in that view.
    Views are compared there, at the object, because at the motion's origin a small turn
    reads as a large shift. A view whose centre lies more than :data:`HOLD_AGREE_M` from
    the median is left out. Find the largest rotational consensus within :data:`HOLD_AGREE_DEG` of a candidate
    orientation, then average and recheck its retained views. Candidates are the observations and their mean:
    the latter covers symmetric noise, but large outliers cannot pull every candidate away from good views.
    A strict majority prevents choosing arbitrarily between competing small clusters. Average position there
    too: the mean hold maps ``centre`` to the mean
    of the retained measured centres, independent of the source coordinate origin.
    Post: ``hold`` (4x4), ``n`` views used of ``views``, and their ``spread_mm`` and
    ``spread_deg`` (medians).
    """
    from scipy.spatial.transform import Rotation

    if not holds:
        return None
    hs = np.stack([np.asarray(h, dtype=float) for h in holds])
    c = np.append(np.asarray(centre, dtype=float)[:3], 1.0)
    centres = (hs @ c)[:, :3]
    rot = Rotation.from_matrix(hs[:, :3, :3])
    off_m = np.linalg.norm(centres - np.median(centres, axis=0), axis=1)
    candidates = []
    for seed in [rot.mean(), *rot]:
        off_deg = np.degrees((rot * seed.inv()).magnitude())
        kept = np.flatnonzero((off_m <= HOLD_AGREE_M) & (off_deg <= HOLD_AGREE_DEG))
        candidates.append((kept, float(off_deg[kept].sum())))
    keep, _ = max(candidates, key=lambda item: (len(item[0]), -item[1]))
    while True:
        if len(keep) < HOLD_MIN_VIEWS or 2 * len(keep) <= len(hs):
            return None
        mean = rot[keep].mean()
        agreeing = np.degrees((rot[keep] * mean.inv()).magnitude()) <= HOLD_AGREE_DEG
        if agreeing.all():
            break
        keep = keep[agreeing]  # shrinking only: finite work, no oscillation between competing fits
    kc = centres[keep]
    out = np.eye(4)
    out[:3, :3] = mean.as_matrix()
    out[:3, 3] = kc.mean(axis=0) - out[:3, :3] @ c[:3]
    return {
        "hold": out,
        "n": len(keep),
        "views": len(hs),
        "spread_mm": float(np.median(np.linalg.norm(kc - np.median(kc, axis=0), axis=1)) * 1000.0),
        "spread_deg": float(np.median(np.degrees((rot[keep] * mean.inv()).magnitude()))),
    }


def interp_rigid(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """The rigid transform ``s`` of the way from ``a`` to ``b``: rotation slerped, translation lerped."""
    from scipy.spatial.transform import Rotation, Slerp

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    s = float(np.clip(s, 0.0, 1.0))
    out = np.eye(4)
    out[:3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))(s).as_matrix()
    out[:3, 3] = (1.0 - s) * a[:3, 3] + s * b[:3, 3]
    return out


def pose_residual(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``(metres, degrees)`` between two poses: translation distance and the rotation angle."""
    from scipy.spatial.transform import Rotation

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ang = np.linalg.norm(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec())
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.degrees(ang))


# An act's step keeps within the walk's own range of speeds (jog.LINEAR_M_S_RANGE, jog.ANGULAR_RAD_S_RANGE): the
# demo's pace over a few jittery samples, or that times the act's speed, cannot ask for a runaway speed, nor a demo
# standing still for none.
LINE_SPEED_M_S = (0.005, 0.30)
LINE_TURN_RAD_S = (np.radians(5.0), np.radians(180.0))


def _walk_seconds(tips: np.ndarray, i0: int, i1: int, lin_m_s: float, ang_rad_s: float) -> np.ndarray:
    """Each of the demo's moves from sample ``i0`` to ``i1`` (``i1 - i0`` of them), as the time the walk takes over
    it: its translation at ``lin_m_s`` or its turn at ``ang_rad_s``, whichever is longer."""
    path = np.asarray(tips[i0 : i1 + 1], dtype=float)
    moved = np.linalg.norm(np.diff(path[:, :3, 3], axis=0), axis=1)
    turn = np.einsum("nji,njk->nik", path[:-1, :3, :3], path[1:, :3, :3])
    turned = np.arccos(np.clip((np.trace(turn, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
    return np.maximum(moved / lin_m_s, turned / ang_rad_s)


def demo_pace(
    t: np.ndarray, tips: np.ndarray, since: float | None, until: float, lin_m_s: float, ang_rad_s: float
) -> float:
    """How many times the walk's speed (``lin_m_s``, ``ang_rad_s``) the demo's fingertip moved at from ``since`` to
    ``until``: the time the walk takes over the demo's own path there, every curve and turn of it (each sample's move
    at the walk's rates), over the time the demo took. 1.0, the walk's own, where there is nothing to measure: no
    key frame before, or no two samples between. Finite; a step bounds what it makes of it (:func:`walk_scale`)."""
    assert lin_m_s > 0.0 and ang_rad_s > 0.0, "a positive walk speed"
    if since is None:
        return 1.0
    i0, i1 = int(np.argmin(np.abs(t - since))), int(np.argmin(np.abs(t - until)))
    took = float(t[i1] - t[i0])
    if i1 <= i0 or took <= 0.0:
        return 1.0
    return float(_walk_seconds(tips, i0, i1, lin_m_s, ang_rad_s).sum()) / took


def walk_scale(pace: float, speed: float, lin_m_s: float, ang_rad_s: float) -> float:
    """The multiple of the walk's speed a step goes at: the demo's ``pace`` there times the act's ``speed``, kept so
    that its speeds stay within :data:`LINE_SPEED_M_S` and :data:`LINE_TURN_RAD_S`. Pre: the walk's speeds within
    those ranges, so the bounds are in order."""
    lo = max(LINE_SPEED_M_S[0] / lin_m_s, LINE_TURN_RAD_S[0] / ang_rad_s)
    hi = min(LINE_SPEED_M_S[1] / lin_m_s, LINE_TURN_RAD_S[1] / ang_rad_s)
    return min(max(pace * speed, lo), hi)


def plan_key_frames(
    steps: list[dict[str, Any]],
    t: np.ndarray,
    tips: np.ndarray,
    grip_cmd: np.ndarray,
    q_demo: np.ndarray,
    start_pose: np.ndarray,
    start_grip: float,
    lin_m_s: float,
    ang_rad_s: float,
    grip_units_s: float,
    speed: float,
    hz: float,
) -> dict[str, Any]:
    """The act as timed samples through ``steps``: key frames (:func:`key_frames`) in order, each resolved for where
    things are now with ``motion``, its frame's motion since the demo (base frame), ``fix``, the hold's correction on
    the gripper's side (the identity when nothing is held), ``held``, whether the demo was holding something as it
    begins (:func:`held_at`), and for a follow ``since``, the moment of the key frame before it.

    Every pose is ``motion . adjust . tip . fix`` (``adjust`` defaults to identity), the demo's fingertip pose then moved, but a lift's: where the step before
    left the arm, ``up_mm`` higher along the base's z, the gripper kept. A line runs from where the arm stands
    at the demo's own pace since the key frame before (:func:`demo_pace`; the walk's speed for the first key frame);
    along it the gripper keeps its opening while held, else goes to the demo's command at the
    key frame, "first" with the arm standing, "along" with the arm. A follow replays the demo's samples after
    ``since`` up to the key frame on the demo's clock with its gripper command, its joint changes as hints and its
    fingertip heights as the floor, each step free to move as fast as the demo did there, and at least at the walk's
    speed, which leaves live corrections room where the demo stood nearly still. Both are scaled by ``speed``: 1.0 is
    the demo's own speed throughout. The first sample is ``start_pose``, where the arm is.

    Pre: speed > 0. Post: ``times`` (N,) from 0, ``poses`` (N, 4, 4) base frame, ``grips`` (N,), ``hints`` (N, J)
    (zero on lines), ``floor_ref`` (N,) (+inf on lines), ``stage`` (N,) the steps' labels, ``walk_scale`` (N,) the
    multiple of the walk's speed a step to each sample may move at (:func:`walk_scale`; ``speed`` where the arm
    stands), ``arrive`` the sample each line ends on and ``spans`` each follow's (first, last) samples, None for one
    with none, both in step order.
    """
    assert speed > 0.0, "a positive speed"
    t = np.asarray(t, dtype=float)
    dt = 1.0 / hz
    nj = np.asarray(q_demo).shape[1]
    times, poses, grips, hints, floor_ref, stage, constraints, scales = (
        [0.0],
        [np.asarray(start_pose, float)],
        [float(start_grip)],
        [np.zeros(nj)],
        [np.inf],
        ["start"],
        ["pose"],
        [float(speed)],
    )
    arrive: list[int] = []
    spans: list[tuple[int, int] | None] = []

    def add(
        dt_s: float,
        pose: np.ndarray,
        grip: float,
        hint: np.ndarray,
        ref: float,
        label: str,
        constraint: str,
        scale: float = speed,
    ) -> None:
        times.append(times[-1] + dt_s)
        poses.append(pose)
        grips.append(float(grip))
        hints.append(hint)
        floor_ref.append(ref)
        stage.append(label)
        constraints.append(constraint)
        scales.append(scale)

    def index(tk: float) -> int:
        return int(np.argmin(np.abs(t - tk)))

    pose, grip = poses[0], grips[0]
    for s in steps:
        motion, fix, label = np.asarray(s["motion"], float), np.asarray(s["fix"], float), s["label"]
        adjusted = motion @ np.asarray(s.get("adjust", np.eye(4)), float)
        constraint = s.get("constraint", "pose")
        if s["move"] == "follow":
            first = len(times)
            i0, i1 = index(s["since"]), index(s["t"])
            walk_s = _walk_seconds(tips, i0, i1, lin_m_s, ang_rad_s) if i1 > i0 else np.zeros(0)
            for i in range(i0 + 1, i1 + 1):
                target = adjusted @ np.asarray(tips[i], float) @ fix
                if constraint == "orientation":
                    target[:3, 3] = pose[:3, 3]
                dt_demo = float(t[i] - t[i - 1])
                pace = walk_s[i - i0 - 1] / dt_demo if dt_demo > 0.0 else 1.0
                add(
                    dt_demo / speed,
                    target,
                    float(grip_cmd[i]),
                    np.asarray(q_demo[i], float) - np.asarray(q_demo[i - 1], float),
                    float(tips[i][2, 3]),
                    label,
                    constraint,
                    walk_scale(max(pace, 1.0), speed, lin_m_s, ang_rad_s),
                )
            spans.append((first, len(times) - 1) if len(times) > first else None)
            pose, grip = poses[-1], grips[-1]
            continue
        i = index(s["t"])
        if s["move"] == "lift":
            target = np.array(pose, dtype=float)
            target[2, 3] += float(s["up_mm"]) / 1000.0
            g = grip
        else:
            target = adjusted @ np.asarray(s.get("target_local", tips[i]), float) @ fix
            if constraint == "orientation":
                target[:3, 3] = pose[:3, 3]
            g = grip if s["held"] else float(grip_cmd[i])
        if s["grip"] == "first":
            n = int(np.ceil(abs(g - grip) / (grip_units_s * speed) * hz))
            for k in range(1, n + 1):
                add(dt, pose, grip + (g - grip) * k / n, np.zeros(nj), np.inf, label, constraint)
            grip = g
        dist_m, ang_deg = pose_residual(pose, target)
        scale = walk_scale(
            demo_pace(t, tips, s.get("since"), s["t"], lin_m_s, ang_rad_s), speed, lin_m_s, ang_rad_s
        )
        dur = max(dist_m / (lin_m_s * scale), np.radians(ang_deg) / (ang_rad_s * scale))
        n = max(1, int(np.ceil(dur * hz)))
        for k in range(1, n + 1):
            add(
                dt,
                interp_rigid(pose, target, k / n),
                grip + (g - grip) * k / n,
                np.zeros(nj),
                np.inf,
                label,
                constraint,
                scale,
            )
        arrive.append(len(times) - 1)
        pose, grip = target, g
    return {
        "times": np.array(times),
        "poses": np.stack(poses),
        "grips": np.array(grips),
        "hints": np.stack(hints),
        "floor_ref": np.array(floor_ref),
        "stage": stage,
        "constraint": constraints,
        "walk_scale": np.array(scales),
        "arrive": arrive,
        "spans": spans,
    }


def solve_plan_joints(
    kin: Any,
    poses: np.ndarray,
    grips: np.ndarray,
    hints: np.ndarray,
    q_start: np.ndarray,
    grip_index: int,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Joints for every planned pose, each solve seeded with the previous answer plus the sample's hint.

    The first seed is the arm's present configuration, so the plan continues from
    where the arm is and keeps its configuration; a hint (the demo's own joint change
    during the grasp) starts each solve where the demo went. ``kin`` maps motor-space
    joints: ``forward_kinematics(q)``, ``inverse_kinematics(seed, pose)``. With servo
    ranges ``lo``/``hi`` (J,; NaN is no limit) no arm joint is solved past its range:
    a solve holds it there and the other joints make up what they can, and the
    residuals say what that costs. Nothing is rejected here. Post: ``q`` (N, J) with
    the planned gripper in its column, ``residual_m``/``residual_deg`` (N,) of the
    solved pose against the planned one, ``step_deg`` (N,) the largest arm-joint change
    from the previous sample (0 first), ``held`` (N,) the joint a solve held at its
    range (-1: none).
    """
    poses = np.asarray(poses, dtype=float)
    n = len(poses)
    q = np.empty((n, len(q_start)))
    res_m, res_deg = np.empty(n), np.empty(n)
    held = np.full(n, -1)
    prev = np.asarray(q_start, dtype=float)
    arm = [k for k in range(len(q_start)) if k != grip_index]
    lo_arm, hi_arm = np.full(len(q_start), -np.inf), np.full(len(q_start), np.inf)
    if lo is not None and hi is not None:
        lo_arm[arm] = np.nan_to_num(np.asarray(lo, dtype=float)[arm], nan=-np.inf)
        hi_arm[arm] = np.nan_to_num(np.asarray(hi, dtype=float)[arm], nan=np.inf)
    for i in range(n):
        qi = prev + np.asarray(hints[i], dtype=float)
        last = (np.inf, np.inf)
        for _ in range(ACT_IK_CALLS):
            wanted = np.asarray(kin.inverse_kinematics(qi, poses[i]), dtype=float)
            qi = np.clip(wanted, lo_arm, hi_arm)
            e_m, e_deg = pose_residual(kin.forward_kinematics(qi), poses[i])
            if e_m <= ACT_SOLVE_TOL_M and e_deg <= ACT_SOLVE_TOL_DEG:
                break
            if e_m > 0.99 * last[0] and e_deg > 0.99 * last[1]:
                break  # no longer improving: out of reach, or as close as this configuration gets
            last = (e_m, e_deg)
        past = np.maximum(lo_arm - wanted, wanted - hi_arm)
        held[i] = int(np.argmax(past)) if past.max() > 0.0 else -1
        qi[grip_index] = float(grips[i])
        q[i], res_m[i], res_deg[i] = qi, e_m, e_deg
        prev = qi
    step = np.zeros(n)
    if n > 1:
        step[1:] = np.abs(np.diff(q[:, arm], axis=0)).max(axis=1)
    return {"q": q, "residual_m": res_m, "residual_deg": res_deg, "step_deg": step, "held": held}


# ── the landing: the turns about the object placed onto that count as the same place ──
# What a place keeps of the demo is the operator's intention, not the object's shape: a key goes into its lock as shown;
# something set on a cube by its middle may land turned any way about that middle; edges laid along a cube's faces may
# land at any of its quarter turns. Of the landings it may make, an act takes one its servos reach whose joints stay
# nearest the demo's own.
LANDINGS = (
    "exact",
    "symmetry",
    "turn",
)  # as shown; any of the target's symmetric turns; any turn about its middle


def turn_about(centre: np.ndarray, deg: float, axis: np.ndarray | None = None) -> np.ndarray:
    """Rigid turn about an observed axis through ``centre``; old callers default to base vertical."""
    from lerobot.showservo import symmetry

    return symmetry.turn_about(np.array([0.0, 0.0, 1.0]) if axis is None else axis, centre, deg)
