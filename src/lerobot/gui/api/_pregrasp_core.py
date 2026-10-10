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

"""One taught pose, transported by the object's rigid motion.

Teach: SIFT keypoints inside a box the operator draws around the object, each
lifted to camera 3D by the aligned depth, plus the fingertip pose the operator
jogged to. Test: the same keypoints found again in a new frame (mutual matches,
a 2D consensus to throw out the impostors, then a robust 3D Kabsch on the
survivors) give the object's rigid motion in the camera frame. Conjugated into
the base frame through the camera-to-base calibration, that motion carries the
taught pose to where the object is now. Same instance only: the descriptors
are the object's own.
"""

from __future__ import annotations

from collections.abc import Callable
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
KEYPOINT_KINDS = ("pregrasp", "grasp_end", "preplace", "place_end", "pose")
GRASP_KINDS = ("pregrasp", "grasp_end")  # follow the object picked, which is then held
PLACE_KINDS = ("preplace", "place_end")  # follow the object it is placed onto
# "pose" is no motion: it sets the frame an object's demo pose is read from, where the operator sees it clearly.
ACT_REACH_TOL_M = (
    0.003  # a planned pose solved to within this is reached: under the hand-eye calibration's own error
)
ACT_REACH_TOL_DEG = 3.0
ACT_SOLVE_TOL_M = 0.0005  # the solve keeps going to this; the reach tolerance above only judges the result
ACT_SOLVE_TOL_DEG = 0.5
ACT_IK_CALLS = 40  # the IK moves a bounded step per call; the first solve from far away needs several
ACT_MAX_JOINT_STEP_DEG = (
    10.0  # a larger change between two samples is a change of arm configuration, not a motion
)
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


def carried_motion(frames: list[tuple[float, np.ndarray]], since: float) -> np.ndarray:
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
    return np.asarray(frames[-1][1], dtype=float) @ np.linalg.inv(_groups_pose_at(frames, since))


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
    can have moved it; pose marks alone are a list of their own.
    """
    if not keypoints:
        return ""
    for k in keypoints:
        if k.get("kind") not in KEYPOINT_KINDS:
            return f"a mark is one of {KEYPOINT_KINDS}"
        tk = k.get("t")
        if not isinstance(tk, (int, float)) or not (t_start <= float(tk) <= t_end):
            return f"a mark's time must lie within the demo ({t_start:.1f} to {t_end:.1f} s)"
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


def grasp_held(
    cmd: float, obs: float, demo_cmd: float, demo_obs: float, closing: float
) -> tuple[bool | None, float]:
    """Did the gripper close on the object? ``(held, how far short of its command it stopped)``.

    ``closing`` is the sign of the demo's closing: +1 when closing raises the gripper's
    reading. Held when the gripper stopped more than :data:`GRASP_HELD_SHORT` short of its
    command, as an object between the fingers stops it. None when the demo's own grasp
    stopped no further short than that, so the reading cannot tell.
    """
    short = float(closing * (cmd - obs))
    if closing * (demo_cmd - demo_obs) <= GRASP_HELD_SHORT:
        return None, short
    return short > GRASP_HELD_SHORT, short


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


# The held object's pose in the gripper is measured with the arm standing still. On the gamepad demo the track's
# hold agreed with itself to 0.6 deg and 0.1 mm (median) over 119 still frames, and was off by a median of 19 mm over
# the 47 frames the arm moved; one-shot finds of the demo's view on those still frames were all strong and agreed to
# 1.0 deg and 0.4 mm. A view further than the limits below from the rest is a bad find, not a slip in the grip.
HOLD_STILL_M_S = 0.005
HOLD_VIEWS = 5
HOLD_MIN_VIEWS = 3
HOLD_AGREE_M = 0.005
HOLD_AGREE_DEG = 5.0


def average_hold(holds: list[np.ndarray], centre: np.ndarray) -> dict[str, Any] | None:
    """One hold from several views of the held object, or None when fewer than :data:`HOLD_MIN_VIEWS` agree.

    Each hold is ``tip^-1 . motion``: the object's motion from its view in the demo, as
    seen from the gripper. ``centre`` (base frame) is the object's centre in that view.
    Views are compared there, at the object, because at the motion's origin a small turn
    reads as a large shift. A view whose centre lies more than :data:`HOLD_AGREE_M` from
    the median, or whose turn is more than :data:`HOLD_AGREE_DEG` from the mean, is left
    out. Post: ``hold`` (4x4), ``n`` views used of ``views``, and their ``spread_mm`` and
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
    off_deg = np.degrees((rot * rot.mean().inv()).magnitude())
    keep = np.flatnonzero((off_m <= HOLD_AGREE_M) & (off_deg <= HOLD_AGREE_DEG))
    if len(keep) < HOLD_MIN_VIEWS:
        return None
    mean = rot[keep].mean()
    out = np.eye(4)
    out[:3, :3] = mean.as_matrix()
    out[:3, 3] = hs[keep, :3, 3].mean(axis=0)
    kc = centres[keep]
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


def plan_pregrasp_grasp(
    keypoints: list[dict[str, Any]],
    t: np.ndarray,
    tips: np.ndarray,
    grip_cmd: np.ndarray,
    q_demo: np.ndarray,
    delta_base: np.ndarray,
    start_pose: np.ndarray,
    start_grip: float,
    lin_m_s: float,
    ang_rad_s: float,
    grip_units_s: float,
    speed: float,
    hz: float,
    skip: int = 0,
) -> dict[str, Any]:
    """The act as timed samples: straight lines through the pre-grasp points, then the grasp exactly as recorded.

    The pre-grasp points and the grasp are the demo's fingertip poses carried by
    ``delta_base``, the object's motion since the demo, so the grasp keeps the demo's
    motion relative to the object sample for sample. The first line starts at
    ``start_pose``, where the arm is. Before each line the gripper walks to that
    pre-grasp's opening where the arm stands, away from the object; during the grasp
    it follows the recorded command. Lines run at the walk's speed and the grasp on
    the demo's clock, both scaled by ``speed``.

    ``skip`` leaves out that many leading pre-grasps, already reached; labels keep
    their numbers. Pre: ``keypoints_problem(keypoints, ...) == ''`` with at least
    one pre-grasp, ``0 <= skip < `` the number of pre-grasps, speed > 0.
    Post: ``times`` (N,) from 0, ``poses`` (N, 4, 4) base frame, ``grips`` (N,), ``hints``
    (N, J) the demo's joint change since the previous sample (zero outside the grasp),
    ``floor_ref`` (N,) the demo's own fingertip height for grasp samples and +inf
    elsewhere, ``stage`` (N,) labels, ``arrive`` the sample index where each pre-grasp is
    reached, ``grasp`` the (first, last) sample indices of the grasp or None.
    """
    assert speed > 0.0, "a positive speed"
    t = np.asarray(t, dtype=float)
    pre = sorted(float(k["t"]) for k in keypoints if k["kind"] == "pregrasp")
    ends = [float(k["t"]) for k in keypoints if k["kind"] == "grasp_end"]
    assert pre, "at least one pre-grasp"
    dt = 1.0 / hz
    nj = np.asarray(q_demo).shape[1]
    times, poses, grips, hints, floor_ref, stage = (
        [0.0],
        [np.asarray(start_pose, float)],
        [float(start_grip)],
        [np.zeros(nj)],
        [np.inf],
        ["start"],
    )
    arrive: list[int] = []

    def add(dt_s: float, pose: np.ndarray, grip: float, hint: np.ndarray, ref: float, label: str) -> None:
        times.append(times[-1] + dt_s)
        poses.append(pose)
        grips.append(float(grip))
        hints.append(hint)
        floor_ref.append(ref)
        stage.append(label)

    def index(tk: float) -> int:
        return int(np.argmin(np.abs(t - tk)))

    assert 0 <= skip < len(pre), "at least one pre-grasp is left to reach"
    pose, grip = poses[0], grips[0]
    for n, tk in enumerate(pre[skip:], start=skip + 1):
        i = index(tk)
        target = np.asarray(delta_base, float) @ np.asarray(tips[i], float)
        g = float(grip_cmd[i])
        steps = int(np.ceil(abs(g - grip) / (grip_units_s * speed) * hz))
        for s in range(1, steps + 1):
            add(dt, pose, grip + (g - grip) * s / steps, np.zeros(nj), np.inf, f"pre-grasp {n}")
        grip = g
        dist_m, ang_deg = pose_residual(pose, target)
        dur = max(dist_m / (lin_m_s * speed), np.radians(ang_deg) / (ang_rad_s * speed))
        steps = max(1, int(np.ceil(dur * hz)))
        for s in range(1, steps + 1):
            add(dt, interp_rigid(pose, target, s / steps), grip, np.zeros(nj), np.inf, f"pre-grasp {n}")
        arrive.append(len(times) - 1)
        pose = target
    grasp = None
    if ends:
        i0, i1 = index(pre[-1]), index(ends[0])
        first = len(times)
        for i in range(i0 + 1, i1 + 1):
            add(
                float(t[i] - t[i - 1]) / speed,
                np.asarray(delta_base, float) @ np.asarray(tips[i], float),
                float(grip_cmd[i]),
                np.asarray(q_demo[i], float) - np.asarray(q_demo[i - 1], float),
                float(tips[i][2, 3]),
                "grasp",
            )
        grasp = (first, len(times) - 1) if len(times) > first else None
    return {
        "times": np.array(times),
        "poses": np.stack(poses),
        "grips": np.array(grips),
        "hints": np.stack(hints),
        "floor_ref": np.array(floor_ref),
        "stage": stage,
        "arrive": arrive,
        "grasp": grasp,
    }


def plan_place(
    keypoints: list[dict[str, Any]],
    t: np.ndarray,
    tips: np.ndarray,
    grip_cmd: np.ndarray,
    q_demo: np.ndarray,
    target_base: np.ndarray,
    hold_fix: np.ndarray,
    start_pose: np.ndarray,
    start_grip: float,
    lin_m_s: float,
    ang_rad_s: float,
    speed: float,
    hz: float,
    skip: int = 0,
) -> dict[str, Any]:
    """The carry and the place as timed samples: straight lines through the pre-place points, then the place
    exactly as recorded.

    Every pose is the demo's fingertip pose carried by ``target_base``, the motion of the
    object placed onto since the demo, and corrected on the gripper's side by
    ``hold_fix``, the change in where the held object sits in the gripper (the demo's
    hold times the inverse of the live one): ``target_base . tip . hold_fix``. Held as in
    the demo, ``hold_fix`` is the identity. The gripper keeps ``start_grip``, the grasp's
    closing, along the lines and follows the recorded command during the place, release
    included. Lines run at the walk's speed and the place on the demo's clock, both scaled
    by ``speed``. ``skip`` leaves out that many leading pre-places, already reached.

    Pre: ``keypoints_problem(keypoints, ...) == ''`` with at least one pre-place,
    ``0 <= skip <`` their number, speed > 0. Post: as :func:`plan_pregrasp_grasp`, with
    ``place`` (the first and last sample indices of the place, or None) in place of ``grasp``.
    """
    assert speed > 0.0, "a positive speed"
    t = np.asarray(t, dtype=float)
    pre = sorted(float(k["t"]) for k in keypoints if k["kind"] == "preplace")
    ends = [float(k["t"]) for k in keypoints if k["kind"] == "place_end"]
    assert pre, "at least one pre-place"
    assert 0 <= skip < len(pre), "at least one pre-place is left to reach"
    carry, fix = np.asarray(target_base, dtype=float), np.asarray(hold_fix, dtype=float)
    dt = 1.0 / hz
    nj = np.asarray(q_demo).shape[1]
    times, poses, grips, hints, floor_ref, stage = (
        [0.0],
        [np.asarray(start_pose, float)],
        [float(start_grip)],
        [np.zeros(nj)],
        [np.inf],
        ["start"],
    )
    arrive: list[int] = []

    def add(dt_s: float, pose: np.ndarray, grip: float, hint: np.ndarray, ref: float, label: str) -> None:
        times.append(times[-1] + dt_s)
        poses.append(pose)
        grips.append(float(grip))
        hints.append(hint)
        floor_ref.append(ref)
        stage.append(label)

    def index(tk: float) -> int:
        return int(np.argmin(np.abs(t - tk)))

    pose, grip = poses[0], grips[0]
    for n, tk in enumerate(pre[skip:], start=skip + 1):
        target = carry @ np.asarray(tips[index(tk)], float) @ fix
        dist_m, ang_deg = pose_residual(pose, target)
        dur = max(dist_m / (lin_m_s * speed), np.radians(ang_deg) / (ang_rad_s * speed))
        steps = max(1, int(np.ceil(dur * hz)))
        for s in range(1, steps + 1):
            add(dt, interp_rigid(pose, target, s / steps), grip, np.zeros(nj), np.inf, f"pre-place {n}")
        arrive.append(len(times) - 1)
        pose = target
    place = None
    if ends:
        i0, i1 = index(pre[-1]), index(ends[0])
        first = len(times)
        for i in range(i0 + 1, i1 + 1):
            add(
                float(t[i] - t[i - 1]) / speed,
                carry @ np.asarray(tips[i], float) @ fix,
                float(grip_cmd[i]),
                np.asarray(q_demo[i], float) - np.asarray(q_demo[i - 1], float),
                float(tips[i][2, 3]),
                "place",
            )
        place = (first, len(times) - 1) if len(times) > first else None
    return {
        "times": np.array(times),
        "poses": np.stack(poses),
        "grips": np.array(grips),
        "hints": np.stack(hints),
        "floor_ref": np.array(floor_ref),
        "stage": stage,
        "arrive": arrive,
        "place": place,
    }


def join_plans(first: dict[str, Any], then: dict[str, Any]) -> dict[str, Any]:
    """``then`` run after ``first`` as one timeline. Pre: ``then`` starts where ``first`` ends, its own first
    sample being that pose; it is dropped. Post: the arrays joined, ``then``'s indices shifted, ``first``'s
    ``grasp`` and ``then``'s ``place`` kept, ``arrive`` the pre-grasps' then the pre-places'."""
    shift = len(first["times"]) - 1

    def moved(span: tuple[int, int] | None) -> tuple[int, int] | None:
        return None if span is None else (span[0] + shift, span[1] + shift)

    return {
        "times": np.concatenate([first["times"], first["times"][-1] + then["times"][1:]]),
        "poses": np.concatenate([first["poses"], then["poses"][1:]]),
        "grips": np.concatenate([first["grips"], then["grips"][1:]]),
        "hints": np.concatenate([first["hints"], then["hints"][1:]]),
        "floor_ref": np.concatenate([first["floor_ref"], then["floor_ref"][1:]]),
        "stage": list(first["stage"]) + list(then["stage"][1:]),
        "arrive": list(first["arrive"]) + [a + shift for a in then["arrive"]],
        "grasp": first.get("grasp"),
        "place": moved(then.get("place")),
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
LANDING_STEP_DEG = (
    5.0  # "turn" is tried every this many degrees; the turn taken is within half a step of the best
)
LANDING_TRIES = 4  # the cheapest landings planned in full before the act gives up


def landing_turns(landing: str, order: int) -> list[float]:
    """The turns, degrees about the vertical through the middle of the object placed onto, that the landing rule
    ``landing`` counts as the same place: only 0 for "exact", the object's own ``order`` turns for "symmetry", every
    ``LANDING_STEP_DEG`` for "turn". Pre: ``landing`` in LANDINGS."""
    assert landing in LANDINGS, f"a landing is one of {LANDINGS}"
    if landing == "turn":
        return [float(a) for a in np.arange(0.0, 360.0, LANDING_STEP_DEG)]
    if landing == "symmetry":
        n = max(1, int(order))
        return [360.0 * k / n for k in range(n)]
    return [0.0]


def turn_about(centre: np.ndarray, deg: float) -> np.ndarray:
    """The rigid turn by ``deg`` degrees about the vertical through ``centre`` (base frame): 4x4."""
    a = np.radians(float(deg))
    out = np.eye(4)
    out[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    p = np.asarray(centre, dtype=float)
    out[:3, 3] = p - out[:3, :3] @ p
    return out


def landed(motion: np.ndarray, centre: np.ndarray, deg: float) -> np.ndarray:
    """The motion that lands a place turned by ``deg`` about the object it goes onto: ``motion`` (that object's since
    the demo, base frame) and then the turn about the vertical through where ``motion`` puts ``centre`` (the object's
    middle in the demo), so the turn goes with the object wherever its track moves it."""
    motion = np.asarray(motion, dtype=float)
    return turn_about(motion[:3, :3] @ np.asarray(centre, dtype=float) + motion[:3, 3], deg) @ motion


def rank_landings(
    kin: Any,
    turns: list[float],
    poses_at: Callable[[float], np.ndarray],
    q_demo: np.ndarray,
    lo: np.ndarray | None,
    hi: np.ndarray | None,
    grip_index: int,
) -> list[tuple[float, float]]:
    """The landing turns the arm can make, as (cost, turn), cheapest first.

    ``poses_at(turn)`` gives the place's judged samples landed by that turn (M, 4, 4), and ``q_demo`` (M, J) the
    demo's own joints at them. Each turn's samples are solved from the demo's joints at the first, with the demo's
    joint changes between them as hints, so the solve stays on the demo's arm configuration, and no joint past its
    servo's range [``lo``, ``hi``] (held there). A turn is dropped when a sample is out of reach as solved; the cost of
    the rest is the mean square of their arm joints' distance from the demo's (motor deg^2).
    """
    q_demo = np.asarray(q_demo, dtype=float)
    arm = [k for k in range(q_demo.shape[1]) if k != grip_index]
    hints = np.vstack([np.zeros(q_demo.shape[1]), np.diff(q_demo, axis=0)])
    ranked = []
    for turn in turns:
        sol = solve_plan_joints(
            kin, poses_at(turn), q_demo[:, grip_index], hints, q_demo[0], grip_index, lo, hi
        )
        if np.any(sol["residual_m"] > ACT_REACH_TOL_M) or np.any(sol["residual_deg"] > ACT_REACH_TOL_DEG):
            continue
        ranked.append((float(np.mean((sol["q"][:, arm] - q_demo[:, arm]) ** 2)), float(turn)))
    return sorted(ranked)
