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

from typing import Any

import numpy as np


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


# ── demo key points: what a replay keeps relative to the object, and what stays put ──────────
KEYPOINT_ANCHORS = ("object", "world")
GRIP_EVENT_SHARE = 0.25  # a gripper move this share of its range over the demo is a grasp or a release
APPROACH_LEAD_S = 1.0  # the suggested approach sits this long before the grasp; the operator moves it
ACT_REACH_TOL_M = (
    0.003  # a corrected pose solved to within this is reached: under the hand-eye calibration's own error
)
ACT_REACH_TOL_DEG = 3.0
ACT_IK_CALLS = 40  # the IK moves a bounded step per call; a correction of a few centimetres needs several
ACT_MAX_JOINT_STEP_DEG = (
    10.0  # a larger change between two samples is a change of arm configuration, not a motion the demo made
)


def suggest_keypoints(t: np.ndarray, grippers: np.ndarray) -> list[dict[str, Any]]:
    """The key points a pickup usually has, read off the gripper channel.

    The grasp is where the gripper settles at its most closed (the start of that
    plateau), the release where it leaves the plateau again, the approach a fixed
    lead before the grasp. Post: sorted by time; the approach and the grasp move
    with the object, the release stays in the world; empty when the gripper never
    moved enough to tell.
    """
    t = np.asarray(t, dtype=float)
    g = np.asarray(grippers, dtype=float)
    if len(t) < 2:
        return []
    span = float(g.max() - g.min())
    if span <= 0.0:
        return []
    band = g.min() + GRIP_EVENT_SHARE * span  # within this of fully closed counts as holding
    closed = int(np.argmin(g))
    grasp = closed
    while grasp > 0 and g[grasp - 1] <= band:
        grasp -= 1
    release = closed
    while release + 1 < len(g) and g[release + 1] <= band:
        release += 1
    approach = int(np.argmin(np.abs(t - (t[grasp] - APPROACH_LEAD_S))))
    out = []
    if approach < grasp:
        out.append({"t": float(t[approach]), "name": "approach", "anchor": "object"})
    out.append({"t": float(t[grasp]), "name": "grasp", "anchor": "object"})
    if release + 1 < len(g):
        out.append({"t": float(t[release]), "name": "release", "anchor": "world"})
    return out


def interp_rigid(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """The rigid transform ``s`` of the way from ``a`` to ``b``: rotation slerped, translation lerped."""
    from scipy.spatial.transform import Rotation, Slerp

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.eye(4)
    rots = Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]]))
    out[:3, :3] = Slerp([0.0, 1.0], rots)(float(np.clip(s, 0.0, 1.0))).as_matrix()
    out[:3, 3] = (1.0 - s) * a[:3, 3] + s * b[:3, 3]
    return out


def replay_corrections(
    t: np.ndarray, keypoints: list[dict[str, Any]], delta_base: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Per-sample corrections for the samples between the first and the last key point.

    An object key point is corrected by ``delta_base`` (it moves with the object),
    a world key point by nothing; between two key points the correction blends
    from one to the other along the demo's own time, so the recorded motion is
    kept and only its anchoring changes. Pre: at least one key point inside the
    demo's span. Post: ``(corrections (M, 4, 4), i0, i1)`` with ``t[i0:i1]`` the
    replayed samples, ``M = i1 - i0 >= 1``.
    """
    t = np.asarray(t, dtype=float)
    kps = sorted(keypoints, key=lambda k: float(k["t"]))
    assert kps, "a replay needs at least one key point"
    anchors = {"object": np.asarray(delta_base, dtype=float), "world": np.eye(4)}
    times = np.array([float(k["t"]) for k in kps])
    corr = [anchors[k["anchor"]] for k in kps]
    i0 = int(np.searchsorted(t, times[0], side="left"))
    i1 = int(np.searchsorted(t, times[-1], side="right"))
    i0 = min(i0, len(t) - 1)
    i1 = max(i1, i0 + 1)
    out = np.empty((i1 - i0, 4, 4))
    for n, tau in enumerate(t[i0:i1]):
        k = int(np.clip(np.searchsorted(times, tau, side="right") - 1, 0, len(kps) - 1))
        if k + 1 >= len(kps) or times[k + 1] <= times[k]:
            out[n] = corr[k]
            continue
        s = (tau - times[k]) / (times[k + 1] - times[k])
        out[n] = interp_rigid(corr[k], corr[k + 1], float(np.clip(s, 0.0, 1.0)))
    return out, i0, i1


def pose_residual(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``(metres, degrees)`` between two poses: translation distance and the rotation angle."""
    from scipy.spatial.transform import Rotation

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ang = np.linalg.norm(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec())
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.degrees(ang))


def solve_replay_joints(kin: Any, q_demo: np.ndarray, poses: np.ndarray) -> dict[str, np.ndarray]:
    """Joints for every corrected pose, solved from the demo's own configuration.

    Each sample is seeded with its recorded joints plus the joint correction the
    previous sample needed, so the arm keeps the configuration the human used and
    each solve starts close. ``kin`` is the arm's motor-space kinematics
    (``forward_kinematics(q)``, ``inverse_kinematics(seed, pose)``; extra columns
    such as the gripper pass through). Post: ``q`` (M, J), ``residual_m`` and
    ``residual_deg`` (M,) of the solved pose against the asked one, ``step_deg``
    (M,) the largest arm-joint change from the previous sample (0 for the first).
    Nothing is rejected here; the caller judges the residuals and the steps.
    """
    q_demo = np.asarray(q_demo, dtype=float)
    poses = np.asarray(poses, dtype=float)
    m = len(poses)
    assert q_demo.shape[0] == m, "one recorded configuration per corrected pose"
    q = np.empty_like(q_demo)
    res_m = np.empty(m)
    res_deg = np.empty(m)
    offset = np.zeros(q_demo.shape[1])
    for i in range(m):
        qi = q_demo[i] + offset
        for _ in range(ACT_IK_CALLS):
            qi = np.asarray(kin.inverse_kinematics(qi, poses[i]), dtype=float)
            e_m, e_deg = pose_residual(kin.forward_kinematics(qi), poses[i])
            if e_m <= ACT_REACH_TOL_M and e_deg <= ACT_REACH_TOL_DEG:
                break
        q[i], res_m[i], res_deg[i] = qi, e_m, e_deg
        offset = qi - q_demo[i]
    step = np.zeros(m)
    if m > 1:
        step[1:] = (
            np.abs(np.diff(q[:, :6], axis=0)).max(axis=1)
            if q.shape[1] >= 6
            else np.abs(np.diff(q, axis=0)).max(axis=1)
        )
    return {"q": q, "residual_m": res_m, "residual_deg": res_deg, "step_deg": step}
