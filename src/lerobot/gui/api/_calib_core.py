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

"""The arithmetic behind the two touch calibrations, free of any device.

Tool point: the fingertip is a fixed offset ``d`` from the arm's anchor link,
and the URDF's guess at it is wrong once the gripper is modified. Touching one
physical point from several orientations gives ``p_i + R_i d = P`` for every
touch, linear in the unknowns ``d`` and ``P``; a least-squares solve returns
both, and the per-touch residual is the first independent check the URDF's
geometry gets.

Camera to base: with the fingertip known, touching marker corners the camera
has located in its own frame gives 3D-3D pairs, and a rigid (Kabsch) fit is
the camera-to-base transform. No planarity assumption anywhere: the corners
may sit at any height. A similarity fit's scale is reported as a diagnostic
for the camera's range source (depth or marker size), never applied.

Marker corners are lifted to 3D two ways: intersecting the corner's pixel ray
with a plane fitted to the marker's interior depth (depth at the corner itself
is the worst place to sample a stereo sensor), or from the marker's printed
size through perspective-n-point. Both go through the same colour intrinsics.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import numpy as np

MIN_TOOL_TOUCHES = 3
MIN_CAMERA_TOUCHES = 3


def solve_tool_point(anchor_poses: list[np.ndarray]) -> dict[str, Any]:
    """Fingertip offset from the anchor link, from touches of one point at several orientations.

    Pre: at least :data:`MIN_TOOL_TOUCHES` 4x4 anchor poses (base frame) whose
    rotations differ; identical rotations leave the offset unobservable and
    raise ``ValueError``.
    Post: ``offset_m`` is the fingertip in the anchor frame, ``point_m`` the
    touched point in the base frame, residuals in metres per touch.
    """
    poses = [np.asarray(p, dtype=float) for p in anchor_poses]
    assert all(p.shape == (4, 4) for p in poses), "anchor poses are 4x4"
    if len(poses) < MIN_TOOL_TOUCHES:
        raise ValueError(f"need at least {MIN_TOOL_TOUCHES} touches, have {len(poses)}")
    n = len(poses)
    a = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    for i, pose in enumerate(poses):
        a[3 * i : 3 * i + 3, 0:3] = pose[:3, :3]
        a[3 * i : 3 * i + 3, 3:6] = -np.eye(3)
        b[3 * i : 3 * i + 3] = -pose[:3, 3]
    sv = np.linalg.svd(a, compute_uv=False)
    # Rank drops when every touch shares an orientation; then d and P trade off freely.
    if sv[-1] < 1e-6 * sv[0]:
        raise ValueError("touch orientations are too alike to separate the fingertip offset from the point")
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    d, point = x[:3], x[3:]
    residuals = [float(np.linalg.norm(p[:3, 3] + p[:3, :3] @ d - point)) for p in poses]
    return {
        "offset_m": d.tolist(),
        "point_m": point.tolist(),
        "residuals_m": residuals,
        "rms_m": float(np.sqrt(np.mean(np.square(residuals)))),
        "max_m": float(max(residuals)),
        "conditioning": float(sv[-1] / sv[0]),
        "n": n,
    }


def rigid_fit(src: np.ndarray, dst: np.ndarray) -> dict[str, Any]:
    """Rigid transform taking ``src`` points onto ``dst`` (Kabsch), plus diagnostics.

    Pre: matching ``(n, 3)`` arrays, ``n`` >= :data:`MIN_CAMERA_TOUCHES`, not collinear.
    Post: ``transform`` is 4x4 with a proper rotation (det +1); ``scale`` is the
    similarity fit's scale, reported for diagnosis only and not applied.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3, (src.shape, dst.shape)
    n = src.shape[0]
    if n < MIN_CAMERA_TOUCHES:
        raise ValueError(f"need at least {MIN_CAMERA_TOUCHES} touches, have {n}")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - mu_s, dst - mu_d
    h = s0.T @ d0
    u, sv, vt = np.linalg.svd(h)
    if sv[1] < 1e-9 * max(sv[0], 1e-12):
        raise ValueError("touched points are collinear; the rotation is not determined")
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    corr = np.diag([1.0, 1.0, sign])
    r = vt.T @ corr @ u.T
    t = mu_d - r @ mu_s
    var_s = float(np.sum(s0**2))
    scale = float(np.trace(np.diag(sv) @ corr) / var_s) if var_s > 0 else float("nan")
    fitted = (r @ src.T).T + t
    residuals = np.linalg.norm(fitted - dst, axis=1)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r, t
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-6), "rotation must be proper"
    return {
        "transform": transform.tolist(),
        "residuals_m": residuals.tolist(),
        "rms_m": float(np.sqrt(np.mean(residuals**2))),
        "max_m": float(residuals.max()),
        "scale": scale,
        "n": n,
    }


def pixel_rays(pixels: np.ndarray, intr: dict[str, float]) -> np.ndarray:
    """Unit-depth ray directions ``(n, 3)`` for ``(n, 2)`` pixels in the colour intrinsics."""
    px = np.asarray(pixels, dtype=float).reshape(-1, 2)
    x = (px[:, 0] - intr["cx"]) / intr["fx"]
    y = (px[:, 1] - intr["cy"]) / intr["fy"]
    return np.stack([x, y, np.ones_like(x)], axis=1)


def deproject(pixels: np.ndarray, depth_m: np.ndarray, intr: dict[str, float]) -> np.ndarray:
    """Camera-frame points ``(n, 3)`` for pixels with metric depths (rays scaled by depth)."""
    return pixel_rays(pixels, intr) * np.asarray(depth_m, dtype=float).reshape(-1, 1)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through ``(n, 3)`` points: ``(unit normal, centroid)``. Pre: n >= 3, not collinear."""
    pts = np.asarray(points, dtype=float)
    assert pts.ndim == 2 and pts.shape[0] >= 3, pts.shape
    c = pts.mean(axis=0)
    _u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    if sv[1] < 1e-9 * max(sv[0], 1e-12):
        raise ValueError("points are collinear; no plane")
    n = vt[2]
    if n[2] > 0:  # face the camera so plane offsets read consistently
        n = -n
    return n, c


def corners_from_depth_plane(
    corners_px: np.ndarray, depth_m: np.ndarray, intr: dict[str, float], min_valid: int = 30
) -> tuple[np.ndarray, dict[str, float]]:
    """Lift a marker's four corners to camera 3D through a plane fitted to its interior depth.

    Pre: ``corners_px`` is ``(4, 2)``; ``depth_m`` the aligned metric depth image
    (zeros are holes). Post: ``(4, 3)`` camera-frame corners; the diagnostics
    carry the number of interior pixels used and the plane fit's rms in metres.
    Raises ``ValueError`` when the interior has too few valid depth pixels.
    """
    import cv2

    corners_px = np.asarray(corners_px, dtype=float).reshape(4, 2)
    h, w = depth_m.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    # Shrink the quad toward its centre so the edge pixels (where stereo depth
    # smears across the border) stay out of the fit.
    centre = corners_px.mean(axis=0)
    inner = centre + 0.7 * (corners_px - centre)
    cv2.fillConvexPoly(mask, np.round(inner).astype(np.int32), 1)
    vs, us = np.nonzero(mask)
    z = np.asarray(depth_m, dtype=float)[vs, us]
    ok = z > 0
    if ok.sum() < min_valid:
        raise ValueError(f"only {int(ok.sum())} valid depth pixels inside the marker")
    pts = deproject(np.stack([us[ok], vs[ok]], axis=1), z[ok], intr)
    normal, centroid = fit_plane(pts)
    dist = (pts - centroid) @ normal
    # One robust pass: drop the far outliers (a finger, a cable) and refit.
    keep = np.abs(dist) < 4.0 * max(float(np.median(np.abs(dist))), 1e-4)
    if keep.sum() >= min_valid:
        normal, centroid = fit_plane(pts[keep])
        dist = (pts[keep] - centroid) @ normal
    rays = pixel_rays(corners_px, intr)
    denom = rays @ normal
    if np.any(np.abs(denom) < 1e-9):
        raise ValueError("a corner ray is parallel to the marker plane")
    t = (centroid @ normal) / denom
    corners = rays * t[:, None]
    return corners, {"pixels_used": int(keep.sum()), "plane_rms_m": float(np.sqrt(np.mean(dist**2)))}


def corners_from_pnp(corners_px: np.ndarray, side_m: float, intr: dict[str, float]) -> np.ndarray:
    """Lift a marker's corners to camera 3D from its printed side length (perspective-n-point on the square).

    Pre: corners in ArUco order (top-left, top-right, bottom-right, bottom-left),
    ``side_m`` > 0. Post: ``(4, 3)`` camera-frame corners.
    """
    import cv2

    assert side_m > 0, "marker side must be positive"
    half = side_m / 2.0
    obj = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=float)
    k = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], dtype=float)
    img = np.asarray(corners_px, dtype=float).reshape(4, 1, 2)
    solve = cv2.solvePnP  # spellchecker:disable-line
    ok, rvec, tvec = solve(obj, img, k, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise ValueError("perspective-n-point failed on the marker")
    r, _ = cv2.Rodrigues(rvec)
    return (r @ obj.T).T + tvec.reshape(1, 3)


def detect_markers(gray: np.ndarray, dictionary_name: str = "DICT_4X4_50") -> list[dict[str, Any]]:
    """ArUco markers in a grayscale image, corners refined to sub-pixel, in ArUco corner order."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _rejected = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    out = []
    if ids is None:
        return out
    for quad, marker_id in zip(corners, ids.ravel(), strict=True):
        out.append({"id": int(marker_id), "corners_px": quad.reshape(4, 2).tolist()})
    return sorted(out, key=lambda m: m["id"])


def marker_sheet_image(dictionary_name: str, side_mm: float, count: int, dpi: int = 300):
    """A printable sheet of ArUco markers at a known physical size, as a grayscale PIL image.

    Pre: ``side_mm`` in 15..80, ``count`` in 1..12. Post: ids ``0..count-1`` in
    a grid, each with a white quiet zone and an id label, plus a 100 mm scale
    bar; the page area fits both A4 and Letter when printed at 100 %.
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    assert 15 <= side_mm <= 80, "marker side must be 15..80 mm"
    assert 1 <= count <= 12, "1..12 markers per sheet"
    px = lambda mm: int(round(mm * dpi / 25.4))  # noqa: E731
    page_w_mm, page_h_mm = 200.0, 270.0
    sheet = Image.new("L", (px(page_w_mm), px(page_h_mm)), 255)
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=px(3))
    except TypeError:  # older Pillow: bitmap default only
        font = ImageFont.load_default()
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    quiet = max(6.0, side_mm / 6.0)  # one code bit of white around the marker, the detector's minimum
    label = 6.0
    cell = side_mm + 2 * quiet + label
    margin = 6.0
    cols = max(1, int((page_w_mm - 2 * margin) // cell))
    rows = -(-count // cols)
    bar_y = page_h_mm - 14.0
    if margin + rows * cell > bar_y - 4:
        raise ValueError(f"{count} markers of {side_mm:g} mm do not fit one page")
    for i in range(count):
        r, c = divmod(i, cols)
        x0 = px(margin + c * cell + quiet)
        y0 = px(margin + r * cell + quiet)
        bitmap = cv2.aruco.generateImageMarker(dictionary, i, px(side_mm))
        sheet.paste(Image.fromarray(bitmap), (x0, y0))
        draw.text(
            (x0, y0 + px(side_mm + 1)), f"id {i}   {side_mm:g} mm   {dictionary_name}", fill=0, font=font
        )
    y_bar = px(bar_y)
    draw.line([(px(margin), y_bar), (px(margin + 100), y_bar)], fill=0, width=px(0.5))
    for mm in (0, 50, 100):
        draw.line([(px(margin + mm), y_bar - px(2)), (px(margin + mm), y_bar + px(2))], fill=0, width=px(0.4))
    draw.text(
        (px(margin), y_bar + px(3)),
        "100 mm scale bar: print at 100 % (actual size) and check it with a ruler",
        fill=0,
        font=font,
    )
    return sheet


def marker_sheet_pdf(dictionary_name: str, side_mm: float, count: int, dpi: int = 300) -> bytes:
    """:func:`marker_sheet_image` as PDF bytes carrying the dpi, so a 100 % print is true to size."""
    import io

    buf = io.BytesIO()
    marker_sheet_image(dictionary_name, side_mm, count, dpi).save(buf, format="PDF", resolution=dpi)
    return buf.getvalue()


# ── persistence ──────────────────────────────────────────────────────────────


def calibration_path(base_dir: pathlib.Path, robot_id: str) -> pathlib.Path:
    return base_dir / "calibration" / f"{robot_id}.json"


def load_calibration(path: pathlib.Path) -> dict[str, Any]:
    """The saved calibration for one arm, or an empty dict when there is none."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    assert isinstance(data, dict), "calibration file is a JSON object"
    return data


def save_calibration(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(data, indent=2))


def tip_offset_from_calibration(data: dict[str, Any]) -> np.ndarray | None:
    """The measured fingertip as a 4x4 anchor->tip transform (pure translation), or None."""
    tool = data.get("tool_point")
    if not tool:
        return None
    off = np.eye(4)
    off[:3, 3] = np.asarray(tool["offset_m"], dtype=float)
    return off
