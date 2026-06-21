# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Camera geometry: pinhole (un)projection, the base<->camera extrinsic, and
SE(3) helpers. ``unproject_pixel`` / ``load_extrinsics`` are lifted verbatim
from the shelved ``experiment/click-target`` branch so this MVP reuses the same
math the 12 mm-RMSE calibration was produced with.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# The top-camera extrinsic the click-target calibration wrote (Kabsch, 12 mm RMSE).
DEFAULT_EXTRINSICS_PATH = Path("~/.config/lerobot/click_target_extrinsics.json").expanduser()


def load_extrinsics(path: str | Path = DEFAULT_EXTRINSICS_PATH) -> np.ndarray | None:
    """Return ``T_base_camera`` (4x4 SE(3)), or ``None`` if missing/malformed.

    ``base_xyz = T[:3,:3] @ cam_xyz + T[:3,3]``.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        T = np.array(data["T_base_camera"], dtype=float)  # noqa: N806
        assert T.shape == (4, 4)
        return T
    except (json.JSONDecodeError, KeyError, AssertionError, ValueError) as e:
        logger.warning("Failed to load extrinsics from %s: %s", path, e)
        return None


def invert_se3(T: np.ndarray) -> np.ndarray:  # noqa: N803 — SE(3) matrix name
    """Inverse of a 4x4 SE(3) (cheaper + stabler than ``np.linalg.inv``)."""
    R = T[:3, :3]  # noqa: N806
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)  # noqa: N806
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def unproject_pixel(u: float, v: float, depth_m: float, intr: dict[str, float]) -> np.ndarray:
    """Pinhole unprojection of one pixel + depth to camera-frame XYZ (meters)."""
    z = float(depth_m)
    x = (float(u) - intr["ppx"]) * z / intr["fx"]
    y = (float(v) - intr["ppy"]) * z / intr["fy"]
    return np.array([x, y, z], dtype=float)


def unproject_points(uv: np.ndarray, depth_m: np.ndarray, intr: dict[str, float]) -> np.ndarray:
    """Vectorized unprojection. ``uv`` is ``(N,2)``, ``depth_m`` is ``(N,)`` meters.

    Returns ``(N,3)`` camera-frame XYZ.
    """
    uv = np.asarray(uv, dtype=float)
    z = np.asarray(depth_m, dtype=float)
    x = (uv[:, 0] - intr["ppx"]) * z / intr["fx"]
    y = (uv[:, 1] - intr["ppy"]) * z / intr["fy"]
    return np.stack([x, y, z], axis=1)


def depth_to_pointcloud(
    depth_m: np.ndarray, intr: dict[str, float], mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a metric depth map to a camera-frame point cloud.

    Args:
        depth_m: ``(H,W)`` depth in meters; non-positive entries are dropped.
        intr: pinhole intrinsics (``fx, fy, ppx, ppy``).
        mask: optional ``(H,W)`` bool; if given, only those pixels are kept.

    Returns ``(pts, uv)`` where ``pts`` is ``(M,3)`` camera XYZ and ``uv`` is the
    matching ``(M,2)`` pixel coords (column, row).
    """
    h, w = depth_m.shape
    vs, us = np.mgrid[0:h, 0:w]
    valid = depth_m > 0
    if mask is not None:
        valid &= mask.astype(bool)
    us, vs, z = us[valid], vs[valid], depth_m[valid]
    x = (us - intr["ppx"]) * z / intr["fx"]
    y = (vs - intr["ppy"]) * z / intr["fy"]
    pts = np.stack([x, y, z], axis=1)
    uv = np.stack([us, vs], axis=1).astype(float)
    return pts, uv


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:  # noqa: N803
    """Apply a 4x4 SE(3) to ``(N,3)`` points."""
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:3, :3].T + T[:3, 3]


def project_points(pts_cam: np.ndarray, intr: dict[str, float]) -> np.ndarray:
    """Project ``(N,3)`` camera-frame XYZ to ``(N,2)`` pixel coords (column, row).

    Points with non-positive Z are projected too (caller filters); the pixel is
    only meaningful for Z > 0.
    """
    pts = np.asarray(pts_cam, dtype=float)
    z = np.where(np.abs(pts[:, 2]) < 1e-9, 1e-9, pts[:, 2])
    u = pts[:, 0] * intr["fx"] / z + intr["ppx"]
    v = pts[:, 1] * intr["fy"] / z + intr["ppy"]
    return np.stack([u, v], axis=1)
