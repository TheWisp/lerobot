# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Top-camera ↔ robot-base extrinsics: Kabsch fit + JSON persistence.

Calibration input is a set of paired 3D points: ``cam_xyz`` is the
camera-frame XYZ unprojected from a clicked pixel + depth + intrinsics;
``base_xyz`` is the gripper tip's position in the robot base frame
(forward kinematics of the live joint positions at click time). Each
pair is one row; the GUI's modal walks the user through capturing
``>=4`` pairs by positioning the gripper at varied spots and clicking
on it in the top-camera view.

:func:`kabsch_se3` solves ``base = R @ cam + t`` in the least-squares
sense via SVD. :func:`save_extrinsics` / :func:`load_extrinsics` store
the resulting 4x4 SE(3) at the configured ``extrinsics_path`` so the
teleop's goto path can apply it on every click.

:func:`unproject_pixel` lives here instead of in the teleop because
both the live goto path AND the GUI's calibration preview want to
project a pixel + depth into camera frame XYZ.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def kabsch_se3(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the SE(3) ``R, t`` mapping ``src -> dst`` in least squares.

    Args:
        src: ``(N, 3)`` source points (camera-frame XYZ).
        dst: ``(N, 3)`` target points (robot-base XYZ).

    Returns:
        ``(T, rmse)`` where ``T`` is the 4x4 SE(3) such that
        ``T @ [src.T; 1] ≈ [dst.T; 1]``, and ``rmse`` is the residual
        RMS error in meters.

    Preconditions:
        * ``src.shape == dst.shape``, ``shape[1] == 3``, ``N >= 4``
          (Kabsch is well-defined for N >= 3; we ask for 4 so the user
          has at least one redundancy point and the RMSE is meaningful).
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    assert src.shape == dst.shape and src.shape[1] == 3, f"shape mismatch: src={src.shape}, dst={dst.shape}"
    assert src.shape[0] >= 4, f"need >=4 paired points, got {src.shape[0]}"

    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    src_dem = src - src_c
    dst_dem = dst - dst_c
    # Uppercase variable names below are mathematical convention for the
    # Kabsch SVD: H (cross-covariance), U/Vt (SVD factors), D (diagonal
    # reflection corrector), R (rotation), T (SE(3) transform). Renaming
    # them to lowercase would make the algorithm harder to read against
    # standard references; noqa per-line keeps the convention explicit.
    H = src_dem.T @ dst_dem  # noqa: N806
    U, _, Vt = np.linalg.svd(H)  # noqa: N806
    # Reflection correction: ensure right-handed rotation (det(R) = +1).
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, 1.0, d])  # noqa: N806
    R = Vt.T @ D @ U.T  # noqa: N806
    t = dst_c - R @ src_c

    T = np.eye(4, dtype=float)  # noqa: N806
    T[:3, :3] = R
    T[:3, 3] = t

    residuals = src @ R.T + t - dst
    rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return T, rmse


def save_extrinsics(
    path: str | Path,
    T_base_camera: np.ndarray,  # noqa: N803 — SE(3) matrix name, see kabsch_se3
    rmse_m: float,
    n_points: int,
    cam_key: str,
) -> None:
    """Persist ``T_base_camera`` (4x4 SE(3)) to JSON.

    The file is human-inspectable. Re-running calibration overwrites
    in place — there's no versioning.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    T = np.asarray(T_base_camera, dtype=float)  # noqa: N806
    assert T.shape == (4, 4), f"T must be 4x4, got {T.shape}"
    payload = {
        "T_base_camera": T.tolist(),
        "rmse_m": float(rmse_m),
        "n_points": int(n_points),
        "cam_key": cam_key,
        "comment": (
            "Kabsch fit of (camera_frame_xyz, base_frame_gripper_xyz) pairs. "
            "Apply: world = T[:3,:3] @ cam_xyz + T[:3,3]."
        ),
    }
    p.write_text(json.dumps(payload, indent=2))


def load_extrinsics(path: str | Path) -> np.ndarray | None:
    """Return ``T_base_camera`` as a 4x4 ndarray, or ``None`` if not present.

    Returns None on missing-file / malformed-JSON / wrong-shape, so the
    caller can decide whether to fall through into calibration mode.
    Logs a warning on malformed files so it isn't silent.
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
        logger.warning("Failed to load click_target extrinsics from %s: %s", path, e)
        return None


def unproject_pixel(u: float, v: float, depth_m: float, intrinsics: dict[str, float]) -> np.ndarray:
    """Pinhole unprojection of one pixel to camera-frame XYZ.

    Args:
        u, v: pixel coordinates (x = column, y = row).
        depth_m: depth in meters at ``(u, v)`` along the camera Z axis.
        intrinsics: dict with ``fx, fy, ppx, ppy`` (see
            :meth:`RealSenseCamera.get_color_intrinsics`).

    Returns:
        ``(3,)`` ndarray of camera-frame XYZ in meters.
    """
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    z = float(depth_m)
    x = (float(u) - ppx) * z / fx
    y = (float(v) - ppy) * z / fy
    return np.array([x, y, z], dtype=float)
