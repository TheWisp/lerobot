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
