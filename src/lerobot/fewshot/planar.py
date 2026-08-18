# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Image plane <-> robot table plane, self-calibrated with the robot's own body.

No extrinsics are assumed. The robot IS the calibration target: touch the
end-effector to the table at a handful of poses, read its XY from forward
kinematics (proprioception) and its pixel position from its SAM mask, and fit a
homography. Registration done in RECTIFIED (robot-plane) coordinates then yields
transforms whose rotation is a true table-frame yaw and whose translation is in
metres — no conjugation error from doing the fit in perspective image space.
"""

from __future__ import annotations

import numpy as np

from lerobot.fewshot.registration import Sim2


def fit_homography(px: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Normalised DLT: image pixels (N, 2) -> table XY metres (N, 2), N >= 4.

    Pre: points are not collinear (asserted via conditioning). Post: H is 3x3 with
    H[2,2] == 1; ``apply_homography(H, px) ≈ xy`` for the calibration set.
    """
    px = np.asarray(px, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    assert px.shape == xy.shape and px.ndim == 2 and px.shape[1] == 2 and len(px) >= 4

    def norm(p):
        mu = p.mean(axis=0)
        scale = np.sqrt(2.0) / max(np.linalg.norm(p - mu, axis=1).mean(), 1e-9)
        tm = np.array([[scale, 0, -scale * mu[0]], [0, scale, -scale * mu[1]], [0, 0, 1.0]])
        return (p - mu) * scale, tm

    a, ta = norm(px)
    b, tb = norm(xy)
    rows = []
    for (x, y), (u, v) in zip(a, b, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    lhs = np.asarray(rows)
    _, sv, vt = np.linalg.svd(lhs)
    assert sv[-2] > 1e-9, "degenerate calibration points (collinear?)"
    h_mat = np.linalg.inv(tb) @ vt[-1].reshape(3, 3) @ ta
    h_mat /= h_mat[2, 2]
    return h_mat


def apply_homography(h_mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Pre: pts (N, 2). Post: (N, 2); asserts no point maps through infinity."""
    pts = np.asarray(pts, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape[1] == 2
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ h_mat.T
    w = ph[:, 2]
    assert np.all(np.abs(w) > 1e-9), "point at the horizon — homography misused"
    return ph[:, :2] / w[:, None]


# ---------------------------------------------------------------------------
# Demo extraction and transfer, in table coordinates
# ---------------------------------------------------------------------------


class PlanarDemo:
    """One teleop demo reduced to: an EE pose at the bottleneck, the trajectory
    after it (in the bottleneck's own frame), and the object observation then.

    The bottleneck is placed BEFORE the interaction event so the replayed segment
    starts clear of contact. The relative trajectory is expressed in the
    bottleneck EE frame — so a transferred bottleneck reproduces the whole
    approach+manipulation with one composition, no per-step matching.
    """

    def __init__(
        self,
        bottleneck_pose: np.ndarray,  # (4,) x, y, z, yaw — robot/table frame
        rel_traj: np.ndarray,  # (M, 5) dx, dy, dz, dyaw, gripper (bottleneck frame)
    ):
        bottleneck_pose = np.asarray(bottleneck_pose, dtype=np.float64)
        rel_traj = np.asarray(rel_traj, dtype=np.float64)
        assert bottleneck_pose.shape == (4,)
        assert rel_traj.ndim == 2 and rel_traj.shape[1] == 5 and len(rel_traj) >= 1
        assert np.allclose(rel_traj[0, :4], 0.0), "trajectory must start AT the bottleneck"
        self.bottleneck_pose = bottleneck_pose
        self.rel_traj = rel_traj

    @classmethod
    def extract(
        cls,
        ee_poses: np.ndarray,  # (T, 4) x, y, z, yaw from forward kinematics
        gripper: np.ndarray,  # (T,)
        event_idx: int,
        lead_frames: int = 15,
    ) -> PlanarDemo:
        """Pre: 0 <= event_idx < T; poses finite. Post: bottleneck at
        ``max(event_idx - lead_frames, 0)``; rel_traj covers bottleneck..end."""
        ee_poses = np.asarray(ee_poses, dtype=np.float64)
        gripper = np.asarray(gripper, dtype=np.float64)
        assert ee_poses.ndim == 2 and ee_poses.shape[1] == 4 and len(gripper) == len(ee_poses)
        assert 0 <= event_idx < len(ee_poses)
        assert np.isfinite(ee_poses).all()
        b = max(event_idx - lead_frames, 0)
        pb = ee_poses[b]
        c, s = np.cos(-pb[3]), np.sin(-pb[3])
        rot_inv = np.array([[c, -s], [s, c]])  # world -> bottleneck frame
        seg = ee_poses[b:]
        dxy = (seg[:, :2] - pb[:2]) @ rot_inv.T
        rel = np.column_stack([dxy, seg[:, 2] - pb[2], _wrap(seg[:, 3] - pb[3]), gripper[b:]])
        return cls(pb, rel)

    def transfer(
        self, object_motion: Sim2, object_centre_demo: np.ndarray, *, trust_rotation: bool
    ) -> np.ndarray:
        """Absolute EE trajectory for the object's new placement.

        Pre: ``object_motion`` maps demo-time table coordinates of the object to
        live-time table coordinates (from registering the two observations in
        RECTIFIED space); its scale must be ~1 — same object on the same plane —
        and this is asserted, because a scale far from 1 means the calibration or
        the match is wrong, and scaling a robot trajectory is never intended.
        ``object_centre_demo`` is the object's centroid in demo table coordinates.
        Post: (M, 5) absolute x, y, z, yaw, gripper. z is carried from the demo:
        same object, same table. When ``trust_rotation`` is False only the
        displacement of the object's CENTRE is applied and yaw is kept from the
        demo (symmetric / texture-poor object: the recovered angle is arbitrary,
        and for such objects yaw is irrelevant by the same symmetry — applying the
        full similarity would swing the approach around the object for nothing).
        """
        assert 0.9 < object_motion.s < 1.1, (
            f"object 'scale' changed by {object_motion.s:.3f}x on a fixed plane — "
            "bad registration or bad homography; refusing to scale a trajectory"
        )
        centre = np.asarray(object_centre_demo, dtype=np.float64)
        assert centre.shape == (2,)
        pb = self.bottleneck_pose
        if trust_rotation:
            dtheta = object_motion.theta
            new_xy = object_motion.apply(pb[:2][None])[0]
        else:
            dtheta = 0.0
            new_xy = pb[:2] + (object_motion.apply(centre[None])[0] - centre)
        new_yaw = _wrap(pb[3] + dtheta)
        c, s = np.cos(new_yaw), np.sin(new_yaw)
        rot = np.array([[c, -s], [s, c]])
        out = np.empty_like(self.rel_traj)
        out[:, 0:2] = new_xy + self.rel_traj[:, 0:2] @ rot.T
        out[:, 2] = pb[2] + self.rel_traj[:, 2]
        out[:, 3] = _wrap(new_yaw + self.rel_traj[:, 3])
        out[:, 4] = self.rel_traj[:, 4]
        return out


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
