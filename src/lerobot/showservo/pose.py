# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Rigid motion between two OBSERVATIONS of an unknown object.

Nothing here knows what an object is. There is no model, no canonical frame, no
"object pose" — only: these tracked points were somewhere in 3D at demo time, they are
somewhere else now, and Kabsch recovers the rigid motion that took one set to the
other. A target that is a surface patch of a machine too large to fit in frame is
handled identically to a cube, because neither is ever identified.

That distinction is the whole premise. Registering against a scanned mesh or a CAD
model would be ground truth smuggled in, and would collapse on exactly the
patch-of-something-larger case.

Depth comes from the rig's RealSense, so 3D is a lift rather than a reconstruction —
the correspondences the tracker already produces, deprojected. Two properties this
buys over the image-plane similarity it replaces:

* **No coplanarity requirement.** The held end may sit at any depth relative to the
  target; the old 2D transport was wrong by ``|d|·(z_t − z_h)/(H − z_t)`` and this is
  not.
* **No degeneracy on flat objects.** Coplanar points break the essential matrix, which
  is the trap waiting on the monocular path. Kabsch on lifted points does not care.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Rigid3:
    """SE(3): ``x -> R @ x + t``. Pre: ``R`` is a proper rotation (det=+1)."""

    rot: np.ndarray  # (3, 3)
    trans: np.ndarray  # (3,)

    def __post_init__(self):
        assert self.rot.shape == (3, 3) and self.trans.shape == (3,)
        assert abs(float(np.linalg.det(self.rot)) - 1.0) < 1e-5, "R must be a proper rotation"

    @classmethod
    def identity(cls) -> Rigid3:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_rotvec(cls, rotvec, t=(0.0, 0.0, 0.0)) -> Rigid3:
        return cls(rotation_matrix(rotvec), np.asarray(t, dtype=np.float64))

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Pre: pts is (N, 3). Post: (N, 3)."""
        pts = np.asarray(pts, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 3
        return pts @ self.rot.T + self.trans

    def compose(self, other: Rigid3) -> Rigid3:
        """``self.compose(other)`` maps x through ``other`` first."""
        return Rigid3(self.rot @ other.rot, self.rot @ other.trans + self.trans)

    def inverse(self) -> Rigid3:
        rt = self.rot.T
        return Rigid3(rt, -rt @ self.trans)

    @property
    def rotvec(self) -> np.ndarray:
        return rotation_vector(self.rot)

    @property
    def angle(self) -> float:
        return float(np.linalg.norm(self.rotvec))


def rotation_matrix(rotvec) -> np.ndarray:
    """Rodrigues. Pre: ``rotvec`` is (3,) axis-angle. Post: proper rotation matrix."""
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def rotation_vector(rot: np.ndarray) -> np.ndarray:
    """Inverse Rodrigues, via the quaternion. Post: (3,) with norm in [0, pi].

    Extracting the quaternion first, pivoting on the largest component, rather than
    inverting Rodrigues directly is what makes this stable at BOTH ends. The
    direct formula divides by sin(theta), which vanishes at a half turn — and patching
    that with a near-pi special case merely moves the precision loss to the seam
    between the branches, where a 179.99 degree re-orientation still comes out wrong.
    The quaternion route has no seam.
    """
    rot = np.asarray(rot, dtype=np.float64)
    assert rot.shape == (3, 3)
    tr = float(np.trace(rot))

    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        v = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / s
    else:
        i = int(np.argmax(np.diag(rot)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k]) * 2.0
        qw = (rot[k, j] - rot[j, k]) / s
        v = np.zeros(3)
        v[i] = 0.25 * s
        v[j] = (rot[j, i] + rot[i, j]) / s
        v[k] = (rot[k, i] + rot[i, k]) / s

    if qw < 0.0:  # keep the half turn in [0, pi] rather than reporting its negative
        qw, v = -qw, -v
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.zeros(3)
    return v * (2.0 * np.arctan2(n, qw) / n)


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics. Pre: focal lengths positive, principal point inside the image."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        assert self.fx > 0 and self.fy > 0

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """Post: (N, 2) pixels. Pre: all points strictly in front of the camera."""
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        assert (xyz[:, 2] > 1e-9).all(), "a point behind the camera has no projection"
        return np.stack(
            [self.fx * xyz[:, 0] / xyz[:, 2] + self.cx, self.fy * xyz[:, 1] / xyz[:, 2] + self.cy],
            axis=1,
        )

    def deproject(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pixels + depth -> camera-frame 3D. Post: (N, 3)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        assert len(uv) == len(z)
        return np.stack([(uv[:, 0] - self.cx) * z / self.fx, (uv[:, 1] - self.cy) * z / self.fy, z], axis=1)


def sample_depth(
    depth_m: np.ndarray, uv: np.ndarray, *, z_min: float = 0.05, z_max: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Read depth at tracked pixels. Post: ``(z (N,), valid (N,) bool)``.

    Pre: ``depth_m`` is a HxW map in METRES with 0 marking "no return", which is what a
    RealSense produces on dark, specular or out-of-range surfaces.

    Nearest-neighbour on purpose. Bilinear interpolation across a depth discontinuity
    averages foreground and background into a distance where no surface exists — the
    classic flying pixel — and a flying pixel on the object's silhouette is precisely
    the sample a tracked corner tends to land on.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    assert depth_m.ndim == 2, "depth must be a single-channel map"
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)

    h, w = depth_m.shape
    col = np.rint(uv[:, 0]).astype(int)
    row = np.rint(uv[:, 1]).astype(int)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)

    z = np.zeros(len(uv))
    z[inside] = depth_m[row[inside], col[inside]]
    return z, inside & (z > z_min) & (z < z_max)


def fit_rigid(src: np.ndarray, dst: np.ndarray, *, estimate_scale: bool = False):
    """Kabsch: least-squares ``dst ≈ R @ src + t``. Post: ``(Rigid3, scale)``.

    Pre: index-corresponding (N, 3) sets with N >= 3 and ``src`` not collinear.

    ``estimate_scale`` is a DIAGNOSTIC, not a modelling choice. A rigid object cannot
    change size, so a fitted scale materially different from 1 means the depth is
    wrong (flying pixels, a mis-scaled depth unit) or the correspondences are — a
    cheap certificate that catches a whole class of silent depth faults.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    assert src.shape == dst.shape and len(src) >= 3, "a rigid fit needs 3+ matched points"

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    x, y = src - mu_s, dst - mu_d
    assert float((x**2).sum()) > 1e-15, "source points are degenerate"

    u, _, vt = np.linalg.svd(x.T @ y)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T  # det=+1: a reflection must never win

    rotated = x @ rot.T
    scale = float((y * rotated).sum() / max((rotated**2).sum(), 1e-15)) if estimate_scale else 1.0
    # The returned transform is ALWAYS rigid: scale is reported, never applied. Folding
    # it into the translation while ``Rigid3.apply`` ignores it makes the two disagree
    # by (scale-1)·|centroid| — a few millimetres at half a metre, which lands right on
    # the inlier threshold and makes RANSAC lose consensus sets it should have kept.
    return Rigid3(rot, mu_d - rot @ mu_s), scale


@dataclass
class RigidFit:
    """A team's 3D motion plus the evidence for it. ``ok=False`` is an abstention."""

    ok: bool
    transform: Rigid3 = field(default_factory=Rigid3.identity)
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rms: float = float("inf")
    scale: float = 1.0

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())

    def scale_is_plausible(self, tol: float = 0.1) -> bool:
        """A rigid object keeps its size; a fitted scale that does not is a depth fault."""
        return abs(self.scale - 1.0) <= tol


def ransac_fit_rigid(
    src: np.ndarray,
    dst: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    inlier_m: float = 0.006,
    min_points: int = 4,
    iters: int = 128,
    seed: int = 0,
) -> RigidFit:
    """Robust Kabsch over index-matched 3D points. Post: never raises; abstains instead.

    Pre: ``src``/``dst`` are (N, 3) in the SAME frame convention (camera frame here),
    ``valid`` marks points with usable depth AND a live track.

    ``min_points`` is 4, one above the algebraic minimum of 3: three points fit a rigid
    transform exactly, leaving no residual and therefore no way for the fit to be
    caught being wrong. ``inlier_m`` defaults to 6 mm, roughly RealSense noise at
    half a metre — tighter than that rejects honest points.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    assert src.shape == dst.shape, "teams must be index-corresponding"
    n = len(src)
    valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(n)
    assert min_points >= 4, "a 3-point fit cannot produce a residual, so it cannot certify itself"

    idx = np.flatnonzero(valid)
    residuals = np.full(n, np.inf)
    if len(idx) < min_points:
        return RigidFit(ok=False, inliers=np.zeros(n, dtype=bool), residuals=residuals)

    s, d = src[idx], dst[idx]
    rng = np.random.default_rng(seed)
    best_inl = None
    for _ in range(iters):
        pick = rng.choice(len(idx), size=3, replace=False)
        try:
            cand, _ = fit_rigid(s[pick], d[pick])
        except AssertionError:
            continue
        inl = np.linalg.norm(cand.apply(s) - d, axis=1) < inlier_m
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl is None or best_inl.sum() < min_points:
        return RigidFit(ok=False, inliers=np.zeros(n, dtype=bool), residuals=residuals)

    transform, scale = fit_rigid(s[best_inl], d[best_inl], estimate_scale=True)
    err = np.linalg.norm(transform.apply(s) - d, axis=1)
    inl = err < inlier_m
    if inl.sum() >= min_points:  # one refit on the consensus set
        transform, scale = fit_rigid(s[inl], d[inl], estimate_scale=True)
        err = np.linalg.norm(transform.apply(s) - d, axis=1)
        inl = err < inlier_m
    if inl.sum() < min_points:
        return RigidFit(ok=False, inliers=np.zeros(n, dtype=bool), residuals=residuals)

    residuals[idx] = err
    inliers = np.zeros(n, dtype=bool)
    inliers[idx[inl]] = True
    return RigidFit(
        ok=True,
        transform=transform,
        inliers=inliers,
        residuals=residuals,
        rms=float(np.sqrt((err[inl] ** 2).mean())),
        scale=scale,
    )
