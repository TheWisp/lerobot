# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Registration between two OBSERVATIONS of the same object — the pose-free core.

Nothing here knows what a pose is. The system never estimates where an object is in
the world; it estimates the similarity transform between two *views* of it (demo vs
live, template vs live), from feature correspondences inside the SAM mask. Composed
with the demo's end-effector pose, that transform is all placement transfer needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Sim2:
    """2D similarity transform ``x -> s * R @ x + t``.

    Pre: ``R`` is a proper rotation (det=+1, orthonormal), ``s > 0``. Composition is
    ``(A @ B)(x) == A(B(x))``.
    """

    s: float
    R: np.ndarray  # (2, 2)
    t: np.ndarray  # (2,)

    def __post_init__(self):
        assert self.R.shape == (2, 2) and self.t.shape == (2,)
        assert self.s > 0, f"scale must be positive, got {self.s}"
        assert abs(float(np.linalg.det(self.R)) - 1.0) < 1e-5, "R must be a proper rotation"

    @classmethod
    def identity(cls) -> Sim2:
        return cls(1.0, np.eye(2), np.zeros(2))

    @classmethod
    def from_angle(cls, theta: float, t=(0.0, 0.0), s: float = 1.0) -> Sim2:
        c, si = np.cos(theta), np.sin(theta)
        return cls(float(s), np.array([[c, -si], [si, c]]), np.asarray(t, dtype=np.float64))

    @property
    def theta(self) -> float:
        """Rotation angle in radians, in (-pi, pi]."""
        return float(np.arctan2(self.R[1, 0], self.R[0, 0]))

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Pre: pts is (N, 2). Post: (N, 2)."""
        pts = np.asarray(pts, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 2
        return self.s * pts @ self.R.T + self.t

    def compose(self, other: Sim2) -> Sim2:
        """``self.compose(other)`` maps x through ``other`` first: (self∘other)(x)."""
        return Sim2(self.s * other.s, self.R @ other.R, self.s * self.R @ other.t + self.t)

    def inverse(self) -> Sim2:
        rot_inv = self.R.T
        return Sim2(1.0 / self.s, rot_inv, -(1.0 / self.s) * rot_inv @ self.t)


def fit_similarity(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True) -> Sim2:
    """Least-squares Sim2 with ``dst ≈ sim.apply(src)`` (Umeyama).

    Pre: matched points, both (N, 2) with N >= 2 and src not all identical.
    Post: exact for noise-free similarity-related inputs.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[0] >= 2 and src.shape[1] == 2
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xc, yc = src - mu_s, dst - mu_d
    var_src = float((xc**2).sum()) / len(src)
    assert var_src > 1e-12, "source points are degenerate (all identical)"
    cov = yc.T @ xc / len(src)
    u, sv, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt)) or 1.0
    fix = np.diag([1.0, sign])  # forces det=+1: a reflection must never fit better
    rot = u @ fix @ vt
    s = float(np.trace(fix @ np.diag(sv)) / var_src) if allow_scale else 1.0
    s = max(s, 1e-9)
    t = mu_d - s * rot @ mu_s
    return Sim2(s, rot, t)


def mutual_matches(
    feats_a: np.ndarray, feats_b: np.ndarray, ratio: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest neighbours in cosine similarity, optional Lowe-style ratio test.

    Pre: L2-normalised feature rows, (Na, D) and (Nb, D). Post: index arrays
    (M,), (M,) with M <= min(Na, Nb); may be empty for unrelated inputs.
    The ratio test is OFF by default (ratio=1.0): on smooth or overlapping-patch
    features the second-best is legitimately close to the best, and a hard ratio
    kills every valid match — identity registration returned zero matches at 0.95.
    Geometric outlier rejection is RANSAC's job; enable the ratio only for scenes
    with repetitive texture, where geometry cannot disambiguate.
    """
    assert feats_a.ndim == 2 and feats_b.ndim == 2 and feats_a.shape[1] == feats_b.shape[1]
    sim = feats_a @ feats_b.T
    nn_ab = sim.argmax(axis=1)
    nn_ba = sim.argmax(axis=0)
    ia = np.arange(len(feats_a))
    mutual = nn_ba[nn_ab] == ia
    if ratio < 1.0 and sim.shape[1] >= 2:
        part = np.partition(sim, -2, axis=1)
        best, second = part[:, -1], part[:, -2]
        # second may be negative; the test only makes sense for positive best.
        distinct = best > 0
        distinct &= second < ratio * best
        mutual &= distinct
    return ia[mutual], nn_ab[mutual]


@dataclass
class RegistrationResult:
    """Outcome of registering observation A onto observation B.

    ``sim2`` maps A-coordinates into B-coordinates. Confidence is carried, not
    hidden: ``inlier_ratio`` is the primary trust signal (texture-poor or symmetric
    objects show up here), and ``rotation_needed`` is False when a translation-only
    fit explains the inliers as well as the full fit — for a symmetric object the
    recovered angle is then arbitrary and a caller should not transfer it.
    """

    ok: bool
    sim2: Sim2 = field(default_factory=Sim2.identity)
    n_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    rms: float = float("inf")
    translation_rms: float = float("inf")
    rotation_needed: bool = False

    @property
    def rotation_trustworthy(self) -> bool:
        """Transfer the angle only when it was both needed and well-supported."""
        return self.ok and self.rotation_needed and self.inlier_ratio >= 0.25 and self.n_inliers >= 6


def ransac_register(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    *,
    allow_scale: bool = True,
    inlier_dist: float = 6.0,
    iters: int = 300,
    seed: int = 0,
    ratio: float = 1.0,
) -> RegistrationResult:
    """Register A onto B from masked patch features. Pure function of its inputs.

    Pre: ``pts_*`` are (N, 2) patch coordinates in each observation's own frame
    (pixels, or metric table coordinates if pre-rectified); ``feats_*`` are the
    L2-normalised features at those points. Post: ``result.sim2.apply(pts_a[i]) ≈
    pts_b[j]`` for inlier matches; ``ok=False`` (never an exception) when there are
    too few matches to fit.
    """
    ia, ib = mutual_matches(feats_a, feats_b, ratio=ratio)
    n = len(ia)
    if n < 2:
        return RegistrationResult(ok=False, n_matches=n)
    src, dst = np.asarray(pts_a, float)[ia], np.asarray(pts_b, float)[ib]

    rng = np.random.default_rng(seed)
    best_inl: np.ndarray | None = None
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        if np.linalg.norm(src[i] - src[j]) < 1e-6:
            continue  # degenerate pair: two coincident points fix no rotation
        try:
            cand = fit_similarity(src[[i, j]], dst[[i, j]], allow_scale=allow_scale)
        except AssertionError:
            continue
        err = np.linalg.norm(cand.apply(src) - dst, axis=1)
        inl = err < inlier_dist
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl is None or best_inl.sum() < 2:
        return RegistrationResult(ok=False, n_matches=n)

    sim = fit_similarity(src[best_inl], dst[best_inl], allow_scale=allow_scale)
    err = np.linalg.norm(sim.apply(src) - dst, axis=1)
    inl = err < inlier_dist
    if inl.sum() >= 2:
        sim = fit_similarity(src[inl], dst[inl], allow_scale=allow_scale)
        err = np.linalg.norm(sim.apply(src) - dst, axis=1)
        inl = err < inlier_dist
    rms = float(np.sqrt((err[inl] ** 2).mean())) if inl.any() else float("inf")

    # Would translation alone have explained the same matches? If yes, the angle is
    # not evidenced by the data (pure translation, or a symmetric object) and must
    # not be transferred as if it were.
    t_only = dst[inl].mean(axis=0) - src[inl].mean(axis=0)
    t_err = np.linalg.norm(src[inl] + t_only - dst[inl], axis=1)
    t_rms = float(np.sqrt((t_err**2).mean())) if inl.any() else float("inf")
    rotation_needed = t_rms > max(1.5 * rms, 1.0)

    return RegistrationResult(
        ok=True,
        sim2=sim,
        n_matches=n,
        n_inliers=int(inl.sum()),
        inlier_ratio=float(inl.sum()) / n,
        rms=rms,
        translation_rms=t_rms,
        rotation_needed=rotation_needed,
    )
