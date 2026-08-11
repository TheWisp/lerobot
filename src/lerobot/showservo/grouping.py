# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Teams: turning a cloud of tracked points into one trustworthy transform.

Two jobs, both classical (invariant 4 — neural nets stay in the image plane):

1. **Fit.** Each team's surviving points are fitted with one Sim2 against their
   taught positions, robustly. The FIT, not any individual point, is the durable
   quantity the servo consumes: points die constantly under occlusion, and a fit
   over the survivors keeps producing the same answer while at least a few remain.
2. **Fission / defission.** Whether the target is currently attached to the world or
   to the holder, decided by which hypothesis explains its motion. This is the grasp
   and release detector on a rig with no force sensing.

Correspondence is KNOWN here — the tracker preserves point identity, so index i of a
team is the same physical feature as taught index i. That is why this module fits
directly rather than going through the binder's feature matching; matching is only
needed to *establish* correspondence, and the binder already did that.

Only Sim2 is fitted, not a homography. A homography needs four points and, on the
~8 noisy points a team actually carries, spends its extra freedom modelling noise as
perspective. Planar similarity is what the fewshot work measured and what the last
centimetre of a servo needs; out-of-plane cases are honestly out of v0's scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lerobot.fewshot.registration import Sim2, fit_similarity


@dataclass
class TeamFit:
    """One team's taught->live transform plus the evidence for it.

    ``ok=False`` means the team lost too many points to fit at all. The servo must
    treat that as an abstention (invariant 5), never as identity.
    """

    ok: bool
    sim2: Sim2 = field(default_factory=Sim2.identity)
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rms: float = float("inf")

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def fit_team(
    taught_uv: np.ndarray,
    live_uv: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    inlier_px: float = 4.0,
    min_points: int = 3,
    allow_scale: bool = True,
    iters: int = 64,
    seed: int = 0,
) -> TeamFit:
    """Robust Sim2 with ``live ≈ fit.apply(taught)`` over index-matched points.

    Pre: ``taught_uv`` and ``live_uv`` are (N, 2) and index-corresponding; ``valid``
    is (N,) bool marking points the tracker still believes. Post: ``ok`` is True only
    when at least ``min_points`` inliers survive; ``residuals`` is (N,) with ``inf``
    at invalid points so a caller can rank eviction candidates without re-deriving.

    ``min_points`` defaults to 3, not the algebraic minimum of 2: two points always
    fit a Sim2 exactly, so a 2-point fit has zero residual and no way to be wrong out
    loud. Three is the smallest team that can produce a certificate.
    """
    taught = np.asarray(taught_uv, dtype=np.float64).reshape(-1, 2)
    live = np.asarray(live_uv, dtype=np.float64).reshape(-1, 2)
    assert taught.shape == live.shape, "teams must be index-corresponding"
    n = len(taught)
    valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(n)
    assert min_points >= 3, "a 2-point fit cannot produce a residual, so it cannot certify itself"

    idx = np.flatnonzero(valid)
    residuals = np.full(n, np.inf)
    if len(idx) < min_points:
        return TeamFit(ok=False, inliers=np.zeros(n, dtype=bool), residuals=residuals)

    src, dst = taught[idx], live[idx]
    rng = np.random.default_rng(seed)
    best_inl = None  # only the consensus SET is kept; the winning sample's fit is refitted below
    for _ in range(iters):
        i, j = rng.choice(len(idx), size=2, replace=False)
        if np.linalg.norm(src[i] - src[j]) < 1e-6:
            continue  # coincident taught points fix no rotation
        try:
            cand = fit_similarity(src[[i, j]], dst[[i, j]], allow_scale=allow_scale)
        except AssertionError:
            continue
        inl = np.linalg.norm(cand.apply(src) - dst, axis=1) < inlier_px
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl is None or best_inl.sum() < min_points:
        return TeamFit(ok=False, inliers=np.zeros(n, dtype=bool), residuals=residuals)

    sim = fit_similarity(src[best_inl], dst[best_inl], allow_scale=allow_scale)
    err = np.linalg.norm(sim.apply(src) - dst, axis=1)
    inl = err < inlier_px
    if inl.sum() >= min_points:  # one refit on the consensus set; more is noise-chasing
        sim = fit_similarity(src[inl], dst[inl], allow_scale=allow_scale)
        err = np.linalg.norm(sim.apply(src) - dst, axis=1)
        inl = err < inlier_px
    if inl.sum() < min_points:
        return TeamFit(ok=False, inliers=np.zeros(n, dtype=bool), residuals=residuals)

    residuals[idx] = err
    inliers = np.zeros(n, dtype=bool)
    inliers[idx[inl]] = True
    return TeamFit(
        ok=True,
        sim2=sim,
        inliers=inliers,
        residuals=residuals,
        rms=float(np.sqrt((err[inl] ** 2).mean())),
    )


def evict(fit: TeamFit, valid: np.ndarray, *, evict_px: float = 8.0) -> np.ndarray:
    """Indices whose residual condemns them. Post: (M,) int, a subset of valid points.

    Separate from ``inlier_px`` and deliberately looser: a point just outside the
    consensus band on one frame is usually noise and will come back, while a point at
    twice that distance has latched onto different texture and never will. Evicting
    on the tight threshold churns the team; evicting on this one prunes it.
    """
    if not fit.ok:
        return np.zeros(0, dtype=int)
    valid = np.asarray(valid, dtype=bool)
    return np.flatnonzero(valid & (fit.residuals > evict_px))


# --- fission / defission ---------------------------------------------------------

WORLD = "world"
HOLDER = "holder"


@dataclass
class AttachmentEvent:
    """A change in what the target is attached to, with the evidence that decided it."""

    kind: str  # "fission" | "defission"
    frame: int
    separation_px: float
    margin: float


class AttachmentMonitor:
    """Is the target moving with the world, or with the holder?

    Every frame two hypotheses predict where the target's points should now be:

    * **world** — they have not moved since the reference frame;
    * **holder** — they moved exactly as the holder team moved.

    Whichever predicts better wins, sustained over ``sustain`` frames. A transition
    world->holder is **fission** (the grasp took), holder->world is **defission**
    (release). No force sensor, no gripper current, no timing assumption.

    The critical guard is ``min_separation_px``: while the holder is still, the two
    hypotheses predict the SAME thing, and whichever wins is deciding on noise. Such
    frames are abstained on rather than voted, which is why a stationary gripper
    hovering over an object never reads as a grasp.
    """

    def __init__(
        self,
        *,
        sustain: int = 3,
        min_separation_px: float = 3.0,
        margin: float = 1.5,
    ):
        assert sustain >= 1 and min_separation_px > 0 and margin > 1.0
        self._sustain = sustain
        self._min_separation = float(min_separation_px)
        self._margin = float(margin)
        self.state = WORLD
        self._run = 0
        self._run_state: str | None = None
        self._ref_uv: np.ndarray | None = None
        self._frame = -1

    def reset(self, target_uv: np.ndarray, state: str = WORLD) -> None:
        """Pin the reference the world hypothesis is measured against.

        Pre: called at chapter start and after every re-bind — a stale reference makes
        the world hypothesis predict a position the object legitimately left.
        """
        assert state in (WORLD, HOLDER)
        self._ref_uv = np.asarray(target_uv, dtype=np.float64).reshape(-1, 2).copy()
        self.state = state
        self._run, self._run_state = 0, None

    def update(
        self,
        target_uv: np.ndarray,
        target_valid: np.ndarray,
        holder_sim2: Sim2 | None,
    ) -> AttachmentEvent | None:
        """Advance one frame. Post: an event on a sustained, well-separated flip.

        Pre: :meth:`reset` has been called; ``holder_sim2`` maps the holder team's
        REFERENCE-frame coordinates to now, or None when the holder's fit failed (in
        which case the frame is abstained on, not guessed).
        """
        assert self._ref_uv is not None, "update() before reset()"
        self._frame += 1
        if holder_sim2 is None:
            self._run, self._run_state = 0, None
            return None

        live = np.asarray(target_uv, dtype=np.float64).reshape(-1, 2)
        ok = np.asarray(target_valid, dtype=bool).reshape(-1)
        assert len(live) == len(self._ref_uv), "target team changed size without a reset()"
        if ok.sum() < 2:
            self._run, self._run_state = 0, None
            return None

        ref = self._ref_uv[ok]
        pred_world = ref
        pred_holder = holder_sim2.apply(ref)

        separation = float(np.linalg.norm(pred_world - pred_holder, axis=1).mean())
        if separation < self._min_separation:
            # The hypotheses are not distinguishable this frame. Abstaining keeps a
            # still gripper from voting; it does NOT reset the run, so a grasp made
            # during a brief pause is still detected once motion resumes.
            return None

        err_world = float(np.linalg.norm(live[ok] - pred_world, axis=1).mean())
        err_holder = float(np.linalg.norm(live[ok] - pred_holder, axis=1).mean())

        if err_holder * self._margin < err_world:
            winner = HOLDER
        elif err_world * self._margin < err_holder:
            winner = WORLD
        else:
            self._run, self._run_state = 0, None
            return None

        if winner != self._run_state:
            self._run, self._run_state = 1, winner
        else:
            self._run += 1
        if self._run < self._sustain or winner == self.state:
            return None

        kind = "fission" if winner == HOLDER else "defission"
        self.state = winner
        self._run, self._run_state = 0, None
        ratio = max(err_world, err_holder) / max(min(err_world, err_holder), 1e-6)
        return AttachmentEvent(kind=kind, frame=self._frame, separation_px=separation, margin=float(ratio))
