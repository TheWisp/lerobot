# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Binding: demo -> live correspondence, once per stage, with a certificate.

"Semantics once per stage, geometry every frame" (invariant 4). This module is the
once-per-stage half: it answers *which live pixel is taught point i*, hands the
answer to the tracker, and then gets out of the fast loop. A ~100 ms budget is
affordable here precisely because it is not per-frame.

Two tiers behind one call:

* :class:`SiftBinder` — v0 default. Same-instance binding for D1-D3, which bind to
  the physical objects we demoed. Fully classical, no GPU, so it never contends with
  the overlay workers for the card.
* :class:`DinoBinder` — the cross-instance tier, delegating to the registration core
  that ``lerobot/fewshot`` already measured (0-1.7 deg over a full rotation sweep with
  an examined bank). Needs a mask, which is SAM3's job as the designator.

Both return the same certificate, and both may ABSTAIN. Abstention is the point: a
binder that always answers turns a mis-bind into a confident wrong motion, and the
fewshot rotation sweep showed exactly that failure carrying 17-38 inliers and a
passing per-registration trust check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lerobot.fewshot.registration import Sim2, mutual_matches
from lerobot.showservo.card import Stage
from lerobot.showservo.grouping import fit_team


@dataclass
class BindResult:
    """Correspondence for one team, plus the evidence that it is real.

    ``sim2`` maps TAUGHT pixels to live pixels, which is what lets
    :meth:`seed_points` place taught points the matcher never found — a taught point
    hidden behind the gripper at bind time still gets a plausible seed, flagged so the
    caller knows it was predicted rather than seen.
    """

    ok: bool
    sim2: Sim2 = field(default_factory=Sim2.identity)
    taught_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    live_uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    n_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    rms: float = float("inf")
    reason: str = ""

    def seed_points(self, taught_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Live seed for every taught point. Post: ``(uv (N,2), measured (N,) bool)``.

        Pre: ``ok`` is True. Measured points sit where the binder found them; the rest
        are transported through ``sim2``.

        **Never lift an unmeasured seed to 3D.** ``sim2`` is a 2D similarity, so a
        transported point is a planar guess at where a taught feature went; sampling
        depth there reads whatever surface happens to lie at a fabricated pixel. Fed to
        a rigid fit that produces a cloud the wrong size — 20% scale error, measured —
        and the servo then abstains on nearly every frame. Callers doing 3D must filter
        on ``measured``, which is why it is returned rather than left implicit.
        """
        assert self.ok, "seeding from a failed bind would fabricate correspondence"
        taught = np.asarray(taught_uv, dtype=np.float64).reshape(-1, 2)
        uv = self.sim2.apply(taught)
        measured = np.zeros(len(taught), dtype=bool)
        if len(self.taught_idx):
            uv[self.taught_idx] = self.live_uv
            measured[self.taught_idx] = True
        return uv, measured


@dataclass
class BindGate:
    """The certificate threshold. Below it the binder abstains and the ladder runs.

    Defaults are deliberately conservative for v0: an abstention costs one rung of the
    retry ladder, a false bind costs a collision.
    """

    min_inliers: int = 6
    min_inlier_ratio: float = 0.25
    max_rms_px: float = 6.0

    def passes(self, n_inliers: int, ratio: float, rms: float) -> bool:
        return n_inliers >= self.min_inliers and ratio >= self.min_inlier_ratio and rms <= self.max_rms_px


def _import_cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError("the SIFT binder needs opencv-python (`uv sync --extra all`)") from e
    return cv2


def sift_keypoints(
    frame: np.ndarray, mask: np.ndarray | None = None, *, max_points: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Detect AND describe in one pass. Post: ``(uv (N, 2), desc (N, 128))`` normalised.

    Pre: ``mask``, when given, confines detection to the team's region (§4 extracts
    keypoints on the SAM3 target mask). ``max_points`` of 0 means unlimited; any other
    value keeps the strongest responses WITHIN THE MASK, which is the
    "self-distinctive" filter §4 asks for, applied by contrast rather than by hand.

    The retention is done here rather than through OpenCV's ``nfeatures``, because
    that parameter is applied to the whole image BEFORE the mask filter: asking for
    the best 40 features returns the best 40 in the frame and then discards those
    outside the region, which yields ZERO points on any object that is not the most
    textured thing in view. A small marker beside a busy background starves silently.

    The compiler and the binder must BOTH come through here. A SIFT descriptor is
    defined relative to the scale and dominant orientation the detector assigned it,
    so descriptors computed at externally chosen pixel coordinates do not live in the
    same space as descriptors from SIFT's own detections — matching across the two
    produces zero mutual matches, which is exactly how this was found.
    """
    cv2 = _import_cv2()
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    cv_mask = None if mask is None else (np.asarray(mask) > 0).astype(np.uint8) * 255

    kps, desc = cv2.SIFT_create().detectAndCompute(gray, cv_mask)
    if desc is None or len(kps) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 128), dtype=np.float32)

    uv = np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float64)
    desc = np.asarray(desc, dtype=np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-9

    if max_points and len(kps) > max_points:
        strongest = np.argsort([-k.response for k in kps])[: int(max_points)]
        uv, desc = uv[strongest], desc[strongest]
    return uv, desc


class SiftBinder:
    """SIFT + mutual-NN + RANSAC. Pre: cards whose descriptors are SIFT descriptors."""

    def __init__(self, *, gate: BindGate | None = None, ratio: float = 0.9, inlier_px: float = 6.0):
        _import_cv2()
        self.gate = gate or BindGate()
        self.ratio = float(ratio)
        self.inlier_px = float(inlier_px)

    def bind(
        self, frame: np.ndarray, stage: Stage, team: str = "target", mask: np.ndarray | None = None
    ) -> BindResult:
        """Locate ``team``'s taught constellation in ``frame``.

        Pre: the stage's team carries descriptors (a card compiled without them can
        only be tracked, never re-bound). Post: ``ok`` is True only if the certificate
        gate passed; ``reason`` explains every abstention.
        """
        taught_uv = stage.team_uv(team)
        taught_desc = stage.team_descriptors(team)
        if len(taught_uv) == 0:
            return BindResult(ok=False, reason=f"stage has no {team} team")
        if taught_desc is None:
            return BindResult(ok=False, reason=f"{team} team has no descriptors to bind with")

        live_xy, desc = sift_keypoints(frame, mask)
        if len(live_xy) < 2:
            return BindResult(ok=False, reason="no SIFT features in the live frame")

        ia, ib = mutual_matches(taught_desc, desc, ratio=self.ratio)
        if len(ia) < 3:
            return BindResult(ok=False, n_matches=len(ia), reason="too few mutual matches")

        fit = fit_team(taught_uv[ia], live_xy[ib], inlier_px=self.inlier_px)
        if not fit.ok:
            return BindResult(ok=False, n_matches=len(ia), reason="no consensus transform")

        ratio = fit.n_inliers / max(len(ia), 1)
        ok = self.gate.passes(fit.n_inliers, ratio, fit.rms)
        inl = fit.inliers
        return BindResult(
            ok=ok,
            sim2=fit.sim2,
            taught_idx=ia[inl],
            live_uv=live_xy[ib][inl],
            n_matches=len(ia),
            n_inliers=fit.n_inliers,
            inlier_ratio=float(ratio),
            rms=fit.rms,
            reason="" if ok else "below certificate gate",
        )


class DinoBinder:
    """Dense-feature binding for cards whose descriptors are ViT patch features.

    Pre: a mask for the region to bind (SAM3 is the designator) and a card compiled in
    the same descriptor space. This is the cross-instance tier; for same-instance v0
    work :class:`SiftBinder` is cheaper and needs no GPU.
    """

    def __init__(self, extractor, *, gate: BindGate | None = None, inlier_px: float = 6.0):
        assert hasattr(extractor, "extract"), "extractor must expose extract(image, mask)"
        self.extractor = extractor
        self.gate = gate or BindGate()
        self.inlier_px = float(inlier_px)

    def bind(
        self, frame: np.ndarray, stage: Stage, team: str = "target", mask: np.ndarray | None = None
    ) -> BindResult:
        """Post: same certificate contract as :class:`SiftBinder`."""
        taught_uv = stage.team_uv(team)
        taught_desc = stage.team_descriptors(team)
        if len(taught_uv) == 0 or taught_desc is None:
            return BindResult(ok=False, reason=f"{team} team is unbindable (no points/descriptors)")
        if mask is None or not np.asarray(mask).any():
            return BindResult(ok=False, reason="dense binding needs a designation mask")

        live_xy, live_desc = self.extractor.extract(frame, np.asarray(mask, dtype=bool))
        if len(live_xy) < 3:
            return BindResult(ok=False, reason="mask too small for the patch grid")

        ia, ib = mutual_matches(taught_desc, np.asarray(live_desc, dtype=np.float32))
        if len(ia) < 3:
            return BindResult(ok=False, n_matches=len(ia), reason="too few mutual matches")

        fit = fit_team(taught_uv[ia], live_xy[ib], inlier_px=self.inlier_px)
        if not fit.ok:
            return BindResult(ok=False, n_matches=len(ia), reason="no consensus transform")

        ratio = fit.n_inliers / max(len(ia), 1)
        ok = self.gate.passes(fit.n_inliers, ratio, fit.rms)
        inl = fit.inliers
        return BindResult(
            ok=ok,
            sim2=fit.sim2,
            taught_idx=ia[inl],
            live_uv=np.asarray(live_xy)[ib][inl],
            n_matches=len(ia),
            n_inliers=fit.n_inliers,
            inlier_ratio=float(ratio),
            rms=fit.rms,
            reason="" if ok else "below certificate gate",
        )
