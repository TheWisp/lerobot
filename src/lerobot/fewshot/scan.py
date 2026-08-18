# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Continuous-scan examination: the SAM3 track as a lightweight mesh-equivalent.

A mesh's real job in pose estimation is DATA ASSOCIATION — certifying that views
from different angles are the same persistent object. SAM3's temporal track gives
that certification directly: the user clicks once, then moves and rotates the
object freely, and every frame of the scan is identity-certified by the tracker.

That certification enables the mechanism synthetic rotation only fakes:
SMALL-BASELINE CHAINING. Adjacent scan frames differ by a couple of degrees,
where feature registration is trivial; views far apart — which share too little
appearance to ever match directly — become connected by composing the chain of
easy registrations. Each promoted keyframe carries real self-occlusion, real
shading and real perspective, none of which warping one view can synthesise.

The output plugs into the SAME :class:`~lerobot.fewshot.templates.TemplateBank`
as everything else: a keyframe is a template whose ``to_source`` is its chained
transform back to the scan's first view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lerobot.fewshot.registration import Sim2, ransac_register
from lerobot.fewshot.templates import Template, TemplateBank


@dataclass
class ScanResult:
    """Outcome of chaining one scan.

    ``drift`` is the loop-closure residual: if the scan returns to (near) its
    starting appearance, the composed chain and a DIRECT first-vs-last
    registration should agree; their disagreement is the accumulated error of the
    chain, and the honest quality metric of the whole scan. ``None`` when the
    last keyframe no longer resembles the first (no closure measurable).
    """

    keyframes: list[Template] = field(default_factory=list)
    frames_used: int = 0
    broke_at: int | None = None  # frame index where the chain was lost, if it was
    drift_theta_deg: float | None = None
    drift_px: float | None = None
    _closure_err: Sim2 | None = None  # raw first->first residual, consumed by close_loop


def chain_scan(
    frames,
    masks,
    extractor,
    *,
    promote_below: float = 0.55,
    min_inliers: int = 10,
    patience: int = 8,
    ransac_kw: dict | None = None,
) -> ScanResult:
    """Chain a continuous examine scan into identity-linked keyframes.

    Pre: ``frames``/``masks`` are equal-length sequences from ONE SAM3 track (that
    is the identity certification — this function never re-establishes identity);
    an empty mask marks the object as not visible in that frame. Post: keyframe 0
    is the first usable frame with ``to_source`` = identity; every later keyframe's
    ``to_source`` maps its coordinates into keyframe 0's frame via the chain.

    Mechanism: each frame registers against the CURRENT keyframe. While inliers
    stay strong the frame is only remembered as "last good". When support decays
    below ``promote_below`` x the keyframe's own patch count — appearance has
    drifted about as far as direct matching can carry — the last good frame is
    promoted as the next keyframe, so consecutive keyframes always overlap well.
    ``patience`` consecutive unregistrable frames (occlusion, motion blur) break
    the chain rather than guessing across the gap.
    """
    assert len(frames) == len(masks) and len(frames) > 0
    ransac_kw = dict(ransac_kw or {})
    ransac_kw.setdefault("allow_scale", True)

    result = ScanResult()
    kf_coords = kf_feats = None
    kf_base_inliers: int | None = None  # this keyframe's first-edge support, the promotion yardstick
    kf_to_first = Sim2.identity()
    last_good: tuple[np.ndarray, np.ndarray, Sim2, int] | None = None  # coords, feats, kf->frame, frame idx
    misses = 0

    for i, (frame, mask) in enumerate(zip(frames, masks, strict=True)):
        if mask is None or not mask.any():
            misses += 1  # occluded: not evidence the chain is wrong, but not progress
            if misses > patience:
                result.broke_at = i
                break
            continue
        coords, feats = extractor.extract(frame, mask)
        if kf_coords is None:
            kf_coords, kf_feats = coords, feats
            result.keyframes.append(
                Template(view_id=f"scan_0@{i}", coords=coords, feats=feats, to_source=Sim2.identity())
            )
            result.frames_used += 1
            continue
        res = ransac_register(kf_coords, coords, kf_feats, feats, **ransac_kw)
        if not res.ok or res.n_inliers < min_inliers:
            misses += 1
            if misses > patience:
                result.broke_at = i
                break
            continue
        misses = 0
        result.frames_used += 1
        if kf_base_inliers is None:
            kf_base_inliers = res.n_inliers  # adjacent-frame support defines "strong" here
        strong = res.n_inliers >= promote_below * kf_base_inliers
        if strong:
            last_good = (coords, feats, res.sim2, i)
            continue
        # Appearance has drifted as far as direct matching carries: promote the last
        # frame that still matched WELL, so keyframe-to-keyframe overlap stays strong
        # (promoting the weak current frame would chain through a marginal edge).
        if last_good is None:
            last_good = (coords, feats, res.sim2, i)
        g_coords, g_feats, g_sim, g_idx = last_good
        kf_to_first = kf_to_first.compose(g_sim.inverse())  # new kf -> old kf -> first
        kf_coords, kf_feats = g_coords, g_feats
        kf_base_inliers = None  # next edge re-baselines against the new keyframe
        result.keyframes.append(
            Template(
                view_id=f"scan_{len(result.keyframes)}@{g_idx}",
                coords=g_coords,
                feats=g_feats,
                to_source=kf_to_first,
            )
        )
        last_good = None

    # Loop closure: does the composed chain agree with a direct first-vs-last match?
    if len(result.keyframes) >= 3:
        first, last = result.keyframes[0], result.keyframes[-1]
        direct = ransac_register(first.coords, last.coords, first.feats, last.feats, **ransac_kw)
        if direct.ok and direct.n_inliers >= min_inliers:
            # chain: last -> first; direct: first -> last, so compare with its inverse.
            err = last.to_source.compose(direct.sim2)  # should be ~identity
            result._closure_err = err
            result.drift_theta_deg = float(abs(np.degrees(err.theta)))
            centre = first.coords.mean(axis=0)
            result.drift_px = float(np.linalg.norm(err.apply(centre[None])[0] - centre))
    return result


def _fractional(sim: Sim2, f: float) -> Sim2:
    """``sim`` raised to a fractional power (first-order in translation) — valid for
    the SMALL corrections loop closure produces, and asserted small."""
    assert 0.0 <= f <= 1.0
    assert abs(sim.theta) < 0.8, "loop-closure correction should be small; refusing a large one"
    return Sim2.from_angle(sim.theta * f, t=sim.t * f, s=sim.s**f)


def close_loop(scan: ScanResult) -> ScanResult:
    """Distribute the loop-closure residual along the chain (pose-graph relaxation).

    Pre: ``scan.drift_theta_deg`` is not None (a closure was measured). Post: the
    LAST keyframe's ``to_source`` equals the directly measured transform exactly,
    intermediate keyframes are corrected proportionally, and the recorded drift is
    zeroed. This is the standard answer to chained error — including the
    systematic shrinkage of grid-quantised small rotations, which no amount of
    per-edge care removes: matches whose arc displacement is below half the patch
    pitch snap to zero displacement and bias every edge toward under-rotation.
    """
    assert scan.drift_theta_deg is not None, "no closure was measured — nothing to distribute"
    assert scan.keyframes and scan._closure_err is not None
    err_inv = scan._closure_err.inverse()
    n = len(scan.keyframes)
    corrected = []
    for k, kf in enumerate(scan.keyframes):
        corr = _fractional(err_inv, k / (n - 1))
        corrected.append(
            Template(
                view_id=kf.view_id, coords=kf.coords, feats=kf.feats, to_source=corr.compose(kf.to_source)
            )
        )
    scan.keyframes = corrected
    scan.drift_theta_deg = 0.0
    scan.drift_px = 0.0
    scan._closure_err = None
    return scan


def bank_from_scan(extractor, scan: ScanResult) -> TemplateBank:
    """A match-ready bank whose templates are the scan's REAL keyframes.

    Pre: at least one keyframe. Every template's ``to_source`` leads to the scan's
    first view, so a match against ANY keyframe yields a transform to the same
    reference observation — exactly the contract synthetic rotations satisfied,
    now with genuinely observed appearance.
    """
    assert scan.keyframes, "scan produced no keyframes"
    bank = TemplateBank(extractor)
    bank.templates.extend(scan.keyframes)
    return bank
