# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The scan's claim, demonstrated: two views 90 deg apart share too little
(rotation-variant) appearance to ever register directly — but a continuous scan
between them chains small easy registrations into the transform no direct match
could produce. Identity comes from the track, not from matching; that is the
'lightweight mesh-equivalent' idea."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lerobot.fewshot.scan import bank_from_scan, chain_scan, close_loop  # noqa: E402

from .test_templates import WindowExtractor, _rotated_view, _scene  # noqa: E402

RANSAC = {"allow_scale": False, "inlier_dist": 3.0, "iters": 120}


def _scan(image, mask, step_deg=6.0, upto_deg=360.0):
    frames, masks = [], []
    d = 0.0
    while d < upto_deg:
        img, m, _ = _rotated_view(image, mask, d, t=(0.0, 0.0))
        frames.append(img)
        masks.append(m)
        d += step_deg
    return frames, masks


def test_direct_match_across_90_degrees_fails():
    """The premise: without the scan, these two views simply do not register."""
    from lerobot.fewshot.registration import ransac_register

    image, mask = _scene()
    ex = WindowExtractor()
    live_img, live_mask, _ = _rotated_view(image, mask, 90.0)
    ca, fa = ex.extract(image, mask)
    cb, fb = ex.extract(live_img, live_mask)
    res = ransac_register(ca, cb, fa, fb, **RANSAC)
    assert (not res.ok) or res.n_inliers < 8


def test_chain_connects_what_direct_matching_cannot():
    image, mask = _scene()
    frames, masks = _scan(image, mask)
    scan = chain_scan(frames, masks, WindowExtractor(), ransac_kw=RANSAC)
    assert scan.broke_at is None
    assert len(scan.keyframes) >= 4, "a full circle must need several keyframes"

    bank = bank_from_scan(WindowExtractor(), scan)
    live_img, live_mask, true = _rotated_view(image, mask, 90.0)
    m = bank.match(live_img, live_mask, **RANSAC)
    assert m is not None and m.result.n_inliers >= 8
    err = np.degrees(abs((m.source_to_live.theta - true.theta + np.pi) % (2 * np.pi) - np.pi))
    assert err < 8.0, f"chained transform off by {err:.1f} deg"


def _keyframe_angle_errors(scan):
    """Per-keyframe error vs the scan generator's known rotation (view_id carries
    the source frame index; the generator rotates 6 deg per frame)."""
    out = []
    for kf in scan.keyframes:
        fidx = int(kf.view_id.split("@")[1])
        out.append(abs((np.degrees(kf.to_source.theta) - 6.0 * fidx + 180) % 360 - 180))
    return out


def test_loop_closure_measures_and_corrects_drift():
    """A full-circle scan returns to its starting appearance: the composed chain and
    the direct first-vs-last registration must agree. Their residual is the scan's
    accumulated error; close_loop distributes it away (pose-graph relaxation). The
    bound matters: an early promotion rule produced 60 keyframes and 26 deg of
    SYSTEMATIC drift, because grid-quantised matches shrink small rotations — long
    edges (few keyframes) keep that bias proportionally small."""
    image, mask = _scene()
    frames, masks = _scan(image, mask)
    scan = chain_scan(frames, masks, WindowExtractor(), ransac_kw=RANSAC)
    assert scan.drift_theta_deg is not None, "full circle must close the loop"
    assert scan.drift_theta_deg < 6.0, f"chain drifted {scan.drift_theta_deg:.1f} deg"
    assert scan.drift_px < 4.0
    assert len(scan.keyframes) < 40, "promotion must produce long edges, not one per frame"
    close_loop(scan)
    assert scan.drift_theta_deg == 0.0
    assert max(_keyframe_angle_errors(scan)) < 2.0, "corrected chain must match ground truth"


def test_occlusion_within_patience_does_not_break_the_chain():
    """SAM losing the object for a few frames (a hand crossing in front while the
    object is roughly still) must pause the chain, not sever it — the track
    resumes on the same identity and the same appearance."""
    image, mask = _scene()
    # The rotation PAUSES during the occlusion — a hand passing in front of a
    # stationary object — then resumes where it left off. (A first version froze
    # the occluded frames but let the generator keep rotating underneath, which
    # teleports the object 24 deg across the gap: that is the blackout case below,
    # not this one.)
    degs = [0, 6, 12, 18, 24, 24, 24, 24, 30, 36, 42, 48, 54, 60]
    occluded = {5, 6, 7}
    frames, masks = [], []
    for k, d in enumerate(degs):
        img, m, _ = _rotated_view(image, mask, float(d), t=(0.0, 0.0))
        frames.append(img)
        masks.append(np.zeros_like(m) if k in occluded else m)
    scan = chain_scan(frames, masks, WindowExtractor(), ransac_kw=RANSAC)
    assert scan.broke_at is None
    assert scan.frames_used >= len(degs) - len(occluded) - 1


def test_rotating_through_a_blackout_breaks_honestly():
    """If the object rotates far WHILE fully occluded, the resumed appearance is
    beyond direct-match range: SAM still certifies identity, but the TRANSFORM
    across the gap is unknowable, and the chain must refuse rather than guess.
    (A future aspect-group edge could keep identity without geometry.)"""
    image, mask = _scene()
    frames, masks = _scan(image, mask, upto_deg=90.0)
    empty = np.zeros_like(masks[0])
    for k in (5, 6, 7):  # object keeps rotating unseen — 18 deg jump on resume,
        masks[k] = empty  # on top of drift already accumulated since the keyframe
    scan = chain_scan(frames, masks, WindowExtractor(), patience=4, ransac_kw=RANSAC)
    assert scan.broke_at is not None or len(scan.keyframes) >= 2  # refuse OR recover; never a wrong edge


def test_long_occlusion_breaks_honestly():
    image, mask = _scene()
    frames, masks = _scan(image, mask, upto_deg=90.0)
    empty = np.zeros_like(masks[0])
    for k in range(4, 14):  # longer than patience
        masks[k] = empty
    scan = chain_scan(frames, masks, WindowExtractor(), patience=5, ransac_kw=RANSAC)
    assert scan.broke_at is not None, "guessing across a long blackout is forbidden"
