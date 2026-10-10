# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Team fitting must be exact on clean data and unmoved by a minority of liars; the
attachment monitor must call the grasp only when the evidence could have said no."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.fewshot.registration import Sim2
from lerobot.showservo.grouping import HOLDER, WORLD, AttachmentMonitor, evict, fit_team

TAUGHT = np.array([[10.0, 10.0], [50.0, 12.0], [48.0, 60.0], [8.0, 58.0], [30.0, 35.0]])


def test_fit_recovers_a_clean_transform_exactly():
    true = Sim2.from_angle(0.4, t=(25.0, -12.0), s=1.15)
    fit = fit_team(TAUGHT, true.apply(TAUGHT))
    assert fit.ok and fit.n_inliers == len(TAUGHT)
    np.testing.assert_allclose(fit.sim2.apply(TAUGHT), true.apply(TAUGHT), atol=1e-6)
    assert fit.rms < 1e-6


def test_fit_survives_a_minority_of_latched_points():
    # Two of seven points have latched onto other texture — the classic KLT failure
    # that survives the status flag. The fit must come from the honest five.
    true = Sim2.from_angle(-0.3, t=(40.0, 5.0), s=1.0)
    taught = np.vstack([TAUGHT, [[70.0, 20.0], [20.0, 70.0]]])
    live = true.apply(taught)
    live[5] += [60.0, -45.0]
    live[6] += [-38.0, 52.0]

    fit = fit_team(taught, live, inlier_px=3.0)
    assert fit.ok
    assert fit.n_inliers == 5
    assert not fit.inliers[5] and not fit.inliers[6]
    np.testing.assert_allclose(fit.sim2.apply(TAUGHT), true.apply(TAUGHT), atol=1e-6)


def test_fit_abstains_rather_than_guessing_when_too_few_points_survive():
    true = Sim2.from_angle(0.2, t=(3.0, 3.0))
    valid = np.array([True, True, False, False, False])
    fit = fit_team(TAUGHT, true.apply(TAUGHT), valid)
    assert not fit.ok, "two survivors fit a Sim2 exactly and can never be caught being wrong"
    assert fit.n_inliers == 0
    assert np.isinf(fit.residuals).all()


def test_invalid_points_are_never_fitted_and_never_evicted():
    true = Sim2.from_angle(0.1, t=(5.0, 5.0))
    live = true.apply(TAUGHT)
    live[4] += [500.0, 500.0]  # a dead point's stale position must not pull the fit
    valid = np.array([True, True, True, True, False])

    fit = fit_team(TAUGHT, live, valid, inlier_px=2.0)
    assert fit.ok and fit.n_inliers == 4
    assert np.isinf(fit.residuals[4])
    assert 4 not in evict(fit, valid), "a point already dead must not consume an eviction"


def test_eviction_is_looser_than_the_consensus_band():
    true = Sim2.identity()
    live = true.apply(TAUGHT).copy()
    live[0] += [5.0, 0.0]  # just outside inlier_px, well inside evict_px: noise, keep it
    live[1] += [30.0, 0.0]  # latched: condemn it
    fit = fit_team(TAUGHT, live, inlier_px=4.0)

    condemned = evict(fit, np.ones(len(TAUGHT), bool), evict_px=8.0)
    assert 1 in condemned
    assert 0 not in condemned


def test_two_point_minimum_is_refused_outright():
    with pytest.raises(AssertionError, match="cannot produce a residual"):
        fit_team(TAUGHT, TAUGHT, min_points=2)


# --- attachment -------------------------------------------------------------------

REF = np.array([[100.0, 100.0], [120.0, 102.0], [110.0, 120.0], [98.0, 118.0]])


def _run(monitor, frames):
    """Feed (target_uv, holder_sim2) pairs; collect the events that fired."""
    out = []
    for uv, holder in frames:
        ev = monitor.update(uv, np.ones(len(REF), bool), holder)
        if ev is not None:
            out.append(ev)
    return out


def test_fission_fires_when_the_target_starts_moving_with_the_holder():
    mon = AttachmentMonitor(sustain=3)
    mon.reset(REF)
    frames = []
    for k in range(1, 7):
        holder = Sim2.from_angle(0.0, t=(4.0 * k, 0.0))
        frames.append((holder.apply(REF), holder))

    events = _run(mon, frames)
    assert [e.kind for e in events] == ["fission"]
    assert mon.state == HOLDER


def test_a_still_gripper_hovering_over_the_object_is_never_a_grasp():
    # The hypotheses coincide while the holder is still. Voting on that frame would
    # decide the grasp on pixel noise, so the monitor must abstain instead.
    mon = AttachmentMonitor(sustain=3, min_separation_px=3.0)
    mon.reset(REF)
    rng = np.random.default_rng(0)
    frames = [
        (REF + rng.normal(scale=0.4, size=REF.shape), Sim2.from_angle(0.0, t=(0.2, -0.1))) for _ in range(40)
    ]
    assert _run(mon, frames) == []
    assert mon.state == WORLD


def test_the_object_staying_put_while_the_gripper_moves_is_not_a_grasp():
    # The distinguishing case: the holder travels, the object does not follow. This is
    # a failed grasp, and reporting fission here would end the stage on a lie.
    mon = AttachmentMonitor(sustain=3)
    mon.reset(REF)
    frames = [(REF.copy(), Sim2.from_angle(0.0, t=(5.0 * k, 0.0))) for k in range(1, 8)]
    assert _run(mon, frames) == []
    assert mon.state == WORLD


def test_defission_fires_on_release():
    mon = AttachmentMonitor(sustain=3)
    mon.reset(REF, state=HOLDER)
    # The holder keeps travelling; the object has been let go and stays where it is.
    frames = [(REF.copy(), Sim2.from_angle(0.0, t=(6.0 * k, 0.0))) for k in range(1, 8)]
    events = _run(mon, frames)
    assert [e.kind for e in events] == ["defission"]
    assert mon.state == WORLD


def test_a_failed_holder_fit_abstains_instead_of_guessing():
    mon = AttachmentMonitor(sustain=2)
    mon.reset(REF)
    holder = Sim2.from_angle(0.0, t=(20.0, 0.0))
    frames = [(holder.apply(REF), None), (holder.apply(REF), None)]
    assert _run(mon, frames) == []
    assert mon.state == WORLD


def test_a_single_frame_of_agreement_is_not_enough():
    mon = AttachmentMonitor(sustain=4)
    mon.reset(REF)
    holder = Sim2.from_angle(0.0, t=(10.0, 0.0))
    assert mon.update(holder.apply(REF), np.ones(len(REF), bool), holder) is None
    assert mon.state == WORLD


def test_reset_repins_the_world_hypothesis_after_a_rebind():
    # After a re-bind the object legitimately sits somewhere new. Without a reset the
    # world hypothesis would keep predicting the old place and fission instantly.
    mon = AttachmentMonitor(sustain=2)
    mon.reset(REF)
    moved = REF + [40.0, 0.0]
    mon.reset(moved)
    still = Sim2.from_angle(0.0, t=(8.0, 0.0))
    frames = [(moved.copy(), still) for _ in range(5)]
    assert _run(mon, frames) == []
    assert mon.state == WORLD
