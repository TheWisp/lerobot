# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The checks' two questions (src/lerobot/showservo/docs/act_execution.md, "What a check answers"): which of two
things an object moved with, and whether it is where a pose puts it. Each answers success only on evidence that tells
the outcomes apart, by the noise measured on that object, never by a fixed number.

Scenes are rendered by ``enginesim`` with the rig's camera: a 40 x 20 x 12 mm box (the dowel's size class) on the
table, its tracked points projected and its depth rendered by z-buffer."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.showservo.engine import predicates as p
from lerobot.showservo.engine.predicates import Answer, Verdict
from lerobot.showservo.engine.scene import Tracks

from .enginesim import WORK, Body, box_points, camera, pose, render_depth, render_tracks, track_points

BOX = (0.040, 0.020, 0.012)
REST = WORK + (0.0, 0.0, 0.006)  # lying on the table under the camera's axis


def _body() -> Body:
    return Body(box_points(BOX, REST))


def _tip_over(body: Body) -> np.ndarray:
    """The fingertip closed on the box: at its centre, pointing down."""
    return pose(body.centre, (180.0, 0.0, 0.0))


def _carried(tip: np.ndarray, up_m: float = 0.05, across_m: float = 0.03) -> np.ndarray:
    """The start of a carry: up and across, as the demo's lift turns into its carry."""
    out = tip.copy()
    out[:3, 3] += (across_m, 0.0, up_m)
    return out


def _shift(cam, centre, motion) -> np.ndarray:
    """Where ``motion`` moves ``centre`` in the image."""
    uv, _ = cam.project(np.stack([centre, (motion @ np.append(centre, 1.0))[:3]]))
    return uv[1] - uv[0]


# ── the decision: the nearer of two predictions, once they are further apart than the noise ──


@pytest.mark.parametrize(
    ("observed", "answer"),
    [
        ((10.0, 0.0), Answer.YES),  # on the new prediction
        ((7.0, 1.0), Answer.YES),  # nearer the new by more than the noise
        ((0.0, 0.0), Answer.NO),  # on the old
        ((5.0, 0.0), Answer.UNKNOWN),  # half way
        ((10.0, 30.0), Answer.NO),  # far from both: something else moved it
    ],
)
def test_decide_takes_the_nearer_prediction_by_more_than_the_noise(observed, answer):
    v = p.decide(np.array(observed), new=np.array([10.0, 0.0]), old=np.zeros(2), noise=1.0, unit="px")
    assert v.answer is answer, v.why


def test_decide_cannot_tell_while_the_predictions_are_within_the_noise_of_each_other():
    v = p.decide(np.array([4.0, 0.0]), new=np.array([4.0, 0.0]), old=np.zeros(2), noise=2.5, unit="px")
    assert v.answer is Answer.UNKNOWN
    assert "apart" in v.why


def test_decide_scales_with_the_noise_not_with_a_constant():
    """The same observation reads success with steady views and can't tell with noisy ones (R9)."""
    obs, new, old = np.array([18.0, 0.0]), np.array([20.0, 0.0]), np.zeros(2)
    assert p.decide(obs, new=new, old=old, noise=2.0, unit="mm").answer is Answer.YES
    assert p.decide(obs, new=new, old=old, noise=12.0, unit="mm").answer is Answer.UNKNOWN


# ── moves_with: tracks against two predictions ──


def test_moves_with_says_the_object_came_with_the_gripper():
    cam, body = camera(), _body()
    rows = track_points(body)
    tip0 = _tip_over(body)
    ref = render_tracks(cam, body, rows, 0.0)
    carry = _carried(tip0) @ np.linalg.inv(tip0)
    body.motion = carry
    v = p.moves_with(
        ref, render_tracks(cam, body, rows, 0.5), _shift(cam, body.centre, carry), np.zeros(2), 1.0
    )
    assert v.answer is Answer.YES, v.why


def test_moves_with_says_the_object_stayed():
    cam, body = camera(), _body()
    rows = track_points(body)
    tip0 = _tip_over(body)
    ref = render_tracks(cam, body, rows, 0.0)
    carry = _carried(tip0) @ np.linalg.inv(tip0)
    v = p.moves_with(
        ref, render_tracks(cam, body, rows, 0.5), _shift(cam, body.centre, carry), np.zeros(2), 1.0
    )
    assert v.answer is Answer.NO, v.why


def test_moves_with_cannot_tell_when_the_gripper_covers_the_tracks():
    cam, body = camera(), _body()
    rows = track_points(body)
    tip0 = _tip_over(body)
    ref = render_tracks(cam, body, rows, 0.0)
    carry = _carried(tip0) @ np.linalg.inv(tip0)
    body.motion = carry
    hidden = np.ones(len(rows), dtype=bool)
    hidden[:2] = False
    now = render_tracks(cam, body, rows, 0.5, hidden=hidden)
    v = p.moves_with(ref, now, _shift(cam, body.centre, carry), np.zeros(2), 1.0)
    assert v.answer is Answer.UNKNOWN
    assert "tracked points" in v.why


def test_moves_with_holds_a_grip_that_is_not_rigid():
    """The object turns 15 degrees and slides 4 mm in the fingers while carried: still came along."""
    cam, body = camera(), _body()
    rows = track_points(body)
    tip0 = _tip_over(body)
    ref = render_tracks(cam, body, rows, 0.0)
    tip1 = _carried(tip0)
    body.motion = tip1 @ pose((0.004, 0.0, 0.0), (0.0, 0.0, 15.0)) @ np.linalg.inv(tip0)
    carry = tip1 @ np.linalg.inv(tip0)
    v = p.moves_with(
        ref, render_tracks(cam, body, rows, 0.6), _shift(cam, body.centre, carry), np.zeros(2), 1.0
    )
    assert v.answer is Answer.YES, v.why


def test_moves_with_uses_only_points_seen_in_both_frames():
    cam, body = camera(), _body()
    rows = track_points(body)
    tip0 = _tip_over(body)
    ref = render_tracks(cam, body, rows, 0.0)
    carry = _carried(tip0) @ np.linalg.inv(tip0)
    body.motion = carry
    now = render_tracks(cam, body, rows, 0.5)
    ids, uv = now.ids.copy(), now.uv.copy()
    ids[::2] += 1000  # re-seeded under new ids at unrelated pixels: must not count
    uv[::2] = 5.0
    v = p.moves_with(
        ref,
        Tracks(t=now.t, ids=ids, uv=uv, visible=now.visible),
        _shift(cam, body.centre, carry),
        np.zeros(2),
        1.0,
    )
    assert v.answer is Answer.YES, v.why
    assert v.evidence["points"] == len(rows) // 2


def test_moves_with_a_release_stays_where_it_was_put():
    """After a release the new reference is where it was put (no motion) and the old one the gripper."""
    cam, body = camera(), _body()
    rows = track_points(body)
    tip0 = _tip_over(body)
    ref = render_tracks(cam, body, rows, 0.0)
    away = _carried(tip0) @ np.linalg.inv(tip0)  # the gripper leaves
    stayed = render_tracks(cam, body, rows, 0.5)
    assert p.moves_with(ref, stayed, np.zeros(2), _shift(cam, body.centre, away), 1.0).answer is Answer.YES
    body.motion = away  # stuck to the fingers
    stuck = render_tracks(cam, body, rows, 0.5)
    assert p.moves_with(ref, stuck, np.zeros(2), _shift(cam, body.centre, away), 1.0).answer is Answer.NO


# ── sits_at: the object's surface where a pose puts it, in the depth image ──


def _surface(body: Body, cam) -> np.ndarray:
    """What the act knows of the object's shape: the points its camera saw of it, at its demo pose."""
    uv, z = cam.project(body.points)
    seen = render_depth(cam, [body.points])
    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    return body.points[np.abs(seen.metres[vi, ui] - z) < 0.002]


DEPTH_NOISE = 0.002  # a synthetic sensor's measured spread; the real one is measured on the object at rest


def test_sits_at_says_yes_where_the_object_is():
    cam, body = camera(), _body()
    v = p.sits_at(_surface(body, cam), np.eye(4), render_depth(cam, [body.now()]), cam, DEPTH_NOISE)
    assert v.answer is Answer.YES, v.why


def test_sits_at_says_no_where_the_object_has_gone():
    cam, body = camera(), _body()
    v = p.sits_at(_surface(body, cam), np.eye(4), render_depth(cam, []), cam, DEPTH_NOISE)
    assert v.answer is Answer.NO, v.why


def test_sits_at_cannot_tell_while_something_covers_the_place():
    cam, body = camera(), _body()
    surface = _surface(body, cam)
    to_camera = cam.t_bc[:3, 3] - body.centre
    over = body.centre + to_camera * (0.08 - body.centre[2]) / to_camera[2]  # on the line of sight, 8 cm up
    gripper = box_points((0.08, 0.08, 0.01), over)
    v = p.sits_at(surface, np.eye(4), render_depth(cam, [body.now(), gripper]), cam, DEPTH_NOISE)
    assert v.answer is Answer.UNKNOWN, v.why


def test_sits_at_allows_an_offset_within_the_lateral_allowance():
    """The pose it is tested at is off by the grasp's own error (4.6 to 7.4 mm, O5): passes when allowed for."""
    cam, body = camera(), _body()
    surface = _surface(body, cam)
    off = pose((0.006, 0.004, 0.0))
    depth = render_depth(cam, [body.now()])
    assert p.sits_at(surface, off, depth, cam, DEPTH_NOISE, lateral_m=0.008).answer is Answer.YES


def test_sits_at_does_not_take_the_table_for_the_object():
    """Gone, with a lateral allowance as large as the object is thick: the table around must not read present."""
    cam, body = camera(), _body()
    surface = _surface(body, cam)
    v = p.sits_at(surface, np.eye(4), render_depth(cam, []), cam, DEPTH_NOISE, lateral_m=0.012)
    assert v.answer is Answer.NO, v.why


def test_sits_at_tells_a_standing_object_from_the_floor_it_was_let_go_over():
    cam, body = camera(), _body()
    surface = _surface(body, cam)
    centre = body.centre
    upright = pose((0.0, 0.0, BOX[0] / 2 - BOX[2] / 2), (0.0, 90.0, 0.0))
    upright[:3, 3] += centre - upright[:3, :3] @ centre  # turned about its centre, then lifted onto its end
    body.motion = upright
    assert p.sits_at(surface, upright, render_depth(cam, [body.now()]), cam, DEPTH_NOISE).answer is Answer.YES
    body.motion = pose((0.03, 0.02, 0.0))  # fell over beside it
    assert p.sits_at(surface, upright, render_depth(cam, [body.now()]), cam, DEPTH_NOISE).answer is Answer.NO


# ── related: a view of the object against where a reference and the demo's relation put it ──


def test_related_uses_the_measured_noise_and_the_operators_allowance():
    centre = np.array([0.0, 0.0, 0.006])
    ring = pose((0.10, 0.05, 0.0), (0.0, 0.0, 40.0))
    rel = pose((0.0, 0.0, 0.03))
    off = ring @ pose((0.010, 0.0, 0.03))
    assert p.related(ring @ rel, ring, rel, centre, noise_m=0.003, noise_deg=1.0).answer is Answer.YES
    assert p.related(off, ring, rel, centre, noise_m=0.003, noise_deg=1.0).answer is Answer.NO
    assert p.related(off, ring, rel, centre, noise_m=0.006, noise_deg=1.0).answer is Answer.UNKNOWN
    # Views noisier than the offset: as strict as the camera can see, and no stricter.
    assert p.related(off, ring, rel, centre, noise_m=0.015, noise_deg=1.0).answer is Answer.YES
    assert p.related(off, ring, rel, centre, noise_m=0.003, noise_deg=1.0, allow_m=0.012).answer is Answer.YES


def test_related_ignores_turns_about_a_declared_symmetry_axis():
    centre = np.array([0.0, 0.0, 0.006])
    ring, rel = np.eye(4), pose((0.0, 0.0, 0.03))
    spun = rel @ pose(rpy_deg=(0.0, 0.0, 90.0))
    turned = spun.copy()
    turned[:3, 3] = rel[:3, 3] + rel[:3, :3] @ centre - spun[:3, :3] @ centre  # about the centre
    assert p.related(turned, ring, rel, centre, noise_m=0.003, noise_deg=1.0).answer is Answer.NO
    free = p.related(
        turned, ring, rel, centre, noise_m=0.003, noise_deg=1.0, free_axis=np.array([0.0, 0.0, 1.0])
    )
    assert free.answer is Answer.YES
    quarter = p.related(
        turned, ring, rel, centre, noise_m=0.003, noise_deg=1.0, turns=(np.array([0.0, 0.0, 1.0]), 4)
    )
    assert quarter.answer is Answer.YES


# ── noise, measured ──


def test_spread_is_the_95th_percentile_from_the_median():
    rng = np.random.default_rng(0)
    xyz = rng.normal(0.0, 0.004, size=(400, 3)) + np.array([0.1, 0.2, 0.3])
    spread = p.spread(xyz)
    assert 0.010 < spread < 0.016  # 95th percentile of a 3D normal's radius with sigma 4 mm: about 11 mm
    assert p.spread(xyz[:2]) is None  # too few to measure


# ── combining evidence ──


@pytest.mark.parametrize(
    ("answers", "out"),
    [
        ((Answer.YES, Answer.UNKNOWN), Answer.YES),
        ((Answer.NO, Answer.UNKNOWN), Answer.NO),
        ((Answer.YES, Answer.NO), Answer.UNKNOWN),
        ((Answer.UNKNOWN, Answer.UNKNOWN), Answer.UNKNOWN),
        ((), Answer.UNKNOWN),
    ],
)
def test_disagreeing_evidence_cannot_tell(answers, out):
    v = p.combine([Verdict(a, f"source {k}") for k, a in enumerate(answers)])
    assert v.answer is out
    if out is Answer.UNKNOWN and Answer.YES in answers and Answer.NO in answers:
        assert "disagree" in v.why
