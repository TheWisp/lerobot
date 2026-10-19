"""What the demo believed of the object it held (lerobot.showservo.engine.demo_beliefs), by the rule an act believes
its own, and the act putting its held object where the demo's was as the demo saw it held, not as it lay at the
grasp."""

from __future__ import annotations

import dataclasses

import numpy as np

from lerobot.showservo.engine import demo_beliefs
from lerobot.showservo.engine.timeline import Timeline

from .enginesim import pick_place_demo, run_act

SHIFT = np.array([0.010, 0.0, 0.0])  # how far the object slid in the demo's fingers, fingertip frame


def _slid(tip_i, tip_grasp):
    """The object's motion since the demo with the fingertip at ``tip_i``, slid SHIFT in the fingers after a grasp
    at ``tip_grasp``."""
    shift = np.eye(4)
    shift[:3, 3] = SHIFT
    return tip_i @ shift @ np.linalg.inv(tip_grasp)


def _setup():
    demo, kf, bodies = pick_place_demo()
    tl = Timeline(demo, kf)
    grasp = next(e for e in tl.events if e.kind == "grasp")
    release = next(e for e in tl.events if e.kind == "release")
    centres = {n: b.points.mean(axis=0) for n, b in bodies.items()}
    return demo, kf, bodies, tl, grasp, release, centres


def _views(demo, grasp, release, slid_from, rng):
    """The demo's views of the block: lying still before the grasp (0.5 mm of noise), held after, slid in the fingers
    from demo time ``slid_from``; none after the release."""
    out = []
    for i, t in enumerate(demo.t):
        if i < grasp.i:
            m = np.eye(4)
        elif i <= release.i:
            m = (
                _slid(demo.tips[i], demo.tips[grasp.i])
                if t >= slid_from
                else demo.tips[i] @ np.linalg.inv(demo.tips[grasp.i])
            )
        else:
            continue
        m = m.copy()
        m[:3, 3] += rng.normal(0.0, 0.0005, 3)
        out.append((float(t), m))
    return {"dowel": out}


def test_the_demos_grip_is_what_its_views_showed_while_held_and_as_it_lay_without_them():
    demo, _kf, _b, tl, grasp, release, centres = _setup()
    slid_from = float(demo.t[grasp.i]) + 0.4
    grips, info = demo_beliefs.grips(
        demo, tl.events, _views(demo, grasp, release, slid_from, np.random.default_rng(1)), centres
    )
    rigid = np.linalg.inv(demo.tips[grasp.i])
    c = np.append(centres["dowel"], 1.0)
    before = int(np.searchsorted(demo.t, slid_from)) - 2
    assert np.linalg.norm((grips["dowel"][before] @ c)[:3] - (rigid @ c)[:3]) < 0.002, (
        "as it lay, until it slid"
    )
    slid = _slid(np.eye(4), demo.tips[grasp.i])
    assert np.linalg.norm((grips["dowel"][release.i] @ c)[:3] - (slid @ c)[:3]) < 0.002, (
        "as the views had it held"
    )
    assert info["dowel"]["views_held"] > 10 and abs(info["dowel"]["moved_mm"] - 10.0) < 2.0
    unseen = {
        "dowel": [
            v
            for v in _views(demo, grasp, release, np.inf, np.random.default_rng(1))["dowel"]
            if v[0] < demo.t[grasp.i]
        ]
    }
    grips, info = demo_beliefs.grips(demo, tl.events, unseen, centres)
    assert info["dowel"]["views_held"] == 0
    np.testing.assert_allclose(
        grips["dowel"][release.i] @ c, rigid @ c, atol=0.002, err_msg="no view held: as it lay"
    )


def test_an_act_puts_its_held_object_where_the_demos_was_as_the_demo_saw_it_held():
    """The demo's block slid 10 mm in its fingers after the grasp, which its views showed. An act whose block sits as
    the demo's did at the grasp sets it down 10 mm (the fingertip's turn of it) from where it would by how the demo's
    lay at the grasp: where the demo's block went."""
    demo, kf, bodies, tl, grasp, release, centres = _setup()
    plain = run_act(demo, kf, bodies)
    grips, _ = demo_beliefs.grips(
        demo,
        tl.events,
        _views(demo, grasp, release, float(demo.t[grasp.i]) + 0.4, np.random.default_rng(2)),
        centres,
    )
    demo2, kf2, bodies2 = pick_place_demo()
    seen = run_act(dataclasses.replace(demo2, grips=grips), kf2, bodies2)
    assert plain.engine.result["ok"] and seen.engine.result["ok"], (plain.engine.result, seen.engine.result)
    moved = seen.world.bodies["dowel"].now().mean(axis=0) - plain.world.bodies["dowel"].now().mean(axis=0)
    expected = demo.tips[release.i][:3, :3] @ SHIFT
    assert np.linalg.norm(moved - expected) < 0.003, (moved, expected)
