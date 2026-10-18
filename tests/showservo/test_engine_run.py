# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The act engine end to end in the simulated world (``enginesim``): the arm, the motor loop, the objects and the
camera, against src/lerobot/showservo/docs/act_execution.md's requirements.

The demo picks a 40 x 20 x 12 mm block off the table and sets it on a 40 x 40 x 20 mm block; the camera is the rig's.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.showservo import motion as core

from .enginesim import FakeKin, pick_place_demo, pose, run_act


def _on_ring(sim, tol_m=0.004):
    """The block lies on the ring as the demo left it: centre over the ring's, resting on its top."""
    dowel = sim.world.bodies["dowel"].now().mean(axis=0)
    ring = sim.world.bodies["ring"].now()
    top = ring[:, 2].max()
    return (
        np.linalg.norm(dowel[:2] - ring.mean(axis=0)[:2]) <= tol_m
        and abs(sim.world.bodies["dowel"].now()[:, 2].min() - top) <= 0.001
    )


def _records(sim, what):
    return [r for r in sim.engine.records if r["what"] == what]


def test_an_undisturbed_act_places_the_block_and_both_checks_succeed():
    demo, kf, bodies = pick_place_demo()
    sim = run_act(demo, kf, bodies)
    assert sim.engine.result["ok"], sim.engine.result
    assert _on_ring(sim), sim.world.events
    status = {m["name"]: m["status"] for m in sim.engine.result["milestones"]}
    assert status == {"grasp of dowel": "success", "release of dowel": "success"}, sim.engine.result
    assert not _records(sim, "failure")
    assert sim.motor.clamped < 1e-6  # no command needed the motor loop's own bound


def _sent_times(sim):
    """When each command index was first sent."""
    out = {}
    for t, k, _, _ in sim.motor.log:
        for j in range(max(out, default=-1) + 1, k + 1):
            out[j] = t
    return out


def _span(sim, from_leg):
    """Seconds from the first command of leg ``from_leg`` being sent to the motion's end."""
    sent = _sent_times(sim)
    cmds = sim.engine.cmds
    first = next(k for k, c in enumerate(cmds) if c.kind == "leg" and c.leg >= from_leg)
    end = next(r["t"] for r in sim.engine.records if r["what"] == "motion ends")
    return end - sent[first]


@pytest.mark.parametrize("fps", [15.0, 29.93, 30.0, 60.0])
@pytest.mark.parametrize("speed", [1.0, 1.7])
def test_the_act_takes_the_demos_time_over_the_speed_at_any_frame_rate(fps, speed):
    """From the first key frame on, the act runs on the demo's clock: (7.6 - 1.5) s / speed (R6). Before it, the arm
    walks from wherever it is, which no demo time covers. Where a joint would move faster than its bound the clock
    slows, by what the engine reports (R12): at 1.7 times the demo's speed the lift and the withdrawal peak above this
    arm's 0.18 m/s; at the demo's own speed nothing does."""
    demo, kf, bodies = pick_place_demo(fps=fps)
    sim = run_act(demo, kf, bodies, speed=speed)
    assert sim.engine.result["ok"], sim.engine.result
    stretched = sim.engine.stretched_s / speed
    expected = (kf[-1]["t"] - kf[0]["t"]) / speed + stretched
    assert abs(_span(sim, 1) - expected) <= 2.0 / 30.0, (_span(sim, 1), expected, stretched)
    if speed == 1.0:
        assert stretched == 0.0


def test_key_frames_on_every_demo_frame_keep_the_demos_time():
    demo, kf, bodies = pick_place_demo(dense=True)
    assert len(kf) > 40
    sim = run_act(demo, kf, bodies, speed=1.3)
    assert sim.engine.result["ok"], sim.engine.result
    assert abs(_span(sim, 1) - (kf[-1]["t"] - kf[0]["t"]) / 1.3) <= 2.0 / 30.0


@pytest.mark.parametrize(("fps", "speed"), [(29.93, 1.0), (30.0, 1.7), (60.0, 0.8)])
def test_the_gripper_reaches_the_demos_full_closing_between_ticks(fps, speed):
    demo, kf, bodies = pick_place_demo(fps=fps)
    sim = run_act(demo, kf, bodies, speed=speed)
    sent = [sim.motor.samples[k][FakeKin.GRIPPER] for k in range(sim.motor.sent_i + 1)]
    assert max(sent) == pytest.approx(demo.grip.max())
    assert min(sent[len(sent) // 2 :]) == pytest.approx(demo.grip.min())


def _demo_still(demo, td):
    """Whether the demo itself stood still, arm and gripper, around demo time ``td``."""
    k = int(np.clip(np.searchsorted(demo.t, td), 1, len(demo.t) - 1))
    m, deg = core.pose_residual(demo.tips[k - 1], demo.tips[k])
    return m < 1e-6 and deg < 1e-4 and abs(demo.grip[k] - demo.grip[k - 1]) < 1e-9


def _standstills(sim, min_ticks=4):
    """Runs of at least ``min_ticks`` sent commands with neither the pose nor the gripper changing, where the demo
    did not stand still itself: (first, last)."""
    cmds, out, run = sim.engine.cmds, [], None
    demo = sim.engine.demo
    for k in range(1, len(cmds)):
        m, deg = core.pose_residual(cmds[k - 1].pose, cmds[k].pose)
        still = m < 1e-5 and deg < 1e-3 and abs(cmds[k].grip - cmds[k - 1].grip) < 1e-6
        still = still and not (cmds[k].td is not None and _demo_still(demo, cmds[k].td))
        if still:
            run = (run[0], k) if run else (k, k)
        else:
            if run and run[1] - run[0] + 1 >= min_ticks:
                out.append(run)
            run = None
    if run and run[1] - run[0] + 1 >= min_ticks:
        out.append(run)
    return out


def test_the_arm_stands_still_only_where_the_demo_did():
    """No run of commands stands still for 4 ticks (133 ms) where the demo was moving (R7)."""
    demo, kf, bodies = pick_place_demo()
    sim = run_act(demo, kf, bodies)
    assert _standstills(sim) == []


def test_a_target_moved_during_the_carry_bends_the_path_without_stopping():
    demo, kf, bodies = pick_place_demo()

    def nudge(world):
        world.at.append(
            (1.0 + 4.3, lambda w, t: setattr(w.bodies["ring"], "motion", pose((0.03, -0.02, 0.0))))
        )

    sim = run_act(demo, kf, bodies, setup=nudge)
    assert sim.engine.result["ok"], sim.engine.result
    assert _on_ring(sim), sim.world.events
    assert _standstills(sim) == []
    assert sim.motor.clamped < 1e-6
    assert not _records(sim, "failure")


def test_the_arm_passes_key_frames_at_the_demos_speed():
    """Where the demo kept moving through a key frame, the commands do too: the command's speed on the tick that
    crosses it is at least half the demo's there (the old runner fell to 3 deg/s at every key frame, E3)."""
    demo, kf, bodies = pick_place_demo()
    sim = run_act(demo, kf, bodies)
    cmds, arm = sim.engine.cmds, sim.engine.arm
    checked = 0
    for n, k in enumerate(kf[:-1]):  # the demo moves through the pre-grasp, the lift's end and the pre-place
        i = int(np.searchsorted(demo.t, k["t"]))
        demo_speed = arm.walk_seconds(demo.tips[i - 1], demo.tips[i]) / (demo.t[i] - demo.t[i - 1])
        if demo_speed <= 0:
            continue
        j = next(j for j in range(1, len(cmds)) if cmds[j].kind == "leg" and cmds[j].leg == n + 1)
        cmd_speed = arm.walk_seconds(cmds[j - 1].pose, cmds[j].pose) * arm.hz
        assert cmd_speed >= 0.5 * demo_speed, (k["label"], cmd_speed, demo_speed)
        checked += 1
    assert checked >= 3


def _open_mid_carry(s=0.4, seconds=0.6):
    from lerobot.showservo.engine.perturb import OpenGripper, Perturbations

    return Perturbations([OpenGripper(leg="pre-place 1", s=s, seconds=seconds)])


def test_a_missed_grasp_goes_back_and_grasps_again():
    demo, kf, bodies = pick_place_demo()
    sim = run_act(demo, kf, bodies, setup=lambda w: setattr(w, "misses", 1))
    assert any("missed" in e for _, e in sim.world.events)
    failures = _records(sim, "failure")
    assert [r["milestone"] for r in failures] == ["grasp of dowel"]
    moved = _records(sim, "progress moved")
    assert len(moved) == 1 and moved[0]["to"][0] <= 1  # back to the approach or the grasp's descent
    assert sim.engine.result["ok"], sim.engine.result
    assert _on_ring(sim), sim.world.events


def test_a_block_dropped_in_the_carry_is_picked_up_where_it_fell():
    """The gripper is opened mid-carry by a perturbation the engine does not know of: it sees the block neither in
    the gripper nor where it lay, waits to see where it is, and picks it up there."""
    demo, kf, bodies = pick_place_demo()
    sim = run_act(demo, kf, bodies, perturb=_open_mid_carry())
    assert any("let go" in e for _, e in sim.world.events[:3]), sim.world.events
    failures = _records(sim, "failure")
    assert [r["milestone"] for r in failures] == ["grasp of dowel"], failures
    assert sim.engine.result["ok"], sim.engine.result
    assert _on_ring(sim), sim.world.events


def test_a_block_knocked_off_after_the_release_is_placed_again():
    demo, kf, bodies = pick_place_demo()

    def knock(world):
        def off(w, t):
            if w.held is None and any("let go" in e for _, e in w.events):
                w.bodies["dowel"].motion = pose(
                    (-0.03, -0.04, 0.0)
                )  # back on the table, beside where it began
                w.events.append((t, "knocked off"))
            else:
                w.at.append((t + 0.1, off))

        world.at.append((1.0 + 2.6 + 5.0, off))  # the release is at about 1.0 + 2.6 + 4.8 s

    sim = run_act(demo, kf, bodies, setup=knock)
    assert any("knocked off" in e for _, e in sim.world.events), sim.world.events
    assert [r["milestone"] for r in _records(sim, "failure")] == ["release of dowel"]
    assert sim.engine.result["ok"], sim.engine.result
    assert _on_ring(sim), sim.world.events


def test_a_grasp_that_fails_twice_gives_up_and_says_why():
    demo, kf, bodies = pick_place_demo()
    sim = run_act(demo, kf, bodies, setup=lambda w: setattr(w, "misses", 5))
    assert not sim.engine.result["ok"]
    assert "grasp of dowel failed 2 times" in sim.engine.result["why"]


def test_noisy_views_and_tracks_fail_no_undisturbed_act():
    """Views wandering by 6 mm and tracks by 1.5 px: the checks can't tell more often, and never fail (R9)."""
    demo, kf, bodies = pick_place_demo()

    def noisy(world):
        world.view_noise_m, world.track_noise_px = 0.006, 1.5

    sim = run_act(demo, kf, bodies, setup=noisy)
    assert not _records(sim, "failure"), _records(sim, "failure")
    assert sim.engine.result["ok"], sim.engine.result


def test_a_target_moved_away_during_the_place_goes_back_up_and_in_again():
    demo, kf, bodies = pick_place_demo()

    def shove(world):
        world.at.append(
            (1.0 + 2.6 + 4.25, lambda w, t: setattr(w.bodies["ring"], "motion", pose((0.035, 0.0, 0.0))))
        )

    sim = run_act(demo, kf, bodies, setup=shove)
    moved = _records(sim, "progress moved")
    assert moved and "ring moved" in moved[0]["cause"], sim.engine.records[-5:]
    assert moved[0]["how"] == "backs out along the demo's path"
    assert sim.engine.result["ok"], sim.engine.result
    assert _on_ring(sim), sim.world.events


def test_the_object_a_leg_leaves_no_longer_moves_it():
    """Going from above the ring to above the block, the ring moving after the leg began changes nothing: the leg's
    start is frozen where the arm was; only the object it heads for is read for each command."""
    demo, kf, bodies = pick_place_demo()
    kf = [dict(k) for k in kf]
    kf[0] = {
        **kf[0],
        "move": "follow",
    }  # the pre-grasp now follows the demo's path from a key frame over the ring
    kf.insert(
        0,
        {
            "t": 0.75,
            "frame": "ring",
            "move": "line",
            "grip": "first",
            "constraint": "pose",
            "label": "over ring",
        },
    )
    still = run_act(demo, kf, pick_place_demo()[2])
    leg1 = lambda sim: [c.pose for c in sim.engine.cmds if c.kind == "leg" and c.leg == 1]  # noqa: E731
    begins = _sent_times(still)[next(k for k, c in enumerate(still.engine.cmds) if c.leg == 1)]

    def push_ring(world):  # a quarter second into the leg from the ring to the block
        world.at.append(
            (begins + 0.25, lambda w, t: setattr(w.bodies["ring"], "motion", pose((0.0, 0.03, 0.0))))
        )

    pushed = run_act(demo, kf, bodies, setup=push_ring)
    assert len(leg1(still)) == len(leg1(pushed)) > 10
    np.testing.assert_allclose(np.stack(leg1(still)), np.stack(leg1(pushed)), atol=1e-9)
    assert pushed.engine.result["ok"] and _on_ring(pushed), (
        pushed.engine.result
    )  # and the place found it moved


@pytest.mark.parametrize(("fps", "speed"), [(29.93, 1.0), (30.0, 1.7), (60.0, 0.8)])
def test_a_gripper_pulse_between_ticks_reaches_its_peak(fps, speed):
    """The gripper's turning points are never smoothed away: a 0.1 s pulse to 60 during the approach, whose peak falls
    between ticks, is sent at 60."""
    import dataclasses

    demo, kf, bodies = pick_place_demo(fps=fps)
    peak_t = 1.913
    pulse = np.clip(60.0 - np.abs(demo.t - peak_t) * 800.0, demo.grip, None)
    demo = dataclasses.replace(
        demo, grip=pulse, grip_obs=np.where(np.abs(demo.t - peak_t) < 0.06, pulse, demo.grip_obs)
    )
    sim = run_act(demo, kf, bodies, speed=speed)
    near = [
        sim.motor.samples[k][FakeKin.GRIPPER]
        for k, c in enumerate(sim.engine.cmds[: sim.motor.sent_i + 1])
        if c.td is not None and abs(c.td - peak_t) < 0.1
    ]
    demo_peak = demo.grip[np.abs(demo.t - peak_t) < 0.1].max()  # the demo's own top sample
    assert max(near) == pytest.approx(demo_peak)


def test_the_grasp_check_does_not_take_the_lifted_block_for_one_left_behind():
    """With fingers the depth camera cannot see, the block is in view at its resting place while the lift begins:
    until it has moved away by more than its own size, its place reading "present" says nothing of a miss."""
    demo, kf, bodies = pick_place_demo()
    sim = run_act(
        demo, kf, bodies, setup=lambda w: setattr(w, "gripper_points", lambda tip: np.zeros((0, 3)))
    )
    assert not _records(sim, "failure"), _records(sim, "failure")
    assert sim.engine.result["ok"], sim.engine.result


def test_the_arm_does_not_drift_with_a_moved_target_while_in_contact():
    """The ring moves during the place: until the second view confirms it, the commands stay on the place's path as
    composed with where the ring was, rather than creeping sideways inside it."""
    demo, kf, bodies = pick_place_demo()
    shove_at = 1.0 + 2.6 + 4.25

    def shove(world):
        world.at.append((shove_at, lambda w, t: setattr(w.bodies["ring"], "motion", pose((0.035, 0.0, 0.0)))))

    sim = run_act(demo, kf, bodies, setup=shove)
    sent = _sent_times(sim)
    back = next(r["t"] for r in sim.engine.records if r["what"] == "progress moved")
    place = sim.engine.tl.legs.index(next(leg for leg in sim.engine.tl.legs if leg.label == "place"))
    ring0 = np.eye(4)
    during = [
        c
        for k, c in enumerate(sim.engine.cmds)
        if c.kind == "leg" and c.leg == place and k in sent and shove_at <= sent[k] <= back
    ]
    assert during
    for c in during:
        expected = ring0 @ sim.engine.demo.pose_at(c.td) @ sim.engine.fix.get("dowel", np.eye(4))
        assert np.linalg.norm(c.pose[:3, 3] - expected[:3, 3]) < 0.0005


def test_a_sticking_joint_is_waited_for_instead_of_driven_into_the_motor_loops_freeze():
    """The lift's joint sticks under load until it is 6 degrees behind or its goal has stood still for 0.6 s, the
    jog's trim then freeing it (on the rig the shoulder was still stuck 0.33 s into a still goal, E21). Run on, the
    motor loop holds at 5 degrees behind and freezes the arm 0.5 s later; waited for whenever a joint trails by more
    than the loop's 3 degree catch-up budget, the goal stands still before the hold and the act completes."""
    demo, kf, bodies = pick_place_demo()
    stuck = run_act(demo, kf, bodies, speed=0.5, stick={2: 6.0}, trim_ticks=18, lag_deg=None)
    assert stuck.motor.halted  # the control: the motor loop freezes the arm
    sim = run_act(demo, kf, pick_place_demo()[2], speed=0.5, stick={2: 6.0}, trim_ticks=18)
    assert not sim.motor.halted
    assert sim.engine.result["ok"], sim.engine.result
    assert _on_ring(sim), sim.world.events
    assert _records(sim, "waiting for the arm")
    lags = [float(np.max(np.abs(q_cmd - q_obs)[:6])) for _, _, q_cmd, q_obs in sim.motor.log]
    assert max(lags) < sim.motor.stop_lag
