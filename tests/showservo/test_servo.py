# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The loop must close on a plant whose Jacobian it was never told correctly — that is
the whole reason the Jacobian is measured instead of derived from a kinematic model."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.fewshot.registration import Sim2
from lerobot.showservo.grouping import TeamFit, fit_team
from lerobot.showservo.servo import (
    ConvergenceCertificate,
    JacobianEstimator,
    PIController,
    servo_error,
)

TAUGHT_HELD = np.array([[200.0, 90.0], [232.0, 96.0], [214.0, 128.0]])
TAUGHT_TARGET = np.array([[100.0, 100.0], [140.0, 104.0], [120.0, 140.0], [96.0, 136.0]])


def _fit_from(sim: Sim2, taught: np.ndarray) -> TeamFit:
    return fit_team(taught, sim.apply(taught))


def test_error_is_zero_exactly_when_the_taught_configuration_is_reproduced():
    # Both ends transported by the same transform is, by construction, the taught
    # relation — wherever the object went.
    moved = Sim2.from_angle(0.5, t=(80.0, -40.0), s=1.0)
    e = servo_error(TAUGHT_HELD, _fit_from(moved, TAUGHT_TARGET), _fit_from(moved, TAUGHT_HELD))
    assert e.ok
    assert e.norm < 1e-6


def test_error_points_from_where_the_held_end_is_to_where_it_should_be():
    target_fit = _fit_from(Sim2.identity(), TAUGHT_TARGET)
    held_fit = _fit_from(Sim2.from_angle(0.0, t=(-30.0, 12.0)), TAUGHT_HELD)
    e = servo_error(TAUGHT_HELD, target_fit, held_fit)
    assert e.ok
    np.testing.assert_allclose(e.e_uv, [30.0, -12.0], atol=1e-6)


def test_moving_the_target_moves_the_goal_with_it():
    # Invariant 3: the goal is object-frame. Slide the target and the held end's
    # required position slides identically, with nothing re-taught.
    shift = Sim2.from_angle(0.0, t=(55.0, 25.0))
    still_held = _fit_from(Sim2.identity(), TAUGHT_HELD)
    e = servo_error(TAUGHT_HELD, _fit_from(shift, TAUGHT_TARGET), still_held)
    np.testing.assert_allclose(e.e_uv, [55.0, 25.0], atol=1e-6)


@pytest.mark.parametrize("broken", ["target", "held"])
def test_a_failed_fit_abstains_and_never_reports_zero_error(broken):
    good = _fit_from(Sim2.identity(), TAUGHT_TARGET)
    good_held = _fit_from(Sim2.identity(), TAUGHT_HELD)
    dead = TeamFit(ok=False)
    e = servo_error(
        TAUGHT_HELD, dead if broken == "target" else good, dead if broken == "held" else good_held
    )
    assert not e.ok and e.reason
    assert e.norm == 0.0, "abstention carries a zero vector, but ok=False is what callers must read"


# --- the control law ---------------------------------------------------------------


def test_command_is_clipped_to_v_max():
    pi = PIController(kp=10.0, v_max=2.0)
    u = pi.step(np.array([50.0, 0.0]), dt=0.02)
    assert np.linalg.norm(u) == pytest.approx(2.0)


def test_commands_below_backlash_are_suppressed():
    pi = PIController(kp=0.001, v_max=1.0, deadband=0.05)
    assert np.allclose(pi.step(np.array([1.0, 0.0]), dt=0.02), 0.0)


def test_the_integral_eventually_breaks_through_backlash():
    # A pure-P loop parks one deadband short of the goal forever. The integral term is
    # what converts a standing error into one real move.
    pi = PIController(kp=0.001, ki=0.01, v_max=1.0, deadband=0.05, integral_limit=100.0)
    e = np.array([1.0, 0.0])
    fired = [k for k in range(1, 9) if np.linalg.norm(pi.step(e, dt=1.0)) > 0]
    assert fired and fired[0] == 5, f"expected breakthrough on step 5, got {fired[:1]}"


def test_a_loop_that_can_never_break_through_is_refused_at_construction():
    with pytest.raises(AssertionError, match="below the deadband"):
        PIController(kp=1.0, ki=0.01, v_max=1.0, deadband=5.0, integral_limit=1.0)


def test_reset_clears_integral_state_earned_against_the_old_pose():
    pi = PIController(kp=0.001, ki=0.01, v_max=1.0, deadband=0.05, integral_limit=100.0)
    for _ in range(6):
        pi.step(np.array([1.0, 0.0]), dt=1.0)
    pi.reset()
    assert np.allclose(pi.step(np.array([1.0, 0.0]), dt=1.0), 0.0)


# --- the measured Jacobian ---------------------------------------------------------

J_TRUE = np.array([[3.0, -1.5, 0.4], [0.8, 2.2, -1.1]])


def _probe(j_true, scale=1.0):
    dq = np.eye(j_true.shape[1]) * scale
    return dq, dq @ j_true.T


def test_the_probe_recovers_the_plant_jacobian():
    est = JacobianEstimator(n_joints=3)
    est.seed_from_probe(*_probe(J_TRUE))
    np.testing.assert_allclose(est.matrix, J_TRUE, atol=1e-9)


def test_damped_solve_stays_bounded_when_a_joint_was_never_excited():
    # An under-excited joint leaves a zero column. Undamped this is a singular inverse;
    # damped it simply declines to use that joint.
    j = J_TRUE.copy()
    j[:, 2] = 0.0
    est = JacobianEstimator(n_joints=3, damping=1e-2)
    est.seed_from_probe(np.eye(3), (np.eye(3) @ j.T))
    dq = est.solve(np.array([10.0, -4.0]))
    assert np.isfinite(dq).all()
    assert abs(dq[2]) < 1e-9


def test_broyden_ignores_steps_too_small_to_carry_information():
    est = JacobianEstimator(n_joints=3)
    est.seed_from_probe(*_probe(J_TRUE))
    before = est.matrix
    est.update(np.array([1e-9, 0.0, 0.0]), np.array([5.0, 5.0]))
    np.testing.assert_allclose(est.matrix, before)


def test_broyden_corrects_a_wrong_seed_from_the_loop_s_own_motion():
    # The claim the design rests on: start from a Jacobian that is materially wrong
    # (as a bad kinematic model would give) and let ordinary servo steps fix it.
    est = JacobianEstimator(n_joints=3)
    est.seed_from_probe(np.eye(3), np.eye(3) @ (J_TRUE * 0.55).T)
    start_err = np.linalg.norm(est.matrix - J_TRUE)

    rng = np.random.default_rng(0)
    for _ in range(60):
        dq = rng.normal(scale=0.3, size=3)
        est.update(dq, J_TRUE @ dq)
    assert np.linalg.norm(est.matrix - J_TRUE) < 0.05 * start_err


def test_the_loop_closes_on_a_plant_whose_jacobian_was_seeded_wrong():
    """M1 in miniature: converge from an offset with a deliberately bad initial model."""
    est = JacobianEstimator(n_joints=3, damping=1e-3)
    est.seed_from_probe(np.eye(3), np.eye(3) @ (J_TRUE * 0.6).T)
    pi = PIController(kp=0.6, v_max=50.0)
    cert = ConvergenceCertificate(window=10, min_improvement=0.02)

    q = np.zeros(3)
    goal_uv = np.array([0.0, 0.0])
    plant = lambda qq: J_TRUE @ qq + np.array([40.0, -25.0])  # noqa: E731

    for _ in range(80):
        e = goal_uv - plant(q)
        cert.update(float(np.linalg.norm(e)))
        u = pi.step(e, dt=0.02)
        dq = est.solve(u)
        before = plant(q)
        q = q + dq
        est.update(dq, plant(q) - before)

    final = float(np.linalg.norm(goal_uv - plant(q)))
    assert final < 0.5, f"loop did not converge: {final:.3f} px"
    assert cert.progressing


# --- the error-decreasing certificate ----------------------------------------------


def test_a_partially_filled_window_is_not_yet_a_failure():
    cert = ConvergenceCertificate(window=10)
    for _ in range(5):
        cert.update(7.0)
    assert cert.progressing


def test_a_stalled_loop_is_reported_rather_than_left_to_grind():
    cert = ConvergenceCertificate(window=10, min_improvement=0.05)
    for _ in range(20):
        cert.update(7.0)
    assert not cert.progressing


def test_energy_separates_a_clean_convergence_from_a_thrashing_one():
    clean, thrash = ConvergenceCertificate(), ConvergenceCertificate()
    for k in range(20):
        clean.update(10.0 * 0.7**k)
        thrash.update(10.0 * 0.7**k + 4.0 * (k % 2))
    assert thrash.energy > clean.energy
    assert clean.best < 0.1
