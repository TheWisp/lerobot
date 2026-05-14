#!/usr/bin/env python

"""Rate-agnosticity tests for the predictive lookahead controller.

These tests verify that the velocity estimator and intent-ring sampling
behave correctly across a wide range of caller publish rates (10 Hz —
200 Hz). The core invariant being tested:

    same underlying trajectory → same controller behaviour,
    regardless of how often the caller publishes samples

This was historically broken: the ``_intent_ring`` was populated at the
controller's 200 Hz tick rate with whatever value happened to be in
``_latest_intent`` at that instant, which for a 30 Hz push source meant
6 of every 7 timestamps held an identical (stale) value at 5 ms apart.
The velocity estimator computed velocities from those degenerate dts
and grossly under-estimated the source's true rate of change.

The fix: populate the intent ring at the source's actual publish
events (in ``set_intent`` on the push path, in ``_tick`` on the pull
path), so the estimator's observed dts always reflect the source's
real cadence.

Tests below verify this fix is in place and stays in place.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.so107_follower_predictive.so107_follower_predictive import (
    _PredictiveLookaheadController as Ctrl,
)
from lerobot.types import ActionChunk

# =============================================================================
# Section A — velocity estimator unit tests (static methods, no controller)
# =============================================================================
#
# These exercise the four estimators directly with synthetic timestamp +
# value sequences. No mock controller / mock bus required.


def _sine_traj(freq_hz, amp, rate_hz, duration_s, n_joints=1, phase=0.0):
    """Discrete samples of a sinusoid at ``rate_hz``."""
    t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    p_one = amp * np.sin(2 * np.pi * freq_hz * t + phase)
    p = np.stack([p_one] * n_joints, axis=1)
    return t, p


def _windowed(t, p, t_now, window_s):
    mask = (t >= t_now - window_s) & (t <= t_now)
    return t[mask], p[mask]


def _analytic_deriv_of_sine(freq_hz, amp, t_now, phase=0.0):
    """d/dt [amp * sin(2π f t + φ)] = amp · 2π f · cos(2π f t + φ)."""
    return amp * 2 * np.pi * freq_hz * np.cos(2 * np.pi * freq_hz * t_now + phase)


# Tolerance in deg/s. The EMA + windowing introduces small residual
# (especially at lower rates where the window contains fewer samples).
# Tuned so the test passes consistently for rates ≥ 10 Hz with a 70 ms
# window on a 0.5 Hz sinusoid. Tighter than this risks flakiness from
# legitimate algorithmic residual rather than a real regression.
_VELOCITY_TOL_DEG_PER_S = 8.0


@pytest.mark.parametrize("rate_hz", [10, 30, 60, 100, 200])
def test_forward_diff_rate_agnostic(rate_hz):
    """forward_diff returns velocity within tolerance of the analytic
    derivative at the same time t, regardless of the input sample rate.

    Same trajectory, different sample density → same answer."""
    t_now = 1.5
    freq, amp = 0.5, 10.0
    expected = _analytic_deriv_of_sine(freq, amp, t_now)
    t, p = _sine_traj(freq, amp, rate_hz=rate_hz, duration_s=3.0)
    tw, pw = _windowed(t, p, t_now, window_s=0.07)
    if len(tw) < 2:
        pytest.skip(f"window has <2 samples at rate_hz={rate_hz}")
    v = Ctrl._velocity_forward_diff(tw, pw)
    assert v is not None
    assert abs(v[0] - expected) < _VELOCITY_TOL_DEG_PER_S, (
        f"rate={rate_hz}: estimated v={v[0]:.2f}, expected≈{expected:.2f}"
    )


@pytest.mark.parametrize("rate_hz", [30, 60, 100, 200])
def test_lsq_quad_end_rate_agnostic(rate_hz):
    """LSQ-quad velocity estimator is rate-agnostic given correct timestamps."""
    t_now = 1.5
    freq, amp = 0.5, 10.0
    expected = _analytic_deriv_of_sine(freq, amp, t_now)
    t, p = _sine_traj(freq, amp, rate_hz=rate_hz, duration_s=3.0)
    tw, pw = _windowed(t, p, t_now, window_s=0.07)
    if len(tw) < 3:
        pytest.skip(f"quad fit needs ≥3 samples; have {len(tw)}")
    v = Ctrl._velocity_lsq_quad_end(tw, pw)
    assert v is not None
    assert abs(v[0] - expected) < _VELOCITY_TOL_DEG_PER_S


@pytest.mark.parametrize("rate_hz", [30, 60, 100, 200])
def test_amp_gated_lp_rate_agnostic(rate_hz):
    """amp_gated_lowpass is rate-agnostic when amplitude gate is open.

    The amplitude in this window is ~10 deg p2p (well above amp_gate_hi=3.0
    default), so the gate is fully open and the lowpassed-forward-diff
    output should match the analytic velocity within tolerance — regardless
    of sample rate."""
    t_now = 1.5
    freq, amp = 0.5, 10.0
    expected = _analytic_deriv_of_sine(freq, amp, t_now)
    t, p = _sine_traj(freq, amp, rate_hz=rate_hz, duration_s=3.0)
    tw, pw = _windowed(t, p, t_now, window_s=0.07)
    if len(tw) < 3:
        pytest.skip(f"amp_gated_lp needs ≥3 samples; have {len(tw)}")
    v = Ctrl._velocity_amp_gated_lowpass(tw, pw, fc_hz=4.0, amp_lo=1.0, amp_hi=3.0)
    assert v is not None
    assert abs(v[0] - expected) < _VELOCITY_TOL_DEG_PER_S, (
        f"rate={rate_hz}: v_est={v[0]:.2f}, expected≈{expected:.2f}"
    )


def test_velocity_estimator_handles_steady_state_hold():
    """Identical-value samples — a real source genuinely holding position —
    must return zero velocity, not undefined or NaN.

    Regression: an earlier proposed fix used value-equality dedup on the
    intent ring, which would have discarded these samples and corrupted
    the velocity estimate during steady-state hold. This test catches
    any re-introduction of that mistake.
    """
    t = np.array([0.0, 0.033, 0.067, 0.1])
    p = np.full((4, 2), 50.0)  # constant — but they ARE four real samples
    for estimator in (
        Ctrl._velocity_forward_diff,
        Ctrl._velocity_lsq_linear,
        Ctrl._velocity_lsq_quad_end,
    ):
        v = estimator(t, p)
        assert v is not None
        assert np.allclose(v, 0.0), f"{estimator.__name__}: expected 0, got {v}"

    v = Ctrl._velocity_amp_gated_lowpass(t, p, fc_hz=4.0, amp_lo=1.0, amp_hi=3.0)
    assert v is not None
    assert np.allclose(v, 0.0)


# =============================================================================
# Section B — intent ring sampling tests (mock controller, no bus)
# =============================================================================
#
# These exercise set_intent and _tick's interaction with _intent_ring
# without spinning a real bus / connection. The minimal mock-robot
# fixture lets us isolate the ring-management logic.


def _make_mock_controller():
    """Build a controller with a stubbed robot/bus suitable for testing
    set_intent + _tick semantics WITHOUT real hardware or threads.

    The controller's __init__ requires a robot whose ``config`` supplies
    a handful of fields. We mock those.
    """
    cfg = MagicMock(name="cfg")
    cfg.lookahead_ms = 80.0
    cfg.max_lookahead_ms = 110.0
    cfg.velocity_window_ms = 70.0
    cfg.corrector_alpha = 1.0
    cfg.control_rate_hz = 200.0
    cfg.adaptive = False  # don't fire the adaptive xcorr in unit tests
    cfg.max_step_deg = 3.0
    cfg.velocity_estimator = "amp_gated_lp"
    cfg.velocity_lowpass_hz = 4.0
    cfg.amp_gate_lo = 1.0
    cfg.amp_gate_hi = 3.0

    bus = MagicMock(name="bus")
    bus.motors = {f"m{i}": MagicMock(id=i + 1) for i in range(7)}
    bus.is_connected = True
    bus.sync_read.return_value = dict.fromkeys(bus.motors, 0.0)
    bus.sync_write = MagicMock()

    robot = MagicMock(name="robot")
    robot.config = cfg
    robot.bus = bus
    robot.__str__ = lambda _self: "mock_robot"

    return Ctrl(robot)


def test_push_path_appends_at_source_rate():
    """30 Hz push of N samples → ring has N entries with ~33 ms spacing."""
    ctrl = _make_mock_controller()
    n = 5
    period = 1.0 / 30.0
    for i in range(n):
        ctrl.set_intent(t=i * period, action={f"m{j}.pos": float(i) for j in range(7)})
    assert len(ctrl._intent_ring) == n
    ts = [t for t, _ in ctrl._intent_ring]
    dts = np.diff(ts)
    assert np.allclose(dts, period, atol=1e-9)


def test_push_path_does_not_dedupe_identical_values():
    """Identical-value samples must NOT be dropped. Steady-state hold IS
    information ('velocity is zero now'), not redundancy."""
    ctrl = _make_mock_controller()
    identical = {f"m{i}.pos": 42.0 for i in range(7)}
    for i in range(5):
        ctrl.set_intent(t=i * 0.033, action=identical)
    assert len(ctrl._intent_ring) == 5  # NOT collapsed to 1


def test_push_path_records_declared_period():
    """When the caller declares period_s, the controller stashes it."""
    ctrl = _make_mock_controller()
    ctrl.set_intent(
        t=0.0,
        action={f"m{i}.pos": 1.0 for i in range(7)},
        period_s=1.0 / 30.0,
    )
    assert ctrl._declared_period_s == pytest.approx(1.0 / 30.0)
    assert ctrl._last_publish_t == pytest.approx(0.0)


def test_push_path_period_none_when_not_declared():
    """No period declared → starvation detection silently disabled."""
    ctrl = _make_mock_controller()
    ctrl.set_intent(t=0.0, action={f"m{i}.pos": 1.0 for i in range(7)})
    assert ctrl._declared_period_s is None


def test_chunk_push_infers_period_from_fps():
    """ActionChunk doesn't need explicit period — auto-inferred as 1/fps."""
    ctrl = _make_mock_controller()
    chunk = ActionChunk(
        fps=30.0,
        frames=tuple({f"m{i}.pos": float(j) for i in range(7)} for j in range(10)),
    )
    ctrl.set_intent(t=0.0, action=chunk)
    assert ctrl._declared_period_s == pytest.approx(1.0 / 30.0)
    assert ctrl._latest_chunk is not None


def test_chunk_push_does_not_populate_ring():
    """Chunk path uses _lookup_in_chunk (exact lookup) — no need to put
    chunk frames into the velocity-estimator ring. Keeping the two paths'
    state isolated avoids polluting the dict path's velocity estimate
    with chunk samples on later dict pushes."""
    ctrl = _make_mock_controller()
    chunk = ActionChunk(
        fps=30.0,
        frames=tuple({f"m{i}.pos": float(j) for i in range(7)} for j in range(10)),
    )
    ctrl.set_intent(t=0.0, action=chunk)
    assert len(ctrl._intent_ring) == 0


def test_dict_after_chunk_clears_chunk_record():
    """Switching from chunk-mode to dict-mode clears _latest_chunk so the
    controller's _tick falls into the velocity-extrapolation path. Without
    this, a stale chunk would keep being interpolated against, ignoring
    the new dict's intent."""
    ctrl = _make_mock_controller()
    chunk = ActionChunk(
        fps=30.0,
        frames=tuple({f"m{i}.pos": float(j) for i in range(7)} for j in range(10)),
    )
    ctrl.set_intent(t=0.0, action=chunk)
    assert ctrl._latest_chunk is not None
    ctrl.set_intent(t=0.033, action={f"m{i}.pos": 5.0 for i in range(7)})
    assert ctrl._latest_chunk is None
    assert len(ctrl._intent_ring) == 1  # the new dict was appended


# =============================================================================
# Section C — HVLA chunk-emission helper
# =============================================================================


def test_remaining_chunk_as_actionchunk_packs_correctly():
    from lerobot.policies.hvla.s1_process import _remaining_chunk_as_actionchunk

    chunk = np.arange(70, dtype=np.float64).reshape(10, 7)
    names = [f"joint{i}.pos" for i in range(7)]
    ac = _remaining_chunk_as_actionchunk(chunk, start_idx=3, joint_names=names, fps=30.0)

    assert isinstance(ac, ActionChunk)
    assert ac.fps == 30.0
    assert len(ac.frames) == 7  # 10 - 3
    # frame[0] of the ActionChunk MUST be chunk[start_idx]
    assert ac.frames[0]["joint0.pos"] == float(chunk[3, 0])
    assert ac.frames[0]["joint6.pos"] == float(chunk[3, 6])
    # tail frame matches chunk's last row
    assert ac.frames[-1]["joint6.pos"] == float(chunk[-1, 6])


def test_remaining_chunk_override_replaces_frame_zero_only():
    """The current_frame_override lets the caller inject a clamped value
    (e.g. POLICY JUMP CLAMP rewrite) into frame 0 without disturbing the
    remaining chunk content."""
    from lerobot.policies.hvla.s1_process import _remaining_chunk_as_actionchunk

    chunk = np.arange(70, dtype=np.float64).reshape(10, 7)
    names = [f"j{i}.pos" for i in range(7)]
    override = np.full(7, 99.0)
    ac = _remaining_chunk_as_actionchunk(
        chunk, start_idx=5, joint_names=names, fps=30.0, current_frame_override=override
    )
    # frame[0] is the override
    assert ac.frames[0]["j0.pos"] == 99.0
    # frame[1] reverts to chunk[6] (the next real frame)
    assert ac.frames[1]["j0.pos"] == float(chunk[6, 0])
