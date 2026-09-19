#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the follower telemetry accumulator.

What is worth pinning here is the arithmetic, not the wording. Three bug
classes would each turn the telemetry into a confident lie:

* including the gripper slot, which runs POS_FORCE rather than MIT and sits at
  a temperature the arm never reaches — a false overheat in the log;
* getting the feedforward subtraction wrong, which makes ``tau_ext`` — the
  external-force reading the telemetry exists to provide — meaningless;
* failing to reset, after which every window reports lifetime maxima rather
  than the window's.

The assertions are therefore on the accumulated numbers. Only one test looks at
the log text, for the single property the format has to keep: it is one line
and says which arm it came from.

Timing is asserted away from the boundary on purpose. ``now = t0 + PERIOD``
does not round-trip in float at every magnitude of ``time.monotonic()`` —
``(1008.684344071 + 30.0) - 1008.684344071 == 29.999999999999886`` — so a test
sitting exactly on it passes or fails according to the machine's uptime, which
is what issue #116 was. The requirement is "reports at or after the period",
never "reports at exactly the period".
"""

import logging

import numpy as np
import pytest

from lerobot.robots.openarm_follower.telemetry import TELEMETRY_PERIOD_SEC, FollowerTelemetry

# Inputs are 8 wide (7 arm joints + gripper); telemetry covers the arm only.
ARM = 7
GRIPPER_JUNK = 95.0


def _telem(**kw) -> FollowerTelemetry:
    return FollowerTelemetry("test_arm", **kw)


def test_nothing_is_reported_before_the_period_elapses():
    t = _telem()
    t.update(np.zeros(8), np.full(8, 0.1), np.zeros(8), np.full(8, 40))
    assert not t.maybe_report(now=t.t0 + TELEMETRY_PERIOD_SEC - 1.0)


def test_nothing_is_reported_when_no_cycle_was_recorded():
    """An idle arm must not log an empty summary, nor divide by zero."""
    t = _telem()
    assert not t.maybe_report(now=t.t0 + TELEMETRY_PERIOD_SEC * 2)


def test_the_gripper_slot_never_reaches_the_telemetry():
    """The gripper runs POS_FORCE and sits far hotter than the arm.

    Including it would report an overheat that is not happening — the reading
    an operator would act on.
    """
    t = _telem()
    t.update(
        q_cmd=np.zeros(8),
        q_pos=np.zeros(8),
        q_torque=np.zeros(8),
        t_mos=[41.0] * ARM + [GRIPPER_JUNK],
    )
    assert t.tmax == 41.0, "the gripper's temperature must not become the arm's max"
    assert len(t.err_max) == ARM
    assert len(t.tau_abs_max) == ARM


def test_feedforward_is_subtracted_from_the_measured_torque():
    """``tau_ext = q_torque - tff``. A sign error here inverts every reading."""
    t = _telem()
    t.update(np.zeros(8), np.zeros(8), np.full(8, 3.0), np.full(8, 40), tff=np.full(ARM, 1.0))
    assert np.allclose(t.tau_abs_max, 2.0)
    assert np.allclose(t.tau_sum, 2.0)


def test_without_feedforward_the_torque_is_taken_as_measured():
    t = _telem()
    t.update(np.zeros(8), np.zeros(8), np.full(8, 3.0), np.full(8, 40))
    assert np.allclose(t.tau_abs_max, 3.0)


def test_the_window_summarises_its_own_cycles_only():
    """Signed mean, absolute mean and absolute max are three different things.

    Two cycles of +2 and -4: the signed mean is negative, the absolute mean is
    3, and the absolute max is 4. Collapsing any pair of them would hide either
    a bias or a spike.
    """
    t = _telem()
    t.update(np.zeros(8), np.full(8, 0.01), np.full(8, 2.0), np.full(8, 41))
    t.update(np.zeros(8), np.full(8, -0.02), np.full(8, -4.0), np.full(8, 47))

    assert t.count == 2
    assert np.allclose(t.err_max, 0.02), "err_max is a max of absolute error"
    assert np.allclose(t.tau_sum / t.count, -1.0)
    assert np.allclose(t.tau_abs_sum / t.count, 3.0)
    assert np.allclose(t.tau_abs_max, 4.0)
    assert t.tmax == 47.0


def test_reporting_resets_the_window():
    """Without this every window reports lifetime maxima, not its own."""
    t = _telem()
    t.update(np.zeros(8), np.full(8, 5.0), np.full(8, 9.0), np.full(8, 60))
    assert t.maybe_report(now=t.t0 + TELEMETRY_PERIOD_SEC * 2)

    assert t.count == 0
    assert np.allclose(t.err_max, 0.0)
    assert np.allclose(t.tau_abs_max, 0.0)
    assert t.tmax == 0.0
    assert not t.maybe_report(now=t.t0 + TELEMETRY_PERIOD_SEC * 2), "the clock restarts too"


@pytest.mark.parametrize("elapsed_factor", [1.5, 2.0, 10.0])
def test_a_window_that_is_well_past_due_reports(elapsed_factor):
    """Asserted away from the boundary — see the module docstring for why."""
    t = _telem()
    t.update(np.zeros(8), np.zeros(8), np.zeros(8), np.full(8, 40))
    assert t.maybe_report(now=t.t0 + TELEMETRY_PERIOD_SEC * elapsed_factor)


def test_the_summary_is_one_line_and_names_the_arm(caplog):
    """The only property of the wording worth pinning.

    It is one line because it is read in a scrolling terminal beside the
    control loop's own output, and it names the arm because a bimanual rig logs
    two of them.
    """
    t = _telem()
    t.update(np.zeros(8), np.zeros(8), np.zeros(8), np.full(8, 40))
    with caplog.at_level(logging.INFO):
        assert t.maybe_report(now=t.t0 + TELEMETRY_PERIOD_SEC * 2)

    line = caplog.text.strip()
    assert "\n" not in line, "telemetry must stay a single line"
    assert line.startswith("[test_arm]") or "[test_arm]" in line


def test_a_custom_period_is_honoured():
    """The period is a constructor argument; a caller that shortens it for a
    short experiment should not have to wait the default."""
    t = _telem(period_secs=1.0)
    t.update(np.zeros(8), np.zeros(8), np.zeros(8), np.full(8, 40))
    assert not t.maybe_report(now=t.t0 + 0.5)
    assert t.maybe_report(now=t.t0 + 2.0)
