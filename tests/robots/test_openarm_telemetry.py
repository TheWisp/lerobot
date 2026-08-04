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

"""Tests for the follower telemetry accumulator."""

import logging

import numpy as np

from lerobot.robots.openarm_follower.telemetry import TELEMETRY_PERIOD_SEC, FollowerTelemetry


def test_no_report_before_period(caplog):
    t = FollowerTelemetry("test_arm")
    t.update(np.zeros(8), np.full(8, 0.1), np.zeros(8), np.full(8, 40))
    with caplog.at_level(logging.INFO):
        assert not t.maybe_report(now=t.t0 + TELEMETRY_PERIOD_SEC - 1e-3)
    assert caplog.text == ""


def test_report_line_and_reset(caplog):
    t = FollowerTelemetry("test_arm")
    cmd = np.zeros(8)
    # Gripper slot (index 7) carries junk: it must be excluded everywhere.
    t.update(cmd, np.full(8, 0.01), np.full(8, 2.0), [41] * 7 + [95])
    t.update(cmd, np.full(8, -0.02), np.full(8, -4.0), np.full(8, 47))
    with caplog.at_level(logging.INFO):
        assert t.maybe_report(now=t.t0 + TELEMETRY_PERIOD_SEC)
    line = caplog.text.strip()
    assert "\n" not in line, "telemetry must be a single line"
    assert "[test_arm] telem n=2 " in line
    assert "err_max=[0.020 0.020 0.020 0.020 0.020 0.020 0.020]" in line
    assert "tau_ext_mean=[-1.00 -1.00 -1.00 -1.00 -1.00 -1.00 -1.00]" in line
    assert "tau_ext_absmean=[3.00 3.00 3.00 3.00 3.00 3.00 3.00]" in line
    assert "tau_ext_absmax=[4.00 4.00 4.00 4.00 4.00 4.00 4.00]" in line
    assert "tmax=47" in line
    assert t.count == 0, "accumulators reset after a report"
    assert not t.maybe_report(now=t.t0 + TELEMETRY_PERIOD_SEC + 1.0)


def test_tff_subtracted_from_tau_ext(caplog):
    t = FollowerTelemetry("test_arm")
    t.update(np.zeros(8), np.zeros(8), np.full(8, 3.0), np.full(8, 40), tff=np.full(7, 1.0))
    with caplog.at_level(logging.INFO):
        assert t.maybe_report(now=t.t0 + TELEMETRY_PERIOD_SEC)
    assert "tau_ext_absmax=[2.00 2.00 2.00 2.00 2.00 2.00 2.00]" in caplog.text
