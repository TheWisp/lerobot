"""Pause evidence must retain long stops that an active-motion window excludes."""

import numpy as np
import pytest

from lerobot.gui.api.manipulation import recording


def test_pause_durations_include_a_long_stationary_middle_and_distinguish_queue_state():
    t = np.arange(61) / 30
    q = np.zeros((61, 7))
    q[:, 0] = np.r_[np.arange(10), np.full(31, 9), np.arange(10, 30)]
    waiting = np.zeros(61, int)
    waiting[10:25] = 1
    depth = np.full(61, 10)
    depth[25:40] = 0
    arm = {"t": t, "q_cmd": q, "q_obs": q, "playback_waiting": waiting, "playback_queue_depth": depth}
    result = recording._control_cadence(arm, [])
    pauses = result["pauses"]
    assert pauses["command_stationary"]["total_s"] == pytest.approx(31 / 30)
    assert pauses["command_stationary"]["longest_s"] == pytest.approx(31 / 30)
    assert pauses["lag_wait"]["total_s"] == pytest.approx(0.5)
    assert pauses["queue_empty"]["total_s"] == pytest.approx(0.5)
    assert not pauses["playback_state_recorded"]
    # Legacy active windows omit the middle of this stop. The new duration
    # must not shrink to those few boundary ticks.
    assert result["active_ticks"] < 60
