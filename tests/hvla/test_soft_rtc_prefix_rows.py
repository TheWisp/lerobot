"""The runtime's RTC prefix slice under Soft RTC.

The delay and the number of prefix rows are the same number under Hard RTC and
different under Soft RTC, which is exactly the kind of distinction that decays
back into a single variable during a refactor. These pin it.
"""

import numpy as np
import pytest

from lerobot.policies.hvla.s1_inference import _rtc_prefix_for_observation

FPS = 30
CHUNK = np.arange(50 * 4, dtype=np.float32).reshape(50, 4)


def call(delay_s, soft_len=0, soft_hmax=8, max_delay=6, obs_t=0.0, origin=0.0, chunk=CHUNK):
    return _rtc_prefix_for_observation(
        old_chunk=chunk,
        old_chunk_origin=origin,
        observation_time=obs_t,
        estimated_delay_s=delay_s,
        fps=FPS,
        max_delay=max_delay,
        soft_len=soft_len,
        soft_hmax=soft_hmax,
    )


def test_hard_rtc_rows_equal_delay():
    """Unchanged behaviour: soft_len=0 returns exactly `delay` rows."""
    for delay_s in (1 / FPS, 2 / FPS, 3 / FPS, 6 / FPS):
        _start, delay, prefix = call(delay_s, soft_len=0)
        assert prefix.shape[0] == delay


def test_soft_rtc_returns_extra_rows_without_changing_the_delay():
    _start, delay, hard = call(2 / FPS, soft_len=0)
    _start2, delay2, soft = call(2 / FPS, soft_len=3)
    assert delay2 == delay, "the hard delay must not move when a soft window is added"
    assert soft.shape[0] == delay + 3
    assert np.array_equal(soft[:delay], hard), "the committed rows must be identical"


def test_soft_rows_are_capped_by_hmax():
    _start, delay, prefix = call(2 / FPS, soft_len=100, soft_hmax=5)
    assert prefix.shape[0] == 5
    assert delay == 2


def test_soft_rows_never_run_past_the_old_chunk():
    short = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    start, delay, prefix = call(2 / FPS, soft_len=6, soft_hmax=8, chunk=short, obs_t=1 / FPS)
    assert start + prefix.shape[0] <= len(short)
    assert prefix.shape[0] >= delay


@pytest.mark.parametrize("soft_len", [0, 1, 4])
def test_prefix_starts_at_the_observation_time(soft_len):
    """Position zero must be the old trajectory's action at the observation."""
    start, _delay, prefix = call(2 / FPS, soft_len=soft_len, obs_t=5 / FPS, origin=0.0)
    assert start == 5
    assert np.array_equal(prefix[0], CHUNK[5])
