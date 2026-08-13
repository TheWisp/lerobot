"""Chunk stitching: resume where the plan continues the prefix.

The offline sweep that motivated this is in the branch notes; what matters here
is that the disabled path is bit-identical to the old behaviour, that the search
picks the continuation rather than the nearest point, and that it cannot walk
off the end of a chunk.
"""

import numpy as np
import pytest

from lerobot.policies.hvla.s1_inference import choose_stitch_index

D = 2


def ramp(n, step, start=0.0, dim=3):
    """A straight constant-velocity trajectory."""
    return np.stack([np.full(dim, start + i * step) for i in range(n)])


def test_search_disabled_returns_the_committed_index():
    """rtc_stitch_search=0 must reproduce the historical resume point exactly."""
    chunk, prefix = ramp(20, 1.0), ramp(D, 1.0)
    assert choose_stitch_index(chunk, prefix, D, search=0) == D


def test_picks_the_index_that_continues_the_prefix():
    """Prefix ends at 1.0 moving +1.0/frame, so the continuation is 2.0."""
    prefix = np.stack([np.zeros(3), np.ones(3)])  # 0.0 -> 1.0, v = +1
    chunk = np.stack([np.full(3, v) for v in [0.0, 1.0, 5.0, 3.0, 2.0, 7.0]])
    # index 4 holds 2.0, the true continuation; index 2 (the default) holds 5.0
    assert choose_stitch_index(chunk, prefix, D, search=4) == 4


def test_a_plan_that_trails_is_skipped_forward():
    """The measured case: the plan reaches the prefix's position ~3 frames late."""
    prefix = ramp(D, 1.0, start=10.0)  # 10, 11 -> continuation is 12
    chunk = ramp(20, 1.0, start=7.0)  # 7,8,9,10,11,12,... -> 12 sits at index 5
    assert choose_stitch_index(chunk, prefix, D, search=8) == 5


def test_never_returns_less_than_the_committed_index():
    """Positions below d are the pinned prefix; resuming there would replay it."""
    prefix = ramp(D, 1.0, start=100.0)
    chunk = ramp(20, 1.0, start=0.0)  # every index is far below the target
    assert choose_stitch_index(chunk, prefix, D, search=8) >= D


def test_bounded_by_the_chunk_length():
    prefix = ramp(D, 1.0)
    chunk = ramp(4, 1.0)
    k = choose_stitch_index(chunk, prefix, D, search=50)
    assert D <= k < len(chunk)


@pytest.mark.parametrize("delay", [0, 1])
def test_degenerate_delays_fall_back(delay):
    """Fewer than two prefix rows gives no direction to continue."""
    assert choose_stitch_index(ramp(10, 1.0), ramp(2, 1.0), delay, search=8) == delay


def test_short_prefix_falls_back():
    assert choose_stitch_index(ramp(10, 1.0), ramp(1, 1.0), D, search=8) == D


def test_joint_subset_is_honoured():
    """Only the selected joints should drive the match."""
    prefix = np.array([[0.0, 0.0], [1.0, 50.0]])  # joint 0: v=+1; joint 1: noise
    chunk = np.array([[0.0, 0.0], [1.0, 0.0], [9.0, 0.0], [2.0, 0.0]])
    # on joint 0 alone the continuation 2.0 is at index 3
    assert choose_stitch_index(chunk, prefix, D, search=4, joint_idx=[0]) == 3
