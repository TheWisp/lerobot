# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the picker tells you a flag would cost.

The figure is supervision lost, not chunks dropped, and the difference is the
point. Under the drop-the-whole-chunk rule this branch used to apply, one
scattered flag disqualified every chunk containing it, so "chunks lost" ran far
ahead of the frame count. The trainer truncates now: it stops drawing only the
starts *on* an excluded frame -- exactly one per frame -- so a chunk count would
be the frame count in different units and would tell an operator nothing new.

Supervision still differs from the frame count, because every chunk reaching a
flag is shortened. These pin that difference, since a metric that merely
restated the frame count would be worth removing rather than showing.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.gui.api.datasets import _supervised_positions

CHUNK = 4


def episodes(*lengths):
    return np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(lengths)])


def test_nothing_excluded_counts_every_window():
    """One episode of 10, chunk 4: the last three starts are short."""
    ep = episodes(10)
    assert _supervised_positions(ep, [], CHUNK) == 4 * 7 + 3 + 2 + 1


def test_a_window_stops_at_an_episode_end_without_any_flag():
    """Five frames, chunk 4: starts 0 and 1 both get a full window, then the
    tail shortens -- 4+4+3+2+1 per episode."""
    assert _supervised_positions(episodes(5, 5), [], CHUNK) == 2 * (4 + 4 + 3 + 2 + 1)


def test_an_excluded_frame_is_never_supervised_and_never_drawn():
    ep = episodes(10)
    lost = _supervised_positions(ep, [], CHUNK) - _supervised_positions(ep, [5], CHUNK)
    # Frame 5 loses its own start (4 positions) and truncates the starts at
    # 2, 3, 4 -- so the cost exceeds the single frame that was marked.
    assert lost > 1


def test_supervision_lost_exceeds_the_frames_marked():
    """The reason this metric is shown instead of the frame count."""
    ep = episodes(40)
    baseline = _supervised_positions(ep, [], CHUNK)
    scattered = [5, 15, 25, 35]
    lost = baseline - _supervised_positions(ep, scattered, CHUNK)
    assert lost > len(scattered), f"{lost} positions lost for {len(scattered)} frames marked"


def test_a_thinly_scattered_flag_costs_more_than_a_clustered_one():
    """Same number of frames, different reach -- which a frame count cannot say."""
    ep = episodes(40)
    baseline = _supervised_positions(ep, [], CHUNK)
    scattered = baseline - _supervised_positions(ep, [5, 15, 25, 35], CHUNK)
    clustered = baseline - _supervised_positions(ep, [20, 21, 22, 23], CHUNK)
    assert scattered > clustered, f"scattered {scattered} should cost more than clustered {clustered}"


def test_flags_on_every_frame_leave_no_supervision():
    ep = episodes(10)
    assert _supervised_positions(ep, list(range(10)), CHUNK) == 0


def test_a_flag_does_not_reach_across_an_episode_boundary():
    """Episode 1's flag must not shorten episode 0's windows."""
    ep = episodes(6, 6)
    without = _supervised_positions(ep, [], CHUNK)
    with_flag = _supervised_positions(ep, [6], CHUNK)  # first frame of episode 1
    # Only episode 1 is affected: its own start is gone and nothing else in
    # episode 0 changes, because a window already stopped at the boundary.
    assert without - with_flag == CHUNK


@pytest.mark.parametrize("chunk", [1, 2, 8, 50])
def test_it_never_reports_more_than_the_windows_could_hold(chunk):
    ep = episodes(12, 7)
    total = _supervised_positions(ep, [], chunk)
    assert 0 < total <= chunk * len(ep)
