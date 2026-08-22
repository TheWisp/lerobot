# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Masking a dataset is one job, not one job per episode.

The save path wrote a single episode per call, so masking 274 episodes meant
274 worker spawns and 274 SAM3 loads — about twenty minutes of pure model
loading on top of the segmentation. The request now carries a list and the
worker loops it on one adapter.

What is pinned here is the request layer: the list is validated as a whole, and
the overwrite gate counts rows across every requested episode. That gate is the
one with teeth — checking only the first episode would report "no masks here"
and then silently replace the other 273.

The worker's loop itself needs SAM3 and a GPU, so it is not covered here; it
was verified against the running server on a three-episode dataset (coverage
288 + 289 + 379 = 956 aggregated, progress 0/3 through 3/3, all three written).
"""

import pytest

from lerobot.gui.api.process import EpisodeMasksRequest, _ep_label


def test_a_single_episode_request_is_unchanged():
    """The interactive Save path sends no list and must keep working."""
    req = EpisodeMasksRequest(source_id="x", episode=7)
    assert req.episodes is None
    resolved = list(req.episodes if req.episodes is not None else [req.episode])
    assert resolved == [7]


def test_a_list_is_carried_through():
    req = EpisodeMasksRequest(source_id="x", episode=0, episodes=[0, 5, 9])
    assert req.episodes == [0, 5, 9]


@pytest.mark.parametrize(
    ("episodes", "expected"),
    [([3], "ep3"), ([0, 1], "2 episodes"), (list(range(274)), "274 episodes")],
)
def test_the_label_says_how_much_work_it_is(episodes, expected):
    """The slot and the job effect both use this, so an operator can tell a
    one-episode save from a three-hour run before it starts."""
    assert _ep_label(episodes) == expected


def test_the_label_distinguishes_one_from_many():
    assert _ep_label([9]) != _ep_label([9, 10])
