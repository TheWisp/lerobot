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
    """A flag costs more than the frames it marks -- but this alone does not
    prove the truncation is modelled, since each marked frame already forfeits
    its own start. The test below is the one that pins truncation."""
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


@pytest.mark.parametrize("chunk", [1, 5, 50])
@pytest.mark.parametrize(
    "flagged",
    [[], [4], [4, 5, 6], [0, 12, 13, 41], list(range(0, 43, 3))],
    ids=["none", "one", "run", "boundaries", "every-third"],
)
def test_it_agrees_with_the_shared_statement_of_the_rule(chunk, flagged):
    """``sampling_trace.window_end`` is where this rule is written down for the
    whole repository -- "the episode end, or the first excluded frame at or
    after the start, whichever comes first".

    This estimator is a third copy of it, vectorised because the form prices
    every label over every frame and a Python loop over the starts would not
    return while a form is open. Vectorised is the only thing it is allowed to
    be: fed the same frames it must produce what summing the shared helper over
    every drawn start produces.
    """
    from lerobot.datasets.sampling_trace import window_end

    lengths = [13, 9, 21]
    ep = episodes(*lengths)
    episode_to = np.cumsum(lengths).astype(np.int64)
    excluded = np.sort(np.asarray(flagged, dtype=np.int64))
    drawn = np.ones(len(ep), dtype=bool)
    drawn[excluded] = False

    oracle = sum(
        max(0, min(start + chunk, window_end(start, episode_to, excluded, len(ep))) - start)
        for start in range(len(ep))
        if drawn[start]
    )
    assert _supervised_positions(ep, flagged, chunk) == oracle


def _dataset_with(tmp_path, features, columns, lengths):
    """A dataset directory holding just what the endpoint reads: info.json and
    the parquet columns. Not a LeRobotDataset -- the endpoint deliberately does
    not open one, so building one here would exercise a path it never takes."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "flagged"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_frames": sum(lengths),
                "total_episodes": len(lengths),
                "features": features,
            }
        )
    )
    ep = episodes(*lengths)
    pq.write_table(pa.table({"episode_index": ep, **columns}), root / "data" / "chunk-000.parquet")
    return root, ep


def test_a_flag_two_columns_declare_is_priced_over_both(tmp_path):
    """``resolve_flag_masks`` excludes a name from every column declaring it,
    and the picker offers one box per name because the vocabulary is deduplicated
    across columns. A row per column would price a choice the operator cannot
    make, and -- keyed by name on the way to the DOM -- would show whichever
    column happened to come last.
    """
    from lerobot.gui.api.datasets import _read_flags_impact

    lengths = (15, 15)
    per_frame = np.zeros(sum(lengths), dtype=np.int64)
    per_frame[[3, 4]] = 0b01  # 'fumble' is bit 0 of "quality"
    per_episode = np.zeros(sum(lengths), dtype=np.int64)
    per_episode[15:] = 0b01  # and bit 0 of "quality.episode", over episode 1
    root, ep = _dataset_with(
        tmp_path,
        {
            "quality": {"dtype": "int64", "shape": (1,), "flags": ["fumble", "blurry"]},
            "quality.episode": {
                "dtype": "int64",
                "shape": (1,),
                "flags": ["fumble"],
                "per_episode": True,
            },
        },
        {"quality": per_frame, "quality.episode": per_episode},
        lengths,
    )

    out = _read_flags_impact(str(root), chunk_size=CHUNK)
    rows = [r for r in out["labels"] if r["label"] == "fumble"]
    assert len(rows) == 1, f"one row per name, got {[r['features'] for r in rows]}"
    row = rows[0]
    assert sorted(row["features"]) == ["quality", "quality.episode"]
    assert row["per_episode"] is True, "a name any per-episode column declares removes whole takes"

    union = np.flatnonzero((per_frame & 1) | (per_episode & 1))
    assert row["frames"] == len(union)
    assert row["positions_lost"] == out["total_positions"] - _supervised_positions(ep, union, CHUNK)

    # And the union must actually cost more than either column alone, or this
    # fixture would pass with the per-column bug still in place.
    only_frame = out["total_positions"] - _supervised_positions(ep, np.flatnonzero(per_frame & 1), CHUNK)
    only_episode = out["total_positions"] - _supervised_positions(ep, np.flatnonzero(per_episode & 1), CHUNK)
    assert row["positions_lost"] > max(only_frame, only_episode)


def test_a_flag_only_one_column_declares_names_that_column(tmp_path):
    """The ordinary case, which the union must not disturb."""
    from lerobot.gui.api.datasets import _read_flags_impact

    lengths = (12,)
    quality = np.zeros(sum(lengths), dtype=np.int64)
    quality[[2, 7]] = 0b10  # 'blurry' is bit 1
    root, _ = _dataset_with(
        tmp_path,
        {"quality": {"dtype": "int64", "shape": (1,), "flags": ["fumble", "blurry"]}},
        {"quality": quality},
        lengths,
    )

    by_label = {r["label"]: r for r in _read_flags_impact(str(root), chunk_size=CHUNK)["labels"]}
    assert by_label["blurry"]["features"] == ["quality"]
    assert by_label["blurry"]["frames"] == 2
    assert by_label["blurry"]["per_episode"] is False
    # Declared but carried by nothing: still offered, priced at zero.
    assert by_label["fumble"]["frames"] == 0
    assert by_label["fumble"]["positions_lost"] == 0
