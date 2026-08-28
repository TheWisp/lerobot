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
"""What a run drew, recorded per frame and readable afterwards.

The claim under test is that the trace is *sufficient*: that "which frames did
this run learn from, and how often" can be answered from the file alone, with no
dataset and no log. Supervision is not recorded, only derived, so the derivation
is what most of these check.
"""

import numpy as np
import pytest

from lerobot.datasets.sampler import EpisodeAwareSampler, ExcludedStartSampler
from lerobot.datasets.sampling_trace import (
    counts_path,
    load_sampling_trace,
    open_draw_counts,
    save_sampling_trace,
    supervised_counts,
    window_end,
)

# Three episodes of ten frames, laid out back to back.
FROM = [0, 10, 20]
TO = [10, 20, 30]


def sampler(excluded=None, **kwargs):
    """A sampler that counts. Counting is opt-in, so the buffer is explicit.

    A bare `EpisodeAwareSampler` deliberately allocates nothing -- an int32 per
    frame is 687 MB at 1000 hours of 50 fps, and charging every sampler for a
    tally most of them discard is what this default avoids.
    """
    kwargs.setdefault("draw_counts", np.zeros(TO[-1], dtype=np.int32))
    if not excluded:
        return EpisodeAwareSampler(FROM, TO, **kwargs)
    return ExcludedStartSampler(FROM, TO, excluded_frames=excluded, **kwargs)


# ── Counting the draws ────────────────────────────────────────────────────────


def test_nothing_is_counted_before_the_first_epoch():
    assert sampler().draw_counts.tolist() == [0] * 30


def test_one_epoch_draws_every_start_exactly_once():
    s = sampler()
    list(s)
    assert s.draw_counts.tolist() == [1] * 30


def test_counts_accumulate_across_epochs():
    s = sampler(shuffle=True, seed=1)
    list(s)
    list(s)
    assert s.draw_counts.tolist() == [2] * 30


def test_counts_are_indexed_by_absolute_frame_not_by_position():
    """A run over a subset of episodes must still be comparable with a run over
    all of them, which it is not if the index means "position in what I loaded"."""
    s = EpisodeAwareSampler(FROM, TO, episode_indices_to_use=[2], draw_counts=np.zeros(30, dtype=np.int32))
    list(s)
    counts = s.draw_counts
    assert counts.size == 30, "the array spans the dataset, not the subset"
    assert counts[:20].tolist() == [0] * 20
    assert counts[20:].tolist() == [1] * 10


def test_a_relative_mapping_does_not_move_the_counts():
    """When only some episodes are loaded the sampler yields relative indices;
    the trace must still be in absolute ones."""
    mapping = {i: i - 10 for i in range(10, 20)}
    s = EpisodeAwareSampler(
        [10], [20], absolute_to_relative_idx=mapping, draw_counts=np.zeros(20, dtype=np.int32)
    )
    drawn = sorted(s)
    assert drawn == list(range(10)), "the consumer still gets relative indices"
    assert np.flatnonzero(s.draw_counts).tolist() == list(range(10, 20))


def test_excluded_frames_are_never_counted():
    s = sampler(excluded=[3, 4, 22])
    list(s)
    assert s.draw_counts[[3, 4, 22]].tolist() == [0, 0, 0]
    assert int(s.draw_counts.sum()) == 27


def test_introspecting_the_indices_is_not_a_draw():
    """`indices` walks every position; counting there would record draws that
    never happened, and the number would quietly be wrong from then on."""
    s = sampler()
    _ = s.indices
    _ = s.indices
    assert s.draw_counts.tolist() == [0] * 30


def test_a_partial_epoch_counts_only_what_it_yielded():
    """Resume replays from an offset; the trace must reflect the run, not the
    plan."""
    s = sampler(shuffle=True, seed=4)
    it = iter(s)
    for _ in range(7):
        next(it)
    assert int(s.draw_counts.sum()) == 7


# ── Round-tripping the file ───────────────────────────────────────────────────


def test_the_file_round_trips(tmp_path):
    s = sampler(excluded=[5])
    list(s)
    path = save_sampling_trace(
        tmp_path / "trace",
        draw_counts=s.draw_counts,
        episode_from=np.array(FROM),
        episode_to=np.array(TO),
        excluded_frames=np.array([5]),
        step=100,
    )
    trace = load_sampling_trace(path)
    assert trace["draw_counts"].tolist() == s.draw_counts.tolist()
    assert trace["excluded_frames"].tolist() == [5]
    assert trace["episode_to"].tolist() == TO
    assert int(trace["step"]) == 100


def test_a_run_that_excluded_nothing_still_writes_a_usable_trace(tmp_path):
    """`excluded_frames` absent must mean "none", not "unknown" -- otherwise
    every unfiltered run's trace is uninterpretable."""
    path = save_sampling_trace(
        tmp_path / "trace",
        draw_counts=np.ones(30, dtype=np.int32),
        episode_from=np.array(FROM),
        episode_to=np.array(TO),
    )
    trace = load_sampling_trace(path)
    assert trace["excluded_frames"].tolist() == []
    assert supervised_counts(trace, chunk_size=1).tolist() == [1] * 30


# ── Deriving supervision, which is the point of the file ──────────────────────


def test_a_window_stops_at_the_episode_end():
    assert window_end(8, np.array(TO), np.array([]), 30) == 10


def test_a_window_stops_at_the_first_excluded_frame():
    assert window_end(8, np.array(TO), np.array([9]), 30) == 9


def test_a_window_takes_whichever_boundary_comes_first():
    """An excluded frame in the *next* episode must not extend this one."""
    assert window_end(8, np.array(TO), np.array([15]), 30) == 10


def test_supervision_is_derived_without_the_dataset(tmp_path):
    """The whole reason draws are stored and supervision is not."""
    s = sampler()
    list(s)
    path = save_sampling_trace(
        tmp_path / "trace",
        draw_counts=s.draw_counts,
        episode_from=np.array(FROM),
        episode_to=np.array(TO),
    )
    counts = supervised_counts(load_sampling_trace(path), chunk_size=4)
    # Frame 0 is covered only by the start at 0; frame 3 by starts 0..3.
    assert counts[0] == 1
    assert counts[3] == 4
    # The episode's last frame is covered by the four starts that reach it.
    assert counts[9] == 4
    # ...and nothing from episode 0 leaks into episode 1's first frame.
    assert counts[10] == 1


def test_no_excluded_frame_is_ever_supervised(tmp_path):
    """By construction rather than by filtering: a window stops *at* the
    excluded frame, so no draw covers it."""
    excluded = [4, 5, 21]
    s = sampler(excluded=excluded)
    list(s)
    path = save_sampling_trace(
        tmp_path / "trace",
        draw_counts=s.draw_counts,
        episode_from=np.array(FROM),
        episode_to=np.array(TO),
        excluded_frames=np.array(excluded),
    )
    counts = supervised_counts(load_sampling_trace(path), chunk_size=6)
    assert counts[excluded].tolist() == [0, 0, 0]
    # Everything else in those episodes is still learned from.
    assert counts[3] > 0 and counts[6] > 0 and counts[20] > 0


def test_the_derived_total_matches_what_the_windows_cover(tmp_path):
    """A cross-check on the derivation itself: summing the per-frame counts must
    equal summing each drawn window's length."""
    excluded = [7, 23]
    s = sampler(excluded=excluded)
    list(s)
    trace = {
        "draw_counts": s.draw_counts,
        "episode_to": np.array(TO),
        "excluded_frames": np.array(excluded),
    }
    chunk = 5
    expected = sum(
        min(start + chunk, window_end(start, np.array(TO), np.array(excluded), 30)) - start
        for start in np.flatnonzero(s.draw_counts)
    )
    assert int(supervised_counts(trace, chunk_size=chunk).sum()) == expected


@pytest.mark.parametrize("chunk", [1, 2, 5, 20])
def test_supervision_never_exceeds_the_frames_that_exist(tmp_path, chunk):
    s = sampler()
    list(s)
    trace = {
        "draw_counts": s.draw_counts,
        "episode_to": np.array(TO),
        "excluded_frames": np.array([], dtype=np.int64),
    }
    counts = supervised_counts(trace, chunk_size=chunk)
    assert counts.size == 30
    assert (counts >= 0).all()


# ── The counter is a file, not a resident array ───────────────────────────────


def test_the_counter_can_be_file_backed(tmp_path):
    """An int32 per frame is 160 KB at 40k frames and 687 MB at 1000 hours of
    50 fps. Mapped, those pages are reclaimable page cache instead of an
    allocation held for the length of the run."""
    counts = open_draw_counts(tmp_path / "trace", 30)
    assert isinstance(counts, np.memmap)
    assert counts.shape == (30,)
    assert counts.tolist() == [0] * 30

    s = EpisodeAwareSampler(FROM, TO, draw_counts=counts)
    list(s)
    counts.flush()
    on_disk = np.fromfile(counts_path(tmp_path / "trace"), dtype=np.int32)
    assert on_disk.tolist() == [1] * 30, "the draws must reach the file, not just the mapping"


def test_reopening_continues_the_same_tally(tmp_path):
    """A resumed run must not start a second count beside the first."""
    counts = open_draw_counts(tmp_path / "trace", 30)
    list(EpisodeAwareSampler(FROM, TO, draw_counts=counts))
    counts.flush()
    del counts

    reopened = open_draw_counts(tmp_path / "trace", 30)
    assert reopened.tolist() == [1] * 30
    list(EpisodeAwareSampler(FROM, TO, draw_counts=reopened))
    assert reopened.tolist() == [2] * 30


def test_a_short_buffer_is_refused(tmp_path):
    """Silently dropping the tail's counts would make the trace wrong in exactly
    the region a partial run is least able to notice."""
    with pytest.raises(ValueError, match="short buffer"):
        EpisodeAwareSampler(FROM, TO, draw_counts=np.zeros(5, dtype=np.int32))


def test_saving_does_not_copy_the_counts(tmp_path):
    """The counts are already on disk; copying them at every checkpoint is the
    cost this design exists to avoid."""
    directory = tmp_path / "trace"
    counts = open_draw_counts(directory, 30)
    list(EpisodeAwareSampler(FROM, TO, draw_counts=counts))
    save_sampling_trace(
        directory,
        draw_counts=counts,
        episode_from=np.array(FROM),
        episode_to=np.array(TO),
    )
    # One counts file, written once, still the live one.
    assert counts_path(directory).stat().st_size == 30 * 4
    trace = load_sampling_trace(directory)
    assert trace["draw_counts"].tolist() == [1] * 30
    assert isinstance(trace["draw_counts"], np.memmap), "reading back must map, not slurp"


def test_every_rank_would_count_the_whole_epoch_so_only_one_counts(tmp_path):
    """Why there is one counter and not one per rank.

    accelerate sheds work by *yielding* a rank's batches, not by restricting
    what the sampler enumerates: `BatchSamplerShard._iter_with_no_split` walks
    the whole underlying sampler on every process. So each rank's tally is the
    entire epoch, and adding them up reports world_size times the truth.
    """
    # accelerate is not in CI's extras; this test explains a decision rather
    # than guarding a contract, so skipping where it is absent is right.
    accelerate_loader = pytest.importorskip("accelerate.data_loader")
    from torch.utils.data import BatchSampler

    world = 2
    tallies, consumed = [], []
    for rank in range(world):
        counts = np.zeros(30, dtype=np.int32)
        sampler = EpisodeAwareSampler(FROM, TO, shuffle=True, seed=0, draw_counts=counts)
        shard = accelerate_loader.BatchSamplerShard(
            BatchSampler(sampler, batch_size=5, drop_last=True),
            num_processes=world,
            process_index=rank,
            split_batches=False,
        )
        consumed.append(sum(len(batch) for batch in shard))
        tallies.append(counts)

    assert consumed == [15, 15], "each rank trains on half the epoch"
    assert all(int(t.sum()) == 30 for t in tallies), "but each one counted all of it"
    assert np.array_equal(tallies[0], tallies[1]), "the tallies are duplicates, not complements"


def test_a_trace_with_no_counts_is_an_error_not_a_zero(tmp_path):
    """All-zeros would be indistinguishable from a run that drew nothing."""
    directory = tmp_path / "trace"
    directory.mkdir()
    np.savez_compressed(
        directory / "meta.npz",
        episode_from=np.array(FROM),
        episode_to=np.array(TO),
        excluded_frames=np.array([], dtype=np.int64),
        num_frames=np.asarray(30),
    )
    with pytest.raises(FileNotFoundError):
        load_sampling_trace(directory)
