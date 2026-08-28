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
"""Excluding starts from the sampler, rather than drawing and wasting them.

A start on a flagged frame produces a wholly padded sample: drawn, decoded,
collated and forward-passed to contribute nothing. Drawing them anyway makes
the sampler uniform over *frames* while the useful draws are a fraction of
that. These pin that the draw is uniform over the starts that can teach
something, and that everything else about the sampler is unchanged.
"""

import numpy as np
import pytest

from lerobot.datasets.sampler import (
    EpisodeAwareSampler,
    ExcludedStartSampler,
    make_start_sampler,
)

# Three episodes of ten frames, laid out back to back.
FROM = [0, 10, 20]
TO = [10, 20, 30]


def sampler(excluded=None, **kwargs):
    """The plain sampler when nothing is excluded, the subclass when something is.

    Mirrors what the trainers do: the exclusion subclass is only reached by a
    run that actually excludes frames, so an ordinary run keeps the parent's
    interval representation and pays nothing.
    """
    if not excluded:
        return EpisodeAwareSampler(FROM, TO, **kwargs)
    return ExcludedStartSampler(FROM, TO, excluded_frames=excluded, **kwargs)


def test_without_exclusions_nothing_changes():
    plain, empty = sampler(), sampler(excluded=[])
    assert plain.indices == list(range(30))
    assert empty.indices == list(range(30))
    assert len(plain) == len(empty) == 30


def test_excluded_frames_are_never_drawn():
    excluded = [3, 4, 5, 22]
    s = sampler(excluded=excluded)
    drawn = set(s.indices)
    assert drawn.isdisjoint(excluded)
    assert len(s) == 30 - len(excluded)


def test_every_frame_that_is_not_excluded_is_still_drawn():
    """The other half: exclusion must not quietly drop bystanders."""
    excluded = [3, 4, 5, 22]
    s = sampler(excluded=excluded)
    assert sorted(s.indices) == [i for i in range(30) if i not in excluded]


def test_each_surviving_start_is_drawn_exactly_once_per_epoch():
    """Uniform means uniform: not merely 'excluded ones are gone'."""
    s = sampler(excluded=[7, 8], shuffle=True, seed=3)
    drawn = list(s)
    assert len(drawn) == 28
    assert sorted(drawn) == [i for i in range(30) if i not in (7, 8)]


def test_the_length_reports_drawable_starts_not_all_frames():
    """`__len__` drives steps-per-epoch; reporting frames would overstate it."""
    assert len(sampler(excluded=list(range(0, 10)))) == 20


def test_exclusion_composes_with_dropping_episode_edges():
    s = sampler(excluded=[5], drop_n_first_frames=1, drop_n_last_frames=1)
    expected = [i for e in (0, 10, 20) for i in range(e + 1, e + 9) if i != 5]
    assert sorted(s.indices) == expected


def test_excluding_everything_is_refused_rather_than_yielding_nothing():
    """An empty sampler trains silently on no data, and the dataloader just
    ends each epoch immediately."""
    with pytest.raises(ValueError, match="nothing left to train on"):
        sampler(excluded=list(range(30)))


def test_duplicate_and_unsorted_exclusions_are_handled():
    s = sampler(excluded=[22, 3, 3, 22, 4])
    assert sorted(s.indices) == [i for i in range(30) if i not in (3, 4, 22)]


def test_an_out_of_range_exclusion_is_ignored():
    """A frame index past the end names no candidate start; it should not
    shift or drop anything."""
    s = sampler(excluded=[999])
    assert s.indices == list(range(30))


def test_shuffling_is_still_a_pure_function_of_seed_and_epoch():
    """The resume guarantee has to survive exclusion: two samplers with the
    same seed must produce the same order."""
    a, b = sampler(excluded=[5], shuffle=True, seed=11), sampler(excluded=[5], shuffle=True, seed=11)
    assert list(a) == list(b)


def test_resume_replays_the_same_epoch_from_an_offset():
    excluded = [2, 17]
    full = sampler(excluded=excluded, shuffle=True, seed=5)
    order = list(full)

    resumed = sampler(excluded=excluded, shuffle=True, seed=5)
    resumed.load_state_dict({"epoch": 0, "start_index": 10})
    assert list(resumed) == order[10:]


def test_relative_indices_are_mapped_after_exclusion():
    """When only some episodes are loaded, the sampler yields relative indices;
    exclusion is expressed in absolute ones and must be applied first."""
    mapping = {i: i - 10 for i in range(10, 20)}
    s = ExcludedStartSampler([10], [20], excluded_frames=[12], absolute_to_relative_idx=mapping)
    assert sorted(s.indices) == [i - 10 for i in range(10, 20) if i != 12]


def test_the_base_sampler_knows_nothing_about_exclusion():
    """The filter lives in the subclass; the parent keeps one representation.

    Asserted on the signature because the point of subclassing was to leave the
    parent's permutation and resume contract untouched, and a parameter
    creeping back would mean the split had quietly collapsed.
    """
    import inspect

    assert "excluded_frames" not in inspect.signature(EpisodeAwareSampler.__init__).parameters
    assert "excluded_frames" in inspect.signature(ExcludedStartSampler.__init__).parameters


def test_the_subclass_inherits_the_resume_contract():
    """Inheriting these rather than reimplementing them is the whole reason for
    subclassing, so it is worth pinning that they are in fact inherited."""
    for name in ("state_dict", "load_state_dict", "set_epoch", "_epoch_generator"):
        assert getattr(ExcludedStartSampler, name) is getattr(EpisodeAwareSampler, name), name


# ── Invariants the integration relies on, checked at runtime ──────────────────


@pytest.mark.parametrize(
    ("from_idx", "to_idx", "drop_first", "drop_last"),
    [
        ([0, 10, 20], [10, 20, 30], 0, 0),
        ([0, 10, 20], [10, 20, 30], 2, 3),  # both edges dropped
        ([0, 5, 7], [5, 7, 30], 1, 0),  # ragged episode lengths
        ([0], [1], 0, 0),  # a single-frame episode
        ([0, 100], [50, 130], 3, 7),  # a gap between episodes
    ],
)
def test_candidates_match_the_per_position_mapping(from_idx, to_idx, drop_first, drop_last):
    """The vectorised enumeration must equal `_absolute_frame_index` position by
    position.

    It replaced a Python generator that cost 153 s to construct on a 180M-frame
    dataset. Equality is asserted rather than assumed because the two are now
    different code: interval arithmetic done once over the range, versus a
    searchsorted per position.
    """
    plain = EpisodeAwareSampler(
        from_idx, to_idx, drop_n_first_frames=drop_first, drop_n_last_frames=drop_last
    )
    expected = [plain._absolute_frame_index(k) for k in range(len(plain))]

    excluding = ExcludedStartSampler(
        from_idx,
        to_idx,
        excluded_frames=[],
        drop_n_first_frames=drop_first,
        drop_n_last_frames=drop_last,
    )
    assert excluding._valid.tolist() == expected


def test_the_filter_postcondition_is_asserted_not_assumed():
    """The sampler must notice if its own filter leaves excluded starts behind.

    Driven by breaking `np.isin` only: the assertion probes with `searchsorted`,
    so it is not the same computation and can still see the damage. An assertion
    that re-ran the filter would pass here, which is the point of it not doing so.
    """
    import lerobot.datasets.sampler as sampler_module

    real_isin = sampler_module.np.isin
    sampler_module.np.isin = lambda a, b: np.zeros(len(a), dtype=bool)
    try:
        with pytest.raises(AssertionError, match="survived the filter"):
            ExcludedStartSampler(FROM, TO, excluded_frames=np.array([3, 4, 5]))
    finally:
        sampler_module.np.isin = real_isin


# ── The factory both trainers call ────────────────────────────────────────────
#
# It exists because the same three decisions -- which class, whether a counter
# is opened, whether the result actually excludes -- were assembled separately
# in two `train()` bodies that no test executes. These are the tests those call
# sites could not have.


def test_no_exclusion_returns_the_plain_sampler():
    """A run that excludes nothing must keep the parent's compact per-episode
    representation and pay for no materialised index array."""
    for excluded in (None, [], np.array([], dtype=np.int64)):
        s = make_start_sampler(FROM, TO, excluded_frames=excluded)
        assert type(s) is EpisodeAwareSampler, excluded
        assert not hasattr(s, "_valid")


def test_exclusion_returns_the_excluding_sampler():
    s = make_start_sampler(FROM, TO, excluded_frames=[3, 4])
    assert isinstance(s, ExcludedStartSampler)
    assert len(s) == 28


def test_no_trace_dir_means_no_counter_at_all(tmp_path):
    """Every rank but the main one passes None. It must write no file, and also
    allocate nothing: a resident int32 per frame is 687 MB at 1000 hours of
    50 fps, of anonymous memory, paid once per rank for a tally only the main
    process keeps."""
    s = make_start_sampler(FROM, TO, excluded_frames=[3])
    list(s)
    assert not list(tmp_path.iterdir()), "nothing may be written without a trace_dir"
    assert s.draw_counts is None, "and nothing may be allocated either"


def test_a_trace_dir_opens_a_file_backed_counter(tmp_path):
    s = make_start_sampler(FROM, TO, excluded_frames=[3], trace_dir=tmp_path / "trace")
    assert isinstance(s.draw_counts, np.memmap)
    list(s)
    s.draw_counts.flush()
    on_disk = np.fromfile(tmp_path / "trace" / "counts.i32", dtype=np.int32)
    assert on_disk.tolist() == [1 if i != 3 else 0 for i in range(30)]


def test_the_counter_is_sized_for_the_dataset_not_the_subset(tmp_path):
    """Indexed by absolute frame, so a run over one episode must still size for
    all of them -- otherwise two runs' traces are not comparable."""
    s = make_start_sampler(
        FROM, TO, excluded_frames=[25], trace_dir=tmp_path / "trace", episode_indices_to_use=[2]
    )
    assert s.draw_counts.size == 30, "sized for the dataset, not the 10 frames loaded"


def test_a_selection_that_removes_nothing_is_refused():
    """The wiring assertion: flags resolved and a sampler that still offers every
    start is a run that reports itself filtered and is not. Reached here by
    naming frames outside the dataset, which is the shape a wrong index space
    would take."""
    with pytest.raises(AssertionError, match="still offers"):
        make_start_sampler(FROM, TO, excluded_frames=[10_000, 10_001])


# ── What this branch must not have changed about EpisodeAwareSampler ──────────


def test_a_plain_sampler_allocates_nothing_for_tracing():
    """Counting must stay opt-in.

    It was not, briefly: `draw_counts=None` allocated a resident int32 per
    frame. That is 687 MB at 1000 hours of 50 fps, of *anonymous* memory --
    unlike the memory-mapped buffer a tracing run passes, those pages are not
    reclaimable. Under accelerate every rank but the main one asks for no trace,
    so each would have paid it for a tally that is thrown away.

    Construction alone would not have shown it: np.zeros is calloc, so the pages
    arrive lazily and only fault in once the epoch is enumerated.
    """
    s = EpisodeAwareSampler(FROM, TO, shuffle=True, seed=0)
    assert s.draw_counts is None
    list(s)
    assert s.draw_counts is None, "iterating must not conjure a counter either"


def test_the_order_is_what_it_was_before_counting_existed():
    """The counter must be observation only. Two samplers with the same seed,
    one counting and one not, must yield the identical sequence."""
    quiet = EpisodeAwareSampler(FROM, TO, shuffle=True, seed=11)
    counting = EpisodeAwareSampler(FROM, TO, shuffle=True, seed=11, draw_counts=np.zeros(30, dtype=np.int32))
    assert list(quiet) == list(counting)
    assert int(counting.draw_counts.sum()) == 30


def test_the_constructor_stayed_backward_compatible():
    """Every parameter this branch found must still be accepted positionally and
    by keyword, in the same order -- the class is upstream's, and a caller
    outside this repo may be passing them positionally."""
    import inspect

    params = list(inspect.signature(EpisodeAwareSampler.__init__).parameters)
    assert params[:9] == [
        "self",
        "dataset_from_indices",
        "dataset_to_indices",
        "episode_indices_to_use",
        "drop_n_first_frames",
        "drop_n_last_frames",
        "shuffle",
        "seed",
        "absolute_to_relative_idx",
    ]
    assert params[9:] == ["draw_counts"], "anything added must come last, with a default"
    assert inspect.signature(EpisodeAwareSampler.__init__).parameters["draw_counts"].default is None
