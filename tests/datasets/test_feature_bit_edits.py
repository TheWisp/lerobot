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
"""Lowering bit edits to constant-value edits.

The property under test throughout is that a bit edit touches the bits it
names and nothing else -- across frames that disagree about their other
flags, across several edits, and across the boundaries of the ranges given.
"""

import itertools

import numpy as np
import pytest

from lerobot.datasets.feature_bit_edits import (
    BitEdit,
    apply_bit_edits,
    is_effective,
    lower_to_value_edits,
    stage,
)

BLURRY = 0b001
FUMBLE = 0b010
OCCLUDED = 0b100
QUALITY = "quality"


def column(values) -> np.ndarray:
    return np.array(values, dtype=np.int64)


def replay(values: np.ndarray, runs: list[dict]) -> np.ndarray:
    """Apply lowered constant-value edits the way set_feature_values would."""
    out = values.astype(np.int64, copy=True)
    for run in runs:
        out[run["from_index"] : run["to_index"]] = run["value"]
    return out


# --- the edit itself --------------------------------------------------------


def test_an_edit_that_sets_and_clears_the_same_bit_is_refused():
    """A contradiction, not a precedence question -- resolving it silently
    would make the result depend on an implementation detail."""
    with pytest.raises(ValueError, match="both set and cleared"):
        BitEdit(QUALITY, 0, 5, set_bits=BLURRY, clear_bits=BLURRY)


def test_an_edit_that_touches_no_bit_is_refused():
    with pytest.raises(ValueError, match="neither sets nor clears"):
        BitEdit(QUALITY, 0, 5)


@pytest.mark.parametrize(("start", "stop"), [(5, 5), (5, 4), (0, 0)])
def test_an_empty_or_inverted_range_is_refused(start, stop):
    with pytest.raises(ValueError, match="empty range"):
        BitEdit(QUALITY, start, stop, set_bits=BLURRY)


def test_a_range_past_the_end_of_the_column_is_refused():
    """Numpy would silently truncate the slice and edit fewer frames than asked."""
    with pytest.raises(ValueError, match="exceeds the column"):
        apply_bit_edits(column([0, 0, 0]), [BitEdit(QUALITY, 1, 9, set_bits=BLURRY)])


# --- the property that motivates the whole design ---------------------------


def test_setting_a_flag_preserves_the_other_flags_on_those_frames():
    """The worked example: ticking `blurry` on 0-5 where 2-5 already carry
    `fumble` must produce two different values, which no constant can express."""
    values = column([0, 0, FUMBLE, FUMBLE, FUMBLE])
    updated = apply_bit_edits(values, [BitEdit(QUALITY, 0, 5, set_bits=BLURRY)])
    assert updated.tolist() == [
        BLURRY,
        BLURRY,
        BLURRY | FUMBLE,
        BLURRY | FUMBLE,
        BLURRY | FUMBLE,
    ]


def test_clearing_a_flag_preserves_the_other_flags():
    values = column([BLURRY | FUMBLE] * 4)
    updated = apply_bit_edits(values, [BitEdit(QUALITY, 0, 4, clear_bits=BLURRY)])
    assert updated.tolist() == [FUMBLE] * 4


@pytest.mark.parametrize("other", [0, BLURRY, FUMBLE, OCCLUDED, BLURRY | OCCLUDED])
@pytest.mark.parametrize("mask", [BLURRY, FUMBLE, OCCLUDED])
def test_an_edit_changes_only_the_bits_it_names(other, mask):
    """Exhaustive over which flag is edited against which pre-existing set."""
    values = column([other] * 3)
    after_set = apply_bit_edits(values, [BitEdit(QUALITY, 0, 3, set_bits=mask)])
    cleared = apply_bit_edits(values, [BitEdit(QUALITY, 0, 3, clear_bits=mask)])
    untouched = ~mask
    assert all(int(v) & untouched == other & untouched for v in after_set)
    assert all(int(v) & untouched == other & untouched for v in cleared)
    assert all(int(v) & mask == mask for v in after_set)
    assert all(int(v) & mask == 0 for v in cleared)


def test_setting_a_flag_that_is_already_set_is_idempotent():
    values = column([BLURRY, BLURRY | FUMBLE])
    once = apply_bit_edits(values, [BitEdit(QUALITY, 0, 2, set_bits=BLURRY)])
    twice = apply_bit_edits(once, [BitEdit(QUALITY, 0, 2, set_bits=BLURRY)])
    assert once.tolist() == values.tolist()
    assert twice.tolist() == values.tolist()


def test_clearing_a_flag_that_is_not_set_is_idempotent():
    values = column([0, FUMBLE])
    updated = apply_bit_edits(values, [BitEdit(QUALITY, 0, 2, clear_bits=BLURRY)])
    assert updated.tolist() == values.tolist()


def test_two_flags_on_overlapping_ranges_both_apply():
    """`blurry` on 0-3 and `fumble` on 2-5 -- the frames in both carry both."""
    values = column([0] * 5)
    updated = apply_bit_edits(
        values,
        [BitEdit(QUALITY, 0, 3, set_bits=BLURRY), BitEdit(QUALITY, 2, 5, set_bits=FUMBLE)],
    )
    assert updated.tolist() == [BLURRY, BLURRY, BLURRY | FUMBLE, FUMBLE, FUMBLE]


def test_edits_on_disjoint_bits_commute():
    """Order cannot matter when no bit is shared, so the staging order of two
    flag edits is not a decision the operator has to think about."""
    values = column([0, FUMBLE, OCCLUDED, BLURRY | OCCLUDED])
    a = BitEdit(QUALITY, 0, 4, set_bits=BLURRY)
    b = BitEdit(QUALITY, 1, 3, clear_bits=OCCLUDED)
    assert apply_bit_edits(values, [a, b]).tolist() == apply_bit_edits(values, [b, a]).tolist()


def test_edits_on_a_shared_bit_are_order_dependent():
    """Which is why `stage` never lets two of them coexist: relying on the
    order a list happens to hold would be resting on nothing."""
    values = column([0] * 3)
    a = BitEdit(QUALITY, 0, 3, set_bits=BLURRY)
    b = BitEdit(QUALITY, 0, 3, clear_bits=BLURRY)
    assert apply_bit_edits(values, [a, b]).tolist() != apply_bit_edits(values, [b, a]).tolist()


def test_the_input_column_is_not_modified():
    values = column([0, FUMBLE])
    before = values.tolist()
    apply_bit_edits(values, [BitEdit(QUALITY, 0, 2, set_bits=BLURRY)])
    assert values.tolist() == before


# --- lowering ---------------------------------------------------------------


def test_lowering_reproduces_the_bitwise_result():
    values = column([0, 0, FUMBLE, FUMBLE, OCCLUDED])
    edits = [BitEdit(QUALITY, 0, 5, set_bits=BLURRY)]
    assert replay(values, lower_to_value_edits(values, edits)).tolist() == (
        apply_bit_edits(values, edits).tolist()
    )


@pytest.mark.parametrize("pattern", list(itertools.product([0, BLURRY, FUMBLE, BLURRY | FUMBLE], repeat=4)))
def test_lowering_reproduces_the_result_for_every_small_column(pattern):
    """Exhaustive over every arrangement of two flags across four frames."""
    values = column(pattern)
    edits = [BitEdit(QUALITY, 1, 4, set_bits=BLURRY), BitEdit(QUALITY, 0, 3, clear_bits=FUMBLE)]
    assert replay(values, lower_to_value_edits(values, edits)).tolist() == (
        apply_bit_edits(values, edits).tolist()
    )


def test_frames_that_do_not_change_are_not_written():
    """Ticking a flag that is already set everywhere should cost nothing --
    otherwise a no-op click rewrites parquet shards."""
    values = column([BLURRY] * 100)
    assert lower_to_value_edits(values, [BitEdit(QUALITY, 0, 100, set_bits=BLURRY)]) == []


def test_a_uniform_range_collapses_to_one_run():
    values = column([0] * 1000)
    runs = lower_to_value_edits(values, [BitEdit(QUALITY, 0, 1000, set_bits=BLURRY)])
    assert runs == [{"feature": QUALITY, "from_index": 0, "to_index": 1000, "value": BLURRY}]


def test_a_split_range_produces_one_run_per_distinct_value():
    values = column([0, 0, FUMBLE, FUMBLE, 0])
    runs = lower_to_value_edits(values, [BitEdit(QUALITY, 0, 5, set_bits=BLURRY)])
    assert runs == [
        {"feature": QUALITY, "from_index": 0, "to_index": 2, "value": BLURRY},
        {"feature": QUALITY, "from_index": 2, "to_index": 4, "value": BLURRY | FUMBLE},
        {"feature": QUALITY, "from_index": 4, "to_index": 5, "value": BLURRY},
    ]


def test_unchanged_frames_split_a_run_rather_than_being_swallowed():
    """Frames 1-2 already carry the flag; emitting one run over 0-4 would be
    correct here by luck, but wrong the moment their other bits differ."""
    values = column([0, BLURRY, BLURRY, 0])
    runs = lower_to_value_edits(values, [BitEdit(QUALITY, 0, 4, set_bits=BLURRY)])
    assert runs == [
        {"feature": QUALITY, "from_index": 0, "to_index": 1, "value": BLURRY},
        {"feature": QUALITY, "from_index": 3, "to_index": 4, "value": BLURRY},
    ]


def test_nothing_outside_the_edited_range_is_emitted():
    values = column([FUMBLE] * 10)
    runs = lower_to_value_edits(values, [BitEdit(QUALITY, 4, 6, set_bits=BLURRY)])
    assert [(r["from_index"], r["to_index"]) for r in runs] == [(4, 6)]


def test_runs_are_disjoint_and_ascending():
    values = column([0, FUMBLE, 0, OCCLUDED, 0, FUMBLE | OCCLUDED])
    runs = lower_to_value_edits(values, [BitEdit(QUALITY, 0, 6, set_bits=BLURRY)])
    ends = [r["to_index"] for r in runs]
    starts = [r["from_index"] for r in runs]
    assert starts == sorted(starts)
    assert all(end <= nxt for end, nxt in zip(ends, starts[1:], strict=False))


def test_lowered_values_are_plain_ints():
    """numpy integers do not survive JSON, and these cross an HTTP boundary."""
    values = column([0, FUMBLE])
    runs = lower_to_value_edits(values, [BitEdit(QUALITY, 0, 2, set_bits=BLURRY)])
    assert all(type(r["value"]) is int for r in runs)
    assert all(type(r["from_index"]) is int and type(r["to_index"]) is int for r in runs)


def test_no_edits_lowers_to_nothing():
    assert lower_to_value_edits(column([0, 1]), []) == []


def test_mixing_features_in_one_lowering_is_refused():
    """One array cannot answer for two columns; lowering them together would
    write plausible values to the wrong frames."""
    values = column([0] * 4)
    with pytest.raises(ValueError, match="one feature"):
        lower_to_value_edits(
            values,
            [BitEdit(QUALITY, 0, 2, set_bits=BLURRY), BitEdit("take", 0, 2, set_bits=BLURRY)],
        )


def test_the_widest_bit_survives_lowering():
    """Bit 62 is inside the 63-flag limit and must not be lost to a narrower
    intermediate type."""
    top = 1 << 62
    values = column([0, 0])
    runs = lower_to_value_edits(values, [BitEdit(QUALITY, 0, 2, set_bits=top)])
    assert runs == [{"feature": QUALITY, "from_index": 0, "to_index": 2, "value": top}]


# --- staging: the per-bit-disjoint invariant --------------------------------
#
# `feature_set` edits are kept order-independent by clipping ranges so none
# overlap. Bit edits get the same guarantee a different way: no two may touch
# the same bit on the same frame. Every test here is ultimately about that.


def assert_per_bit_disjoint(edits):
    for a, b in itertools.combinations(edits, 2):
        if a.feature != b.feature:
            continue
        frames_overlap = a.from_index < b.to_index and b.from_index < a.to_index
        assert not (frames_overlap and (a.touched_bits & b.touched_bits)), (
            f"{a} and {b} both touch bit(s) {a.touched_bits & b.touched_bits:#b}"
        )


def stage_all(edits):
    pending: list[BitEdit] = []
    for edit in edits:
        pending = stage(pending, edit)
    return pending


def test_staging_keeps_the_set_per_bit_disjoint():
    pending = stage_all(
        [
            BitEdit(QUALITY, 0, 10, set_bits=BLURRY),
            BitEdit(QUALITY, 5, 15, set_bits=BLURRY | FUMBLE),
            BitEdit(QUALITY, 8, 12, clear_bits=BLURRY),
            BitEdit(QUALITY, 0, 20, set_bits=OCCLUDED),
        ]
    )
    assert_per_bit_disjoint(pending)


def test_a_disjoint_pending_set_is_order_independent():
    """The invariant's whole purpose: with it, applying in any sequence gives
    the same column, so nothing depends on list order."""
    values = column([0, FUMBLE, OCCLUDED, BLURRY, 0, FUMBLE | OCCLUDED])
    pending = stage_all(
        [
            BitEdit(QUALITY, 0, 6, set_bits=BLURRY),
            BitEdit(QUALITY, 2, 5, clear_bits=FUMBLE),
            BitEdit(QUALITY, 1, 4, set_bits=OCCLUDED),
        ]
    )
    assert_per_bit_disjoint(pending)
    results = {
        tuple(apply_bit_edits(values, list(order)).tolist()) for order in itertools.permutations(pending)
    }
    assert len(results) == 1, "a per-bit-disjoint set must apply the same in any order"


def test_ticking_then_unticking_the_same_range_leaves_the_older_edit_gone():
    pending = stage_all([BitEdit(QUALITY, 0, 5, set_bits=BLURRY), BitEdit(QUALITY, 0, 5, clear_bits=BLURRY)])
    assert pending == [BitEdit(QUALITY, 0, 5, clear_bits=BLURRY)]


def test_ticking_then_unticking_stages_nothing_at_all():
    """The question this design exists to answer: 0 pending edits, not 2.

    `stage` removes the superseded tick; `is_effective` then drops the untick,
    because on this column it changes nothing.
    """
    values = column([0, 0, FUMBLE, FUMBLE, FUMBLE])
    pending = stage_all([BitEdit(QUALITY, 0, 5, set_bits=BLURRY), BitEdit(QUALITY, 0, 5, clear_bits=BLURRY)])
    effective = [e for e in pending if is_effective(values, e)]
    assert effective == []
    assert lower_to_value_edits(values, effective) == []


def test_unticking_then_reticking_also_stages_nothing():
    """The mirror case, on a column that already carries the flag."""
    values = column([BLURRY] * 5)
    pending = stage_all([BitEdit(QUALITY, 0, 5, clear_bits=BLURRY), BitEdit(QUALITY, 0, 5, set_bits=BLURRY)])
    assert [e for e in pending if is_effective(values, e)] == []


def test_each_pending_edit_can_be_judged_against_the_stored_column_alone():
    """Because the set is per-bit disjoint, no pending edit changes the bits
    another one reads -- so `is_effective` needs no knowledge of the others."""
    values = column([0, BLURRY, FUMBLE, BLURRY | FUMBLE])
    pending = stage_all([BitEdit(QUALITY, 0, 4, set_bits=BLURRY), BitEdit(QUALITY, 0, 4, set_bits=FUMBLE)])
    for edit in pending:
        alone = apply_bit_edits(values, [edit])
        assert is_effective(values, edit) == (alone.tolist() != values.tolist())


def test_a_different_flag_on_the_same_frames_is_kept():
    pending = stage_all(
        [BitEdit(QUALITY, 10, 21, set_bits=BLURRY), BitEdit(QUALITY, 15, 26, set_bits=FUMBLE)]
    )
    assert len(pending) == 2
    assert_per_bit_disjoint(pending)


def test_a_partial_frame_overlap_splits_the_older_edit():
    """The older edit keeps the flag where the new one does not reach."""
    pending = stage_all([BitEdit(QUALITY, 0, 10, set_bits=BLURRY), BitEdit(QUALITY, 4, 6, clear_bits=BLURRY)])
    assert sorted((e.from_index, e.to_index, e.set_bits, e.clear_bits) for e in pending) == [
        (0, 4, BLURRY, 0),
        (4, 6, 0, BLURRY),
        (6, 10, BLURRY, 0),
    ]


def test_an_older_edit_keeps_the_bits_the_new_one_does_not_touch():
    pending = stage_all(
        [
            BitEdit(QUALITY, 0, 10, set_bits=BLURRY | FUMBLE),
            BitEdit(QUALITY, 0, 10, clear_bits=BLURRY),
        ]
    )
    assert sorted((e.set_bits, e.clear_bits) for e in pending) == [(0, BLURRY), (FUMBLE, 0)]


def test_an_older_edit_fully_covered_and_fully_shadowed_is_dropped():
    pending = stage_all([BitEdit(QUALITY, 2, 5, set_bits=BLURRY), BitEdit(QUALITY, 0, 10, clear_bits=BLURRY)])
    assert pending == [BitEdit(QUALITY, 0, 10, clear_bits=BLURRY)]


def test_edits_on_another_column_are_never_disturbed():
    other = BitEdit("take", 0, 10, set_bits=BLURRY)
    pending = stage_all([other, BitEdit(QUALITY, 0, 10, clear_bits=BLURRY)])
    assert other in pending


def test_abutting_ranges_leave_each_other_alone():
    """Ranges are half-open, so 0-10 and 10-20 do not touch."""
    a = BitEdit(QUALITY, 0, 10, set_bits=BLURRY)
    b = BitEdit(QUALITY, 10, 20, set_bits=BLURRY)
    assert stage_all([a, b]) == [a, b]


def test_the_newest_edit_is_never_the_one_that_gives_way():
    pending = stage_all(
        [BitEdit(QUALITY, 0, 10, set_bits=BLURRY), BitEdit(QUALITY, 0, 10, clear_bits=BLURRY)]
    )
    assert pending[-1] == BitEdit(QUALITY, 0, 10, clear_bits=BLURRY)


@pytest.mark.parametrize("ranges", list(itertools.product([(0, 4), (2, 6), (4, 8)], repeat=3)))
def test_the_invariant_survives_many_range_arrangements(ranges):
    masks = [BLURRY, BLURRY | FUMBLE, FUMBLE]
    pending = stage_all([BitEdit(QUALITY, *r, set_bits=m) for r, m in zip(ranges, masks, strict=True)])
    assert_per_bit_disjoint(pending)


@pytest.mark.parametrize("pattern", list(itertools.product([0, BLURRY, FUMBLE, BLURRY | FUMBLE], repeat=4)))
def test_staging_preserves_what_the_operator_asked_for(pattern):
    """Collapsing must not change the outcome: the staged set applied to the
    column must equal applying the raw sequence in order."""
    values = column(pattern)
    sequence = [
        BitEdit(QUALITY, 0, 4, set_bits=BLURRY),
        BitEdit(QUALITY, 1, 3, clear_bits=BLURRY),
        BitEdit(QUALITY, 2, 4, set_bits=FUMBLE),
    ]
    assert apply_bit_edits(values, stage_all(sequence)).tolist() == (
        apply_bit_edits(values, sequence).tolist()
    )


# --- is_effective -----------------------------------------------------------


def test_an_edit_that_changes_nothing_is_not_effective():
    assert not is_effective(column([BLURRY] * 5), BitEdit(QUALITY, 0, 5, set_bits=BLURRY))
    assert not is_effective(column([0] * 5), BitEdit(QUALITY, 0, 5, clear_bits=BLURRY))


def test_an_edit_changing_a_single_frame_is_effective():
    assert is_effective(column([BLURRY, BLURRY, 0]), BitEdit(QUALITY, 0, 3, set_bits=BLURRY))


def test_clearing_a_flag_that_is_set_is_effective():
    """Guards the half of the expression a set-only check would satisfy: with
    clear_bits ignored, removing a flag reads as changing nothing and the
    operator's edit is silently discarded."""
    assert is_effective(column([BLURRY] * 3), BitEdit(QUALITY, 0, 3, clear_bits=BLURRY))
    assert is_effective(column([BLURRY | FUMBLE, FUMBLE]), BitEdit(QUALITY, 0, 2, clear_bits=BLURRY))


def test_clearing_is_effective_when_only_one_frame_carries_the_flag():
    assert is_effective(column([0, 0, BLURRY, 0]), BitEdit(QUALITY, 0, 4, clear_bits=BLURRY))


def test_effectiveness_looks_only_inside_the_range():
    values = column([0, BLURRY, BLURRY, BLURRY])
    assert not is_effective(values, BitEdit(QUALITY, 1, 4, set_bits=BLURRY))
    assert is_effective(values, BitEdit(QUALITY, 0, 4, set_bits=BLURRY))
