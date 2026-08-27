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
"""The bitset feature contract: several flags can hold on one frame."""

import numpy as np
import pytest

from lerobot.datasets.feature_utils import (
    MAX_FLAGS,
    decode_flags,
    encode_flags,
    flag_bit,
    flags_vocabulary_error,
    is_categorical_feature,
    is_flags_feature,
    validate_feature_numeric_bounds,
)

QUALITY = {"dtype": "int64", "shape": [1], "flags": ["blurry", "fumble", "occluded"]}
CONTROL_MODE = {"dtype": "int64", "shape": [1], "names": ["ee", "joint"]}


def test_a_bitset_is_not_mistaken_for_a_categorical():
    """The two contracts must not overlap: a categorical value is an index,
    a bitset value is a bit pattern, and reading one as the other is wrong
    without erroring."""
    assert is_flags_feature(QUALITY)
    assert not is_categorical_feature(QUALITY)
    assert is_categorical_feature(CONTROL_MODE)
    assert not is_flags_feature(CONTROL_MODE)


@pytest.mark.parametrize(
    "feature",
    [
        {"dtype": "int64", "shape": [1], "flags": []},  # empty vocabulary
        {"dtype": "int64", "shape": [1]},  # no vocabulary
        {"dtype": "float32", "shape": [1], "flags": ["a"]},  # not an integer
        {"dtype": "int64", "shape": [3], "flags": ["a"]},  # not a scalar
        {"dtype": "int64", "shape": [1], "flags": "blurry"},  # not a list
    ],
)
def test_only_a_scalar_integer_with_a_vocabulary_is_a_bitset(feature):
    assert not is_flags_feature(feature)


def test_flags_round_trip_through_the_integer():
    for flags in ([], ["blurry"], ["fumble", "occluded"], ["blurry", "fumble", "occluded"]):
        value = encode_flags(QUALITY, flags)
        assert decode_flags(QUALITY, value) == sorted(flags, key=QUALITY["flags"].index)


def test_several_flags_hold_at_once():
    """The whole reason this is not a categorical."""
    value = encode_flags(QUALITY, ["blurry", "occluded"])
    assert decode_flags(QUALITY, value) == ["blurry", "occluded"]
    assert value == 0b101


def test_setting_the_same_flag_twice_is_idempotent():
    assert encode_flags(QUALITY, ["blurry", "blurry"]) == encode_flags(QUALITY, ["blurry"])


def test_a_bit_keeps_its_meaning_when_the_vocabulary_grows():
    """Appending is the only safe edit: every stored value keeps its meaning."""
    stored = encode_flags(QUALITY, ["fumble"])
    grown = {**QUALITY, "flags": [*QUALITY["flags"], "mistimed"]}
    assert decode_flags(grown, stored) == ["fumble"]
    assert flag_bit(grown, "fumble") == flag_bit(QUALITY, "fumble")


def test_an_unknown_flag_is_refused_rather_than_matching_nothing():
    with pytest.raises(ValueError, match="unknown flag 'nope'"):
        flag_bit(QUALITY, "nope")


def test_the_widest_vocabulary_is_63_flags_not_64():
    """int64 is signed, so bit 63 is the sign. A value setting it reads back
    negative and is then rejected as setting bits no flag declares, which
    makes 64 the obvious wrong answer."""
    assert MAX_FLAGS == 63
    full = {"dtype": "int64", "shape": [1], "flags": [f"f{i}" for i in range(MAX_FLAGS)]}
    assert flags_vocabulary_error(full) == ""
    assert flag_bit(full, f"f{MAX_FLAGS - 1}") == MAX_FLAGS - 1

    widest = (1 << MAX_FLAGS) - 1
    stored = np.array([widest], dtype="int64")
    assert int(stored[0]) == widest, "the largest legal value does not fit the column"
    assert validate_feature_numeric_bounds("q", full, stored) == ""


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "uint8", "float32"])
def test_only_an_int64_column_is_a_bitset(dtype):
    """Narrower integers are not accepted as bitsets of their own width.

    The saving is negligible -- one int64 per frame is 8 bytes -- and a
    per-column capacity is a second rule that has to be right everywhere the
    first one is. A column that is not int64 stays a plain column rather than
    becoming a bitset with a different limit.
    """
    assert not is_flags_feature({"dtype": dtype, "shape": [1], "flags": ["a"]})


def test_an_overwide_vocabulary_is_reported_by_the_validator_too():
    """flag_bit guards the write path; the validator guards values arriving
    from anywhere else."""
    too_wide = {"dtype": "int64", "shape": [1], "flags": [f"f{i}" for i in range(MAX_FLAGS + 1)]}
    error = validate_feature_numeric_bounds("q", too_wide, np.array([1]))
    assert f"holds {MAX_FLAGS}" in error


def test_an_undeclared_bit_is_rejected_at_the_boundary():
    """Bit 3 has no meaning in a 3-flag vocabulary. Stored, it would survive
    every round trip while decoding to nothing."""
    error = validate_feature_numeric_bounds("quality", QUALITY, np.array([0b1000]))
    assert "outside the 3 declared flags" in error


def test_every_declared_bit_pattern_is_accepted():
    for value in range(1 << len(QUALITY["flags"])):
        assert validate_feature_numeric_bounds("quality", QUALITY, np.array([value])) == ""


def test_a_negative_value_is_rejected():
    """Two's complement would set every high bit, none of them declared."""
    assert validate_feature_numeric_bounds("quality", QUALITY, np.array([-1])) != ""


def test_a_feature_without_a_vocabulary_is_unaffected():
    """Existing datasets must validate exactly as before."""
    plain = {"dtype": "int64", "shape": [1]}
    assert validate_feature_numeric_bounds("plain", plain, np.array([1 << 40])) == ""


# --- vocabulary validity: one home, two callers -----------------------------


def test_one_flag_past_the_limit_is_refused():
    over = {"dtype": "int64", "shape": [1], "flags": [f"f{i}" for i in range(MAX_FLAGS + 1)]}
    assert f"holds {MAX_FLAGS}" in flags_vocabulary_error(over)


def test_a_single_flag_vocabulary_works():
    """The smallest usable vocabulary; bit 0 only."""
    one = {"dtype": "int64", "shape": [1], "flags": ["only"]}
    assert flags_vocabulary_error(one) == ""
    assert flag_bit(one, "only") == 0
    assert decode_flags(one, 0) == []
    assert decode_flags(one, 1) == ["only"]
    assert validate_feature_numeric_bounds("q", one, np.array([0b10])) != ""


def test_zero_means_no_flags_not_a_missing_value():
    """Every column starts filled with 0, so 0 must be a valid decode, not an
    error and not a sentinel."""
    assert decode_flags(QUALITY, 0) == []
    assert encode_flags(QUALITY, []) == 0
    assert validate_feature_numeric_bounds("q", QUALITY, np.array([0])) == ""


def test_an_empty_vocabulary_is_not_a_bitset_at_all():
    """Zero flags cannot describe any value, so the feature is a plain integer
    column rather than a bitset with nothing in it."""
    empty = {"dtype": "int64", "shape": [1], "flags": []}
    assert not is_flags_feature(empty)
    assert validate_feature_numeric_bounds("q", empty, np.array([7])) == ""


def test_a_repeated_flag_is_refused_because_it_breaks_the_round_trip():
    """With ['a', 'b', 'a'], bit 2 decodes to 'a' and 'a' encodes to bit 0, so a
    stored value changes just by being read and written back."""
    dup = {"dtype": "int64", "shape": [1], "flags": ["a", "b", "a"]}
    assert "repeats flag(s) ['a']" in flags_vocabulary_error(dup)
    with pytest.raises(ValueError, match="repeats"):
        flag_bit(dup, "a")
    assert "repeats" in validate_feature_numeric_bounds("q", dup, np.array([1]))


@pytest.mark.parametrize("vocabulary", [["ok", 7], ["ok", ""], ["ok", None]])
def test_a_flag_that_is_not_a_non_empty_string_is_refused(vocabulary):
    """Nothing can select such a flag by name, so a run naming it would look
    like a typo rather than like a malformed column."""
    bad = {"dtype": "int64", "shape": [1], "flags": vocabulary}
    assert "not a non-empty string" in flags_vocabulary_error(bad)


def test_the_validator_and_the_write_path_agree_on_validity():
    """Both callers route through flags_vocabulary_error, so a vocabulary
    cannot be writable but unreadable, or the reverse."""
    for vocabulary in (["a", "a"], ["a", ""], [f"f{i}" for i in range(64)]):
        feature = {"dtype": "int64", "shape": [1], "flags": vocabulary}
        problem = flags_vocabulary_error(feature)
        assert problem
        assert problem in validate_feature_numeric_bounds("q", feature, np.array([1]))
        with pytest.raises(ValueError, match="flags vocabulary"):
            flag_bit(feature, "a")


# --- algebraic properties, enumerated exhaustively --------------------------
#
# The vocabulary is small enough to check every bit pattern rather than sample
# them, which is stronger than random generation at this size: 3 flags is 8
# values, 5 is 32.

SMALL = {"dtype": "int64", "shape": [1], "flags": ["a", "b", "c"]}
ALL_SUBSETS = [[flag for i, flag in enumerate(SMALL["flags"]) if value & (1 << i)] for value in range(1 << 3)]


@pytest.mark.parametrize("flags", ALL_SUBSETS)
def test_every_subset_round_trips(flags):
    """decode(encode(x)) == x over the whole domain, not a sampled corner."""
    assert decode_flags(SMALL, encode_flags(SMALL, flags)) == flags


@pytest.mark.parametrize("value", range(1 << 3))
def test_every_value_round_trips(value):
    """encode(decode(v)) == v, the other direction -- this is what a repeated
    flag would break."""
    assert encode_flags(SMALL, decode_flags(SMALL, value)) == value


@pytest.mark.parametrize("flags", ALL_SUBSETS)
def test_encoding_does_not_depend_on_order(flags):
    """Commutativity: a set of flags, not a sequence."""
    assert encode_flags(SMALL, flags) == encode_flags(SMALL, list(reversed(flags)))


@pytest.mark.parametrize("flags", ALL_SUBSETS)
def test_encoding_is_idempotent(flags):
    """Setting what is already set changes nothing."""
    once = encode_flags(SMALL, flags)
    assert encode_flags(SMALL, [*flags, *flags]) == once


@pytest.mark.parametrize("value", range(1 << 3))
@pytest.mark.parametrize("flag", SMALL["flags"])
def test_setting_one_flag_disturbs_no_other(value, flag):
    """Bit independence: the flags share a column but not their meanings."""
    before = decode_flags(SMALL, value)
    after = decode_flags(SMALL, value | (1 << flag_bit(SMALL, flag)))
    assert set(after) == set(before) | {flag}


@pytest.mark.parametrize("value", range(1 << 3))
@pytest.mark.parametrize("flag", SMALL["flags"])
def test_clearing_one_flag_disturbs_no_other(value, flag):
    after = decode_flags(SMALL, value & ~(1 << flag_bit(SMALL, flag)))
    assert set(after) == set(decode_flags(SMALL, value)) - {flag}


def test_decoding_follows_declaration_order():
    """Not an accident of bit order: a reader showing flags to a person should
    show them in the order the vocabulary declares."""
    assert decode_flags(SMALL, 0b111) == ["a", "b", "c"]


def test_an_undeclared_bit_is_rejected_not_silently_stripped():
    """Python's Flag calls these boundary policies STRICT / CONFORM / EJECT /
    KEEP. This contract is STRICT: an out-of-range bit is an error, never
    quietly masked off, because stripping would let a writer believe it stored
    something it did not.
    """
    stray = 0b1000  # bit 3, with three flags declared
    assert validate_feature_numeric_bounds("q", SMALL, np.array([stray])) != ""
    # And decoding does not invent a flag for it, so an accepted value and a
    # rejected one cannot decode the same way.
    assert decode_flags(SMALL, stray) == []
    assert decode_flags(SMALL, stray | 0b1) == ["a"]


def test_the_widest_vocabulary_round_trips_end_to_end():
    """The 63-flag boundary, exercised rather than only asserted."""
    full = {"dtype": "int64", "shape": [1], "flags": [f"f{i}" for i in range(MAX_FLAGS)]}
    every = full["flags"]
    value = encode_flags(full, every)
    assert value == (1 << MAX_FLAGS) - 1
    assert decode_flags(full, value) == every
    assert int(np.array([value], dtype="int64")[0]) == value
