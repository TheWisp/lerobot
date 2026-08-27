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
"""Turning "tick this flag on these frames" into ordinary value edits.

Setting one flag on a range is a read-modify-write -- ``new = (old | set) &
~clear`` -- and the frames in a range need not agree on their other flags, so
no single constant expresses the result. Ticking ``blurry`` on frames 10-20
where 15-20 already carry ``fumble`` must produce ``0b01`` on some frames and
``0b11`` on others.

Rather than teach the value-edit writer a second mode, bit edits are *lowered*
here: read the column as it stands, apply the masks, and emit the constant-value
edits that produce the same result. ``set_feature_values`` -- the primitive every
other feature edit in the GUI goes through -- is left exactly as it was, so a
mistake in this file cannot affect editing reward, success, or subtasks.

Lowering happens at save time, against the column as it is then. That is what
makes two flags on overlapping frames compose instead of contest: each edit
records which bits it touches, not what the whole value should become.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class BitEdit:
    """Set and/or clear specific bits over a half-open global frame range.

    ``set_bits`` and ``clear_bits`` are masks over the same column. A bit in
    both is a contradiction rather than a precedence question, and is refused.
    """

    feature: str
    from_index: int
    to_index: int
    set_bits: int = 0
    clear_bits: int = 0

    def __post_init__(self) -> None:
        if self.to_index <= self.from_index:
            raise ValueError(f"empty range [{self.from_index}, {self.to_index}) for {self.feature!r}")
        if self.from_index < 0:
            raise ValueError(f"negative from_index {self.from_index} for {self.feature!r}")
        if self.set_bits < 0 or self.clear_bits < 0:
            raise ValueError(f"bit masks must be non-negative for {self.feature!r}")
        if not (self.set_bits or self.clear_bits):
            raise ValueError(f"edit for {self.feature!r} neither sets nor clears any bit")
        overlap = self.set_bits & self.clear_bits
        if overlap:
            raise ValueError(
                f"bit(s) {overlap:#b} are both set and cleared for {self.feature!r}; "
                "an edit must not contradict itself"
            )

    @property
    def touched_bits(self) -> int:
        return self.set_bits | self.clear_bits


def _without_bits(edit: BitEdit, start: int, stop: int, bits: int) -> list[BitEdit]:
    """``edit`` with ``bits`` removed over ``[start, stop)``, as 1-3 pieces.

    The parts of the range outside the overlap keep the edit unchanged; the
    overlapping part keeps only the bits not being taken away. A piece left
    touching no bits is dropped rather than kept as an edit that does nothing.
    """
    pieces: list[BitEdit] = []
    if edit.from_index < start:
        pieces.append(dataclasses.replace(edit, to_index=min(start, edit.to_index)))
    overlap_from, overlap_to = max(edit.from_index, start), min(edit.to_index, stop)
    if overlap_from < overlap_to:
        remaining_set = edit.set_bits & ~bits
        remaining_clear = edit.clear_bits & ~bits
        if remaining_set or remaining_clear:
            pieces.append(
                dataclasses.replace(
                    edit,
                    from_index=overlap_from,
                    to_index=overlap_to,
                    set_bits=remaining_set,
                    clear_bits=remaining_clear,
                )
            )
    if edit.to_index > stop:
        pieces.append(dataclasses.replace(edit, from_index=max(stop, edit.from_index)))
    return pieces


def stage(pending: Sequence[BitEdit], new: BitEdit) -> list[BitEdit]:
    """``pending`` with ``new`` added, keeping the set per-bit disjoint.

    The invariant is that no two pending edits touch the same bit on the same
    frame. It is what makes the pending set order-independent: with it, the
    result of applying them does not depend on the sequence they are applied
    in, so nothing rests on the order a list happens to hold them in. The
    constant-value edits alongside these keep the same property a different
    way, by clipping prior ranges so they never overlap at all.

    Where ``new`` touches bits an older edit also touches, the older edit gives
    those bits up over the shared frames -- the operator has just said what
    they want there. Unlike clipping a range, this discards nothing they still
    mean: they re-specified exactly those bits. That is why staging a bit edit
    needs no confirmation, while overwriting a range with a constant does.

    Post: the returned list is per-bit disjoint, and ``new`` is its last entry.
    """
    kept: list[BitEdit] = []
    for edit in pending:
        if edit.feature != new.feature or not (edit.touched_bits & new.touched_bits):
            kept.append(edit)
            continue
        kept.extend(_without_bits(edit, new.from_index, new.to_index, new.touched_bits))
    kept.append(new)
    return kept


def is_effective(values: np.ndarray, edit: BitEdit) -> bool:
    """True if ``edit`` would change at least one frame of ``values``.

    Staging an edit that changes nothing -- re-ticking a flag every frame in
    the range already carries -- would leave the operator with a pending edit
    that does nothing on save, which reads as unsaved work that isn't there.
    """
    window = values[edit.from_index : edit.to_index].astype(np.int64, copy=False)
    return bool(np.any((window | edit.set_bits) & ~edit.clear_bits != window))


def apply_bit_edits(values: np.ndarray, edits: Sequence[BitEdit]) -> np.ndarray:
    """The column after ``edits``, in order. Does not modify ``values``.

    Pre: ``values`` is indexed by global frame index and every edit's range
    lies inside it. Post: a new array of the same dtype and length.
    """
    for edit in edits:
        if edit.to_index > len(values):
            raise ValueError(
                f"edit range [{edit.from_index}, {edit.to_index}) for {edit.feature!r} "
                f"exceeds the column's {len(values)} frames"
            )
    updated = values.astype(np.int64, copy=True)
    for edit in edits:
        window = updated[edit.from_index : edit.to_index]
        updated[edit.from_index : edit.to_index] = (window | edit.set_bits) & ~edit.clear_bits
    return updated


def lower_to_value_edits(values: np.ndarray, edits: Sequence[BitEdit]) -> list[dict]:
    """Constant-value edits reproducing ``edits`` against ``values``.

    Only frames whose value actually changes are emitted, and consecutive
    frames sharing a new value collapse into one run -- so ticking a flag
    across a thousand frames that already agree costs one edit, not a thousand.

    Pre: ``values`` is the column of a single feature, every edit names that
    same feature, and every range lies inside it. Post: each run is
    ``{feature, from_index, to_index, value}`` with ``to_index`` exclusive,
    ranges disjoint and ascending, and ``value`` a plain ``int`` so it survives
    JSON.

    Raises:
        ValueError: edits naming more than one feature. One array cannot answer
            for two columns, and silently lowering them against the wrong one
            would write plausible values to the wrong frames.
    """
    if not edits:
        return []

    features = {edit.feature for edit in edits}
    if len(features) != 1:
        raise ValueError(f"expected edits for one feature, got {sorted(features)}")
    feature = features.pop()

    updated = apply_bit_edits(values, edits)
    original = values.astype(np.int64, copy=False)

    changed = np.flatnonzero(updated != original)
    if not changed.size:
        return []

    # A run ends where the frames stop being consecutive or the value changes;
    # either alone would merge frames that must stay separate.
    breaks = np.flatnonzero((np.diff(changed) != 1) | (updated[changed[1:]] != updated[changed[:-1]]))
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks + 1, [changed.size]])
    return [
        {
            "feature": feature,
            "from_index": int(changed[start]),
            "to_index": int(changed[stop - 1]) + 1,
            "value": int(updated[changed[start]]),
        }
        for start, stop in zip(starts, stops, strict=True)
    ]
