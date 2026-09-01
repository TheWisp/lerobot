# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Lossless run-length coding for per-frame segmentation masks.

Masks are sparse and blobby, so a bitmap wastes almost all of its bytes: a
224x224 binary mask is 6 KB packed, while the same region as run lengths is a
few hundred. Storing them as video was the other candidate and is rejected here
— a lossy codec perturbs pixel values, and an object-index image cannot survive
that, while lossless video encoders give up the compression that made video
attractive in the first place.

The encoding is COCO's, byte-for-byte, rather than something bespoke:

  * Runs are counted in COLUMN-major (Fortran) order, starting with a run of
    zeros, which is what ``pycocotools`` does and what every consumer of a COCO
    ``segmentation`` field expects.
  * Each run length is delta-coded against the run two positions back and
    written LEB128-style in 5-bit groups, biased into printable ASCII. That is
    why the result is a ``str`` and not ``bytes``: COCO chose it so masks embed
    in JSON, and it is what lets these live in a dataset ``string`` feature
    without a new dtype or a base64 wrapper.

Interoperability is the point of matching it: a mask written here can be read by
``pycocotools.mask.decode`` unchanged, and any COCO tooling can consume the
dataset. Correctness is pinned by a round-trip test over real SAM3 masks, not
just synthetic ones — see ``tests/datasets/test_mask_codec.py``.

Encoding is exact. ``decode(encode(m)) == m`` for every binary mask, so nothing
here is an approximation of the segmenter's output.

A frame's stored value is a JSON array of entries, one per label the frame
carries::

    [[label_id, counts], [label_id, counts, 0], ...]

``label_id`` indexes the column's ``mask_labels``, so positions are the
contract: a label may be appended or renamed in place, never moved or deleted.

The optional third element is the ENABLED flag, and it is what makes three
states distinguishable rather than two:

===========  ======================  ==================  ===================
state        stored                  reaches training    a gap-filling write
===========  ======================  ==================  ===================
detected     ``[id, counts]``        yes                 leaves it alone
disabled     ``[id, counts, 0]``     no                  leaves it alone
absent       no entry                nothing there       fills it
===========  ======================  ==================  ===================

Without the middle row, "this detection is wrong here" and "nothing was ever
found here" are the same bytes, so any later pass puts the wrong detection
straight back — and the only way to suppress it would be to delete the mask,
which cannot be undone. Disabling keeps the pixels and withholds them.

It is written INLINE rather than as a parallel per-frame bitset column so the
flag cannot desync from the mask it describes: one write carries both, and
there is no second column to migrate, rename, or forget to update. The cost is
two bytes on a disabled entry beside a run-length string of a few hundred.

The flag is omitted when a label is enabled: the writer emits the narrow form
and the reader accepts either. So `[id, counts]`, `[id, counts, 1]` and
`[id, counts, true]` all read as enabled, and only `[id, counts]` is ever
written.

That asymmetry is deliberate. Rows written before the flag existed must stay
readable — datasets are on disk and on the Hub — so the two-element form can
never be retired, and a reader must carry the default no matter what the writer
does. Emitting `1` therefore removes no code path; it only makes a dataset's
bytes depend on which version wrote it, so two datasets with identical masks
would no longer compare equal. Writing the default buys nothing and costs
that.

Why this is hand-rolled rather than ``pycocotools.mask``: there is no blocker,
only a trade. pycocotools is a C extension built at install time, and this is
~40 lines against a format that has not changed in a decade, so the dependency
buys little. It does buy speed — the C encoder is far faster than this pure
Python one — which does not matter while encoding happens once per frame in an
offline pass beside a 50 ms segmentation, and would matter if RLE ever lands in
a hot path. The wire format is byte-identical either way, so the swap is
mechanical: replace ``encode_mask``/``decode_mask`` with
``pycocotools.mask.encode/``decode`` and delete the two helpers. Revisit that
if profiling ever shows this module in a live loop.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np

#: Value stored in a frame's mask feature when nothing was detected. An empty
#: list rather than an empty string, so a reader never has to distinguish
#: "no instances" from "field not written".
EMPTY = "[]"


def _counts_to_string(counts: list[int]) -> str:
    """COCO's ``rleToString``: delta-code, then 5 bits per character."""
    out: list[str] = []
    for i, count in enumerate(counts):
        x = int(count)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            # Bit 0x10 is the sign bit of the 5-bit group: when it is set the
            # value continues only while x is not -1 (arithmetic shift of a
            # negative), otherwise while x is not 0.
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def _string_to_counts(s: str) -> list[int]:
    """COCO's ``rleFrString``: inverse of :func:`_counts_to_string`.

    Vectorized because it sits on the hot path twice: every composited playback
    frame decodes one of these per label, and so does every training sample.
    Character-at-a-time in Python cost 1.3 ms on a five-thousand-character row,
    more than the blur it feeds.

    The format: 5 bits per character, low group first, 0x20 continues a group,
    and 0x10 on a group's final character means the value is negative. Counts
    are delta-coded against the value two positions back, from the fourth
    onwards — which makes the decode a cumulative sum along each of two chains
    (indices 1,3,5,… and 2,4,6,…) with the first count standing alone.
    """
    if not s:
        return []
    c = np.frombuffer(s.encode("ascii"), dtype=np.uint8).astype(np.int64) - 48
    ends = (c & 0x20) == 0  # last character of each group
    end_idx = np.flatnonzero(ends)
    if end_idx.size == 0:
        raise ValueError("truncated RLE: no group terminator")
    starts = np.concatenate(([0], end_idx[:-1] + 1))
    within = np.arange(c.size) - np.repeat(starts, np.diff(np.append(starts, c.size)))
    raw = np.add.reduceat((c & 0x1F) << (5 * within), starts)
    # Sign lives on the group's last character, extending above its width.
    widths = within[end_idx] + 1
    negative = (c[end_idx] & 0x10) != 0
    raw[negative] |= -1 << (5 * widths[negative])

    out = raw.copy()
    odd = np.arange(1, out.size, 2)  # 1, 3, 5, … chain
    if odd.size:
        out[odd] = np.cumsum(raw[odd])
    even = np.arange(2, out.size, 2)  # 2, 4, 6, … chain; index 0 alone
    if even.size:
        out[even] = np.cumsum(raw[even])
    return out.tolist()


def encode_mask(mask: np.ndarray) -> str:
    """One boolean HxW mask -> COCO RLE ``counts`` string.

    Pre: ``mask`` is 2-D and castable to bool. Post: the returned string decodes
    back to exactly this mask via :func:`decode_mask` given the same shape.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    flat = np.asfortranarray(mask.astype(bool)).ravel(order="F")
    n = flat.size
    # Run boundaries, then lengths. Taking boundaries from value CHANGES (rather
    # than diffing a padded array) is what keeps a uniform mask correct: an
    # all-zero frame has no changes and must still emit one run covering every
    # pixel, since the counts have to sum to h*w for the frame to decode.
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [n]))).tolist()
    # The format defines the first run as zeros; a mask that starts set needs a
    # leading empty run so the alternation lines up.
    if n and flat[0]:
        counts = [0, *counts]
    return _counts_to_string(counts)


def decode_mask(counts: str, shape: tuple[int, int]) -> np.ndarray:
    """COCO RLE ``counts`` string -> boolean HxW mask. Inverse of :func:`encode_mask`."""
    h, w = shape
    runs = np.asarray(_string_to_counts(counts), dtype=np.int64)
    # One vectorized expansion instead of a Python loop over runs. Irrelevant
    # for blocky masks (a rectangle is ~2 runs) and decisive for real
    # segmentation boundaries: ~2,000 runs per label on 720p SAM masks made
    # the loop the decode cost (0.55 -> 0.30 ms per 6-label frame, measured on
    # real rows; output verified identical).
    values = np.zeros(len(runs), dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, runs)
    if flat.size != h * w:
        raise ValueError(f"RLE covers {flat.size} pixels, but {shape} needs {h * w}")
    return flat.reshape((h, w), order="F")


def encode_frame(
    masks_by_label: dict[str, np.ndarray],
    labels: list[str],
    *,
    disabled: Iterable[str] = (),
) -> str:
    """All instances in one frame -> the string stored in the feature column.

    Pre: every key of ``masks_by_label`` is in ``labels``; ``labels`` is the
    vocabulary recorded in the feature metadata, so the stored rows carry small
    integer ids rather than repeating the label text on every frame.
    ``disabled`` names labels whose mask is stored but muted. Pre: every name in
    it is in ``labels``; a name with no mask on this frame is accepted and
    stores nothing, since muting is a property of a mask.

    Post: returns JSON ``[[label_id, counts], ...]``, ordered by label id so two
    frames with the same content encode identically and diff cleanly. A muted
    entry carries a third element, ``[label_id, counts, 0]``.

    The flag is emitted ONLY when the entry is disabled. Readers accept the
    explicit ``[label_id, counts, 1]`` too, but nothing here writes it: the
    two-element form can never be retired while pre-flag datasets exist, so a
    reader carries the default regardless and emitting it would remove no code
    path -- only make a dataset's bytes depend on which version wrote it.
    """
    index = {name: i for i, name in enumerate(labels)}
    muted = set(disabled)
    if unknown := muted - set(index):
        # Silently ignoring this would leave a mask enabled and reaching
        # training, with nothing said -- the same shape as a typo'd mask label,
        # which is refused two lines down.
        raise KeyError(f"disabled labels {sorted(unknown)} are not in the declared vocabulary {labels}")
    rows: list[list] = []
    for name, mask in masks_by_label.items():
        if name not in index:
            raise KeyError(f"label {name!r} is not in the declared vocabulary {labels}")
        if mask is None or not mask.any():
            continue  # an absent object is absent, not an empty run
        entry: list = [index[name], encode_mask(mask)]
        if name in muted:
            entry.append(0)
        rows.append(entry)
    rows.sort(key=lambda r: r[0])
    return json.dumps(rows, separators=(",", ":"))


def _entries(value: str, labels: list[str]):
    """``(label, counts, enabled)`` per stored entry, validated."""
    for entry in json.loads(value or EMPTY):
        label_id, counts = entry[0], entry[1]
        if not 0 <= label_id < len(labels):
            raise ValueError(f"label id {label_id} outside the declared vocabulary of {len(labels)}")
        yield labels[label_id], counts, bool(entry[2]) if len(entry) > 2 else True


def frame_states(value: str, labels: list[str]) -> dict[str, bool]:
    """``{label: enabled}`` for the labels this frame carries, without decoding.

    The timeline needs presence and mutedness per frame and nothing else, and
    expanding the RLE to answer that would cost more than drawing the track.
    """
    return {name: enabled for name, _counts, enabled in _entries(value, labels)}


def decode_frame(
    value: str,
    labels: list[str],
    shape: tuple[int, int],
    *,
    include_disabled: bool = False,
) -> dict[str, np.ndarray]:
    """Inverse of :func:`encode_frame`: the stored string -> ``{label: mask}``.

    Labels absent from the frame are absent from the result rather than present
    and empty, so a caller can tell "not detected" from "detected nothing".

    Disabled entries are omitted by default. That is the safe direction: the
    compositor calls this, and a muted mask must not reach training. Pass
    ``include_disabled`` where mutedness is the thing being displayed.
    """
    out: dict[str, np.ndarray] = {}
    for name, counts, enabled in _entries(value, labels):
        if enabled or include_disabled:
            out[name] = decode_mask(counts, shape)
    return out


def feature_spec(labels: list[str], shape: tuple[int, int]) -> dict:
    """The dataset feature declaration for one camera's masks.

    Mirrors how ``quality.human_flags`` carries its ``flags`` vocabulary: the
    per-frame rows stay small and the meaning lives once, in the metadata. The
    mask resolution is recorded because it is the SEGMENTED resolution, which is
    deliberately not the stored video's — masks are computed at source scale,
    where SAM3 can still see a 50 px object, and may be composited against a
    downscaled frame later.
    """
    return {
        "dtype": "string",
        "shape": [1],
        "names": None,
        "mask_encoding": "coco_rle",
        "mask_labels": list(labels),
        "mask_size": [int(shape[0]), int(shape[1])],
    }
