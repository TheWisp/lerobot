# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Contracts for the mask codec.

Masks are labels: a lossy round trip corrupts the annotation rather than
degrading it visibly, so exactness is the property worth pinning, and it is
pinned on the shapes that actually break run-length coders — uniform masks with
no runs at all, alternating masks where every pixel is a run, and the
column-major convention that makes the output readable by COCO tooling.
"""

import json

import numpy as np
import pytest

from lerobot.datasets.mask_codec import (
    EMPTY,
    _string_to_counts,
    decode_frame,
    decode_mask,
    encode_frame,
    encode_mask,
    feature_spec,
)


def _roundtrip(mask: np.ndarray) -> np.ndarray:
    return decode_mask(encode_mask(mask), mask.shape)


@pytest.mark.parametrize(
    "name,mask",
    [
        ("empty", np.zeros((17, 23), bool)),
        ("full", np.ones((17, 23), bool)),
        ("checkerboard", np.indices((17, 23)).sum(0) % 2 == 0),
        ("random", np.random.default_rng(0).random((64, 64)) > 0.5),
        ("one_pixel", np.eye(9, dtype=bool)[:1].repeat(9, 0) & np.eye(9, dtype=bool)),
        ("single_column", np.zeros((32, 8), bool)),
        ("wide", np.zeros((3, 500), bool)),
    ],
)
def test_roundtrip_is_exact(name: str, mask: np.ndarray):
    """Uniform and alternating masks are where run-length coders go wrong.

    An all-zero mask has no value changes at all and still has to emit one run
    covering every pixel, or the frame does not decode; a checkerboard makes
    every pixel its own run, which is the worst case for the delta coding.
    """
    if name == "single_column":
        mask[:, 3] = True
    if name == "wide":
        mask[1, 100:400] = True
    assert np.array_equal(_roundtrip(mask), mask), name


def test_counts_follow_coco_column_major_convention():
    """Interoperability: pycocotools reads column-major runs starting with zeros."""
    mask = np.zeros((4, 3), bool)
    mask[:, 0] = True  # a full FIRST COLUMN, which is contiguous only in column-major
    assert _string_to_counts(encode_mask(mask)) == [0, 4, 8]

    row = np.zeros((4, 3), bool)
    row[0, :] = True  # a full first ROW is strided in column-major
    assert _string_to_counts(encode_mask(row)) == [0, 1, 3, 1, 3, 1, 3]


def test_counts_always_cover_the_whole_frame():
    """The decoder relies on the runs summing to h*w; a short RLE must not pass."""
    for mask in (np.zeros((5, 7), bool), np.ones((5, 7), bool)):
        assert sum(_string_to_counts(encode_mask(mask))) == mask.size


def test_decode_rejects_an_rle_that_does_not_fit_the_shape():
    counts = encode_mask(np.zeros((4, 4), bool))
    with pytest.raises(ValueError, match="RLE covers"):
        decode_mask(counts, (8, 8))


def test_frame_roundtrip_keeps_labels_and_drops_absent_ones():
    labels = ["tray", "ball", "box"]
    tray = np.zeros((12, 10), bool)
    tray[2:8, 1:9] = True
    ball = np.zeros((12, 10), bool)
    ball[5, 5] = True

    value = encode_frame({"tray": tray, "ball": ball, "box": np.zeros((12, 10), bool)}, labels)
    out = decode_frame(value, labels, (12, 10))

    # "box" detected nothing, so it is absent rather than present-and-empty:
    # a consumer must be able to tell those apart.
    assert set(out) == {"tray", "ball"}
    assert np.array_equal(out["tray"], tray)
    assert np.array_equal(out["ball"], ball)


def test_frame_encoding_is_stable_for_the_same_content():
    """Ordered by label id, so equal frames encode equal and diffs stay readable."""
    labels = ["a", "b"]
    m = np.zeros((6, 6), bool)
    m[1:3, 1:3] = True
    first = encode_frame({"b": m, "a": m}, labels)
    second = encode_frame({"a": m, "b": m}, labels)
    assert first == second
    assert [row[0] for row in json.loads(first)] == [0, 1]


def test_unknown_label_is_rejected_rather_than_silently_dropped():
    with pytest.raises(KeyError, match="vocabulary"):
        encode_frame({"ghost": np.ones((4, 4), bool)}, ["tray"])


def test_empty_frame_is_representable():
    assert decode_frame(EMPTY, ["tray"], (4, 4)) == {}
    assert decode_frame("", ["tray"], (4, 4)) == {}


def test_feature_spec_declares_a_string_column_carrying_its_vocabulary():
    """The vocabulary lives in metadata, like quality.human_flags' `flags`.

    Per-frame rows then carry small integer ids instead of repeating label text
    47k times, and a reader can interpret the column without external context.
    """
    spec = feature_spec(["tray", "ball"], (720, 1280))
    assert spec["dtype"] == "string"
    assert spec["shape"] == [1]
    assert spec["mask_encoding"] == "coco_rle"
    assert spec["mask_labels"] == ["tray", "ball"]
    assert spec["mask_size"] == [720, 1280]


def _reference_string_to_counts(s: str) -> list[int]:
    """The straightforward character-at-a-time reader, kept as the oracle.

    The shipped decoder is vectorized because it runs per label per frame in
    playback and in training; the delta-against-two-back and the sign bit are
    exactly the parts that are easy to get subtly wrong in array form, so they
    are checked against the obvious implementation rather than against
    themselves.
    """
    counts: list[int] = []
    p, n = 0, len(s)
    while p < n:
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[len(counts) - 2]
        counts.append(x)
    return counts


@pytest.mark.parametrize("seed", range(12))
def test_vectorized_parse_matches_the_scalar_reader(seed: int):
    """Random masks, including the shapes that stress the coder: long runs
    (large multi-character groups) and alternating pixels (every run length 1).
    """
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(4, 90)), int(rng.integers(4, 90))
    style = seed % 3
    if style == 0:
        mask = rng.random((h, w)) > 0.5  # every run tiny
    elif style == 1:
        mask = np.zeros((h, w), bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True  # few long runs
    else:
        mask = rng.random((h, w)) > 0.9  # sparse
    s = encode_mask(mask)
    assert _string_to_counts(s) == _reference_string_to_counts(s)
    assert np.array_equal(decode_mask(s, (h, w)), mask)


def test_vectorized_parse_handles_the_degenerate_rows():
    for mask in (np.zeros((7, 5), bool), np.ones((7, 5), bool)):
        s = encode_mask(mask)
        assert _string_to_counts(s) == _reference_string_to_counts(s)
    assert _string_to_counts("") == []
