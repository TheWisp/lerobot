# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The RLE codec, against the cases COCO implementations are known to fail.

The cases are not chosen for typical masks -- a rectangle is two runs and tells
you nothing. They are the shapes that break run-length coders, taken from what
goes wrong in this format generally:

* a uniform mask has no value changes at all, yet its counts must still sum to
  h*w or the frame does not decode;
* a mask that starts SET needs a leading empty run, because the format defines
  the first run as zeros;
* a checkerboard makes every pixel its own run, the maximum the encoder emits;
* first-row and first-column are indistinguishable unless the column-major
  convention is right, which is the single most common way to get COCO RLE
  wrong;
* a 720p uniform mask is one run of 921,600, which only encodes correctly if
  the variable-length count encoding carries across characters -- the classic
  break, and invisible on small test masks;
* a probability map and a 0/255 uint8 mask are what callers actually hold.

Every case asserts a round trip, because a decoder that agrees with a broken
encoder is the failure this cannot afford: it looks like a slightly-off overlay
rather than a bug, and gets read as poor segmentation.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.mask_codec import (
    decode_frame,
    decode_mask,
    encode_frame,
    encode_mask,
    feature_spec,
    frame_states,
)


def _roundtrip(mask: np.ndarray) -> np.ndarray:
    return decode_mask(encode_mask(mask), mask.shape)


def _checker(h: int, w: int) -> np.ndarray:
    return np.indices((h, w)).sum(0) % 2 == 0


@pytest.mark.parametrize(
    ("name", "mask"),
    [
        ("all zero -- no value changes, must still cover every pixel", np.zeros((8, 8), bool)),
        ("all one -- needs a leading empty run", np.ones((8, 8), bool)),
        ("single pixel, first", np.pad(np.ones((1, 1), bool), ((0, 7), (0, 7)))),
        ("single pixel, last", np.pad(np.ones((1, 1), bool), ((7, 0), (7, 0)))),
        ("checkerboard -- every pixel its own run", _checker(8, 8)),
        ("1xN", np.zeros((1, 16), bool)),
        ("Nx1", np.zeros((16, 1), bool)),
        ("1x1", np.zeros((1, 1), bool)),
        ("odd dimensions", np.pad(np.ones((1, 1), bool), ((3, 3), (5, 7)))),
    ],
)
def test_round_trip(name, mask):
    assert np.array_equal(_roundtrip(mask), mask), name


def test_first_row_and_first_column_are_distinguishable():
    """The column-major convention, which is the usual way to get this wrong.
    Row-major would encode these two identically."""
    row, col = np.zeros((8, 8), bool), np.zeros((8, 8), bool)
    row[0] = True
    col[:, 0] = True
    assert encode_mask(row) != encode_mask(col)
    assert np.array_equal(_roundtrip(row), row)
    assert np.array_equal(_roundtrip(col), col)


def test_a_run_longer_than_one_character_encodes():
    """A 720p uniform mask is a single run of 921,600. Counts are variable
    length, so this is where an encoder that does not carry across characters
    breaks -- and it cannot be seen on a small mask."""
    mask = np.ones((720, 1280), bool)
    assert np.array_equal(_roundtrip(mask), mask)
    assert len(encode_mask(mask)) < 20, "a single run should stay tiny"


def test_a_single_pixel_at_720p_is_two_huge_runs():
    mask = np.zeros((720, 1280), bool)
    mask[359, 639] = True
    assert np.array_equal(_roundtrip(mask), mask)
    assert int(_roundtrip(mask).sum()) == 1


@pytest.mark.parametrize(
    ("name", "mask"),
    [
        ("uint8 0/1", _checker(8, 8).astype(np.uint8)),
        ("uint8 0/255 -- the cv2 convention", _checker(8, 8).astype(np.uint8) * 255),
        ("non-contiguous view", np.ones((8, 8), bool)[::2]),
        ("already Fortran-ordered", np.asfortranarray(_checker(8, 8))),
    ],
)
def test_what_callers_actually_hold(name, mask):
    assert np.array_equal(_roundtrip(mask), mask.astype(bool)), name


def test_a_probability_map_is_truthy_not_thresholded():
    """The codec takes any nonzero as set. Thresholding a probability map is
    the CALLER's job -- `mask_store.write_episode` does it at 0.5 -- and this
    pins the codec's own rule so the two cannot silently disagree."""
    soft = np.zeros((4, 4), np.float32)
    soft[0] = 0.9
    soft[1] = 0.01
    assert int(_roundtrip(soft).sum()) == 8, "the codec thresholds at 0, not 0.5"


def test_a_mask_that_is_not_2d_is_refused():
    with pytest.raises(ValueError, match="2-D"):
        encode_mask(np.zeros((4, 4, 3), bool))


def test_decoding_against_the_wrong_shape_is_refused():
    """Silently reshaping would put a mask over the wrong pixels."""
    m = np.zeros((4, 4), bool)
    m[0] = True
    with pytest.raises(ValueError, match="covers"):
        decode_mask(encode_mask(m), (8, 8))


# ── the frame level: label ids, ordering, and the empty answers ─────────────


def test_a_frame_round_trips_by_label():
    labels = ["ball", "tray"]
    masks = {"ball": np.zeros((4, 4), bool), "tray": np.zeros((4, 4), bool)}
    masks["ball"][0] = True
    masks["tray"][:, 0] = True
    got = decode_frame(encode_frame(masks, labels), labels, (4, 4))
    assert set(got) == set(labels)
    for name in labels:
        assert np.array_equal(got[name], masks[name])


def test_rows_are_ordered_by_label_id():
    """Two frames with the same content must encode identically, or every row
    differs from its neighbour for no reason and the column stops compressing."""
    labels = ["a", "b"]
    m = {n: np.zeros((4, 4), bool) for n in labels}
    for n in labels:
        m[n][0] = True
    assert encode_frame({"a": m["a"], "b": m["b"]}, labels) == encode_frame(
        {"b": m["b"], "a": m["a"]}, labels
    )


def test_an_absent_object_is_absent_not_an_empty_run():
    """Storing an empty run for every declared label would cost bytes on every
    frame and make "not found" indistinguishable from "found nothing"."""
    labels = ["ball", "tray"]
    row = encode_frame({"ball": np.zeros((4, 4), bool)}, labels)
    assert row == "[]"
    assert decode_frame(row, labels, (4, 4)) == {}


@pytest.mark.parametrize("value", ["", "[]", None])
def test_an_empty_row_decodes_to_nothing(value):
    """`""` is never written and `"[]"` is segmented-found-nothing; both read as
    no masks, and neither may raise."""
    assert decode_frame(value, ["ball"], (4, 4)) == {}


def test_a_label_outside_the_vocabulary_is_refused_on_encode():
    with pytest.raises(KeyError, match="not in the declared vocabulary"):
        encode_frame({"ghost": np.ones((4, 4), bool)}, ["ball"])


def test_a_label_id_outside_the_vocabulary_is_refused_on_decode():
    """A row written against a longer vocabulary must not silently index past
    the end -- that would rename every object after the missing one."""
    with pytest.raises(ValueError, match="outside the declared vocabulary"):
        decode_frame('[[3,"PPk0"]]', ["ball"], (4, 4))


def test_the_feature_spec_declares_what_a_reader_needs():
    spec = feature_spec(["ball", "tray"], (240, 320))
    assert spec["dtype"] == "string"
    assert spec["shape"] == [1]
    assert spec["mask_encoding"] == "coco_rle"
    assert spec["mask_labels"] == ["ball", "tray"]
    assert spec["mask_size"] == [240, 320]


# ── the disabled state ──────────────────────────────────────────────────────
# A muted mask is stored but ignored: it stops reaching training and no
# gap-filling write may replace it. Without a state of its own, "this detection
# is wrong here" and "nothing was found here" are identical in storage, so the
# next pass puts the wrong detection straight back.


def _one(name="ball"):
    m = np.zeros((4, 4), bool)
    m[0] = True
    return {name: m}


def test_an_enabled_entry_is_byte_identical_to_what_it_was():
    """Every row written before the flag existed must encode the same, or the
    column stops diffing cleanly and every old frame looks changed."""
    assert encode_frame(_one(), ["ball"]) == '[[0,"013000000"]]'


def test_a_disabled_entry_carries_a_third_element():
    assert encode_frame(_one(), ["ball"], disabled=["ball"]) == '[[0,"013000000",0]]'


def test_decode_omits_a_disabled_mask_by_default():
    """The safe direction: the compositor calls decode_frame, and a muted mask
    must not reach training."""
    row = encode_frame(_one(), ["ball"], disabled=["ball"])
    assert decode_frame(row, ["ball"], (4, 4)) == {}


def test_decode_can_be_asked_for_disabled_masks():
    """The GUI draws them muted, so it needs the pixels."""
    row = encode_frame(_one(), ["ball"], disabled=["ball"])
    got = decode_frame(row, ["ball"], (4, 4), include_disabled=True)
    assert set(got) == {"ball"}
    assert got["ball"][0].all()


def test_a_disabled_mask_keeps_its_pixels():
    """Muting is not deleting -- the mask survives so it can be unmuted."""
    plain = decode_frame(encode_frame(_one(), ["ball"]), ["ball"], (4, 4))["ball"]
    muted = decode_frame(
        encode_frame(_one(), ["ball"], disabled=["ball"]), ["ball"], (4, 4), include_disabled=True
    )["ball"]
    assert np.array_equal(plain, muted)


def test_states_reports_presence_and_mutedness_without_decoding():
    labels = ["ball", "tray"]
    masks = {"ball": np.zeros((4, 4), bool), "tray": np.zeros((4, 4), bool)}
    masks["ball"][0] = True
    masks["tray"][:, 0] = True
    row = encode_frame(masks, labels, disabled=["tray"])
    assert frame_states(row, labels) == {"ball": True, "tray": False}


def test_states_of_an_empty_row_is_empty():
    for value in ("", "[]", None):
        assert frame_states(value, ["ball"]) == {}


def test_a_two_element_entry_reads_as_enabled():
    """Rows written before the flag existed have no third element."""
    assert frame_states('[[0,"013000000"]]', ["ball"]) == {"ball": True}


def test_mixed_enabled_and_disabled_in_one_frame():
    labels = ["a", "b", "c"]
    masks = {n: np.zeros((4, 4), bool) for n in labels}
    for i, n in enumerate(labels):
        masks[n][i] = True
    row = encode_frame(masks, labels, disabled=["b"])
    assert frame_states(row, labels) == {"a": True, "b": False, "c": True}
    assert set(decode_frame(row, labels, (4, 4))) == {"a", "c"}
    assert set(decode_frame(row, labels, (4, 4), include_disabled=True)) == {"a", "b", "c"}


def test_disabling_a_label_that_is_not_present_stores_nothing():
    """Muting is a property of a mask; with no mask there is nothing to mute."""
    row = encode_frame(_one("ball"), ["ball", "tray"], disabled=["tray"])
    assert frame_states(row, ["ball", "tray"]) == {"ball": True}


def test_the_explicit_enabled_form_is_accepted_on_read():
    """`[id, counts, 1]` is never written here, but a reader must take it: the
    default has to be carried for pre-flag rows anyway, so tolerating the
    explicit form costs nothing and makes another writer's output readable."""
    for third in ("1", "true"):
        row = f'[[0,"013000000",{third}]]'
        assert frame_states(row, ["ball"]) == {"ball": True}
        assert set(decode_frame(row, ["ball"], (4, 4))) == {"ball"}


@pytest.mark.parametrize("n_labels", [1, 3, 7])
def test_an_enabled_entry_never_carries_a_flag(n_labels):
    """The default is not written, so a dataset's bytes do not depend on which
    version made it. An invariant over the vocabulary rather than one golden
    string, which would pass for a single shape and say nothing about the rest.
    """
    labels = [f"l{i}" for i in range(n_labels)]
    masks = {}
    for i, name in enumerate(labels):
        m = np.zeros((4, 4), bool)
        m[i % 4] = True
        masks[name] = m
    assert all(len(e) == 2 for e in json.loads(encode_frame(masks, labels)))


def test_only_the_disabled_entries_carry_a_flag():
    labels = ["a", "b", "c"]
    masks = {}
    for i, name in enumerate(labels):
        m = np.zeros((4, 4), bool)
        m[i] = True
        masks[name] = m
    row = json.loads(encode_frame(masks, labels, disabled=["b"]))
    assert [len(e) for e in row] == [2, 3, 2]


def test_a_disabled_name_outside_the_vocabulary_is_refused():
    """A typo'd name would otherwise mute nothing and say nothing, leaving the
    mask reaching training -- the same failure a typo'd mask label is refused
    for."""
    with pytest.raises(KeyError, match="not in the declared vocabulary"):
        encode_frame(_one("ball"), ["ball"], disabled=["bal"])


# ── against the reference implementation, not against ourselves ─────────────


def _coco_reference() -> dict:
    return json.loads((Path(__file__).parent / "coco_reference.json").read_text())


def _build(spec: dict) -> np.ndarray:
    """Materialise a reference case's mask from its description.

    The fixture stores masks as the rectangles they are made of rather than as
    pixel lists: one 720x1280 case written per-pixel is 11 MB of fixture. This
    mirrors `regen_coco_reference.py` and uses only numpy slicing, so nothing
    about building the input borrows from the encoding under test.
    """
    h, w = spec["h"], spec["w"]
    if spec["fill"] == "pixels":
        return np.array(spec["pixels"], dtype=bool).reshape(h, w)
    if spec["fill"] == "checker":
        return np.indices((h, w)).sum(axis=0) % 2 == 0
    m = np.zeros((h, w), bool)
    for r0, r1, c0, c1 in spec["rects"]:
        m[r0:r1, c0:c1] = True
    return m


def test_the_encoder_is_byte_for_byte_coco():
    """The one test the round trips above cannot be.

    Every other assertion in this file is encode -> our own decode, which an
    encoder and a decoder wrong in the SAME way both pass -- and being COCO
    byte-for-byte is the entire reason this is hand-rolled, since it is what
    lets `pycocotools.mask.decode` and any other COCO consumer read a dataset
    written here. The expectations come from pycocotools itself (see
    `regen_coco_reference.py`), captured as data so the repo needs no C
    extension at test time.
    """
    cases = _coco_reference()
    assert len(cases) >= 8, f"the reference fixture looks thin: {sorted(cases)}"
    for name, case in cases.items():
        assert encode_mask(_build(case)) == case["counts"], (
            f"{name}: the encoder no longer matches pycocotools, so masks written here "
            f"are no longer readable by COCO tooling"
        )


def test_the_decoder_reads_what_coco_wrote():
    """The other half of interop: reading a mask this repo did not write.

    Encoding agreement alone would still allow a decoder that only understands
    our own output.
    """
    for name, case in _coco_reference().items():
        got = decode_mask(case["counts"], (case["h"], case["w"]))
        assert np.array_equal(got, _build(case)), f"{name}: decoded a COCO string wrongly"


def test_the_reference_covers_what_breaks_run_length_coders():
    """A fixture regenerated without these is not an oracle for much.

    Uniform masks emit no value changes; the checkerboard makes every pixel its
    own run; first_row and first_column differ only under the column-major
    convention; and a 720p mask has runs too long for a single count character.
    """
    cases = _coco_reference()
    assert {"empty", "full", "checker", "first_row", "first_column"} <= set(cases), sorted(cases)
    assert [c for c in cases.values() if c["h"] * c["w"] > 500_000], (
        "no case large enough to need a multi-character count"
    )
    # The two column-major cases must actually differ, or the convention is
    # untested however many cases there are.
    assert cases["first_row"]["counts"] != cases["first_column"]["counts"]
