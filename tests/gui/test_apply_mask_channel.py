# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The channel that carries a run's masks from the overlay worker to the server.

Apply is the preview loop with frame-skipping removed, so a run's masks are the
ones already computed for the picture — they only have to reach the server,
which is the only process allowed to write a dataset. This is that hop.

Two properties it exists for, both invisible if they break:

* the block is a single-slot latch, so masks travel in BATCHES; a per-frame write
  the reader had not yet picked up would be overwritten and that frame would
  vanish, looking exactly like a frame that was never segmented;
* each camera carries the obs-stream sequence the worker CONSUMED, so the server
  attributes masks to the frame they came from rather than to wherever the
  playhead has since moved — per camera, because the image blocks' counters
  diverge as soon as one camera is missing from a published frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.datasets.mask_codec import decode_mask, encode_mask

pytest.importorskip("lerobot.overlays.overlay_ipc")
from lerobot.overlays.overlay_ipc import SharedOverlayBuffer  # noqa: E402


@pytest.fixture
def buffers():
    """A writer (worker side) and a reader (server side) over one segment."""
    cams = {"observation.images.top": (8, 12)}
    writer = SharedOverlayBuffer(cameras=cams, model="sam3_track", create=True)
    reader = SharedOverlayBuffer(create=False)
    yield writer, reader
    writer.close() if hasattr(writer, "close") else None


def _rle(h=8, w=12, rows=(2, 5)):
    m = np.zeros((h, w), bool)
    m[rows[0] : rows[1]] = True
    return encode_mask(m), m


def test_a_batch_survives_the_trip(buffers):
    writer, reader = buffers
    counts, mask = _rle()
    ok = writer.write_masks([{"observation.images.top": {"seq": 41, "rle": {"ring": counts}}}])
    assert ok, "the batch did not fit"

    got = reader.read_masks()
    assert len(got) == 1
    cam = got[0]["observation.images.top"]
    assert cam["seq"] == 41, "the sequence the worker consumed was not carried"
    assert np.array_equal(decode_mask(cam["rle"]["ring"], mask.shape), mask), "the mask did not survive"


def test_several_frames_travel_in_one_write(buffers):
    """The latch hazard: one write, many frames.

    A per-frame write is what this shape exists to avoid, so the batch has to be
    able to hold more than one — and every frame in it must keep its own seq.
    """
    writer, reader = buffers
    counts, _ = _rle()
    batch = [{"observation.images.top": {"seq": s, "rle": {"ring": counts}}} for s in (7, 8, 9)]
    assert writer.write_masks(batch)
    got = reader.read_masks()
    assert [e["observation.images.top"]["seq"] for e in got] == [7, 8, 9]


def test_an_oversized_batch_is_refused_rather_than_truncated(buffers):
    """A clipped JSON body is unparsable, so the whole batch would read as
    frames that produced nothing. The writer must say no instead, and leave what
    was there alone so the caller can flush a smaller batch."""
    writer, reader = buffers
    counts, _ = _rle()
    assert writer.write_masks([{"observation.images.top": {"seq": 1, "rle": {"ring": counts}}}])

    huge = [
        {"observation.images.top": {"seq": i, "rle": {f"label{j}": "x" * 4000 for j in range(20)}}}
        for i in range(50)
    ]
    assert writer.write_masks(huge) is False, "an oversized batch was accepted"
    # And the previous batch is still readable, not half-overwritten.
    still = reader.read_masks()
    assert [e["observation.images.top"]["seq"] for e in still] == [1], still


def test_the_reader_can_tell_a_new_batch_from_one_it_has(buffers):
    """Without this the server would re-stage the same frames on every poll."""
    writer, reader = buffers
    counts, _ = _rle()
    writer.write_masks([{"observation.images.top": {"seq": 1, "rle": {"ring": counts}}}])
    first = reader.masks_seq()
    assert reader.read_masks()
    assert reader.masks_seq() == first, "reading must not advance the write counter"

    writer.write_masks([{"observation.images.top": {"seq": 2, "rle": {"ring": counts}}}])
    assert reader.masks_seq() != first, "a new batch did not advance the counter"


def test_each_camera_keeps_its_own_sequence(buffers):
    """Two cameras in one sweep may sit at different obs sequences: an image
    block's counter only advances when that camera is present in the frame, so a
    single seq for the entry would file one camera's masks against another
    camera's frame."""
    writer, reader = buffers
    counts, _ = _rle()
    writer.write_masks(
        [
            {
                "observation.images.top": {"seq": 11, "rle": {"ring": counts}},
                "observation.images.wrist": {"seq": 9, "rle": {"ring": counts}},
            }
        ]
    )
    entry = reader.read_masks()[0]
    assert entry["observation.images.top"]["seq"] == 11
    assert entry["observation.images.wrist"]["seq"] == 9


def test_nothing_published_reads_as_nothing(buffers):
    """The complement: an empty channel must not look like a frame with no masks,
    which the server would stage as 'segmented, found nothing'."""
    _writer, reader = buffers
    assert reader.read_masks() == []


def test_an_empty_batch_is_a_programming_error(buffers):
    writer, _reader = buffers
    with pytest.raises(AssertionError):
        writer.write_masks([])
