"""Masks cross at each camera's encoded resolution (design: O4, O5, C3, R11).

Masks at the stored resolution were a third of a chunk's bytes and stalled
playback; resized to the encoded resolution they fit. A resized mask must still
cover the same part of the frame -- it was computed at full resolution and is
only drawn smaller.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

pytest.importorskip("av")

import requests  # noqa: E402

from tests.gui.chunk_fixtures import (  # noqa: E402
    CAM_NARROW,
    CAM_WIDE,
    SIZES,
    GuiServer,
    blob_mask,
    build_dataset,
    chunk_url,
    parse_chunk,
)


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("masks") / "masks")


@pytest.fixture(scope="module")
def server(dataset_root, tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    ds_id = srv.open_dataset(dataset_root)
    yield srv, ds_id
    srv.stop()


@pytest.fixture(scope="module")
def chunk(server):
    srv, ds_id = server
    r = requests.get(chunk_url(srv.base, ds_id, 0, 0), timeout=120)
    assert r.status_code == 200, r.text
    return parse_chunk(r.content)


def _rows(part, data):
    assert part["encoding"] == "gzip"
    return json.loads(gzip.decompress(data))


def test_rows_are_resized_to_the_cameras_encoded_resolution(chunk):
    """The mask rows for the wide camera come at the resolution its video was
    encoded at, and cover the same fraction of the frame as the stored mask."""
    from lerobot.datasets.mask_codec import decode_mask

    header, parts = chunk
    video, _ = parts[("video", CAM_WIDE)]
    part, data = parts[("masks", CAM_WIDE)]
    assert part["size"] == [video["height"], video["width"]] == [160, 320]
    rows = _rows(part, data)
    assert len(rows) == header["frames"]
    h, w = SIZES[CAM_WIDE]
    stored = blob_mask(h, w)
    sy, sx = np.nonzero(stored)
    for entries in rows:
        assert len(entries) == 1, entries
        label, counts, *_ = entries[0]
        small = decode_mask(counts, (160, 320))
        assert small.shape == (160, 320)
        assert abs(small.mean() - stored.mean()) < 0.005, (small.mean(), stored.mean())
        # It is the same region drawn smaller, not a different region: its bounding
        # box is the stored one scaled by 2/3.
        ys, xs = np.nonzero(small)
        assert (ys.min(), ys.max(), xs.min(), xs.max()) == pytest.approx(
            (sy.min() * 2 / 3, sy.max() * 2 / 3, sx.min() * 2 / 3, sx.max() * 2 / 3), abs=1.5
        )


def test_every_masked_camera_in_a_chunk_carries_its_own_rows(chunk):
    """The page composites from the chunk, so a chunk with video only cannot be
    drawn the way the tab draws it today."""
    header, parts = chunk
    assert ("masks", CAM_WIDE) in parts
    assert ("masks", CAM_NARROW) not in parts, "a camera with no mask column carries no rows"
    assert parts[("masks", CAM_WIDE)][0]["labels"] == ["ball"]


def test_the_bytes_are_bounded_by_the_encoded_resolution(chunk):
    """The measured failure: at the stored resolution masks were a third of the
    chunk. Resized rows are smaller than the same rows at the stored size."""
    from lerobot.datasets.mask_codec import encode_mask

    header, parts = chunk
    _, data = parts[("masks", CAM_WIDE)]
    h, w = SIZES[CAM_WIDE]
    stored = blob_mask(h, w)
    full = gzip.compress(
        json.dumps([[["ball", encode_mask(stored)]] for _ in range(header["frames"])]).encode()
    )
    assert len(data) < len(full), (len(data), len(full))
    video_bytes = parts[("video", CAM_WIDE)][0]["length"]
    assert len(data) < video_bytes, ("masks outweigh the video they annotate", len(data), video_bytes)
