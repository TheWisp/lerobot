"""A chunk is built, and its pixels are the stored video's (design: R2-R5, C1, C2, C5).

The one thing that cannot be checked by looking at the picture: whether the
frames in a chunk are the frames the file holds at those indices, in that
order. Every frame here is a flat grey unique to (episode, frame), so the
check is exact and an off-by-one chunk -- which decodes perfectly well --
fails it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("av")

import requests  # noqa: E402

from tests.gui.chunk_fixtures import (  # noqa: E402
    CAM_NARROW,
    CAM_WIDE,
    CAMS,
    EPISODES,
    FPS,
    FRAMES,
    SIZES,
    GuiServer,
    build_dataset,
    chunk_url,
    decode_h264,
    frame_ids,
    nal_types,
    parse_chunk,
)

CHUNK_FRAMES = 20  # the design's 2 s chunk at this fixture's 10 fps


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("chunks") / "chunks")


@pytest.fixture(scope="module")
def server(dataset_root, tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    ds_id = srv.open_dataset(dataset_root)
    yield srv, ds_id
    srv.stop()


def _get(server, episode, start, profile="low"):
    srv, ds_id = server
    return requests.get(chunk_url(srv.base, ds_id, episode, start, profile), timeout=120)


def test_a_chunk_holds_the_stored_frames(server):
    """Frame n of the chunk is frame start+n of the file, on every camera."""
    r = _get(server, 1, 0)
    assert r.status_code == 200, r.text
    header, parts = parse_chunk(r.content)
    assert header["episode_index"] == 1
    assert header["first_frame"] == 0
    assert header["frames"] == CHUNK_FRAMES
    assert header["fps"] == FPS
    for cam in CAMS:
        part, data = parts[("video", cam)]
        assert part["codec"] == "h264"
        frames = decode_h264(data)
        assert len(frames) == CHUNK_FRAMES, cam
        assert len(part["frame_sizes"]) == CHUNK_FRAMES and sum(part["frame_sizes"]) == len(data), cam
        assert frame_ids(frames) == [(n, 1) for n in range(CHUNK_FRAMES)], cam


def test_each_camera_is_scaled_from_its_own_resolution(server):
    """The profile is a target width, not one output resolution: the wide camera
    comes down to it, the narrow one is not upscaled, and the header says so."""
    r = _get(server, 0, 0)
    assert r.status_code == 200, r.text
    header, parts = parse_chunk(r.content)
    assert header["profile"] == "low"
    part_w, data_w = parts[("video", CAM_WIDE)]
    part_n, data_n = parts[("video", CAM_NARROW)]
    fw = decode_h264(data_w)[0]
    fn = decode_h264(data_n)[0]
    assert fw.shape[:2] == (160, 320), fw.shape  # 480x240 scaled to the 320 target, aspect kept
    assert fn.shape[:2] == SIZES[CAM_NARROW], fn.shape  # 200x100 left alone
    assert (part_w["width"], part_w["height"]) == (320, 160)
    assert (part_n["width"], part_n["height"]) == (200, 100)
    # And each camera's declared size rides beside it: the page draws the tile at
    # that size, as the JPEG path does, and scales the picture into it.
    assert (part_w["stored_width"], part_w["stored_height"]) == (480, 240)
    assert (part_n["stored_width"], part_n["stored_height"]) == (200, 100)


def test_the_first_frame_is_a_keyframe(server):
    """A chunk decodes on its own: parameter sets and an IDR frame come first,
    so the page never depends on the previous chunk having arrived."""
    r = _get(server, 0, CHUNK_FRAMES)
    header, parts = parse_chunk(r.content)
    for cam in CAMS:
        part, data = parts[("video", cam)]
        first_au = data[: part["frame_sizes"][0]]
        types = nal_types(first_au)
        assert 7 in types and 8 in types, (cam, types)  # SPS, PPS
        assert 5 in types, (cam, types)  # IDR
        assert types[0] == 9, (cam, types)  # the access-unit delimiter the page splits on


def test_a_chunk_past_the_last_frame_is_short_not_wrong(server):
    """The last chunk of an episode is partial. The next episode's frames sit in
    the same file right after it, and must not be served as this episode's."""
    r = _get(server, 0, CHUNK_FRAMES)  # frames 20..34 of 35
    assert r.status_code == 200, r.text
    header, parts = parse_chunk(r.content)
    assert header["frames"] == FRAMES - CHUNK_FRAMES == 15
    for cam in CAMS:
        _, data = parts[("video", cam)]
        ids = frame_ids(decode_h264(data))
        assert ids == [(i, 0) for i in range(CHUNK_FRAMES, FRAMES)], (cam, ids)


def test_an_episode_is_read_through_the_dataset_accessors(dataset_root):
    """A dataset opened with an episode filter holds a subset of rows. A builder
    that dissects metadata by hand reads the wrong rows for it; one that reads
    through the accessors gets episode 1's frames."""
    import gzip
    import json

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.gui.api import chunk_playback

    ds = LeRobotDataset("tests/chunks", root=dataset_root, episodes=[1])
    body, header = chunk_playback.build_chunk(ds, 1, CHUNK_FRAMES, "low")
    _, parts = parse_chunk(body)
    _, data = parts[("video", CAM_WIDE)]
    assert frame_ids(decode_h264(data)) == [(CHUNK_FRAMES + n, 1) for n in range(CHUNK_FRAMES - 5)]
    # The rows are where a hand-rolled global index goes wrong on a subset table:
    # episode 1's rows are the table's first rows here, not rows 45 onward.
    part, rows = parts[("masks", CAM_WIDE)]
    rows = json.loads(gzip.decompress(rows))
    assert len(rows) == header["frames"] == CHUNK_FRAMES - 5, len(rows)
    assert all(len(r) == 1 and r[0][0] == 0 for r in rows), "every frame of the chunk carries its stored mask"


@pytest.mark.parametrize(
    ("episode", "start", "profile", "status"),
    [
        (0, 7, "low", 400),  # off the chunk grid: a client bug, not a request to honour
        (0, 0, "medium", 400),  # a profile the design does not offer
        (0, 0, "high", 400),  # high is the JPEG path, not an encode
        (EPISODES, 0, "low", 404),  # no such episode
        (0, FRAMES, "low", 404),  # start past the episode's last frame
    ],
)
def test_bad_requests_are_refused(server, episode, start, profile, status):
    r = _get(server, episode, start, profile)
    assert r.status_code == status, (r.status_code, r.text)


def test_responses_are_not_browser_cacheable(server):
    """Like the JPEG path: the browser asks the server every time, and the server's
    cache is what an edit invalidates (design: C7, R7)."""
    r = _get(server, 0, 0)
    cc = r.headers.get("Cache-Control", "")
    assert "no-store" in cc, cc
