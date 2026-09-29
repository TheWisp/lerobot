# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A depth camera on both playback paths: the JPEG tile and the Low Bandwidth
chunk. Each renders the stored codes through the display mapping the page
asked for, keyed apart in its cache, instead of drawing the raw 12-bit codes
as luma, which was a near-black picture.
"""

from __future__ import annotations

import asyncio
import io

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.api.chunk_playback import _cache_key, build_chunk
from lerobot.gui.depth_display import DepthMapping
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState
from tests.gui.chunk_fixtures import decode_h264, parse_chunk

FPS, FRAMES, H, W = 10, 25, 96, 128
CAM = "observation.images.d"
CHUNK_FRAMES = 20


def _depth_frame(i: int) -> np.ndarray:
    """A ramp from 200 to 1000 mm left to right, drifting a millimetre per
    frame, with a hole of no measurement in the top-left corner."""
    d = np.tile(np.linspace(200, 1000, W, dtype=np.float32), (H, 1)) + i
    d[:8, :8] = 0
    return d.astype(np.uint16)[..., None]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("depth") / "ds"
    ds = LeRobotDataset.create(
        repo_id="tests/depth_playback",
        fps=FPS,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (1,), "names": ["a"]},
            "action": {"dtype": "float32", "shape": (1,), "names": ["a"]},
            CAM: {
                "dtype": "video",
                "shape": (H, W, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True},
            },
        },
        use_videos=True,
    )
    for i in range(FRAMES):
        ds.add_frame(
            {
                "observation.state": np.array([float(i)], np.float32),
                "action": np.array([0.0], np.float32),
                "task": "depth",
                CAM: _depth_frame(i),
            }
        )
    ds.save_episode()
    ds.finalize()
    return LeRobotDataset("tests/depth_playback", root=root)


def _first_depth_frame(body: bytes) -> np.ndarray:
    _header, parts = parse_chunk(body)
    return decode_h264(parts[("video", CAM)][1])[0]


def _turbo(code: int) -> np.ndarray:
    """The palette's colour for a nearness code, as the renderer produces it."""
    import cv2

    return cv2.applyColorMap(np.array([[code]], np.uint8), cv2.COLORMAP_TURBO)[0, 0, ::-1]


def _near(px: np.ndarray, ref: np.ndarray, tol: int = 60) -> bool:
    """Same colour to within what a lossy codec does to it."""
    return int(np.abs(px.astype(int) - ref.astype(int)).max()) <= tol


class TestChunk:
    def test_the_chunk_is_a_picture_not_the_raw_codes(self, dataset):
        body, header = build_chunk(dataset, 0, 0, "low")
        _header, parts = parse_chunk(body)
        part, data = parts[("video", CAM)]
        frames = decode_h264(data)
        assert header["frames"] == CHUNK_FRAMES and len(frames) == CHUNK_FRAMES
        assert part["codec"] == "h264" and len(part["frame_sizes"]) == CHUNK_FRAMES
        f = frames[0]
        assert f.shape[:2] == (H, W)  # narrower than the profile target: left at its size
        assert f.mean() > 60  # the raw codes averaged 22 of 255 on the real recording
        assert np.all(f[:5, :5] < 60)  # the hole stays black
        # Under the episode's own range the ramp's ends sit at the palette's ends.
        assert _near(f[H // 2, 4], _turbo(255)) and _near(f[H // 2, W - 4], _turbo(0))

    def test_a_fixed_mapping_changes_the_picture_and_its_cache_key(self, dataset):
        fixed = {"*": DepthMapping("fixed", 0.9, 1.0)}
        auto = _first_depth_frame(build_chunk(dataset, 0, 0, "low")[0])
        narrow = _first_depth_frame(build_chunk(dataset, 0, 0, "low", fixed)[0])
        # Everything nearer than 0.9 m saturates at the warm end under the narrow range.
        assert _near(narrow[H // 2, W // 2], _turbo(255))
        assert not _near(auto[H // 2, W // 2], _turbo(255))
        assert _cache_key("d", 0, 0, "low", "__d" + fixed["*"].variant) != _cache_key("d", 0, 0, "low")


@pytest.fixture
def app_with_state():
    app = FastAPI()
    app.include_router(datasets_module.router)
    state = AppState(frame_cache=FrameCache(max_bytes=50_000_000))
    original_state = datasets_module._app_state
    original_indices = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)
    # What the server does at startup: an in-process server in an earlier
    # module shuts the decode and prefetch pools down on exit, and a frame
    # request then dies with "cannot schedule new futures after shutdown".
    datasets_module.ensure_executors()
    yield app, state
    datasets_module._app_state = original_state
    datasets_module._episode_start_indices.clear()
    datasets_module._episode_start_indices.update(original_indices)


def _get_frame(app, dataset_id: str, frame: int, query: str = ""):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get(
                f"/api/datasets/{dataset_id}/episodes/0/frame/{frame}?camera={CAM}{query}"
            )

    return asyncio.run(run())


def _jpeg_array(body: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(body)).convert("RGB"))


class TestTile:
    def test_mappings_render_apart_and_bad_specs_are_refused(self, app_with_state, dataset):
        app, state = app_with_state
        dataset_id = str(dataset.root)
        state.datasets[dataset_id] = dataset

        auto = _get_frame(app, dataset_id, 3)
        assert auto.status_code == 200 and auto.headers["content-type"] == "image/jpeg"
        narrow = _get_frame(app, dataset_id, 3, "&depth=fixed:0.9:1.0")
        assert narrow.status_code == 200
        a, n = _jpeg_array(auto.content), _jpeg_array(narrow.content)
        assert a.mean() > 60
        assert _near(n[H // 2, W // 2], _turbo(255), tol=40)
        assert not _near(a[H // 2, W // 2], _turbo(255), tol=40)
        # The narrow mapping is a variant of the tile: asking for auto again
        # still serves the auto pixels, not the cached narrow ones.
        assert _get_frame(app, dataset_id, 3).content == auto.content

        assert _get_frame(app, dataset_id, 3, "&depth=fixed:1:0.5").status_code == 400
        assert _get_frame(app, dataset_id, 3, "&depth=bogus").status_code == 400


def _get(app, url: str):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get(url)

    return asyncio.run(run())


class TestPointCloudFeeds:
    def test_depth_bytes_are_the_stored_millimetres_with_holes_zeroed(self, app_with_state, dataset):
        app, state = app_with_state
        dataset_id = str(dataset.root)
        state.datasets[dataset_id] = dataset

        r = _get(app, f"/api/datasets/{dataset_id}/episodes/0/frame/3/depth?camera={CAM}")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/octet-stream"
        assert (r.headers["x-depth-height"], r.headers["x-depth-width"]) == (str(H), str(W))
        mm = np.frombuffer(r.content, dtype="<u2").reshape(H, W)  # httpx already inflated the gzip
        assert np.all(mm[:8, :8] == 0)  # no measurement stays zero
        ramp = _depth_frame(3)[..., 0].astype(int)
        assert np.abs(mm[H // 2, 8:].astype(int) - ramp[H // 2, 8:]).max() <= 1  # within the quantizer

        assert (
            _get(app, f"/api/datasets/{dataset_id}/episodes/0/frame/3/depth?camera=nope").status_code == 400
        )
        assert (
            _get(app, f"/api/datasets/{dataset_id}/episodes/0/frame/{FRAMES}/depth?camera={CAM}").status_code
            == 404
        )
        # The bytes are cached beside the JPEGs, so the next request decodes nothing.
        assert state.frame_cache.contains(dataset_id, 0, 3, CAM, datasets_module.DEPTH_BYTES_VARIANT)
        again = _get(app, f"/api/datasets/{dataset_id}/episodes/0/frame/3/depth?camera={CAM}")
        assert again.content == r.content

    def test_cameras_report_what_the_recording_carries(self, app_with_state, dataset, tmp_path, monkeypatch):
        app, state = app_with_state
        dataset_id = str(dataset.root)
        state.datasets[dataset_id] = dataset
        monkeypatch.setenv("LEROBOT_GUI_CONFIG_DIR", str(tmp_path))

        cams = _get(app, f"/api/datasets/{dataset_id}/episodes/0/urdf-viz/cameras").json()["cameras"]
        assert [c["depth_key"] for c in cams] == [CAM]
        assert cams[0]["intrinsics"] is None and cams[0]["extrinsics"] is None
        assert (cams[0]["height"], cams[0]["width"]) == (H, W) and cams[0]["invalid_below_mm"] == 10.0

        dataset.meta.features[CAM]["info"]["camera"] = {
            "source": "device", "intrinsics": {"width": W, "height": H, "fx": 100.0, "fy": 100.0, "cx": 64.0, "cy": 48.0},
        }  # fmt: skip
        (tmp_path / "click_target_extrinsics.json").write_text(
            '{"cam_key": "d", "T_base_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0.5],[0,0,0,1]], "rmse_m": 0.01}'
        )
        cams = _get(app, f"/api/datasets/{dataset_id}/episodes/0/urdf-viz/cameras").json()["cameras"]
        assert cams[0]["intrinsics"]["fx"] == 100.0
        assert cams[0]["extrinsics"]["source"] == "click_target_file"
        assert cams[0]["extrinsics"]["T_parent_camera"][2][3] == 0.5


class TestAutoRange:
    def test_the_episode_range_is_the_recorded_percentiles_in_metres(self, app_with_state, dataset):
        """What the Auto mapping colours with, so the tile can show the numbers:
        the episode's own 1st and 99th percentile, from the statistics the
        writer recorded, in metres; a colour camera is not a depth camera."""
        app, state = app_with_state
        dataset_id = str(dataset.root)
        state.datasets[dataset_id] = dataset

        r = _get(app, f"/api/datasets/{dataset_id}/episodes/0/depth-range?camera={CAM}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["camera"] == CAM
        lo, hi = body["lo_m"], body["hi_m"]
        assert 0 < lo < hi
        frames = np.stack([_depth_frame(i) for i in range(FRAMES)]).astype(np.float64) / 1000.0
        valid = frames[frames > 0]
        assert valid.min() <= lo <= np.percentile(valid, 5) + 0.05
        assert np.percentile(valid, 95) - 0.05 <= hi <= valid.max() + 0.001

        not_depth = "observation.images.colour"  # a key that is not a depth camera of this dataset
        assert (
            _get(app, f"/api/datasets/{dataset_id}/episodes/0/depth-range?camera={not_depth}").status_code
            == 404
        )
        assert (
            _get(app, f"/api/datasets/{dataset_id}/episodes/99/depth-range?camera={CAM}").status_code == 404
        )
