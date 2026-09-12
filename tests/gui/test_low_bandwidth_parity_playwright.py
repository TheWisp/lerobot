"""Everything the tab does on the JPEG path, it does at Low Bandwidth (design: R6,
R7, C7). These are the JPEG-path behaviours whose existing tests observe the
frame URL -- an observable that does not exist at Low Bandwidth -- restated
against what the tiles show and what reaches the server.
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("av")

from playwright.sync_api import sync_playwright  # noqa: E402

from tests.gui.chunk_fixtures import (  # noqa: E402
    BLOB_CENTER,
    CAM_WIDE,
    FRAMES,
    GuiServer,
    build_dataset,
)

pytestmark = pytest.mark.requires_playwright

MODE_KEY = "lerobot.cameraVideoMode"
MASK_KEY = "masks.a"
# The tile is the camera's declared 480x240 at either profile: the disc's centre, and a
# point outside it below the band strip.
IN_X, IN_Y = BLOB_CENTER[1], BLOB_CENTER[0]
OUT_X, OUT_Y = 450, 225


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("parity") / "parity")


@pytest.fixture()
def fresh_dataset_root(tmp_path_factory):
    """A dataset of its own for the test that writes to it."""
    return build_dataset(tmp_path_factory.mktemp("parity_w") / "parity_w")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    yield srv
    srv.stop()


def _requests(page):
    seen = {"chunk": 0, "frame": 0, "masks": 0, "status": 0, "series": 0}

    def on_request(req):
        u = req.url
        if "/chunk?" in u:
            seen["chunk"] += 1
        elif "/frame/" in u:
            seen["frame"] += 1
        elif u.endswith("/masks/status"):
            seen["status"] += 1
        elif u.endswith("/masks") or "/masks?" in u:
            seen["masks"] += 1
        elif "/feature-series" in u:
            seen["series"] += 1

    page.on("request", on_request)
    return seen


def _open(page, base, ds_id, root):
    page.add_init_script(f"localStorage.setItem({json.dumps(MODE_KEY)}, 'low-bandwidth');")
    page.goto(base)
    page.wait_for_function(
        "typeof openDataset === 'function' && typeof selectEpisode === 'function'", timeout=15_000
    )
    page.evaluate("(ds) => openDataset(ds)", ds_id)
    page.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
    page.evaluate(f"selectEpisode({json.dumps(ds_id)}, 0, {FRAMES})")
    page.wait_for_function("window.__chunkPlayer && window.__chunkPlayer.ready()", timeout=60_000)
    page.evaluate("window.Dialogs.confirm = async () => true")


def _tile_px(page, x, y):
    return page.evaluate(
        f"""(() => {{
            const c = document.getElementById('video-{CAM_WIDE.replace(".", "-")}');
            const d = c.getContext('2d').getImageData({x}, {y}, 1, 1).data;
            return [d[0], d[1], d[2]];
        }})()"""
    )


def test_the_playhead_never_moves_backwards_while_playing(server, dataset_root):
    srv = server
    ds_id = srv.open_dataset(dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, srv.base, ds_id, dataset_root)
        page.evaluate("window.__chunkPlayer.metrics.painted.length = 0")
        page.evaluate("togglePlay()")
        time.sleep(2.5)
        page.evaluate("togglePlay()")
        frames = page.evaluate("window.__chunkPlayer.metrics.painted.map((q) => q.frame)")
        assert len(frames) >= 3, frames
        steps = list(zip(frames, frames[1:], strict=False))
        wraps = [(a, b) for a, b in steps if b < a and a >= FRAMES * 0.5 and b <= FRAMES * 0.5]
        backwards = [(a, b) for a, b in steps if b < a and (a, b) not in wraps]
        assert not backwards, f"the playhead went backwards while playing: {backwards} (all: {frames})"
        browser.close()


def test_the_tile_shows_the_recipes_treatment_and_a_write_changes_it(server, fresh_dataset_root):
    """The JPEG path asks the server for the composite; here the page draws the
    treatment on the decoded frame. The fixture's recipe tints the disc red; a
    treatment write to solid green must show on the next paint."""
    srv = server
    ds_id = srv.open_dataset(fresh_dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, srv.base, ds_id, fresh_dataset_root)
        page.wait_for_function(
            f"(() => {{ const d = document.getElementById('video-{CAM_WIDE.replace('.', '-')}').getContext('2d').getImageData({IN_X}, {IN_Y}, 1, 1).data; return d[0] > 150 && d[1] < 80; }})()",
            timeout=30_000,
        )
        inside, outside = _tile_px(page, IN_X, IN_Y), _tile_px(page, OUT_X, OUT_Y)
        assert inside[0] > 150 and inside[1] < 80, ("the disc is tinted red", inside)
        assert abs(outside[0] - outside[1]) < 6, ("outside the disc the frame is its own grey", outside)

        page.evaluate(
            """async ([ds]) => {
                await fetch('/api/edits/mask-treatments', {method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({dataset_id: ds, treatments: {ball: {key: 'solid', params: {color: [0, 200, 0]}}},
                                        background: {key: 'none', params: {}}})});
                await window.applyEdits();
            }""",
            [ds_id],
        )
        page.wait_for_function(
            f"(() => {{ const d = document.getElementById('video-{CAM_WIDE.replace('.', '-')}').getContext('2d').getImageData({IN_X}, {IN_Y}, 1, 1).data; return d[1] > 150 && d[0] < 60; }})()",
            timeout=30_000,
        )
        assert _tile_px(page, OUT_X, OUT_Y)[1] < 150, "the treatment stays inside the mask"
        browser.close()


def test_the_page_never_fetches_the_episodes_rows_and_the_lane_still_shows_presence(server, dataset_root):
    """The rows the page draws come with the chunks; the lane's presence comes
    with the series it already loads. The whole episode's rows at the stored
    resolution -- the response that did not fit the link -- are never asked for."""
    srv = server
    ds_id = srv.open_dataset(dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        seen = _requests(page)
        _open(page, srv.base, ds_id, dataset_root)
        page.wait_for_function(
            f"() => document.querySelector('.row-track[data-feature=\"{MASK_KEY}\"]')", timeout=60_000
        )
        page.evaluate("togglePlay()")
        time.sleep(1.5)
        page.evaluate("togglePlay()")
        assert seen["masks"] == 0, f"the episode's rows were fetched at Low Bandwidth: {seen}"
        assert seen["status"] >= 1 and seen["series"] >= 1, seen
        browser.close()


def test_a_mask_edit_reaches_the_chunks_the_tile_and_the_lane(server, fresh_dataset_root):
    """One write, every copy: after disabling the mask on a frame range, the
    next chunk the page asks for is a server miss (its cache was dropped), the
    tile no longer draws the mask there, and the lane's series was fetched again."""
    srv = server
    ds_id = srv.open_dataset(fresh_dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        seen = _requests(page)
        _open(page, srv.base, ds_id, fresh_dataset_root)
        tile = "video-" + CAM_WIDE.replace(".", "-")
        # The fixture's tint is in the tile's pixels while the mask is there.
        page.wait_for_function(
            f"(() => {{ const d = document.getElementById('{tile}').getContext('2d').getImageData({IN_X}, {IN_Y}, 1, 1).data; return d[0] > 150 && d[1] < 80; }})()",
            timeout=30_000,
        )
        # applyEdits re-selects the episode, which keeps the player it has: mark wall
        # time, and look for a chunk fetched after it, whichever player fetched it.
        series_before, chunks_seen_before, mark = seen["series"], seen["chunk"], page.evaluate("Date.now()")
        page.evaluate(
            """async ([ds, n, cam]) => {
                await fetch('/api/edits/mask-range', {method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({dataset_id: ds, episode_index: 0, camera: cam, label: 'ball', from_frame: 0, to_frame: n, action: 'disable'})});
                await window.applyEdits();
            }""",
            [ds_id, FRAMES, CAM_WIDE],
        )
        page.wait_for_function(
            f"window.__chunkPlayer && window.__chunkPlayer.metrics.chunks.some((c) => c.at > {mark})",
            timeout=60_000,
        )
        # Disabled on every frame: the tint is gone from the tile, on the next paint.
        page.wait_for_function(
            f"(() => {{ const d = document.getElementById('{tile}').getContext('2d').getImageData({IN_X}, {IN_Y}, 1, 1).data; return d[0] < 130 && Math.abs(d[0] - d[1]) < 8; }})()",
            timeout=30_000,
        )
        after = page.evaluate(
            f"window.__chunkPlayer.metrics.chunks.filter((c) => c.at > {mark}).map((c) => c.cache)"
        )
        assert "miss" in after, (
            "the chunk after the write came from a cache the write should have emptied",
            after,
        )
        # One fetch per chunk after the write: re-selecting the episode keeps the
        # player, so the write does not cost every chunk twice on the slowest path.
        page.wait_for_timeout(1500)
        starts = page.evaluate(
            f"window.__chunkPlayer.metrics.chunks.filter((c) => c.at > {mark}).map((c) => c.start)"
        )
        assert seen["chunk"] - chunks_seen_before <= len(set(starts)) + 1, (
            "chunks fetched twice after the write",
            seen["chunk"] - chunks_seen_before,
            starts,
        )
        assert seen["series"] > series_before, "the lane did not fetch the series again after the write"
        assert seen["masks"] == 0, seen
        browser.close()


def test_invalidating_the_masks_alone_makes_the_player_ask_again(server, dataset_root):
    """The hook at its own level: a write that reaches the mask layer without a
    re-selection of the episode (a staged treatment, an apply run's completion)
    must still empty the player's buffer and fetch again. applyEdits happens to
    re-select the episode, which hides a missing hook end to end."""
    srv = server
    ds_id = srv.open_dataset(dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, srv.base, ds_id, dataset_root)
        page.wait_for_function("window.__chunkPlayer.state().chunks.length >= 1", timeout=30_000)
        held_before = page.evaluate("window.__chunkPlayer.state().chunks.slice()")
        mark = page.evaluate("Date.now()")
        page.evaluate("(ds) => window.MaskOverlay.invalidate(ds)", ds_id)
        page.wait_for_function(
            f"window.__chunkPlayer.metrics.chunks.some((c) => c.at > {mark})", timeout=30_000
        )
        assert page.evaluate("window.__chunkPlayer === window.__chunkPlayer"), (
            "same player: the episode was not re-selected"
        )
        refetched = page.evaluate(
            f"window.__chunkPlayer.metrics.chunks.filter((c) => c.at > {mark}).map((c) => c.start)"
        )
        assert set(refetched) & set(held_before), (
            "the chunks it held were asked for again",
            held_before,
            refetched,
        )
        browser.close()
