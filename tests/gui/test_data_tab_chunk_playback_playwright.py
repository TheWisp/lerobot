"""The Data tab at Low Bandwidth: video through chunks, driven as the operator
drives it (design: R1 behaviours, R4-R6, R9, C1).

Frames are flat greys unique to (episode, frame), so a canvas identifies the
frame it shows. Nothing is inferred from a control's state: every assertion
reads the painted tiles, or counts what reached the server.
"""

from __future__ import annotations

import json
import struct
import time

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("av")

from playwright.sync_api import sync_playwright  # noqa: E402

from tests.gui.chunk_fixtures import (  # noqa: E402
    BANDS,
    BLOB_CENTER,
    BLOB_RADIUS,
    CAM_NARROW,
    CAM_WIDE,
    CAMS,
    FRAMES,
    GuiServer,
    build_dataset,
    read_ids,
    wait_for_player,
)

pytestmark = pytest.mark.requires_playwright

MODE_KEY = "lerobot.cameraVideoMode"


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("tab") / "tab")


@pytest.fixture(scope="module")
def server(dataset_root, tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    ds_id = srv.open_dataset(dataset_root)
    yield srv, ds_id
    srv.stop()


def _tile_id(cam: str) -> str:
    return "video-" + cam.replace(".", "-")


def _ids(page) -> dict[str, tuple[int, int] | None]:
    """(frame, episode) every video tile identifies itself as, read from the
    band strip along its top at the canvas's own resolution."""
    raw = page.evaluate(
        """([ids, bands]) => {
            const out = {};
            for (const [cam, id] of Object.entries(ids)) {
                const c = document.getElementById(id);
                if (!c || !c.width) { out[cam] = null; continue; }
                const ctx = c.getContext('2d');
                const w = c.width;
                const s = [];
                for (let k = 0; k < bands; k++) {
                    const cx = Math.floor((2 * k + 1) * w / (2 * bands));
                    const d = ctx.getImageData(cx - 1, 2, 3, 3).data;
                    let r = 0, n = 0;
                    for (let i = 0; i < d.length; i += 4) { r += d[i]; n++; }
                    s.push(r / n);
                }
                out[cam] = s;
            }
            return out;
        }""",
        [{cam: _tile_id(cam) for cam in CAMS}, BANDS],
    )
    return {cam: (None if v is None else read_ids(v)) for cam, v in raw.items()}


def _open(page, base, ds_id, mode="low-bandwidth", episode=0):
    page.add_init_script(f"localStorage.setItem({json.dumps(MODE_KEY)}, {json.dumps(mode)});")
    page.goto(base)
    page.wait_for_function(
        "typeof openDataset === 'function' && typeof selectEpisode === 'function'", timeout=15_000
    )
    # As the operator does: open the dataset in the tab, which is what fills window.datasets.
    page.evaluate("(ds) => openDataset(ds)", ds_id)
    page.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
    page.evaluate(f"selectEpisode({json.dumps(ds_id)}, {episode}, {FRAMES})")


def _requests(page):
    seen = {"chunk": 0, "frame": 0}

    def on_request(req):
        if "/chunk?" in req.url:
            seen["chunk"] += 1
        elif "/frame/" in req.url:
            seen["frame"] += 1

    page.on("request", on_request)
    return seen


def test_at_low_bandwidth_the_tiles_paint_from_chunks_and_never_ask_for_a_still(server):
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        seen = _requests(page)
        _open(page, srv.base, ds_id)
        wait_for_player(
            page,
            f"(() => {{ const c = document.getElementById('{_tile_id(CAM_WIDE)}'); return c && c.width > 0 && window.__chunkPlayer && window.__chunkPlayer.ready(); }})()",
        )
        ids = _ids(page)
        assert ids[CAM_WIDE] == ids[CAM_NARROW] == (0, 0), ids
        assert seen["chunk"] >= 1
        assert seen["frame"] == 0, "the JPEG endpoint was asked for a still at Low Bandwidth"
        # The <img> tiles the JPEG path paints are out of the way; the canvases are what shows.
        assert (
            page.evaluate(
                f"getComputedStyle(document.getElementById('frame-{CAM_WIDE.replace('.', '-')}')).display"
            )
            == "none"
        )
        browser.close()


def test_at_full_quality_the_tab_is_the_jpeg_path_and_never_asks_for_a_chunk(server):
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        seen = _requests(page)
        _open(page, srv.base, ds_id, mode="full-quality")
        page.wait_for_function(
            f"document.getElementById('frame-{CAM_WIDE.replace('.', '-')}').complete && document.getElementById('frame-{CAM_WIDE.replace('.', '-')}').naturalWidth > 0",
            timeout=60_000,
        )
        assert seen["frame"] >= len(CAMS)
        assert seen["chunk"] == 0, "a chunk was fetched at Full Quality"
        assert page.evaluate("window.__chunkPlayer") is None
        browser.close()


def test_a_step_paints_the_next_frame_on_every_camera_with_its_readout(server):
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, srv.base, ds_id)
        wait_for_player(page, "window.__chunkPlayer && window.__chunkPlayer.ready()")
        page.keyboard.press("ArrowRight")
        wait_for_player(page, "window.__chunkPlayer.frame() === 1 && window.currentFrame === 1")
        wait_for_player(page, "window.__chunkPlayer.metrics.painted.some((p) => p.frame === 1)")
        assert _ids(page) == {CAM_WIDE: (1, 0), CAM_NARROW: (1, 0)}
        assert page.text_content("#frame-info").strip().startswith("2 /")


def test_a_camera_missing_a_frame_holds_every_camera(server):
    """Withhold one camera's video from the chunk that holds frame 20: no tile
    may advance into it (R4). Then let it through: they all advance."""
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        block = {"on": True}

        def strip_narrow(route, request):
            resp = route.fetch()
            body = resp.body()
            if not block["on"] or "start=20" not in request.url:
                route.fulfill(response=resp, body=body)
                return
            hl = struct.unpack("<I", body[:4])[0]
            header = json.loads(body[4 : 4 + hl])
            header["parts"] = [
                pt for pt in header["parts"] if not (pt["kind"] == "video" and pt["camera"] == CAM_NARROW)
            ]
            hb = json.dumps(header).encode()
            route.fulfill(response=resp, body=struct.pack("<I", len(hb)) + hb + body[4 + hl :])

        page.route("**/chunk?*", strip_narrow)
        _open(page, srv.base, ds_id)
        wait_for_player(page, "window.__chunkPlayer && window.__chunkPlayer.ready()")
        page.evaluate("loadAllFrames(19)")
        wait_for_player(page, "window.__chunkPlayer.metrics.painted.some((p) => p.frame === 19)")
        page.evaluate("togglePlay()")
        time.sleep(2.0)
        ids = _ids(page)
        assert ids[CAM_WIDE] == ids[CAM_NARROW] == (19, 0), ("a tile advanced without the other", ids)
        assert page.evaluate("window.__chunkPlayer.state().held") == True  # noqa: E712
        block["on"] = False
        page.evaluate("window.__chunkPlayer.retry()")
        # A paint is what puts a frame on the tiles; the counter runs ahead of it
        # between animation frames, so wait for the paint, then pause and read.
        wait_for_player(page, "window.__chunkPlayer.metrics.painted.some((p) => p.frame >= 21)")
        page.evaluate("togglePlay()")
        last = page.evaluate("window.__chunkPlayer.metrics.painted.at(-1).frame")
        assert last >= 21, last
        ids = _ids(page)
        assert ids[CAM_WIDE] == ids[CAM_NARROW] == (last, 0), (
            "the tiles advanced together once the chunk arrived",
            ids,
            last,
        )
        browser.close()


def test_play_wraps_within_the_episode_and_within_a_trim(server):
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, srv.base, ds_id)
        wait_for_player(page, "window.__chunkPlayer && window.__chunkPlayer.ready()")
        page.select_option("#speed-select", "2")
        page.evaluate("togglePlay()")
        wait_for_player(page, "window.__chunkPlayer.metrics.wraps.length >= 1")
        wait_for_player(
            page,
            f"window.__chunkPlayer.metrics.painted.some((q) => q.frame === {FRAMES - 1}) && window.__chunkPlayer.metrics.painted.some((q) => q.frame === 0 && q.t > window.__chunkPlayer.metrics.wraps[0].t)",
        )
        assert page.evaluate("window.currentEpisode") == 0, "playback left the episode on its own"
        page.evaluate("togglePlay()")
        # A trim set while paused: play wraps inside it and never paints outside.
        page.evaluate("window.__setTrimForTest(10, 30)")
        page.evaluate("window.__chunkPlayer.metrics.painted.length = 0")
        page.evaluate("togglePlay()")
        wait_for_player(page, "window.__chunkPlayer.metrics.wraps.length >= 2")
        page.evaluate("togglePlay()")
        painted = page.evaluate("window.__chunkPlayer.metrics.painted.map((q) => q.frame)")
        assert painted and all(10 <= f < 30 for f in painted), painted
        browser.close()


def test_switching_episodes_paints_the_other_episode(server):
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, srv.base, ds_id)
        wait_for_player(page, "window.__chunkPlayer && window.__chunkPlayer.ready()")
        page.evaluate("navigateEpisode(1)")
        wait_for_player(
            page,
            "window.currentEpisode === 1 && window.__chunkPlayer.episode() === 1 && window.__chunkPlayer.metrics.painted.some((q) => q.episode === 1)",
        )
        assert _ids(page) == {CAM_WIDE: (0, 1), CAM_NARROW: (0, 1)}
        browser.close()


def test_saved_masks_are_drawn_from_the_chunk_and_a_label_toggle_costs_no_request(server):
    """The fixture's recipe tints the disc: the tint is in the tile's pixels and
    the mask layer draws the disc's outline over it, as the JPEG composite view
    does. Hiding the label removes the outline without a request."""
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, srv.base, ds_id)
        wait_for_player(page, "window.__chunkPlayer && window.__chunkPlayer.ready()")
        mask_id = "mask-" + CAM_WIDE.replace(".", "-")
        tile_id = _tile_id(CAM_WIDE)
        # The tile is the camera's declared 480x240, as on the JPEG path: the disc's
        # centre and a window across its right edge, in those pixels.
        cx, cy, r = BLOB_CENTER[1], BLOB_CENTER[0], BLOB_RADIUS
        edge = f"c.getContext('2d').getImageData({cx + r - 3}, {cy - 3}, 7, 7).data.filter((v, i) => i % 4 === 3 && v > 0).length"
        page.wait_for_function(
            f"(() => {{ const t = document.getElementById('{tile_id}'); const d = t.getContext('2d').getImageData({cx}, {cy}, 1, 1).data; return d[0] > 150 && d[1] < 80; }})()",
            timeout=30_000,
        )
        page.wait_for_function(
            f"(() => {{ const c = document.getElementById('{mask_id}'); return c && c.width > 0 && getComputedStyle(c).display !== 'none' && {edge} > 0; }})()",
            timeout=30_000,
        )
        centre_alpha = page.evaluate(
            f"document.getElementById('{mask_id}').getContext('2d').getImageData({cx}, {cy}, 1, 1).data[3]"
        )
        assert centre_alpha == 0, "over a treated recipe the mask is an outline, not a fill"
        seen = _requests(page)
        page.evaluate("window.MaskOverlay.setLabelHidden(0, true)")
        page.wait_for_function(
            f"(() => {{ const c = document.getElementById('{mask_id}'); return !c || getComputedStyle(c).display === 'none' || {edge} === 0; }})()",
            timeout=10_000,
        )
        assert seen["chunk"] == 0 and seen["frame"] == 0, "a label toggle is a repaint, not a request"
        browser.close()


def test_playback_makes_one_request_per_chunk_and_none_per_frame(server):
    """C1, enforced: over a few seconds of play the server sees one request per
    chunk the range needs, and nothing per frame."""
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        seen = _requests(page)
        _open(page, srv.base, ds_id)
        wait_for_player(page, "window.__chunkPlayer && window.__chunkPlayer.ready()")
        page.evaluate("togglePlay()")
        wait_for_player(page, "window.__chunkPlayer.metrics.wraps.length >= 1")
        page.evaluate("togglePlay()")
        chunks_needed = -(-FRAMES // 20)  # 3 for 45 frames of 20-frame chunks
        assert seen["frame"] == 0
        assert seen["chunk"] <= chunks_needed + 1, ("more requests than chunks", seen)
        browser.close()


def test_without_a_decoder_low_bandwidth_is_unavailable_with_a_reason(server):
    """127.0.0.1 is a secure context, so the missing capability is injected: no
    VideoDecoder, as on a plain-HTTP origin. The option is disabled and says why,
    and the tab is the JPEG path."""
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.add_init_script("delete window.VideoDecoder;")
        seen = _requests(page)
        _open(page, srv.base, ds_id)
        page.wait_for_function(
            f"document.getElementById('frame-{CAM_WIDE.replace('.', '-')}').naturalWidth > 0", timeout=60_000
        )
        opt = page.locator("#video-mode-select option[value='low-bandwidth']")
        assert opt.is_disabled()
        assert "secure" in (page.get_attribute("#video-mode-select", "title") or "").lower()
        assert page.evaluate("document.getElementById('video-mode-select').value") == "full-quality"
        assert seen["chunk"] == 0 and seen["frame"] >= len(CAMS)
        browser.close()
