# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The live overlay owns the tiles at Low Bandwidth the way it does on the JPEG
path (design: R6), at production sizes.

Reported from the rig with `sam3_track` on a 2 x 1280x720 dataset at Low
Bandwidth: the stored masks' outlines and names stacked on the overlay's own,
and the stream's picture sat smaller than the base tile, which stayed visible
under it. The JPEG path is the oracle for every assertion here -- each runs in
both modes -- and nothing is a toy: two cameras of different resolution and
aspect, thirty frames a second, stored masks with a recipe, the real stream
endpoint, ffmpeg, fragmented MP4 and MediaSource. Only the SAM3 worker is
faked, at the seams the endpoint reaches it through.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("av")

import uvicorn  # noqa: E402
from playwright.sync_api import TimeoutError as PWTimeout, sync_playwright  # noqa: E402

from lerobot.datasets.mask_codec import encode_mask  # noqa: E402
from tests.gui.chunk_fixtures import (  # noqa: E402
    LABELS,
    TINT_RECIPE,
    frame_image,
    redirect_gui_config,
    wait_for_player,
)

pytestmark = pytest.mark.requires_playwright

CAM_TOP = "observation.images.top"  # 16:9, the rig's head cameras
CAM_WRIST = "observation.images.left_wrist"  # 16:10, the rig's wrist cameras
SIZES = {CAM_TOP: (720, 1280), CAM_WRIST: (600, 960)}
FPS = 30
FRAMES = 120
MODE_KEY = "lerobot.cameraVideoMode"
VIEWPORT = {
    "width": 2560,
    "height": 1400,
}  # the rig's screen: tiles wider than the stream's 360-line atlas rects
DISC = (360, 640, 90)  # a mask the recipe treats and the layer outlines


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeWorker:
    """What the endpoint needs of the SAM3 subprocess: its cameras, a sequence
    that moves when a frame is published, and no overlay -- the dataset's own
    pixels fill the atlas, which is all the geometry here needs."""

    def __init__(self, cameras):
        self.cameras = list(cameras)
        self._seq = dict.fromkeys(self.cameras, 0)

    def overlay_seq(self, cam):
        return self._seq.get(cam, 0)

    def read_overlay(self, cam):
        return None

    def published(self):
        for cam in self._seq:
            self._seq[cam] += 1


def _disc(h, w):
    cy, cx, r = DISC
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt, write_episode

    root = tmp_path_factory.mktemp("stream") / "prod"
    feats = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    for cam, (h, w) in SIZES.items():
        feats[cam] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(
        repo_id="tests/prodstream", fps=FPS, root=root, features=feats, use_videos=True
    )
    for i in range(FRAMES):
        ds.add_frame(
            {
                "observation.state": np.zeros(2, np.float32),
                "action": np.zeros(2, np.float32),
                "task": "stream",
                **{cam: frame_image(0, i, h, w) for cam, (h, w) in SIZES.items()},
            }
        )
    ds.save_episode()
    ds.finalize()
    ds = LeRobotDataset("tests/prodstream", root=root)
    h, w = SIZES[CAM_TOP]
    adopt(ds, [CAM_TOP], LABELS, (h, w), treatments={"ball": TINT_RECIPE})
    write_episode(ds, 0, CAM_TOP, [{"ball": _disc(h, w)} for _ in range(FRAMES)])
    return root


@pytest.fixture(scope="module")
def server(dataset_root, tmp_path_factory):
    from lerobot.gui import server as gui_server_mod
    from lerobot.gui.api import overlays as ovl

    mp = pytest.MonkeyPatch()
    redirect_gui_config(mp, tmp_path_factory.mktemp("config"))
    mp.setenv("LEROBOT_CHUNK_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    worker = _FakeWorker(list(SIZES))
    mp.setattr(ovl, "_get_live_reader", lambda: worker)
    mp.setattr(ovl, "_data_publisher_active", lambda: True)
    mp.setattr(ovl, "publish_data_frame", lambda *a, **k: worker.published())
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    import requests

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if requests.get(base, timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        srv.should_exit = True
        pytest.fail("GUI server did not come up")
    yield base
    srv.should_exit = True
    thread.join(timeout=10)
    mp.undo()


def _open(browser, base, ds_id, mode):
    pg = browser.new_page(viewport=VIEWPORT)
    pg.warnings = []
    pg.on("console", lambda m: pg.warnings.append(m.text[:200]) if m.type in ("warning", "error") else None)
    pg.add_init_script(f"localStorage.setItem({json.dumps(MODE_KEY)}, {json.dumps(mode)});")
    pg.goto(base)
    pg.wait_for_function("typeof openDataset === 'function' && window.OverlayStream", timeout=15_000)
    pg.evaluate("(ds) => openDataset(ds)", ds_id)
    pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
    pg.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [ds_id, FRAMES])
    if mode == "low-bandwidth":
        wait_for_player(pg, "window.__chunkPlayer && window.__chunkPlayer.ready()")
        pg.wait_for_function(f"document.getElementById('{_id('video', CAM_TOP)}').width > 0", timeout=30_000)
    else:
        pg.wait_for_function(
            f"document.getElementById('{_id('frame', CAM_TOP)}').naturalWidth > 0", timeout=60_000
        )
    return pg


def _id(kind, cam):
    return f"{kind}-{cam.replace('.', '-')}"


def _base_id(mode, cam):
    return _id("video" if mode == "low-bandwidth" else "frame", cam)


def _rect(pg, element_id):
    return pg.evaluate(
        f"(() => {{ const r = document.getElementById('{element_id}').getBoundingClientRect(); return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; }})()"
    )


def _stream_rect(pg, cam):
    return pg.evaluate(
        f"(() => {{ const c = document.getElementById('{_id('frame', cam)}').parentElement.querySelector('canvas.stream-layer'); if (!c) return null; const r = c.getBoundingClientRect(); return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; }})()"
    )


def _mask_shown(pg, cam):
    return pg.evaluate(
        f"(() => {{ const c = document.getElementById('{_id('mask', cam)}'); return !!c && getComputedStyle(c).display !== 'none' && c.width > 0; }})()"
    )


def _visibility(pg, element_id):
    return pg.evaluate(f"getComputedStyle(document.getElementById('{element_id}')).visibility")


def _worker_active(pg, on):
    """What the badge renders while the SAM3 worker is up: the layer's and the transport's signal."""
    cls, text = ("overlays-badge ok", "12 fps") if on else ("overlays-badge off", "off")
    pg.evaluate(
        f"() => {{ const b = document.getElementById('overlays-badge'); b.className = {json.dumps(cls)}; b.textContent = {json.dumps(text)}; }}"
    )


def _play_stream(pg):
    """Press Play with the worker active: the transport hands the tiles to the stream."""
    _worker_active(pg, True)
    pg.evaluate("() => togglePlay()")
    try:
        pg.wait_for_function("() => window.OverlayStream._debug().streaming", timeout=15_000)
        pg.wait_for_function(
            "() => { const d = window.OverlayStream._debug(); return d.started && d.ct > 0.2; }",
            timeout=45_000,
        )
    except PWTimeout:
        raise AssertionError(
            f"the stream did not show a picture: {pg.evaluate('window.OverlayStream._debug()')}; console: {pg.warnings[-6:]}"
        ) from None


def _wait_chrome(pg, cam, shown):
    pg.wait_for_function(
        f"(() => {{ const c = document.getElementById('{_id('mask', cam)}'); return (!!c && getComputedStyle(c).display !== 'none' && c.width > 0) === {json.dumps(shown)}; }})()",
        timeout=30_000,
    )


@pytest.mark.parametrize("mode", ["full-quality", "low-bandwidth"])
def test_the_live_layer_owns_the_tiles_and_the_stored_chrome_stays_off(server, dataset_root, mode):
    """Stored masks draw only while the live layer is off: with the worker
    active (the badge) or the stream playing, the mask canvases stay hidden --
    also while the chunk player keeps painting, which is where they stacked."""
    ds_id = str(dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = _open(browser, server, ds_id, mode)
        _wait_chrome(pg, CAM_TOP, True)  # the complement: the chrome exists when the live layer is off
        # The one draw entry both picture paths use, called directly: under
        # the live layer it paints nothing whoever calls it, and paints again
        # once the layer is off.
        h, w = SIZES[CAM_TOP]
        rows = [[0, encode_mask(_disc(h, w))]]
        draw = (
            f"(rows) => window.MaskOverlay.drawCamera({json.dumps(CAM_TOP)}, rows, [{h}, {w}], "
            f"window.MaskOverlay.chromeOptions({{ hasAny: true, labels: ['ball'] }}))"
        )
        _worker_active(pg, True)
        assert pg.evaluate(draw, rows) == 0 and not _mask_shown(pg, CAM_TOP), (
            "drew stored chrome under the live layer"
        )
        _worker_active(pg, False)
        assert pg.evaluate(draw, rows) > 0 and _mask_shown(pg, CAM_TOP), (
            "the entry draws when the layer is off"
        )
        _worker_active(pg, True)
        pg.evaluate("() => { loadAllFrames(3); }")
        if mode == "low-bandwidth":
            wait_for_player(pg, "window.__chunkPlayer.metrics.painted.some((q) => q.frame === 3)")
            # The repaint paths that reach the layer without a playhead change.
            pg.evaluate("() => { window.__chunkPlayer.repaintMasks(); }")
            pg.wait_for_timeout(300)
        _wait_chrome(pg, CAM_TOP, False)
        assert not pg.evaluate("window.MaskOverlay.isDrawing()")
        _worker_active(pg, False)
        pg.evaluate("() => { loadAllFrames(4); }")
        _wait_chrome(pg, CAM_TOP, True)  # and back, so the hide is the arbitration and not a one-way latch
        # Playing already; the stream takes the transport from the running player.
        pg.evaluate("() => togglePlay()")
        if mode == "low-bandwidth":
            wait_for_player(pg, "window.__chunkPlayer.metrics.painted.length > 5")
        _play_stream(pg)
        if mode == "low-bandwidth":
            pg.evaluate(
                "() => { window.__chunkPlayer.repaintMasks(); window.MaskOverlay.invalidate(window.currentDataset); }"
            )
        seen = []
        for _ in range(6):
            seen.append((_mask_shown(pg, CAM_TOP), pg.evaluate("window.MaskOverlay.isDrawing()"), None))
            pg.wait_for_timeout(250)
        assert all(not shown and not drawing for shown, drawing, _s in seen), (
            "stored chrome under the live layer",
            seen,
        )
        stacked = [w for w in pg.warnings if "stored masks painted" in w]
        assert not stacked, stacked
        pg.evaluate("() => window.OverlayStream.stop({resume: true})")
        pg.wait_for_function("() => !window.OverlayStream._debug().streaming", timeout=15_000)
        # The still overlay's last PNG is on the tile (as after a pull) when the
        # stream takes over from a paused transport: the still path does not
        # touch it while the stream plays, so only the stream can hide it.
        still = _id("overlay", CAM_TOP)
        # Put it up and read it back without yielding: the page hides this
        # overlay itself, which is the very thing asserted below, so a separate
        # read can find it already gone and fail the setup because the product
        # was quick rather than because it was wrong.
        up = pg.evaluate(
            f"() => {{ const i = document.getElementById('{still}'); i.src = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='; i.style.display = 'block'; return getComputedStyle(i).display; }}"
        )
        assert up == "block", f"the still overlay could not be put up to begin with: {up}"
        _play_stream(pg)
        shown = [
            pg.evaluate(f"getComputedStyle(document.getElementById('{still}')).display") for _ in range(3)
        ]
        assert all(s == "none" for s in shown), ("the still overlay's PNG under the stream", shown)
        pg.evaluate("() => window.OverlayStream.stop({resume: true})")
        browser.close()


@pytest.mark.parametrize("mode", ["full-quality", "low-bandwidth"])
def test_the_stream_picture_occupies_the_base_picture_rectangle(server, dataset_root, mode):
    """Each camera's stream canvas takes exactly the rectangle its base picture
    takes -- the JPEG <img> or the video canvas -- at both aspects, in a
    viewport whose tiles are wider than the stream's own atlas rects."""
    ds_id = str(dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = _open(browser, server, ds_id, mode)
        base = {cam: _rect(pg, _base_id(mode, cam)) for cam in SIZES}
        assert all(r[2] > 640 for r in base.values()), (
            "tiles must exceed the atlas rects for this to mean anything",
            base,
        )
        _play_stream(pg)

        def check(label):
            for cam in SIZES:
                b = _rect(pg, _base_id(mode, cam))
                stream = _stream_rect(pg, cam)
                assert stream is not None, (mode, cam, label, "no stream canvas on the tile")
                assert all(abs(x - y) <= 1 for x, y in zip(stream, b, strict=True)), (
                    mode,
                    cam,
                    label,
                    "base",
                    b,
                    "stream",
                    stream,
                )

        check("at the rig's viewport")
        # The tiles resize under the stream (a narrower window): the canvas follows the base.
        pg.set_viewport_size({"width": 1800, "height": 1000})
        # Wait for the layout to follow rather than for a spell long enough to
        # assume it did. Both layers have to have moved: waiting only on the
        # base returns while the stream canvas is still at its old size, and
        # `check` then compares a settled rectangle against an unsettled one.
        # That the two then coincide is the assertion, and stays one.
        pg.wait_for_function(
            """(w) => { const b = document.getElementById(w.id);
               const c = b && b.parentElement.querySelector('canvas.stream-layer');
               return b && c && b.getBoundingClientRect().width < w.was
                      && c.getBoundingClientRect().width < w.was; }""",
            arg={"id": _base_id(mode, CAM_TOP), "was": base[CAM_TOP][2]},
            timeout=30_000,
        )
        assert _rect(pg, _base_id(mode, CAM_TOP))[2] < base[CAM_TOP][2], "the resize did not move the tiles"
        check("after a resize")
        pg.evaluate("() => window.OverlayStream.stop({resume: true})")
        browser.close()


@pytest.mark.parametrize("mode", ["full-quality", "low-bandwidth"])
def test_the_base_picture_is_hidden_while_the_stream_plays(server, dataset_root, mode):
    """The stream recomputes the whole picture: the base under it is not shown
    while it plays and is back the moment it stops."""
    ds_id = str(dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = _open(browser, server, ds_id, mode)
        before = {cam: _visibility(pg, _base_id(mode, cam)) for cam in SIZES}
        assert all(v == "visible" for v in before.values()), before
        _play_stream(pg)
        during = {cam: _visibility(pg, _base_id(mode, cam)) for cam in SIZES}
        pg.evaluate("() => window.OverlayStream.stop({resume: true})")
        pg.wait_for_function("() => !window.OverlayStream._debug().streaming", timeout=15_000)
        after = {cam: _visibility(pg, _base_id(mode, cam)) for cam in SIZES}
        browser.close()
    assert all(v == "hidden" for v in during.values()), (mode, "base shown under the stream", during)
    assert all(v == "visible" for v in after.values()), (mode, "base not restored", after)


def test_stopping_the_stream_lands_the_player_on_the_frame_it_reached(server, dataset_root):
    """At Low Bandwidth the stream's transport hands the frame it reached back
    to the chunk player, which paints it; the player does not run alongside."""
    ds_id = str(dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = _open(browser, server, ds_id, "low-bandwidth")
        pg.evaluate("() => togglePlay()")
        wait_for_player(pg, "window.__chunkPlayer.metrics.painted.length > 5")
        _play_stream(pg)
        n0 = pg.evaluate("window.__chunkPlayer.metrics.painted.length")
        pg.wait_for_timeout(800)
        n1 = pg.evaluate("window.__chunkPlayer.metrics.painted.length")
        assert n1 == n0, ("the player kept painting under the stream", n0, n1)
        reached = pg.evaluate("window.currentFrame")
        pg.evaluate("() => window.OverlayStream.stop({resume: true})")
        wait_for_player(pg, f"() => window.__chunkPlayer.metrics.painted.some((q) => q.frame === {reached})")
        assert reached > 0, "the stream never advanced the playhead"
        assert pg.evaluate("window.currentFrame") == reached
        browser.close()
