# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Playback survives a burst of treatment saves at Low Bandwidth (design: R7, R10).

Reported from the rig: after several treatment changes saved in quick
succession while playing, the tiles stopped and only switching datasets and
back recovered them. The server log showed two saves 0.9 s apart, each
dropping the chunk cache twice (the save's own invalidation, then the
metadata-change reload the page's refresh triggers), the page re-requesting
one chunk, and then no request at all for sixteen seconds.

Production shape: the rig's three cameras at their resolutions, thirty frames
a second, a stored mask with a recipe, the real edits pipeline driven through
the inspector's own buttons, under an emulated link so rebuilt chunks arrive
slowly enough for the saves to overlap the fetches, as they did on the rig.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("av")

from playwright.sync_api import sync_playwright  # noqa: E402

from tests.gui.chunk_fixtures import LABELS, TINT_RECIPE, GuiServer, frame_image  # noqa: E402

pytestmark = pytest.mark.requires_playwright

CAM_TOP = "observation.images.top"
# Two fetches stand at once and a third chunk can still be decoding from an
# earlier arrival, since decode starts on arrival and is not gated by the
# fetch limit. Three chunks' worth of decoders is the ceiling; what must never
# return is one per camera per *buffered* chunk, which grows with the buffer.
DECODING_CHUNKS = 3

CAMS = {
    CAM_TOP: (720, 1280),
    "observation.images.left_wrist": (600, 960),
    "observation.images.right_wrist": (600, 960),
}
FPS = 30
FRAMES = 600
MODE_KEY = "lerobot.cameraVideoMode"
LINK = {
    "latency": 250,
    "downloadThroughput": 150_000,
    "uploadThroughput": 150_000,
}  # the rig link's order of magnitude

# Every VideoDecoder the page makes and closes, so a leak of decoders left
# running for dropped chunks is a number and not a guess.
COUNT_DECODERS = """
(() => {
  const Real = window.VideoDecoder;
  if (!Real) return;
  window.__decoders = { made: 0, closed: 0, live: 0, maxLive: 0, errors: [] };
  window.VideoDecoder = class extends Real {
    constructor(init) {
      const err = init.error;
      super({ ...init, error: (e) => { window.__decoders.errors.push(String(e && e.message || e)); err && err(e); } });
      const d = window.__decoders; d.made++; d.live++; d.maxLive = Math.max(d.maxLive, d.live);
    }
    close() { const d = window.__decoders; d.closed++; d.live--; return super.close(); }
  };
})();
"""


def _disc(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt, write_episode

    root = tmp_path_factory.mktemp("storm") / "prod"
    feats = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    for cam, (h, w) in CAMS.items():
        feats[cam] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    rng = np.random.default_rng(1)
    tex = {cam: rng.integers(0, 256, (h, w * 2, 3), np.uint8) for cam, (h, w) in CAMS.items()}
    ds = LeRobotDataset.create(repo_id="tests/storm", fps=FPS, root=root, features=feats, use_videos=True)
    for i in range(FRAMES):
        ds.add_frame(
            {
                "observation.state": np.zeros(2, np.float32),
                "action": np.zeros(2, np.float32),
                "task": "storm",
                **{cam: frame_image(0, i % 128, h, w, texture=tex[cam]) for cam, (h, w) in CAMS.items()},
            }
        )
    ds.save_episode()
    ds.finalize()
    ds = LeRobotDataset("tests/storm", root=root)
    h, w = CAMS[CAM_TOP]
    adopt(ds, [CAM_TOP], LABELS, (h, w), treatments={"ball": TINT_RECIPE})
    write_episode(ds, 0, CAM_TOP, [{"ball": _disc(h, w, 360, 400 + (i % 300), 90)} for i in range(FRAMES)])
    return root


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    yield srv
    srv.stop()


def _open(browser, base, ds_id):
    pg = browser.new_page(viewport={"width": 1600, "height": 1000})
    pg.console = []
    pg.on("console", lambda m: pg.console.append(f"{m.type}: {m.text[:220]}"))
    pg.on("pageerror", lambda e: pg.console.append(f"pageerror: {str(e)[:220]}"))
    pg.add_init_script(f"localStorage.setItem({json.dumps(MODE_KEY)}, 'low-bandwidth');")
    pg.add_init_script(COUNT_DECODERS)
    pg.goto(base)
    pg.wait_for_function("typeof openDataset === 'function'", timeout=15_000)
    pg.evaluate("(ds) => openDataset(ds)", ds_id)
    pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=120_000)
    pg.evaluate(f"selectEpisode({json.dumps(ds_id)}, 0, {FRAMES})")
    pg.wait_for_function("window.__chunkPlayer && window.__chunkPlayer.ready()", timeout=120_000)
    return pg


def _state(pg):
    return pg.evaluate(
        """() => { const p = window.__chunkPlayer; const m = p.metrics; return {
          painted: m.painted.length, last: m.painted.length ? m.painted[m.painted.length - 1].frame : -1,
          errors: m.errors.slice(), chunks: m.chunks.length, stalls: m.stalls.slice(-5), frame: window.currentFrame,
          decoders: window.__decoders }; }"""
    )


def _press(pg, selector):
    """Press a control, checking what matters and not what the compositor is doing.

    A plain click also waits for the element to hold still between two
    animation frames. This tab is playing video the whole time these run, so on
    a runner whose main thread is saturated that settles slowly or not at all,
    and the press fails on an element the log shows it had already found. The
    conditions worth keeping -- it is there, shown, and enabled -- are checked
    here; the press itself does not need a steady bounding box.
    """
    pg.wait_for_selector(selector, timeout=30_000)
    ready = pg.evaluate(
        """(sel) => { const el = document.querySelector(sel); if (!el) return 'missing';
           if (el.disabled) return 'disabled';
           const r = el.getBoundingClientRect();
           return (r.width && r.height) ? 'ok' : 'not shown'; }""",
        selector,
    )
    assert ready == "ok", f"{selector} is {ready}"
    pg.dispatch_event(selector, "click")


def _save_background(pg, key):
    """What the operator does: pick a background treatment in the inspector, press Save."""
    _press(pg, f'.ds-treat[data-label="__background__"] .ds-treat-btn[data-key="{key}"]')
    _press(pg, ".ds-treat-save")


def test_playback_survives_a_burst_of_treatment_saves(server, dataset_root):
    ds_id = server.open_dataset(dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = _open(browser, server.base, ds_id)
        cdp = pg.context.new_cdp_session(pg)
        cdp.send("Network.enable")
        cdp.send("Network.emulateNetworkConditions", {"offline": False, **LINK})
        pg.evaluate("() => togglePlay()")
        pg.wait_for_function("window.__chunkPlayer.metrics.painted.length > 40", timeout=60_000)
        _save_background(pg, "blur")
        pg.wait_for_timeout(900)
        _save_background(pg, "none")
        after = _state(pg)
        # The saves are done and the picture must move on: within the time the
        # server needs to rebuild and the link to deliver a couple of chunks --
        # on a shared CI runner a 720p chunk of three cameras took 5-9 s to
        # arrive -- the player paints three more seconds of media.
        try:
            pg.wait_for_function(
                f"window.__chunkPlayer.metrics.painted.length >= {after['painted'] + 3 * FPS}", timeout=45_000
            )
        except Exception:
            final = _state(pg)
            player_log = [c for c in pg.console if "[chunk-player]" in c][-30:]
            errs = [c for c in pg.console if c.startswith(("error", "pageerror"))][-10:]
            raise AssertionError(
                f"playback wedged after the saves: painted {after['painted']} -> {final['painted']} "
                f"(frame {final['frame']}), errors {final['errors']}, decoders {final['decoders']}, "
                f"stalls {final['stalls']}\\nplayer log: {player_log}\\nconsole errors: {errs}"
            ) from None
        final = _state(pg)
        assert not final["errors"], final["errors"]
        assert not [c for c in pg.console if c.startswith("pageerror")], pg.console[-10:]
        print("DECODERS", final["decoders"], "chunks", final["chunks"], "stalls", final["stalls"])
        browser.close()


def _corrupt_once(pg, start):
    """The chunk at `start` arrives once with its first camera's video bytes
    zeroed -- a transfer the decoder cannot use -- and intact after that."""
    hits = {"n": 0}

    def handle(route, request):
        if f"start={start}&" in request.url and hits["n"] == 0:
            hits["n"] += 1
            resp = route.fetch()
            body = bytearray(resp.body())
            hl = int.from_bytes(body[:4], "little")
            header = json.loads(bytes(body[4 : 4 + hl]))
            part = next(p for p in header["parts"] if p["kind"] == "video")
            off = 4 + hl + part["offset"]
            body[off : off + part["length"]] = bytes(part["length"])
            route.fulfill(response=resp, body=bytes(body))
            return
        route.continue_()

    pg.route(re.compile(r".*/chunk\?.*"), handle)
    return hits


def test_a_chunk_that_never_becomes_ready_is_fetched_again(server, dataset_root, caplog):
    """A chunk the page holds but can never show -- here its video arrived
    unusable -- is given up on after a deadline and asked for again, the
    decoders it opened are closed, and the event reaches the server log.
    Before, the tile held on it forever with nothing in any log."""
    import logging

    caplog.set_level(logging.WARNING, logger="lerobot.gui.api.chunk_playback")
    ds_id = server.open_dataset(dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = _open(browser, server.base, ds_id)
        # A chunk beyond the look-ahead the player already holds at ready().
        bad = 180
        assert not pg.evaluate(f"window.__chunkPlayer.metrics.chunks.some((c) => c.start === {bad})"), (
            "already fetched: corrupt a later chunk"
        )
        hits = _corrupt_once(pg, bad)
        pg.evaluate("() => togglePlay()")
        pg.wait_for_function(
            f"window.__chunkPlayer.metrics.painted.some((q) => q.frame >= {bad + 1})", timeout=60_000
        )
        assert hits["n"] == 1, "the corrupt transfer was never served"
        st = _state(pg)
        fetched = pg.evaluate(f"window.__chunkPlayer.metrics.chunks.filter((c) => c.start === {bad}).length")
        assert fetched >= 2, ("the chunk was not asked for again", fetched, st)
        retries = pg.evaluate("window.__chunkPlayer.metrics.retries")
        assert retries and retries[0]["start"] == bad, retries
        pg.wait_for_timeout(500)
        dec = _state(pg)["decoders"]
        # Not `live == 0`: decoders come and go with every chunk, so a count
        # sampled at an instant catches whichever chunk happens to be decoding
        # -- the same weakness that let three versions of the recovery suite's
        # decoder test pass with their bug in place. What must not happen is a
        # decoder being forgotten: the dropped chunk's outliving the chunk.
        assert dec["made"] - dec["closed"] <= DECODING_CHUNKS * len(CAMS), (
            "more decoders are open than one chunk's worth: the dropped chunk's were left",
            dec,
        )
        browser.close()
    warned = [rec.getMessage() for rec in caplog.records if "chunk-playback client" in rec.getMessage()]
    assert any("never-ready" in m and "chunk 180" in m for m in warned), warned


def test_chunk_requests_carry_the_players_state(server, dataset_root):
    """Every chunk request says what the page holds, has in flight and has
    painted, so the server's chunk line is a record of the page as well."""
    ds_id = server.open_dataset(dataset_root)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        seen = []
        pg = browser.new_page(viewport={"width": 1600, "height": 1000})
        pg.on("request", lambda r: seen.append(r.headers.get("x-player")) if "/chunk?" in r.url else None)
        pg.add_init_script(f"localStorage.setItem({json.dumps(MODE_KEY)}, 'low-bandwidth');")
        pg.goto(server.base)
        pg.wait_for_function("typeof openDataset === 'function'", timeout=15_000)
        pg.evaluate("(ds) => openDataset(ds)", ds_id)
        pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=120_000)
        pg.evaluate(f"selectEpisode({json.dumps(ds_id)}, 0, {FRAMES})")
        pg.wait_for_function("window.__chunkPlayer && window.__chunkPlayer.ready()", timeout=120_000)
        pg.evaluate("() => togglePlay()")
        pg.wait_for_function("window.__chunkPlayer.metrics.chunks.length >= 3", timeout=60_000)
        browser.close()
    assert seen and all(s for s in seen), seen
    assert all(
        all(k in s for k in ("cur=", "held=", "inflight=", "painted=", "holds=", "errors=")) for s in seen
    ), seen
    assert any("held=0" in s and "inflight=" in s for s in seen), (
        "the second request names what the first left held",
        seen,
    )
