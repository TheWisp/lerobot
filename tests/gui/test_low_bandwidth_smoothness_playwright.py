"""Playback stays smooth across many chunks on a link that can carry them
(design: R1), under Chromium's network emulation.

The operator saw 5 s holds at certain frames over the tailnet. The tab's other
tests play two chunks locally, which cannot see a fetcher that runs dry at the
third. This plays a 30 s episode -- fifteen 2 s chunks -- through a 250 ms
link capped relative to the content's own bytes, and reads the player's stalls.
The complement runs the same episode on a link that cannot carry it, so the
harness is known to see the holds it is asked about.
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("av")

import requests  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from tests.gui.chunk_fixtures import (  # noqa: E402
    FPS,
    GuiServer,
    build_dataset,
    chunk_url,
)

pytestmark = pytest.mark.requires_playwright

MODE_KEY = "lerobot.cameraVideoMode"
FRAMES = 300  # 30 s at 10 fps: fifteen chunks per episode
LATENCY_MS = 250


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    # Noise under the band strip, so a chunk costs what real footage costs and the
    # cap below is a link, not the throttle's floor (flat frames were 4 kB a chunk).
    return build_dataset(tmp_path_factory.mktemp("smooth") / "smooth", frames=FRAMES, noise=True)


@pytest.fixture(scope="module")
def server(dataset_root, tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    ds_id = srv.open_dataset(dataset_root)
    yield srv, ds_id
    srv.stop()


@pytest.fixture(scope="module")
def bytes_per_second(server):
    """What the content costs per second of media, measured from its own chunks
    (and warming the server's cache, so the link is the only variable)."""
    srv, ds_id = server
    total = 0
    for start in range(0, FRAMES, 20):
        r = requests.get(chunk_url(srv.base, ds_id, 0, start), timeout=120)
        assert r.status_code == 200, r.text
        total += len(r.content)
    per_s = total / (FRAMES / FPS)
    assert per_s >= 20_000, (
        f"the content costs {per_s / 1000:.1f} kB/s: too little for the cap to behave as a link"
    )
    return per_s


def _play(server, rate_bytes_per_s: float, seconds: float, speed: str = "1"):
    srv, ds_id = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        page.add_init_script(f"localStorage.setItem({json.dumps(MODE_KEY)}, 'low-bandwidth');")
        page.goto(srv.base)
        page.wait_for_function("typeof openDataset === 'function'", timeout=30_000)
        page.evaluate("(ds) => openDataset(ds)", ds_id)
        page.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=120_000)
        page.evaluate(f"selectEpisode({json.dumps(ds_id)}, 0, {FRAMES})")
        page.wait_for_function("window.__chunkPlayer && window.__chunkPlayer.ready()", timeout=120_000)
        # The link is capped only now: the page and its scripts are not the
        # transport under test. A seek into media the player does not hold
        # restarts the buffer under the cap, so every chunk played is fetched
        # through it.
        cdp = context.new_cdp_session(page)
        cdp.send("Network.enable")
        cdp.send(
            "Network.emulateNetworkConditions",
            {
                "offline": False,
                "latency": LATENCY_MS,
                "downloadThroughput": rate_bytes_per_s,
                "uploadThroughput": -1,
            },
        )
        page.evaluate("loadAllFrames(100)")
        page.wait_for_function(
            "window.__chunkPlayer.metrics.painted.some((q) => q.frame === 100)", timeout=120_000
        )
        page.select_option("#speed-select", speed)
        page.evaluate(
            "window.__chunkPlayer.metrics.stalls.length = 0; window.__chunkPlayer.metrics.painted.length = 0"
        )
        page.evaluate("togglePlay()")
        t0 = time.perf_counter()
        f0 = page.evaluate("window.__chunkPlayer.frame()")
        time.sleep(seconds)
        f1 = page.evaluate("window.__chunkPlayer.frame()")
        wall = time.perf_counter() - t0
        page.evaluate("togglePlay()")
        m = page.evaluate(
            "({stalls: window.__chunkPlayer.metrics.stalls, chunks: window.__chunkPlayer.metrics.chunks, wraps: window.__chunkPlayer.metrics.wraps.length})"
        )
        browser.close()
    advanced = (f1 - f0) + m["wraps"] * FRAMES
    # What the link delivered per chunk, so a failure says whether the cap or the fetcher was short.
    timeline = [
        (c["start"], c["bytes"], round(c["ms"]), round(c["bytes"] / max(1, c["ms"]))) for c in m["chunks"]
    ]
    return {
        "ratio": advanced / FPS / (float(speed) * wall),
        "stalls": m["stalls"],
        "chunks": len(m["chunks"]),
        "nominal_kB_s": round(rate_bytes_per_s / 1000, 1),
        "per_chunk (start, bytes, ms, kB/s)": timeline,
    }


def test_playback_keeps_time_across_many_chunks_on_a_link_that_carries_them(server, bytes_per_second):
    """1.5x the content's bytes, 250 ms latency, 1x for 16 s: eight chunks or
    more fetched, no hold over 500 ms, media time within 5% of wall time."""
    m = _play(server, 1.5 * bytes_per_second, 16.0)
    assert m["chunks"] >= 8, m
    long = [s for s in m["stalls"] if s["ms"] > 500]
    assert not long, f"playback held: {long}; {m}"
    assert m["ratio"] >= 0.95, m


def test_at_2x_on_a_link_that_carries_it(server, bytes_per_second):
    """3x the content's bytes -- 1.5x what 2x needs -- and 2x for 12 s."""
    m = _play(server, 3.0 * bytes_per_second, 12.0, speed="2")
    long = [s for s in m["stalls"] if s["ms"] > 500]
    assert not long, f"playback held at 2x: {long}"
    assert m["ratio"] >= 0.95, m


def test_the_harness_sees_holds_on_a_link_that_cannot_carry_the_content(server, bytes_per_second):
    """The complement: at 0.4x the content's bytes the player must hold, or the
    assertions above pass because nothing is measured."""
    m = _play(server, 0.4 * bytes_per_second, 10.0)
    assert any(s["ms"] > 500 for s in m["stalls"]) or m["ratio"] < 0.7, m
