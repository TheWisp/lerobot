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

from lerobot.gui.api.chunk_playback import chunk_frames  # noqa: E402
from lerobot.gui.link_class import CLASS_LINK  # noqa: E402
from tests.gui.chunk_fixtures import (  # noqa: E402
    FPS,
    GuiServer,
    build_dataset,
    chunk_url,
    wait_for_player,
)

pytestmark = pytest.mark.requires_playwright

MODE_KEY = "lerobot.cameraVideoMode"
FRAMES = 300  # 30 s at 10 fps: fifteen chunks per episode
# The link both low-bandwidth paths are built for, named once in code and
# shared with the Run tab's live stream: the round trip is the class's, and
# the rates below are stated against its downlink rather than against a
# number chosen here.
LATENCY_MS = CLASS_LINK.rtt_ms
CLASS_BYTES_PER_S = CLASS_LINK.down_kbit_s * 1000 / 8


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
        wait_for_player(page, "window.__chunkPlayer && window.__chunkPlayer.ready()")
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
        wait_for_player(page, "window.__chunkPlayer.metrics.painted.some((q) => q.frame === 100)")
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


@pytest.fixture(scope="module")
def unthrottled(server, bytes_per_second):
    """How well this machine keeps time when the link is not the constraint.

    Every claim below is measured against this rather than against a number.
    "Media time within a few percent of wall time" is a statement about the
    runner as much as the player: one that cannot decode in real time misses it
    however fast the link is, and one that can will beat it on a link that
    carries the content. Comparing the two runs takes the machine out.
    """
    m = _play(server, 50.0 * bytes_per_second, 8.0)
    # Relative claims are only worth as much as what they are relative to: if
    # the uncapped run played nothing, every comparison below passes by
    # default. This is the one floor here, and it is "it played at all".
    assert m["ratio"] > 0 and m["chunks"] > 0, ("the uncapped run played nothing to compare against", m)
    return m


def _worst_hold(m):
    return max((s["ms"] for s in m["stalls"]), default=0)


def test_playback_keeps_time_across_many_chunks_on_the_class_link(server, bytes_per_second, unthrottled):
    """The link the profile is chosen for costs the player nothing it does not
    already cost itself.

    Stated against the shared constant rather than against the content's own
    bytes: what the operator has is a link of a certain class, and what this
    answers is whether the profile fits it.
    """
    m = _play(server, CLASS_BYTES_PER_S, 16.0)

    # Chunks are what the media advanced demanded, not a count fixed in advance.
    wanted = m["ratio"] * 16.0 * FPS / chunk_frames(FPS)
    assert m["chunks"] >= wanted * 0.8, (m, f"expected about {wanted:.0f} chunks for what it played")
    assert _worst_hold(m) <= max(_worst_hold(unthrottled) * 2, 1000 / FPS * 4), (
        m,
        f"the uncapped run's worst hold was {_worst_hold(unthrottled)} ms",
    )
    assert m["ratio"] >= unthrottled["ratio"] * 0.95, (m, unthrottled)


def test_at_2x_on_a_link_that_carries_it(server, bytes_per_second, unthrottled):
    """2x for 12 s, on twice the class link: reviewing at speed needs twice
    the bytes per second of wall time, so this is the same claim as above."""
    m = _play(server, 2.0 * CLASS_BYTES_PER_S, 12.0, speed="2")
    assert _worst_hold(m) <= max(_worst_hold(unthrottled) * 2, 1000 / FPS * 4), (m, unthrottled)
    assert m["ratio"] >= unthrottled["ratio"] * 0.95, (m, unthrottled)


def test_the_harness_sees_holds_on_a_link_that_cannot_carry_the_content(
    server, bytes_per_second, unthrottled
):
    """The complement: at 0.4x the content's bytes the player must hold, or the
    assertions above pass because nothing is measured.

    Stated against the uncapped run so that a machine slow enough to miss the
    others does not quietly satisfy this one too.
    """
    m = _play(server, 0.4 * bytes_per_second, 10.0)  # relative: the floor is the content's
    assert _worst_hold(m) > _worst_hold(unthrottled) * 2 or m["ratio"] < unthrottled["ratio"] * 0.7, (
        m,
        unthrottled,
    )


def test_the_profile_fits_the_class_links_budget(bytes_per_second):
    """What anchors the profile: its bytes per second of media are inside the
    share of the class link the pictures may use.

    The width and the quality are empirical, but the link they were chosen
    for is not a number typed here — it is the constant both tabs derive
    from, and this is what would fail if either moved.
    """
    budget = CLASS_LINK.stream_budget_kbit_s * 1000 / 8
    assert bytes_per_second <= budget, (
        f"the profile costs {bytes_per_second / 1000:.1f} kB/s of media and the "
        f"{CLASS_LINK.name} budget is {budget / 1000:.1f} kB/s"
    )
