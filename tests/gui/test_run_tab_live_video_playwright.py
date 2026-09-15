"""The Run tab at Low Bandwidth: video tiles, no polls, and the way back.

The operator's two paths through the same tab, driven in a real browser
against a real run's tap. What matters here is not that pictures exist —
the server tests cover that — but that choosing Low Bandwidth changes what
the tab does: video elements in place of polled images, the readouts and the
visualizer fed from the stream, and Full Quality untouched beside it.
"""

from __future__ import annotations

import base64
import json
import os
import time

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import sync_playwright  # noqa: E402

import lerobot.robots.obs_stream as obs_stream  # noqa: E402
from tests.gui.chunk_fixtures import GuiServer  # noqa: E402

pytestmark = pytest.mark.requires_playwright

MODE_KEY = "lerobot.cameraVideoMode"


@pytest.fixture(scope="module", autouse=True)
def _own_shm_names():
    before = obs_stream.SHM_PREFIX
    obs_stream.SHM_PREFIX = f"lerobot_obs_rt{os.getpid()}_"
    yield
    obs_stream.SHM_PREFIX = before


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = GuiServer(tmp_path_factory.mktemp("config"), tmp_path_factory.mktemp("cache"))
    yield srv
    srv.stop()


class Tap:
    """The run's tap, which every test in this module shares.

    ``restart`` is what a run ending and another starting does to it: the
    segments are unlinked and new ones are created under the same names.
    It is a method here rather than something a test does for itself,
    because a test that stopped the shared tap would leave every test after
    it with nothing to watch.
    """

    def __init__(self) -> None:
        self._tap = None
        self.restart()

    def __getattr__(self, name):
        return getattr(self._tap, name)

    def restart(self) -> None:
        from tests.gui.test_live_video_pipeline import SyntheticTap

        if self._tap is not None:
            self._tap.stop()
        self._tap = SyntheticTap()
        self._tap.start()
        deadline = time.time() + 5.0
        while self._tap.cycles_written < 2 and time.time() < deadline:
            time.sleep(0.01)

    def stop(self) -> None:
        self._tap.stop()


@pytest.fixture(scope="module")
def tap(server):
    """After the server: its startup sweeps every segment in our namespace."""
    t = Tap()
    yield t
    t.stop()


class Tab:
    """The Run tab in a browser, with the requests it makes recorded."""

    def __init__(self, page, requests):
        self.page = page
        self.requests = requests

    def since(self, marker: int) -> list[str]:
        return self.requests[marker:]

    def mark(self) -> int:
        return len(self.requests)

    def wait_for_tiles(self, kind: str, count: int, timeout: float = 30.0) -> None:
        """Wait for the tiles to be showing something, not merely to exist."""
        selector = f".obs-cam-grid [data-cam-cell] {kind}"
        self.page.wait_for_function(
            """([sel, n]) => {
                const els = [...document.querySelectorAll(sel)];
                if (els.length < n) return false;
                return els.every((e) => e.tagName === 'VIDEO' ? e.videoWidth > 0 : true);
            }""",
            arg=[selector, count],
            timeout=timeout * 1000,
        )

    def state(self) -> dict:
        return self.page.evaluate("() => window.__liveVideoState || null")


@pytest.fixture(scope="module")
def browser_page(server, tap):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        requests: list[str] = []
        page.on("request", lambda r: requests.append(r.url))
        page.goto(server.base, wait_until="domcontentloaded")
        yield Tab(page, requests)
        browser.close()


def _open_run_tab(tab: Tab, mode: str) -> None:
    tab.page.evaluate(
        "(m) => { localStorage.setItem('lerobot.cameraVideoMode', m); }",
        mode,
    )
    tab.page.reload(wait_until="domcontentloaded")
    tab.page.evaluate("() => switchTab('run')")
    tab.page.evaluate("() => startObsStreamViewer()")


def test_low_bandwidth_draws_the_cameras_as_video(browser_page, tap):
    tab = browser_page
    _open_run_tab(tab, "low-bandwidth")
    tab.wait_for_tiles("video", len(tap.stream.image_keys))
    # Playing, at the profile's width, and named by the answer. `videoWidth`
    # alone said none of that: it is non-zero for an element that decoded one
    # frame and then stopped, which is what a tile looks like when the element
    # was never started -- four of them, three frozen, and this passed.
    sizes = tab.page.evaluate(
        """() => [...document.querySelectorAll('.obs-cam-grid [data-cam-cell] video')]
              .map(v => ({w: v.videoWidth, h: v.videoHeight, cam: v.dataset.camera,
                          paused: v.paused, t: v.currentTime}))"""
    )
    assert len(sizes) == len(tap.stream.image_keys)
    for s in sizes:
        assert s["w"] == 320, s
        assert s["h"] > 0, s
        assert s["cam"] in tap.stream.image_keys, s

    tab.page.wait_for_timeout(1500)
    playing = tab.page.evaluate(
        """() => [...document.querySelectorAll('.obs-cam-grid [data-cam-cell] video')]
              .map(v => ({cam: v.dataset.camera, paused: v.paused, t: v.currentTime}))"""
    )
    stalled = [v for v in playing if v["paused"] or v["t"] == 0]
    assert not stalled, f"tiles that never started: {stalled}; all: {playing}"
    # The picture is the video; the polled image layer is not also there.
    assert (
        tab.page.evaluate(
            "() => document.querySelectorAll('.obs-cam-grid [data-cam-cell] img:not(.overlay-layer)').length"
        )
        == 0
    )


def test_low_bandwidth_stops_the_polls(browser_page, tap):
    tab = browser_page
    _open_run_tab(tab, "low-bandwidth")
    tab.wait_for_tiles("video", len(tap.stream.image_keys))
    marker = tab.mark()
    tab.page.wait_for_timeout(2000)
    after = tab.since(marker)
    assert not [u for u in after if "/obs-stream/image/" in u], "a picture was polled"
    assert not [u for u in after if "/urdf-viz?source=" in u], "the visualizer polled"
    assert not [u for u in after if "/obs-stream/state" in u], "the readouts polled"


def test_the_controls_bar_reports_the_stream_and_its_age(browser_page, tap):
    tab = browser_page
    _open_run_tab(tab, "low-bandwidth")
    tab.wait_for_tiles("video", len(tap.stream.image_keys))
    tab.page.wait_for_function("() => (window.__liveVideoState || {}).name === 'streaming'", timeout=30000)
    text = tab.page.inner_text("#run-live-video-state")
    assert "Streaming" in text, text
    assert "320" in text, text
    # The age appears once enough frames have been painted to place them.
    tab.page.wait_for_function(
        "() => /\\d+\\s*ms/.test(document.getElementById('run-live-video-state').innerText)",
        timeout=30000,
    )
    age = int(
        tab.page.evaluate(
            "() => /([0-9]+)\\s*ms/.exec(document.getElementById('run-live-video-state').innerText)[1]"
        )
    )
    assert 0 < age < 2000, age


def test_the_readouts_and_the_visualizer_follow_the_stream(browser_page, tap):
    tab = browser_page
    _open_run_tab(tab, "low-bandwidth")
    tab.wait_for_tiles("video", len(tap.stream.image_keys))
    tab.page.wait_for_function("() => (window.__liveVideoCycles || 0) > 10", timeout=30000)
    first = tab.page.evaluate("() => window.__liveVideoLastCycle")
    tab.page.wait_for_timeout(1000)
    second = tab.page.evaluate("() => window.__liveVideoLastCycle")
    assert second > first, (first, second)
    # The visualizer tile is drawing what the stream sent it, not its own poll.
    tab.page.wait_for_function(
        """() => {
            const f = [...document.querySelectorAll('iframe')].find(f => f.title === 'Robot visualizer');
            return f && f.contentWindow && f.contentWindow.__urdfApplied;
        }""",
        timeout=30000,
    )


def test_full_quality_is_the_tab_as_it_was(browser_page, tap):
    tab = browser_page
    _open_run_tab(tab, "full-quality")
    tab.wait_for_tiles("img:not(.overlay-layer)", len(tap.stream.image_keys))
    marker = tab.mark()
    tab.page.wait_for_timeout(1500)
    after = tab.since(marker)
    assert [u for u in after if "/obs-stream/image/" in u], "the pictures were not polled"
    assert not [u for u in after if "live-video/offer" in u], "a stream was opened anyway"
    assert tab.page.evaluate("() => document.querySelectorAll('.obs-cam-grid video').length") == 0


def test_the_control_switches_between_them_mid_run(browser_page, tap):
    tab = browser_page
    _open_run_tab(tab, "full-quality")
    tab.wait_for_tiles("img:not(.overlay-layer)", len(tap.stream.image_keys))
    assert tab.page.input_value("#run-video-mode-select") == "full-quality"

    tab.page.select_option("#run-video-mode-select", "low-bandwidth")
    tab.wait_for_tiles("video", len(tap.stream.image_keys))
    marker = tab.mark()
    tab.page.wait_for_timeout(1500)
    assert not [u for u in tab.since(marker) if "/obs-stream/image/" in u]
    # And the choice is the one the Data tab keeps, not a second setting.
    assert tab.page.evaluate("() => localStorage.getItem('lerobot.cameraVideoMode')") == "low-bandwidth"

    tab.page.select_option("#run-video-mode-select", "full-quality")
    tab.wait_for_tiles("img:not(.overlay-layer)", len(tap.stream.image_keys))
    marker = tab.mark()
    tab.page.wait_for_timeout(1500)
    assert [u for u in tab.since(marker) if "/obs-stream/image/" in u]


def test_a_second_run_streams_after_the_first_one_ends(browser_page, tap):
    """Stop a run, start another: the commonest thing an operator does.

    The tap's segments are recreated, so a reader attached to the old ones
    sees a sequence that never advances again. Nothing in the page asks for
    anything at that point — the tiles are rebuilt with the same camera names
    and the connection is deliberately still open — so the freeze is silent.
    """
    tab = browser_page
    _open_run_tab(tab, "low-bandwidth")
    tab.wait_for_tiles("video", len(tap.stream.image_keys))
    tab.page.wait_for_function("() => (window.__liveVideoCycles || 0) > 10", timeout=30000)

    tab.page.evaluate("() => stopObsStreamViewer()")
    tap.restart()
    tab.page.evaluate("() => startObsStreamViewer()")
    tab.wait_for_tiles("video", len(tap.stream.image_keys))

    # A frozen video element still has a width and is still playing, so what
    # says the second run is arriving is the count moving from where it is
    # now — not where it is, which the first run already raised.
    was = tab.page.evaluate("() => window.__liveVideoCycles || 0")
    tab.page.wait_for_function(f"() => (window.__liveVideoCycles || 0) > {was} + 20", timeout=30000)
    painted = tab.page.evaluate(
        """() => [...document.querySelectorAll('.obs-cam-grid [data-cam-cell] video')]
              .every(v => v.videoWidth > 0 && !v.paused)"""
    )
    assert painted, "the second run's tiles are not showing anything"


# A 1x1 PNG: what the overlay worker's endpoint returns, reduced to the one
# thing the tile does with it — load it and show it.
_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _serve_overlay(tab: Tab) -> None:
    """Answer the overlay endpoint, and make the page ask for it.

    The gate that decides whether a camera has an overlay lives in the
    Overlays panel and needs the worker; the URL it produces does not, and
    the URL is what the tile consumes.
    """
    tab.page.route(
        "**/api/overlays/live/frame/**",
        lambda route: route.fulfill(status=200, content_type="image/png", body=_PIXEL_PNG),
    )
    tab.page.evaluate(
        """() => {
            window.Overlays = window.Overlays || {};
            window.Overlays.liveFrameUrl = (cam, seq) =>
                `/api/overlays/live/frame/${encodeURIComponent(cam)}?_=${seq}`;
        }"""
    )


def test_an_overlay_is_drawn_at_full_quality(browser_page, tap):
    """The JPEG path composites the worker's result over each tile. This is
    the path the branch did not set out to change and did touch: the tick
    that refreshes both the picture and its overlay was rewritten."""
    tab = browser_page
    _open_run_tab(tab, "full-quality")
    tab.wait_for_tiles("img:not(.overlay-layer)", len(tap.stream.image_keys))
    _serve_overlay(tab)

    tab.page.wait_for_function(
        """() => {
            const els = [...document.querySelectorAll('.obs-cam-grid .overlay-layer')];
            return els.length > 0 && els.every((e) => e.src && e.style.display === 'block');
        }""",
        timeout=30000,
    )
    # And it keeps up with the pictures rather than being drawn once.
    marker = tab.mark()
    tab.page.wait_for_timeout(1500)
    asked = [u for u in tab.since(marker) if "/api/overlays/live/frame/" in u]
    assert len(asked) >= len(tap.stream.image_keys), asked


def test_the_overlay_is_not_fetched_a_second_time_at_low_bandwidth(browser_page, tap):
    """At Low Bandwidth the overlay is already in the pixels — the pipeline
    takes it from the worker's buffer and blends it before the encode, which
    `tests/gui/test_live_video_pipeline.py` measures on the decoded frames.
    Fetching it again as a PNG would spend the link on a picture the operator
    is already looking at, so the layer stays hidden and nothing asks for
    it."""
    tab = browser_page
    _open_run_tab(tab, "low-bandwidth")
    tab.wait_for_tiles("video", len(tap.stream.image_keys))
    _serve_overlay(tab)

    marker = tab.mark()
    tab.page.wait_for_timeout(2000)
    assert not [u for u in tab.since(marker) if "/api/overlays/live/frame/" in u]
    hidden = tab.page.evaluate(
        """() => [...document.querySelectorAll('.obs-cam-grid .overlay-layer')]
              .every((e) => e.style.display === 'none')"""
    )
    assert hidden, "the polled overlay layer is showing over a video tile"


def test_the_other_tab_s_control_changes_this_one(browser_page, tap):
    """One setting, two controls. They share a key, which is not the same as
    following each other: a Run tab left showing Low Bandwidth after the
    operator turned it off on the Data tab goes on streaming, and its
    dropdown goes on disagreeing with the one they just used."""
    tab = browser_page
    _open_run_tab(tab, "low-bandwidth")
    tab.wait_for_tiles("video", len(tap.stream.image_keys))

    # The Data tab's own control, changed as a person changes it. Playwright
    # will not touch it while the Run tab is the one shown, so the event the
    # browser would raise is raised here; the listener under test is the same.
    tab.page.evaluate(
        """() => {
            const sel = document.getElementById('video-mode-select');
            sel.value = 'full-quality';
            sel.dispatchEvent(new Event('change'));
        }"""
    )

    tab.wait_for_tiles("img:not(.overlay-layer)", len(tap.stream.image_keys))
    assert tab.page.input_value("#run-video-mode-select") == "full-quality"
    marker = tab.mark()
    tab.page.wait_for_timeout(1500)
    assert [u for u in tab.since(marker) if "/obs-stream/image/" in u], "the JPEG path is back"
    assert tab.state()["name"] == "off"


def test_a_failure_says_why_and_keeps_the_tab_usable(browser_page, tap):
    """With the server refusing to answer, the bar carries the reason and the
    operator still has the control to switch paths."""
    tab = browser_page
    tab.page.route(
        "**/api/run/live-video/offer",
        lambda route: route.fulfill(
            status=503,
            content_type="application/json",
            body=json.dumps({"detail": "the encoder is unavailable"}),
        ),
    )
    try:
        _open_run_tab(tab, "low-bandwidth")
        tab.page.wait_for_function("() => (window.__liveVideoState || {}).name === 'failed'", timeout=30000)
        text = tab.page.inner_text("#run-live-video-state")
        assert "encoder is unavailable" in text, text
        assert tab.page.is_enabled("#run-video-mode-select")
    finally:
        tab.page.unroute("**/api/run/live-video/offer")
