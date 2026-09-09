"""The Data tab plays through windows, not stills.

Frames are recorded as distinct flat greys, so a canvas identifies the frame
it shows. The tab is driven the way the operator drives it: select an
episode, step with the keyboard, play, switch episodes. What the still path
did per frame the player now does per paint, and nothing asks the JPEG
endpoint for anything.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")

import uvicorn  # noqa: E402
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeout,  # noqa: E402
    sync_playwright,  # noqa: E402
)

pytestmark = pytest.mark.requires_playwright

CAMS = ["observation.images.a", "observation.images.b"]
H, W = 240, 480
FPS = 10
FRAMES = 40


def grey(ep: int, i: int) -> int:
    return 30 + 60 * ep + 4 * i


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    """Two episodes; camera a carries a saved ball mask with a red tint, so the
    tab shows the recipe's composite on it."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt, write_episode

    root = tmp_path_factory.mktemp("tab") / "tab"
    ds = LeRobotDataset.create(
        repo_id="tests/tab",
        fps=FPS,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            **{
                c: {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
                for c in CAMS
            },
        },
        use_videos=True,
    )
    for ep in range(2):
        for i in range(FRAMES):
            ds.add_frame(
                {
                    "observation.state": np.array([ep, i], np.float32),
                    "action": np.array([i, ep], np.float32),
                    "task": "tab",
                    **{c: np.full((H, W, 3), grey(ep, i), np.uint8) for c in CAMS},
                }
            )
        ds.save_episode()
    ds.finalize()
    ds = LeRobotDataset("tests/tab", root=root)
    adopt(
        ds,
        [CAMS[0]],
        ["ball"],
        (H, W),
        treatments={"ball": {"key": "tint", "params": {"color": [255, 0, 0]}}},
    )
    blob = np.zeros((H, W), bool)
    blob[40:160, 40:240] = True
    for ep in range(2):
        write_episode(ds, ep, CAMS[0], [{"ball": blob} for _ in range(FRAMES)])
    return root


@pytest.fixture(scope="module")
def server(dataset_root, tmp_path_factory):
    import os

    from lerobot.gui import server as gui_server_mod

    prev = {k: os.environ.get(k) for k in ("LEROBOT_GUI_CONFIG_DIR", "LEROBOT_WINDOW_CACHE_DIR")}
    os.environ["LEROBOT_GUI_CONFIG_DIR"] = str(tmp_path_factory.mktemp("config"))
    os.environ["LEROBOT_WINDOW_CACHE_DIR"] = str(tmp_path_factory.mktemp("wcache"))
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()

    import requests

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if requests.get(base, timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        srv.should_exit = True
        pytest.fail("GUI server did not become ready")
    r = requests.post(f"{base}/api/datasets", json={"local_path": str(dataset_root)}, timeout=60)
    assert r.status_code == 200, r.text
    yield base, r.json()["id"]
    srv.should_exit = True
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


PROBE = """() => {
  const out = { frame: window.currentFrame, playing: window.__windowPlayer.playing(), cams: {} };
  for (const c of document.querySelectorAll('canvas.frame-canvas')) {
    const ctx = c.getContext('2d');
    const at = (fx, fy) => ctx.getImageData(Math.floor(fx * c.width), Math.floor(fy * c.height), 1, 1).data;
    // The blob covers rows 40..160 and columns 40..240 of 240x480: one probe well inside, one well outside.
    const centre = at(0.8, 0.85), blob = at(0.2, 0.4);
    out.cams[c.dataset.cam] = { w: c.width, h: c.height, centre: [...centre].slice(0, 3), blob: [...blob].slice(0, 3) };
  }
  return out;
}"""


def _wait_paint(page, pred: str, timeout: int = 30_000):
    page.wait_for_function(
        f"() => window.__windowPlayer && window.__windowPlayer.metrics.painted.length > 0 && ({pred})",
        timeout=timeout,
    )


def test_the_tab_plays_the_episode_through_windows(server):
    base, did = server
    stills: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("request", lambda r: stills.append(r.url) if "/frame/" in r.url else None)
        page.goto(base)
        page.wait_for_function(
            "() => typeof openDataset === 'function' && window.WindowPlayer", timeout=30_000
        )
        page.evaluate("(ds) => openDataset(ds)", did)
        page.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=did, timeout=60_000)
        page.evaluate("([ds, n]) => selectEpisode(ds, 1, n)", [did, FRAMES])
        _wait_paint(page, "window.currentFrame === 0")
        # Saved masks exist on camera a, so the tab asks for composited windows and the
        # canvas shows the tint inside the blob and the stored grey outside.
        page.wait_for_function(
            "() => (window.__windowPlayer.lastWindowUrl() || '').includes('masks=composited') && window.__windowPlayer.metrics.painted.length > 1",
            timeout=30_000,
        )
        page.wait_for_timeout(500)
        first = page.evaluate(PROBE)

        page.keyboard.press("ArrowRight")
        _wait_paint(page, "window.currentFrame === 1")
        stepped = page.evaluate(PROBE)

        done = page.evaluate("() => window.loadAllFrames(10).then(() => window.currentFrame)")
        assert done == 10

        page.evaluate("() => togglePlay()")
        page.wait_for_function("() => window.__windowPlayer.metrics.wraps.length >= 1", timeout=60_000)
        page.evaluate("() => togglePlay()")
        page.wait_for_timeout(300)
        played = page.evaluate(PROBE)

        page.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [did, FRAMES])
        page.wait_for_function(
            "() => window.__windowPlayer.episode() === 0 && window.__windowPlayer.metrics.painted.length > 0 && window.currentEpisode === 0",
            timeout=30_000,
        )
        page.wait_for_timeout(300)
        switched = page.evaluate(PROBE)
        errors = page.evaluate("() => window.__windowPlayer.metrics.errors")
        browser.close()

    assert not errors, errors
    assert not stills, f"the tab still asked the JPEG endpoint: {stills[:3]}"
    a, b = CAMS
    # Frame 0 of episode 1 on both tiles; the tint only inside camera a's blob.
    assert abs(first["cams"][b]["centre"][0] - grey(1, 0)) <= 8, first
    assert abs(first["cams"][a]["centre"][0] - grey(1, 0)) <= 8, first
    assert first["cams"][a]["blob"][0] - first["cams"][a]["blob"][1] > 60, first
    assert abs(first["cams"][b]["blob"][0] - first["cams"][b]["blob"][1]) < 12, first
    # A step is one frame; the playhead readouts follow.
    assert stepped["frame"] == 1 and abs(stepped["cams"][b]["centre"][0] - grey(1, 1)) <= 8, stepped
    # Play wrapped within the episode and paused where it was.
    assert not played["playing"] and 0 <= played["frame"] < FRAMES, played
    assert abs(played["cams"][b]["centre"][0] - grey(1, played["frame"])) <= 8, played
    # The other episode's frames after the switch.
    assert switched["frame"] == 0 and abs(switched["cams"][b]["centre"][0] - grey(0, 0)) <= 8, switched


def test_the_tile_keeps_one_size_while_the_rung_changes(server):
    """A rung is a bitrate, not a tile size: the canvases keep the camera's
    stored resolution at every rung, or the tiles grow and shrink as the ladder
    moves and the mask layer leaves register.

    The rung is driven by hand here and the held windows dropped after each
    change (what the tab does when an edit makes them wrong), because on
    localhost the automatic rule reaches the top of the ladder within the first
    second and a sampled test would see one rung."""
    base, did = server
    sizes: set[tuple] = set()
    seen: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(base)
        page.wait_for_function(
            "() => typeof openDataset === 'function' && window.WindowPlayer", timeout=30_000
        )
        page.evaluate("(ds) => openDataset(ds)", did)
        page.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=did, timeout=60_000)
        page.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [did, FRAMES])
        _wait_paint(page, "window.currentFrame === 0")
        rungs = page.evaluate("() => window.__windowPlayer.bundle().rungs")
        sizes_js = (
            "() => [...document.querySelectorAll('canvas.frame-canvas')].map((c) => [c.width, c.height])"
        )
        for rung in rungs:
            page.evaluate(
                "(r) => { window.__windowPlayer.rung(r); window.__windowPlayer.masksChanged(); }", rung
            )
            page.wait_for_function(
                "(r) => (window.__windowPlayer.metrics.painted.slice(-1)[0] || {}).rung === r",
                arg=rung,
                timeout=30_000,
            )
            seen.append(rung)
            sizes.add(tuple(tuple(x) for x in page.evaluate(sizes_js)))
        errors = page.evaluate("() => window.__windowPlayer.metrics.errors")
        browser.close()

    assert not errors, errors
    assert seen == rungs and len(seen) >= 2, seen
    assert sizes == {((W, H), (W, H))}, f"the tiles changed size across rungs {seen}: {sizes}"


def test_window_zero_is_requested_once(server):
    """Window 0 goes out beside the bundle, outside the fetch walk. Without a
    place held for it the walk asks for the very same window again the moment
    the bundle lands, which over a link is a second copy of it on the wire.
    (A later request for frame 0 at a higher rung is the upgrade, not this.)"""
    base, did = server
    asked: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("request", lambda r: asked.append(r.url) if "/window?" in r.url else None)
        page.goto(base)
        page.wait_for_function(
            "() => typeof openDataset === 'function' && window.WindowPlayer", timeout=30_000
        )
        page.evaluate("(ds) => openDataset(ds)", did)
        page.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=did, timeout=60_000)
        page.evaluate("([ds, n]) => selectEpisode(ds, 1, n)", [did, FRAMES])
        _wait_paint(page, "window.currentFrame === 0")
        page.wait_for_timeout(1500)
        browser.close()

    twice = sorted({u for u in asked if asked.count(u) > 1})
    assert not twice, f"the same window was fetched more than once: {twice}"


def test_play_wraps_within_the_trim_range(server):
    base, did = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(base)
        page.wait_for_function(
            "() => typeof openDataset === 'function' && window.WindowPlayer", timeout=30_000
        )
        page.evaluate("(ds) => openDataset(ds)", did)
        page.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=did, timeout=60_000)
        page.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [did, FRAMES])
        _wait_paint(page, "window.currentFrame === 0")
        page.evaluate("() => { trimStart = 10; trimEnd = 20; }")
        # Everything from here on is what the trim is supposed to govern.
        page.evaluate("() => { window.__windowPlayer.metrics.painted.length = 0; }")
        page.evaluate("() => togglePlay()")
        # Ten frames at 10 fps: two laps take about two seconds. The wait is
        # long because a loaded machine starves both the page's animation
        # frames and the server's encoder; a timeout reports what the player
        # was doing, so a slow run and a stalled one are told apart.
        try:
            page.wait_for_function("() => window.__windowPlayer.metrics.wraps.length >= 2", timeout=120_000)
        except PlaywrightTimeout:  # pragma: no cover - only on a starved machine
            m = page.evaluate("window.__windowPlayer.metrics")
            pytest.fail(
                f"playback did not lap the trim range twice: painted {len(m['painted'])} frames, "
                f"wraps {m['wraps']}, stalls {m['stalls'][:4]}, windows {len(m['windows'])}, "
                f"errors {m['errors'][:2]}, state {page.evaluate('window.__windowPlayer.state()')}"
            )
        page.evaluate("() => togglePlay()")
        frames = page.evaluate("() => window.__windowPlayer.metrics.painted.map((f) => f.frame)")
        browser.close()
    # Every frame painted after the trim was set lies inside it. Skipping to the
    # first in-range frame instead would pass while the player walked the whole
    # episode to reach the range -- which is what it did, painting frames the
    # trim excludes and then waiting on a window the fetch planner, reading the
    # clock as clamped, never asked for.
    assert frames, "nothing was painted after the trim was set"
    assert all(10 <= f < 20 for f in frames), (
        f"frames outside the trim range were painted: {sorted({f for f in frames if not 10 <= f < 20})}"
    )
    assert frames.count(10) >= 2, frames
