"""Playback after an edit that rewrites the dataset.

A trim renumbers an episode's frames and a delete renumbers its episodes, so
every window built before either one describes something else afterwards. The
server drops its own cached windows on an edit; the browser holds window
responses for an hour and answers from them unless the URL changes. Both paths
are checked here by driving the tab: edit, then read the canvases.

Frames are recorded as distinct flat greys, so a canvas says which frame it is
showing and a stale window is visible as the wrong grey.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")

import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright

CAM = "observation.images.top"
H, W = 120, 160
FPS = 10
FRAMES = 40
EPISODES = 3


def grey(ep: int, i: int) -> int:
    """A frame's own shade: distinct across episodes and frames, and inside the
    byte a video frame can carry."""
    return 20 + 70 * ep + 2 * i


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def dataset_root(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "edits"
    ds = LeRobotDataset.create(
        repo_id="tests/edits",
        fps=FPS,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            CAM: {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
        },
        use_videos=True,
    )
    for ep in range(EPISODES):
        for i in range(FRAMES):
            ds.add_frame(
                {
                    "observation.state": np.array([ep, i], np.float32),
                    "action": np.array([i, ep], np.float32),
                    "task": "edits",
                    CAM: np.full((H, W, 3), grey(ep, i), np.uint8),
                }
            )
        ds.save_episode()
    ds.finalize()
    return root


@pytest.fixture
def gui(dataset_root, tmp_path):
    import os

    from lerobot.gui import server as gui_server_mod

    prev = {k: os.environ.get(k) for k in ("LEROBOT_GUI_CONFIG_DIR", "LEROBOT_WINDOW_CACHE_DIR")}
    os.environ["LEROBOT_GUI_CONFIG_DIR"] = str(tmp_path / "config")
    os.environ["LEROBOT_WINDOW_CACHE_DIR"] = str(tmp_path / "wcache")
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
  const c = document.querySelector('canvas.frame-canvas');
  const px = c.getContext('2d').getImageData(c.width >> 1, c.height >> 1, 1, 1).data;
  return { frame: window.currentFrame, total: window.totalFrames, grey: px[0],
           episode: window.__windowPlayer.episode(), url: window.__windowPlayer.lastWindowUrl() };
}"""


def _open(page, base, did, episode=0, frames=FRAMES):
    page.asked = []
    page.on("request", lambda r: page.asked.append(r.url) if "/window?" in r.url else None)
    page.goto(base)
    page.wait_for_function("() => typeof openDataset === 'function' && window.WindowPlayer", timeout=30_000)
    page.evaluate("(ds) => openDataset(ds)", did)
    page.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=did, timeout=60_000)
    page.evaluate("([ds, n]) => selectEpisode(ds, n[0], n[1])", [did, [episode, frames]])
    page.wait_for_function(
        "() => window.__windowPlayer && window.__windowPlayer.metrics.painted.length > 0", timeout=30_000
    )


def _apply(page, did, *, trim=None, delete=None):
    """Stage one edit and apply it, the way the bottom bar does."""
    page.evaluate(
        """async ([ds, trim, del_]) => {
            window.confirm = () => true;
            if (trim) {
                await fetch('/api/edits/trim', {method: 'POST', headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({dataset_id: ds, episode_index: trim[0], start_frame: trim[1], end_frame: trim[2]})});
            }
            if (del_ !== null) {
                await fetch('/api/edits/delete', {method: 'POST', headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({dataset_id: ds, episode_index: del_})});
            }
            await window.applyEdits();
        }""",
        [did, trim, delete],
    )
    page.wait_for_timeout(3000)


def _pin(page, rung="160"):
    """Hold the rung still, so a window's URL is the same before and after an
    edit but for what the edit changes. With the rung free to move, two asks
    for the same frames carry different URLs anyway and prove nothing about
    the browser's cache."""
    page.evaluate("(r) => window.__windowPlayer.rung(r)", rung)
    page.evaluate("() => window.__windowPlayer.masksChanged()")  # drop what was fetched at another rung
    page.evaluate("() => window.__windowPlayer.seek(0)")
    page.evaluate("() => window.__windowPlayer.play()")
    page.wait_for_timeout(700)
    page.evaluate("() => window.__windowPlayer.pause()")
    page.evaluate("() => window.__windowPlayer.seek(0)")
    page.wait_for_function(
        "(r) => { const p = window.__windowPlayer.metrics.painted.slice(-1)[0];"
        "        return p && p.frame === 0 && p.rung === r; }",
        arg=rung,
        timeout=30_000,
    )


def test_a_trim_is_visible_in_the_picture_at_once(gui):
    """After trimming an episode's first ten frames, frame 0 is the eleventh
    recorded frame. Every other term of the window's URL is the same as before
    the trim, so without the dataset's generation in it the browser answers
    from the window it cached and the operator watches the frames that were
    cut."""
    base, did = gui
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, base, did, episode=1)
        _pin(page)
        before = page.evaluate(PROBE)
        assert before["grey"] == pytest.approx(grey(1, 0), abs=8), before
        assert before["total"] == FRAMES

        _apply(page, did, trim=(1, 10, FRAMES), delete=None)
        page.wait_for_function(f"() => window.totalFrames === {FRAMES - 10}", timeout=30_000)
        page.wait_for_function(
            "() => window.__windowPlayer.metrics.painted.slice(-1)[0].frame === 0", timeout=30_000
        )
        page.wait_for_timeout(400)
        # What the operator sees the moment the edit lands: window 0, fetched
        # beside the bundle before either names the generation. A later fetch
        # would replace it, so this is read before anything else runs.
        first_paint = page.evaluate(PROBE)
        _pin(page)
        after = page.evaluate(PROBE)
        errors = page.evaluate("() => window.__windowPlayer.metrics.errors")
        # A window fetched after the bundle, so its URL carries the version and
        # the generation. (Window 0 goes out before the bundle names either and
        # is fetched past the browser's cache instead; the grey below is what
        # checks that.)
        asked = [u for u in page.asked if "episodes/1/window?start=5&" in u and "rung=160" in u]
        browser.close()

    assert not errors, errors
    assert after["total"] == FRAMES - 10, after
    # The picture is the episode's new first frame, not the one that was cut.
    assert first_paint["grey"] == pytest.approx(grey(1, 10), abs=8), (
        f"the first frame after the trim is the one that was cut: {first_paint}"
    )
    assert after["grey"] == pytest.approx(grey(1, 10), abs=8), after
    assert len(asked) >= 2, asked
    assert asked[0] != asked[-1], f"the same URL was asked before and after the trim: {asked[0]}"


def test_a_delete_renumbers_the_episodes_and_the_picture_follows(gui):
    """Deleting episode 0 makes what was episode 1 into episode 0, so a window
    cached under episode 0 shows frames that no longer exist."""
    base, did = gui
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, base, did, episode=0)
        _pin(page)
        before = page.evaluate(PROBE)
        assert before["grey"] == pytest.approx(grey(0, 0), abs=8), before

        _apply(page, did, trim=None, delete=0)
        page.wait_for_function("(ds) => window.datasets[ds].total_episodes === 2", arg=did, timeout=30_000)
        page.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [did, FRAMES])
        page.wait_for_function(
            "() => window.__windowPlayer.episode() === 0 && window.__windowPlayer.metrics.painted.length > 0",
            timeout=30_000,
        )
        _pin(page)
        after = page.evaluate(PROBE)
        errors = page.evaluate("() => window.__windowPlayer.metrics.errors")
        asked = [u for u in page.asked if "episodes/0/window?start=5&" in u and "rung=160" in u]
        browser.close()

    assert not errors, errors
    # Episode 0 is now what was episode 1: its own shade, not the deleted one's.
    assert after["grey"] == pytest.approx(grey(1, 0), abs=8), after
    assert len(asked) >= 2 and asked[0] != asked[-1], asked


def test_the_server_drops_its_cached_windows_on_an_edit(gui, tmp_path):
    """The other half, and the one that does not depend on a URL: the windows
    the server built before the edit are dropped from its disk cache, so it
    rebuilds rather than serving them. (That the key itself carries the
    generation is checked in test_window_units.py, where no invalidation runs
    to hide it.)"""
    import requests

    base, did = gui
    url = f"{base}/api/datasets/{did}/episodes/1/window?start=0&len=0.5&rung=320&codec=h264&rc=cbr&masks=none"
    first = requests.get(url, timeout=60)
    assert first.status_code == 200 and first.headers["x-window-cache"] == "miss"
    assert requests.get(url, timeout=60).headers["x-window-cache"] == "hit"

    cache = tmp_path / "wcache"
    assert list(cache.glob("*.bin")), "nothing was cached"
    requests.post(
        f"{base}/api/edits/trim",
        json={"dataset_id": did, "episode_index": 1, "start_frame": 10, "end_frame": FRAMES},
        timeout=30,
    )
    assert (
        requests.post(f"{base}/api/edits/apply", params={"dataset_id": did}, timeout=300).status_code == 200
    )

    after = requests.get(url, timeout=60)
    assert after.status_code == 200
    assert after.headers["x-window-cache"] == "miss", "a window built before the trim was served after it"
    assert after.content != first.content, "the trimmed episode returned the same bytes"
