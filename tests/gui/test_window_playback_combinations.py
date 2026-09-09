"""Windowed playback crossed with the controls that sit beside it.

The window player replaced the per-frame JPEG path, so every control that used
to act on a still now acts on a decoding pipeline with a buffer, a rung ladder
and a clock of its own. Playing through windows is covered next door; what is
covered here is playing *while something else changes underneath it* -- the
speed selector, a different dataset, and an episode switch made at a speed
other than 1x.

Frames are recorded as distinct flat greys and the two datasets use bands that
cannot be confused, so a canvas identifies which dataset, episode and frame is
on screen rather than merely that something painted.
"""

from __future__ import annotations

import socket
import threading
import time
from urllib.parse import unquote

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")

import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright

CAMS = ["observation.images.a", "observation.images.b"]
H, W = 240, 480
FPS = 10
FRAMES = 60


def grey_a(ep: int, i: int) -> int:
    """Dataset A: two episodes, low band."""
    return 20 + 50 * ep + 2 * i


def grey_b(i: int) -> int:
    """Dataset B: one episode, high band, disjoint from A's."""
    return 190 + i


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build(root, episodes, shade, masked: bool):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(
        repo_id=f"tests/{root.name}",
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
    for ep in range(episodes):
        for i in range(FRAMES):
            ds.add_frame(
                {
                    "observation.state": np.array([ep, i], np.float32),
                    "action": np.array([i, ep], np.float32),
                    "task": "combo",
                    **{c: np.full((H, W, 3), shade(ep, i), np.uint8) for c in CAMS},
                }
            )
        ds.save_episode()
    ds.finalize()

    if masked:
        from lerobot.datasets.mask_store import adopt, write_episode

        ds = LeRobotDataset(f"tests/{root.name}", root=root)
        adopt(
            ds,
            [CAMS[0]],
            ["ball"],
            (H, W),
            treatments={"ball": {"key": "tint", "params": {"color": [255, 0, 0]}}},
        )
        blob = np.zeros((H, W), bool)
        blob[40:160, 40:240] = True
        for ep in range(episodes):
            write_episode(ds, ep, CAMS[0], [{"ball": blob} for _ in range(FRAMES)])
    return root


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    base = tmp_path_factory.mktemp("combo")
    a = _build(base / "combo_a", 2, lambda ep, i: grey_a(ep, i), masked=True)
    b = _build(base / "combo_b", 1, lambda ep, i: grey_b(i), masked=False)
    return a, b


@pytest.fixture(scope="module")
def server(datasets, tmp_path_factory):
    import os

    import requests

    from lerobot.gui import server as gui_server_mod

    prev = {k: os.environ.get(k) for k in ("LEROBOT_GUI_CONFIG_DIR", "LEROBOT_WINDOW_CACHE_DIR")}
    os.environ["LEROBOT_GUI_CONFIG_DIR"] = str(tmp_path_factory.mktemp("config"))
    os.environ["LEROBOT_WINDOW_CACHE_DIR"] = str(tmp_path_factory.mktemp("wcache"))
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            if requests.get(base, timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        srv.should_exit = True
        pytest.fail("GUI server did not become ready")

    ids = []
    for root in datasets:
        r = requests.post(f"{base}/api/datasets", json={"local_path": str(root)}, timeout=120)
        assert r.status_code == 200, r.text
        ids.append(r.json()["id"])
    yield base, ids[0], ids[1]
    srv.should_exit = True
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


PROBE = """() => {
  const out = { frame: window.currentFrame, playing: window.__windowPlayer.playing(),
                rate: window.__windowPlayer.rate(), cams: {} };
  for (const c of document.querySelectorAll('canvas.frame-canvas')) {
    const ctx = c.getContext('2d');
    const at = (fx, fy) => ctx.getImageData(Math.floor(fx * c.width), Math.floor(fy * c.height), 1, 1).data;
    // The mask blob covers rows 40..160, columns 40..240 of 240x480.
    const centre = at(0.8, 0.85), blob = at(0.2, 0.4);
    out.cams[c.dataset.cam] = { centre: [...centre].slice(0, 3), blob: [...blob].slice(0, 3) };
  }
  return out;
}"""


def _open(page, base, ds, episode=0):
    page.goto(base)
    page.wait_for_function("() => typeof openDataset === 'function' && window.WindowPlayer", timeout=30_000)
    page.evaluate("(d) => openDataset(d)", ds)
    page.wait_for_function("(d) => window.datasets && window.datasets[d]", arg=ds, timeout=90_000)
    page.evaluate("([d, n]) => selectEpisode(d, e_, n)".replace("e_", str(episode)), [ds, FRAMES])
    page.wait_for_function(
        "() => window.__windowPlayer && window.__windowPlayer.metrics.painted.length > 0", timeout=60_000
    )


def _advance_per_second(painted):
    """Frames the player actually moved through, over the time it took.

    Counted as changes between consecutive paints so a repeated paint of one
    frame is not read as progress, and so a wrap costs one change rather than a
    negative jump."""
    moves = sum(1 for x, y in zip(painted, painted[1:], strict=False) if y["frame"] != x["frame"])
    span = (painted[-1]["t"] - painted[0]["t"]) / 1000.0
    return moves / span if span > 0 else 0.0, moves, span


@pytest.mark.parametrize("speed", ["0.5", "2"])
def test_the_speed_selector_reaches_the_player(server, speed):
    """The selector is the operator's control; `rate` is what the player and the
    rung rule read. A selector wired to a variable the player never sees looks
    identical on screen until the buffer is sized for the wrong speed."""
    base, ds_a, _ = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, base, ds_a)
        page.select_option("#speed-select", speed)
        seen = page.evaluate(
            "() => ({ player: window.__windowPlayer.rate(), select: document.getElementById('speed-select').value })"
        )
        browser.close()
    assert seen["select"] == speed, seen
    assert seen["player"] == float(speed), f"the selector says {speed} and the player is at {seen['player']}"


def test_a_faster_speed_moves_the_picture_faster(server):
    """Both speeds are measured in the same browser on the same machine, so the
    comparison holds on a loaded runner where an absolute frame rate would not.
    The complement matters as much as the claim: a player that ignores the
    selector paints at one rate for both and fails here."""
    base, ds_a, _ = server
    samples = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, base, ds_a)
        # Let the ladder settle before timing, so a rung change is not read as speed.
        page.wait_for_timeout(1500)
        for speed in ("0.5", "2"):
            page.select_option("#speed-select", speed)
            page.evaluate("() => { window.__windowPlayer.metrics.painted.length = 0; }")
            page.evaluate("() => { if (!window.__windowPlayer.playing()) togglePlay(); }")
            page.wait_for_timeout(4000)
            page.evaluate("() => { if (window.__windowPlayer.playing()) togglePlay(); }")
            painted = page.evaluate("() => window.__windowPlayer.metrics.painted")
            samples[speed] = _advance_per_second(painted)
            page.wait_for_timeout(300)
        errors = page.evaluate("() => window.__windowPlayer.metrics.errors")
        browser.close()

    assert not errors, errors
    slow_fps, slow_moves, slow_span = samples["0.5"]
    fast_fps, fast_moves, fast_span = samples["2"]
    assert slow_moves > 3, f"the slow pass barely moved, nothing to compare: {samples}"
    assert fast_moves > 3, f"the fast pass barely moved, nothing to compare: {samples}"
    # True ratio is 4; a floor of 2 leaves room for a starved machine while
    # still failing a player that paints at one speed whatever the selector says.
    assert fast_fps > slow_fps * 2, (
        f"2x advanced {fast_fps:.1f} frames/s and 0.5x advanced {slow_fps:.1f}; "
        f"the selector is not reaching the clock ({samples})"
    )


def test_a_dataset_switch_while_playing_lands_on_the_new_dataset(server):
    """A switch mid-play has to retarget the whole pipeline: the clock, the
    bundle, the windows in flight and the tiles. The greys of the two datasets
    do not overlap, so a tile still showing the old dataset is visible here
    rather than merely suspected."""
    base, ds_a, ds_b = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, base, ds_a, episode=1)
        page.evaluate("() => { if (!window.__windowPlayer.playing()) togglePlay(); }")
        page.wait_for_function("() => window.__windowPlayer.metrics.painted.length > 4", timeout=60_000)
        before = page.evaluate(PROBE)

        page.evaluate("(d) => openDataset(d)", ds_b)
        page.wait_for_function("(d) => window.datasets && window.datasets[d]", arg=ds_b, timeout=90_000)
        page.evaluate("([d, n]) => selectEpisode(d, 0, n)", [ds_b, FRAMES])
        page.wait_for_function(
            "(d) => (window.__windowPlayer.lastWindowUrl() || '').includes(encodeURIComponent(d))",
            arg=ds_b,
            timeout=60_000,
        )
        page.wait_for_timeout(1200)
        after = page.evaluate(PROBE)
        url = page.evaluate("() => window.__windowPlayer.lastWindowUrl()")
        errors = page.evaluate("() => window.__windowPlayer.metrics.errors")
        browser.close()

    assert not errors, errors
    a, b = CAMS
    # Before: dataset A episode 1, whose band is 70..188 on camera b.
    lo, hi = grey_a(1, 0) - 10, grey_a(1, FRAMES - 1) + 10
    assert lo <= before["cams"][b]["centre"][0] <= hi, f"the tile was not showing dataset A: {before}"
    # After: dataset B's band, which A cannot produce on any episode or frame.
    lo, hi = grey_b(0) - 10, grey_b(FRAMES - 1) + 10
    assert lo <= after["cams"][b]["centre"][0] <= hi, f"the tile still shows the old dataset: {after}"
    assert lo <= after["cams"][a]["centre"][0] <= hi, f"one tile lagged the switch: {after}"
    # B has no masks, so nothing tints camera a any more.
    assert abs(after["cams"][a]["blob"][0] - after["cams"][a]["blob"][1]) < 14, (
        f"the old dataset's mask recipe is still being composited: {after}"
    )
    # The path is percent-encoded into the window URL.
    decoded = unquote(url)
    assert ds_b in decoded and ds_a not in decoded, decoded


def test_the_saved_masks_survive_an_episode_switch_at_a_faster_speed(server):
    """Speed, the mask composite and an episode switch in one pass. Each is
    covered alone; the combination is where the retarget happens while the
    buffer holds composited windows built for the episode being left."""
    base, ds_a, _ = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, base, ds_a, episode=0)
        page.wait_for_function(
            "() => (window.__windowPlayer.lastWindowUrl() || '').includes('masks=composited')",
            timeout=60_000,
        )
        page.select_option("#speed-select", "2")
        page.evaluate("() => { if (!window.__windowPlayer.playing()) togglePlay(); }")
        page.wait_for_function("() => window.__windowPlayer.metrics.painted.length > 6", timeout=60_000)

        page.evaluate("([d, n]) => selectEpisode(d, 1, n)", [ds_a, FRAMES])
        page.wait_for_function(
            "() => window.__windowPlayer.episode() === 1 && window.currentEpisode === 1", timeout=60_000
        )
        page.wait_for_timeout(1200)
        after = page.evaluate(PROBE)
        errors = page.evaluate("() => window.__windowPlayer.metrics.errors")
        browser.close()

    assert not errors, errors
    a, b = CAMS
    assert after["rate"] == 2.0, f"the switch reset the speed: {after}"
    lo, hi = grey_a(1, 0) - 10, grey_a(1, FRAMES - 1) + 10
    assert lo <= after["cams"][b]["centre"][0] <= hi, f"the picture did not follow the switch: {after}"
    # The tint is still on camera a and still only on camera a.
    assert after["cams"][a]["blob"][0] - after["cams"][a]["blob"][1] > 60, (
        f"the mask composite was lost across the switch: {after}"
    )
    assert abs(after["cams"][b]["blob"][0] - after["cams"][b]["blob"][1]) < 14, (
        f"a camera with no masks was tinted: {after}"
    )


def test_changing_the_speed_while_playing_takes_effect_without_a_pause(server):
    """Starting playback applies the speed on its way in, so a selector wired to
    nothing still looks right as long as the operator sets it before pressing
    Play. The only moment the two paths differ is a change made mid-play, which
    is also the way the control is used: watch, decide it is too slow, change
    it. The player must not stop to take the change."""
    base, ds_a, _ = server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open(page, base, ds_a)
        page.wait_for_timeout(1500)
        page.select_option("#speed-select", "0.5")
        page.evaluate("() => { if (!window.__windowPlayer.playing()) togglePlay(); }")
        page.evaluate("() => { window.__windowPlayer.metrics.painted.length = 0; }")
        page.wait_for_timeout(3500)
        slow = _advance_per_second(page.evaluate("() => window.__windowPlayer.metrics.painted"))

        # No Play press here: the selector alone has to carry it.
        page.select_option("#speed-select", "2")
        stayed_playing = page.evaluate("() => window.__windowPlayer.playing()")
        page.evaluate("() => { window.__windowPlayer.metrics.painted.length = 0; }")
        page.wait_for_timeout(3500)
        fast = _advance_per_second(page.evaluate("() => window.__windowPlayer.metrics.painted"))
        page.evaluate("() => { if (window.__windowPlayer.playing()) togglePlay(); }")
        errors = page.evaluate("() => window.__windowPlayer.metrics.errors")
        browser.close()

    assert not errors, errors
    assert stayed_playing, "the speed change stopped playback"
    assert slow[1] > 3 and fast[1] > 3, f"one leg barely moved, nothing to compare: {slow} {fast}"
    assert fast[0] > slow[0] * 2, (
        f"after changing to 2x mid-play the picture advanced {fast[0]:.1f} frames/s, "
        f"against {slow[0]:.1f} frames/s at 0.5x; the change did not reach the running clock"
    )
