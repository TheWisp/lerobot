# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Play and Pause, driven through the real button while the SAM3 worker's
state changes underneath -- the flow reported as "it plays but fails to pause".

The operator enables SAM3 and presses Play while the model is still loading,
so the still-frame loop starts. The badge then goes live. Pause was routed the
way Play is, by asking whether the overlay was live at that moment, so it went
to the stream module, which had nothing to stop and started a stream instead;
when the server refused that stream the loop ran on under a button that said
Pause, and the refusal went to the console only.

Only the SAM3 worker is faked, at the seams the endpoints reach it through,
and its reported phase is a dial the tests turn: loading, loaded (model up,
not yet bound to a stream), active. The panel, the badge, the transport, the
stream endpoint, ffmpeg and the MediaSource pipeline are real. Every test ends
by asserting the transport's own invariant checks never fired.
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

H, W = 48, 64
FPS, FRAMES = 30, 240  # eight seconds: long enough to play, pause and play again
CAMS = ["observation.images.top"]
PANEL = "#overlays-panel"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeWorker:
    """The seams the endpoints use: cameras, a per-camera sequence that moves
    when a frame is published, a blank overlay, and control/latency stubs."""

    def __init__(self, cams):
        self.cameras = list(cams)
        self._seq = dict.fromkeys(cams, 0)
        self.controls = []

    def overlay_seq(self, cam):
        return self._seq.get(cam, 0)

    def read_overlay(self, cam):
        return None

    def read_latency(self):
        return {"compute_ms": 9.0}

    def write_control(self, block):
        self.controls.append(block)

    def published(self):
        for c in self._seq:
            self._seq[c] += 1


@pytest.fixture
def dataset_root(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "routing"
    feats = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    for cam in CAMS:
        feats[cam] = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id="tests/routing", fps=FPS, root=root, features=feats, use_videos=True)
    img = np.full((H, W, 3), 90, np.uint8)
    for _ in range(FRAMES):
        ds.add_frame(
            {
                "observation.state": np.zeros(2, np.float32),
                "action": np.zeros(2, np.float32),
                "task": "routing",
                **dict.fromkeys(CAMS, img),
            }
        )
    ds.save_episode()
    ds.finalize()
    return root


@pytest.fixture
def gui(dataset_root, monkeypatch):
    """The GUI with a controllable worker. Yields (page, dial); the dial's
    ``phase`` is what the worker reports and ``bound`` whether its frame buffer
    exists (the two facts the badge and the stream endpoint read)."""
    from lerobot.gui import server as gui_server_mod
    from lerobot.gui.api import overlays as ovl
    from lerobot.overlays.overlay_state import Event

    worker = _FakeWorker(CAMS)
    dial = {"phase": "loading", "bound": False}

    class _Proc:
        returncode = None
        pid = 4242

    async def fake_spawn(model, **kw):
        if ovl._live_proc is None:
            ovl._live_model = model
            ovl._live_proc = _Proc()
            ovl._machine(model).fire(Event.START)

    async def fake_teardown():
        ovl._live_proc = None
        ovl._live_model = None
        ovl._machines.clear()

    monkeypatch.setattr(ovl, "_spawn_worker", fake_spawn)
    monkeypatch.setattr(ovl, "_teardown_current", fake_teardown)
    monkeypatch.setattr(ovl, "start_data_publisher", lambda *a, **k: True)
    monkeypatch.setattr(ovl, "_data_publisher_active", lambda: True)
    monkeypatch.setattr(ovl, "publish_data_frame", lambda *a, **k: worker.published())
    monkeypatch.setattr(ovl, "_get_live_reader", lambda: worker if dial["bound"] else None)
    monkeypatch.setattr(ovl, "_proc_sm", lambda pid: 0)
    monkeypatch.setattr(
        ovl,
        "_read_status",
        lambda: {"phase": dial["phase"], "fps": 4.0 if dial["phase"] == "active" else 0.0, "vram": 1.0},
    )
    ovl._live_proc = None
    ovl._live_model = None
    ovl._machines.clear()

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
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
        server.should_exit = True
        pytest.fail("GUI server did not come up")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1600, "height": 1000})
        pg.console = []
        pg.on("console", lambda m: pg.console.append(f"{m.type}: {m.text[:200]}"))
        pg.goto(base)
        pg.wait_for_function("typeof openDataset === 'function' && window.OverlayStream", timeout=15_000)
        ds_id = str(dataset_root)
        pg.evaluate("(ds) => openDataset(ds)", ds_id)
        pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
        pg.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [ds_id, FRAMES])
        pg.wait_for_function(
            "() => document.querySelectorAll('img[id^=\"frame-\"]').length > 0", timeout=30_000
        )
        # SAM3 on, a label named: the panel configures the worker on its own.
        pg.evaluate(
            "(s) => { const p = document.querySelector(s + ' .overlays-picker');"
            " p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); }",
            PANEL,
        )
        pg.wait_for_function(
            "(s) => !!document.querySelector(s + ' .overlays-obj-name')", arg=PANEL, timeout=10_000
        )
        pg.evaluate(
            "(s) => { const r = document.querySelector(s + ' .overlays-obj-name');"
            " r.value = 'robot arm'; r.dispatchEvent(new Event('input', {bubbles: true})); }",
            PANEL,
        )
        _wait_badge(pg, "loading")
        pg.ds_id = ds_id
        yield pg, dial
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)
    ovl._live_proc = None
    ovl._live_model = None
    ovl._machines.clear()


PROBE = """() => ({
  isPlaying: window.__streamIsPlaying(),
  engine: window.__transportEngine(),
  streaming: window.OverlayStream.streaming,
  btn: document.getElementById('play-btn').textContent.trim(),
  frame: window.currentFrame,
  readout: document.getElementById('frame-info').textContent,
  eligible: window.OverlayStream.eligible(),
  badge: document.getElementById('overlays-badge').className.replace('overlays-badge', '').trim()
         + ':' + document.getElementById('overlays-badge').textContent,
  violations: window.__transportViolations.slice(),
  text: document.body.innerText,
})"""


def _probe(pg):
    return pg.evaluate(PROBE)


def _wait_badge(pg, cls, timeout=15_000):
    pg.wait_for_function(
        "(cls) => { const b = document.getElementById('overlays-badge');"
        " return !!b && b.className.split(' ').includes(cls); }",
        arg=cls,
        timeout=timeout,
    )


def _click_play(pg):
    pg.click("#play-btn")


def _wait_frames(pg, n, timeout=10_000):
    """Playback advanced by at least n frames from where it is now."""
    start = pg.evaluate("() => window.currentFrame")
    pg.wait_for_function("(t) => window.currentFrame >= t", arg=start + n, timeout=timeout)


def _assert_paused(pg):
    s = _probe(pg)
    assert (s["isPlaying"], s["engine"], s["streaming"], s["btn"]) == (False, None, False, "▶ Play"), s
    frame = s["frame"]
    pg.wait_for_timeout(700)
    assert _probe(pg)["frame"] == frame, "the playhead moved after Pause"


def _assert_healthy(pg):
    s = _probe(pg)
    assert s["violations"] == [], s["violations"]
    assert not [c for c in pg.console if "[transport]" in c], pg.console[-8:]


def test_pause_stops_a_still_loop_started_while_the_worker_was_loading(gui):
    """The reported flow. Play during 'loading…' plays stills; the worker comes
    up and binds; Pause then stops that loop -- whatever the badge says now."""
    pg, dial = gui
    _click_play(pg)
    _wait_frames(pg, 5)
    s = _probe(pg)
    assert (s["isPlaying"], s["engine"], s["btn"]) == (True, "still", "⏸ Pause"), s

    dial["phase"] = "loaded"
    pg.wait_for_function(
        "() => document.getElementById('overlays-badge').textContent.startsWith('loaded')", timeout=10_000
    )
    assert not _probe(pg)["eligible"], "a worker without a frame buffer must not be offered the stream"
    _wait_frames(pg, 5)  # still playing stills through the phase change

    dial["phase"] = "active"
    dial["bound"] = True
    _wait_badge(pg, "ok")
    assert _probe(pg)["eligible"]
    _wait_frames(pg, 5)  # a badge going live does not hijack a running still loop

    _click_play(pg)  # Pause
    _assert_paused(pg)
    _assert_healthy(pg)


def test_play_after_that_pause_takes_the_stream_now_that_the_worker_is_bound(gui):
    pg, dial = gui
    _click_play(pg)
    _wait_frames(pg, 3)
    dial["phase"] = "active"
    dial["bound"] = True
    _wait_badge(pg, "ok")
    _click_play(pg)  # Pause
    _assert_paused(pg)

    _click_play(pg)  # Play again
    pg.wait_for_function("() => window.OverlayStream.streaming", timeout=15_000)
    s = _probe(pg)
    assert (s["isPlaying"], s["engine"], s["btn"]) == (True, "stream", "⏸ Pause"), s
    _wait_frames(pg, 5, timeout=45_000)  # the picture, hence the playhead, moves
    _click_play(pg)  # Pause the stream
    _assert_paused(pg)
    _assert_healthy(pg)


def test_a_refused_stream_plays_stills_and_says_why(gui):
    """The worker reports active but has no frame buffer (the 503 the operator
    saw). Play must still play, and the refusal must reach the operator."""
    pg, dial = gui
    dial["phase"] = "active"
    dial["bound"] = False
    _wait_badge(pg, "ok")
    assert _probe(pg)["eligible"]

    _click_play(pg)
    _wait_frames(pg, 5)
    s = _probe(pg)
    assert (s["isPlaying"], s["engine"], s["streaming"], s["btn"]) == (True, "still", False, "⏸ Pause"), s
    assert "Live preview did not start" in s["text"], "the refusal never reached the operator"
    assert "no frame buffer" in s["text"], "the server's reason was dropped"

    _click_play(pg)  # Pause
    _assert_paused(pg)
    _assert_healthy(pg)


def test_a_bound_worker_streams_and_pause_stops_the_stream(gui):
    pg, dial = gui
    dial["phase"] = "active"
    dial["bound"] = True
    _wait_badge(pg, "ok")

    _click_play(pg)
    pg.wait_for_function("() => window.OverlayStream.streaming", timeout=15_000)
    s = _probe(pg)
    assert (s["isPlaying"], s["engine"], s["btn"]) == (True, "stream", "⏸ Pause"), s
    assert "Live preview did not start" not in s["text"]
    _wait_frames(pg, 5, timeout=45_000)

    _click_play(pg)  # Pause
    _assert_paused(pg)
    _assert_healthy(pg)


def test_a_scrub_during_the_stream_pauses_onto_the_still(gui):
    pg, dial = gui
    dial["phase"] = "active"
    dial["bound"] = True
    _wait_badge(pg, "ok")
    _click_play(pg)
    pg.wait_for_function("() => window.OverlayStream.streaming", timeout=15_000)

    pg.evaluate("() => seekTimeline(7)")
    pg.wait_for_function("() => !window.OverlayStream.streaming", timeout=5_000)
    s = _probe(pg)
    assert (s["isPlaying"], s["engine"], s["btn"], s["readout"]) == (False, None, "▶ Play", "8 / 240"), s
    _assert_healthy(pg)


def test_an_armed_apply_run_owns_play_and_pause(gui):
    """An armed Apply run drives playback itself (lock-step with the worker), so
    Play must hand it the transport and Pause must stop it -- not the stream,
    not a still loop. The run is stubbed: its own tests live elsewhere."""
    pg, dial = gui
    dial["phase"] = "active"
    dial["bound"] = True
    _wait_badge(pg, "ok")
    pg.evaluate(
        "() => { window.__applyCalls = []; window.Overlays.applyArmed = () => true;"
        " window.Overlays.applyOnTransport = (p) => window.__applyCalls.push(p); }"
    )
    _click_play(pg)
    s = _probe(pg)
    assert (s["isPlaying"], s["engine"], s["streaming"], s["btn"]) == (True, "apply", False, "⏸ Pause"), s
    pg.wait_for_timeout(600)
    assert _probe(pg)["frame"] == s["frame"], "the app must not also run its own still loop"
    assert pg.evaluate("() => window.__applyCalls") == [True]

    _click_play(pg)  # Pause
    _assert_paused(pg)
    assert pg.evaluate("() => window.__applyCalls") == [True, False]

    # A run that reaches the episode's end hands the transport back.
    _click_play(pg)
    assert _probe(pg)["engine"] == "apply"
    pg.evaluate("() => window.__transportEngineEnded('apply')")
    _assert_paused(pg)
    _assert_healthy(pg)


def test_the_badge_is_not_live_while_the_worker_is_only_loaded(gui):
    pg, dial = gui
    dial["phase"] = "loaded"
    pg.wait_for_function(
        "() => document.getElementById('overlays-badge').textContent.startsWith('loaded')", timeout=10_000
    )
    s = _probe(pg)
    assert s["badge"].startswith("loading:") and "waiting for frames" in s["badge"], s["badge"]
    assert not s["eligible"]
