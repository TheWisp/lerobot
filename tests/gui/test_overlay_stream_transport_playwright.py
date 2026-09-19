# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The composited overlay preview owns the tiles; the rest must follow it.

Reported from the rig as "playback is broken as it shows overlay out of sync".
While this stream plays, the server composites every selected camera into one
H.264 atlas and the page slices it onto per-camera canvases -- so no still is
fetched, and nothing in the still path moves the playhead. The stream reports
where it has reached by calling ``window.__streamSetPlayhead``, which was
called here and defined nowhere: the picture advanced and the frame readout,
the timeline, the saved-mask layer and the feature rows all stayed at the frame
play started on.

Its own runtime check could not see this either. ``assertTransport`` reads
``window.__streamIsPlaying ? window.__streamIsPlaying() : true`` -- also never
defined -- so the invariant "the stream runs while the transport reports
paused" compared the stream against the constant ``true`` and could not fail.
An observation that is a constant is not an observation, so this file asserts
the supply of state as well as the state.

Only the SAM3 worker is faked, at the three seams the endpoint reaches it
through. Everything else is real: the endpoint, ffmpeg, the fragmented MP4, the
MediaSource pipeline and the canvases.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import TimeoutError as PWTimeout, sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright

H, W = 48, 64
FPS = 30  # the stream's encoder rate, so stream time IS episode time
FRAMES = 150  # five seconds at 1x
CAMS = ["observation.images.top", "observation.images.wrist"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeWorker:
    """What the endpoint needs of the SAM3 subprocess: a camera list, a
    per-camera sequence number that moves when a frame is published, and an
    overlay to composite.

    The sequence is the endpoint's own handshake -- it publishes a frame and
    waits for every camera's ``overlay_seq`` to move before compositing -- so a
    fake that never moves it stalls the stream for five seconds and then gives
    up. Returning no overlay leaves the dataset's own pixels in the atlas,
    which is what the frame assertions read.
    """

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


@pytest.fixture
def dataset_root(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "stream"
    feats = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    for cam in CAMS:
        feats[cam] = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id="tests/stream", fps=FPS, root=root, features=feats, use_videos=True)
    img = np.full((H, W, 3), 100, np.uint8)
    for _ in range(FRAMES):
        ds.add_frame(
            {
                "observation.state": np.zeros(2, np.float32),
                "action": np.zeros(2, np.float32),
                "task": "stream",
                **dict.fromkeys(CAMS, img),
            }
        )
    ds.save_episode()
    ds.finalize()
    return root


@pytest.fixture
def page(dataset_root, monkeypatch):
    from lerobot.gui import server as gui_server_mod
    from lerobot.gui.api import overlays as ovl

    worker = _FakeWorker(CAMS)
    monkeypatch.setattr(ovl, "_get_live_reader", lambda: worker)
    monkeypatch.setattr(ovl, "_data_publisher_active", lambda: True)
    monkeypatch.setattr(ovl, "publish_data_frame", lambda *a, **k: worker.published())

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
        pg.warnings = []
        pg.on(
            "console", lambda m: pg.warnings.append(m.text[:160]) if m.type in ("warning", "error") else None
        )
        pg.goto(base)
        pg.wait_for_function("typeof openDataset === 'function' && window.OverlayStream", timeout=15_000)
        ds_id = str(dataset_root)
        pg.evaluate("(ds) => openDataset(ds)", ds_id)
        pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
        pg.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [ds_id, FRAMES])
        pg.wait_for_function(
            "() => document.querySelectorAll('img[id^=\"frame-\"]').length > 0", timeout=30_000
        )
        pg.ds_id = ds_id
        yield pg
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


# What the stream's own clock says, beside what the app believes. Read together
# in one evaluation so the two cannot be a frame apart because of the sampling.
PROBE = """() => {
  const d = window.OverlayStream._debug();
  const streamFrame = (d.layout && d.ct !== null)
      ? d.layout.from_frame + Math.floor(d.ct * d.layout.fps) : null;
  return {
    streaming: d.streaming, started: d.started, ct: d.ct, cams: d.cams,
    streamFrame,
    playhead: window.currentFrame,
    readout: (document.getElementById('frame-info') || {}).textContent || '',
    progress: (document.getElementById('timeline-progress') || {style: {}}).style.width,
    playBtn: (document.getElementById('play-btn') || {}).textContent || '',
    isPlaying: window.__streamIsPlaying ? window.__streamIsPlaying() : null,
    hasSetPlayhead: typeof window.__streamSetPlayhead,
  };
}"""


DIAGNOSIS = """() => ({
  debug: window.OverlayStream._debug(),
  mseH264: !!window.MediaSource && MediaSource.isTypeSupported('video/mp4; codecs="avc1.42C01E"'),
})"""


def _play(pg):
    """Start the composited stream and wait for it to be showing pictures.

    Waited in two stages, because one wait on "playing" times out identically
    whether the endpoint refused the request, the encoder produced nothing, or
    the browser cannot decode what it sent -- and a bare timeout names none of
    them. The first of those actually happened: another test left the aux-GPU
    slot held, every request here was answered 409, and `start` returned
    without a word.
    """
    pg.evaluate("() => window.OverlayStream.start()")
    try:
        pg.wait_for_function("() => window.OverlayStream._debug().streaming", timeout=15_000)
    except PWTimeout:
        raise AssertionError(
            f"the stream never started -- the request was refused or the codec is missing. "
            f"{pg.evaluate(DIAGNOSIS)}; console: {pg.warnings[-6:]}"
        ) from None
    try:
        pg.wait_for_function(
            "() => { const d = window.OverlayStream._debug(); return d.started && d.ct > 0.2; }",
            timeout=45_000,
        )
    except PWTimeout:
        raise AssertionError(
            f"the stream started but never decoded a picture. "
            f"{pg.evaluate(DIAGNOSIS)}; console: {pg.warnings[-6:]}"
        ) from None


def _samples(pg, n=6, gap_ms=450):
    out = []
    for _ in range(n):
        out.append(pg.evaluate(PROBE))
        pg.wait_for_timeout(gap_ms)
    return out


def test_the_playhead_follows_the_picture_while_the_stream_plays(page):
    """The reported desync. The tiles are painted from the stream's own clock,
    so the frame readout, the timeline and every module that follows the
    playhead have to be told where it has reached -- nothing else will."""
    pg = page
    _play(pg)
    obs = _samples(pg)
    pg.evaluate("() => window.OverlayStream.stop()")

    assert all(s["streaming"] for s in obs), f"the stream stopped mid-test: {obs}"
    assert all(len(s["cams"]) == len(CAMS) for s in obs), f"not every camera got a canvas: {obs}"

    # It moved: an assertion that the two agree is satisfied by both being stuck.
    moved = obs[-1]["streamFrame"] - obs[0]["streamFrame"]
    assert moved >= 10, f"the stream itself did not advance, so nothing is under test: {obs}"

    lag = [s["streamFrame"] - s["playhead"] for s in obs]
    assert all(abs(d) <= 1 for d in lag), (
        f"the playhead is not tracking the picture; stream-minus-playhead {lag}: {obs}"
    )

    # And the readouts the operator actually looks at moved with it.
    assert obs[0]["readout"] != obs[-1]["readout"], f"the frame readout never changed: {obs}"
    assert obs[0]["progress"] != obs[-1]["progress"], f"the timeline never moved: {obs}"


def test_the_transport_reports_playing_while_the_stream_runs(page):
    """The stream's runtime check asks the app whether it is playing. With
    nothing to ask, it substituted `true` -- so its "running while paused"
    invariant compared the stream against a constant and could never fire.
    Pin the supply, not just the value: the check is only worth its assertion
    above if what it observes is real."""
    pg = page
    _play(pg)
    obs = pg.evaluate(PROBE)
    pg.evaluate("() => window.OverlayStream.stop()")
    stopped = pg.evaluate(PROBE)

    assert obs["isPlaying"] is True, f"the stream is running and the app does not report playing: {obs}"
    assert "Pause" in obs["playBtn"], f"the transport button does not offer Pause: {obs}"
    # The complement, without which "isPlaying is True" is satisfied by a stub
    # that always says True -- which is exactly the defect.
    assert stopped["isPlaying"] is False, f"the app still reports playing after the stream stopped: {stopped}"
    assert "Play" in stopped["playBtn"], f"the button did not go back to Play: {stopped}"


def test_stopping_leaves_the_playhead_where_the_picture_reached(page):
    """Pause is a hand-off: the still path takes the tiles back and must take
    them back at the frame that was on screen. A playhead that is written while
    streaming but reset on stop looks identical during play and wrong the
    moment you pause."""
    pg = page
    _play(pg)
    pg.wait_for_function(
        "() => { const d = window.OverlayStream._debug(); return d.ct > 1.2; }", timeout=30_000
    )
    at_stop = pg.evaluate(PROBE)
    pg.evaluate("() => window.OverlayStream.stop()")
    pg.wait_for_timeout(800)
    after = pg.evaluate(PROBE)

    assert at_stop["streamFrame"] > 20, f"the stream had not got far enough to tell: {at_stop}"
    assert abs(after["playhead"] - at_stop["streamFrame"]) <= 2, (
        f"pausing at frame {at_stop['streamFrame']} left the playhead at {after['playhead']}"
    )


# ── one publication, two movers ─────────────────────────────────────────────
#
# Moving the playhead and PUBLISHING it were one function, and it lived in the
# still-fetching path -- so only that path could publish, and the composited
# stream had no way to. The narrow repair for that is for the stream to write
# the frame counter itself, which is the same publication in a second place and
# reaches none of the modules that also follow the playhead.
#
# So the check is not "the counter moved". It is that both movers notify the
# SAME set of consumers: an enumeration, compared, rather than a spot check of
# the one readout a band-aid would have fixed.

CONSUMERS = """() => {
  window.__seen = [];
  const spy = (obj, name, key) => {
    if (!obj) return;
    const inner = obj[name];
    obj[name] = function (...a) { window.__seen.push(key); return inner && inner.apply(this, a); };
  };
  spy(window.FeatureEditing, 'onPlayheadChanged', 'FeatureEditing.onPlayheadChanged');
  spy(window.Overlays, 'onFrame', 'Overlays.onFrame');
  spy(window.MaskOverlay, 'onPlayheadChanged', 'MaskOverlay.onPlayheadChanged');
}"""

READOUT = """() => ({
  seen: [...new Set(window.__seen)].sort(),
  frame: window.currentFrame,
  readout: document.getElementById('frame-info').textContent,
  progress: document.getElementById('timeline-progress').style.width,
  scrubber: document.getElementById('timeline-scrubber').style.left,
  time: document.getElementById('time-info').textContent,
})"""


def test_the_stream_publishes_the_playhead_the_same_way_the_still_path_does(page):
    """Enumerate what each mover notifies, and compare.

    A repair that taught the stream to write the frame counter itself would pass
    any assertion about the counter and fail this one, because the modules that
    follow the playhead would hear nothing.
    """
    pg = page

    pg.evaluate(CONSUMERS)
    pg.evaluate("() => window.__seen = []")
    pg.evaluate("() => loadAllFrames(5)")
    pg.wait_for_timeout(500)
    still = pg.evaluate(READOUT)

    pg.evaluate("() => window.__seen = []")
    # Far enough apart that the elapsed-time readout differs too: at 30 fps,
    # frames 5 and 9 both format to 0:00 and would compare equal for the
    # wrong reason.
    pg.evaluate("() => window.__streamSetPlayhead(100)")
    pg.wait_for_timeout(300)
    stream = pg.evaluate(READOUT)

    assert still["seen"], f"the still path notified nobody, so there is nothing to compare: {still}"
    assert stream["seen"] == still["seen"], (
        f"the stream notifies {stream['seen']}, the still path notifies {still['seen']}"
    )
    # And each mover actually moved the playhead it published.
    assert still["frame"] == 5 and stream["frame"] == 100, (still, stream)
    # Every readout the still path writes, the stream writes too -- and to a
    # different value, so "both wrote something" cannot pass by writing nothing.
    for field in ("readout", "progress", "scrubber", "time"):
        assert still[field] and stream[field], f"{field} empty: {still} / {stream}"
        assert still[field] != stream[field], f"{field} did not follow the second mover: {still[field]!r}"
