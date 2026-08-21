# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Playback must survive the end of the clip.

`_followVideoClock` maps the <video> clock onto frame indices and loops at the
end of the trim range, and it used to do both only inside
`requestVideoFrameCallback`. That callback stops firing at end of stream, and
the final frame's callback is regularly missed: the loop then died one frame
short of its own wrap test, leaving the episode frozen with isPlaying still
true and the button still reading Pause. Observed on a 120-frame episode as
`ended` arriving while the playhead read 118 — reported from the field as
"sometimes playback gets stuck at the end of the episode".

What is pinned here is that the loop subscribes to `ended` at all, and
exactly once per element — the defect was that nothing did, so the only path
back was a callback that had already stopped coming. The wrap's arithmetic
cannot be driven from a test: `isPlaying` and `currentFrame` are module-scoped
and only real playback sets them, so that half was verified by reproduction
against the running server (six consecutive plays to the end, no stall, with
`ended` at frame 118 followed by the clip restarting).
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def page():
    from lerobot.gui import server as gui_server_mod

    port = _free_port()
    config = uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import requests

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if requests.get(base_url, timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        pytest.fail("GUI server did not come up")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.goto(base_url)
        pg.wait_for_function("typeof _followVideoClock === 'function'", timeout=10_000)
        yield pg
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


#: A <video> stand-in that records its listeners. It never presents frames,
#: which is the state at end of stream — the loop must still have a way back.
_HARNESS = """
() => {
    const fake = {
        currentTime: 3.9667,
        dataset: {},
        listeners: {},
        addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
        play() { return Promise.resolve(); },
        pause() {},
    };
    window.__fakeVideo = fake;
    window._camVideoEls = () => [fake];
    _followVideoClock();
    return Object.keys(fake.listeners);
}
"""


def test_the_clock_subscribes_to_the_end_of_the_clip(page):
    events = page.evaluate(_HARNESS)
    assert "ended" in events, (
        "nothing listened for end of stream, so a missed final frame callback "
        f"leaves playback frozen with no way back: {events}"
    )


def test_the_end_handler_is_wired_once_per_element(page):
    """`_followVideoClock` runs on every play; stacking listeners would wrap
    once per press and race the seeks against each other."""
    page.evaluate(_HARNESS)
    page.evaluate("() => { _followVideoClock(); _followVideoClock(); }")
    count = page.evaluate("() => window.__fakeVideo.listeners.ended.length")
    assert count == 1, f"{count} end handlers on one element"
