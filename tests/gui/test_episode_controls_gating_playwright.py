# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Episode controls must be live only while a *record* run is active.

The buttons used to enable on ``isRunning`` alone. During a teleop session
that handed the operator a live "Next episode" whose ``exit_early`` does not
advance an episode — it ends the whole run — and the N hotkey did the same
invisibly. Greying out per run-kind is the existing mechanism; this pins that
it is strictly executed for every kind, not just record.
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
        pg.wait_for_function("typeof switchTab === 'function'", timeout=10_000)
        pg.evaluate("switchTab('run')")
        pg.wait_for_selector("#run-ctrl-next", timeout=10_000)
        # These tests inject run state via updateRunUI(); the 1 Hz backstop
        # poll would overwrite it with the server's idle state mid-assert.
        pg.wait_for_function("window._runStatusPollTimer !== undefined", timeout=10_000)
        pg.evaluate("clearInterval(window._runStatusPollTimer)")
        yield pg
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


# (isRunning, command) -> may the operator touch episode controls?
CASES = [
    (True, "record", True),
    (True, "teleoperate", False),  # the field failure: live button during teleop
    (True, "replay", False),
    (True, "hvla", False),
    (False, "record", False),  # stale command after exit must not enable
    (False, None, False),
]


@pytest.mark.parametrize(("running", "command", "expected"), CASES)
def test_buttons_track_run_kind(page, running, command, expected):
    page.evaluate(f"updateRunUI({str(running).lower()}, {command!r})".replace("None", "null"))
    for btn in ("#run-ctrl-next", "#run-ctrl-rerecord"):
        enabled = not page.is_disabled(btn)
        assert enabled == expected, (
            f"{btn} enabled={enabled} for (running={running}, command={command}); expected enabled={expected}"
        )


def test_hotkey_is_gated_by_run_kind_not_just_running(page):
    """N during a teleop run must send nothing — it would end the session."""
    page.evaluate("window._controlCalls = []")
    page.evaluate(
        "window.fetch = new Proxy(window.fetch, { apply: (t, self, args) => {"
        "  if (String(args[0]).includes('/api/run/control')) {"
        "    window._controlCalls.push(String(args[0]));"
        "    return Promise.resolve(new Response('{}', {status: 200}));"
        "  }"
        "  return Reflect.apply(t, self, args);"
        "}})"
    )

    page.evaluate("updateRunUI(true, 'teleoperate')")
    page.keyboard.press("n")
    page.wait_for_timeout(200)
    assert page.evaluate("window._controlCalls.length") == 0, (
        "the N hotkey fired during a teleop run — exit_early would have ended the session"
    )

    page.evaluate("updateRunUI(true, 'record')")
    page.keyboard.press("n")
    page.wait_for_timeout(200)
    assert page.evaluate("window._controlCalls.length") == 1, (
        "the N hotkey stopped working during a record run"
    )
