# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Overlays panel in a real browser — the wiring unit tests cannot reach.

The defect this pins: the "Process dataset…" gate lived only at the end of
``renderObjects()``, but typing an object name updates the model and calls
``renderAction()`` WITHOUT re-rendering the rows. So a user who named an object
(with the Background already defaulting to Random) saw the button stay greyed
out, its tooltip telling them to do the thing they had just done — the feature
looked broken until they happened to click a treatment button. No dataset, GPU
or overlay worker is needed: the gate is pure panel state.
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

NAME_SEL = '#overlays-panel .overlays-obj-name[data-i="0"]'
PROC_SEL = "#overlays-panel .overlays-process"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def overlays_gui_server(monkeypatch):
    from lerobot.gui import server as gui_server_mod
    from lerobot.gui.api import datasets as datasets_api

    # Keep the page hermetic: no user-persisted datasets restored on load.
    monkeypatch.setattr(datasets_api, "_read_opened", lambda: [])
    monkeypatch.setattr(datasets_api, "_read_sources", lambda: [])

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
            if requests.get(f"{base_url}/api/overlays/models", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        server.should_exit = True
        pytest.fail("GUI server did not become ready within 15 seconds")

    yield base_url

    server.should_exit = True
    thread.join(timeout=10)


def test_process_button_enables_when_an_object_is_named(overlays_gui_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(overlays_gui_server, wait_until="networkidle")
            page.evaluate(
                "(() => { const p = document.querySelector('#overlays-panel .overlays-picker');"
                " p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); })()"
            )
            page.wait_for_selector(PROC_SEL, timeout=5000)

            # The data tab's Background defaults to Random, so naming one object is the
            # only thing standing between a fresh panel and a runnable job.
            assert (
                page.eval_on_selector(
                    "#overlays-panel .overlays-objrow.bg .overlays-treat-btn.sel", "e => e.dataset.key"
                )
                == "random"
            )
            assert page.eval_on_selector(PROC_SEL, "e => e.disabled") is True, "gate must start closed"

            page.click(NAME_SEL)
            page.type(NAME_SEL, "robot arm", delay=20)
            # Naming alone must open the gate — no row re-render, no other click.
            page.wait_for_function(
                f"() => {{ const b = document.querySelector('{PROC_SEL}'); return b && !b.disabled; }}",
                timeout=5000,
            )
            assert "Apply these per-region treatments" in page.eval_on_selector(PROC_SEL, "e => e.title")

            # And clearing the name must close it again.
            page.fill(NAME_SEL, "")
            page.wait_for_function(
                f"() => {{ const b = document.querySelector('{PROC_SEL}'); return b && b.disabled; }}",
                timeout=5000,
            )
        finally:
            browser.close()
