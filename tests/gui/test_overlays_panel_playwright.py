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


def test_last_row_clears_in_place_instead_of_being_immortal(overlays_gui_server):
    """The x used to be offered only with 2+ rows, so the panel's only row could
    never be cleared with one click — inconsistently, since every OTHER row was.
    Now the x is always there: with several rows it removes, on the last row it
    clears in place (a text row needs an input to type into)."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(overlays_gui_server, wait_until="networkidle")
            page.evaluate(
                "(() => { const p = document.querySelector('#overlays-panel .overlays-picker');"
                " p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); })()"
            )
            page.wait_for_selector(NAME_SEL, timeout=5000)

            rm_sel = '#overlays-panel .overlays-obj-rm[data-i="0"]'
            assert page.eval_on_selector(rm_sel, "e => e.title") == "clear", (
                "the single row must offer x, labelled as a clear"
            )

            page.click(NAME_SEL)
            page.type(NAME_SEL, "robot arm", delay=20)
            page.click(rm_sel)
            page.wait_for_function(
                f"() => {{ const i = document.querySelector('{NAME_SEL}'); return i && i.value === ''; }}",
                timeout=5000,
            )
            # Still exactly one (empty) row — cleared, not deleted.
            assert page.eval_on_selector_all("#overlays-panel .overlays-obj-name", "els => els.length") == 1

            # With a second row, the x removes rather than clears.
            page.click("#overlays-panel .overlays-add-obj")
            page.wait_for_function(
                "() => document.querySelectorAll('#overlays-panel .overlays-obj-name').length === 2",
                timeout=5000,
            )
            assert page.eval_on_selector(rm_sel, "e => e.title") == "remove"
            page.click('#overlays-panel .overlays-obj-rm[data-i="1"]')
            page.wait_for_function(
                "() => document.querySelectorAll('#overlays-panel .overlays-obj-name').length === 1",
                timeout=5000,
            )
        finally:
            browser.close()
