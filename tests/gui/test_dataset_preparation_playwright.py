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
"""Browser guardrails for the HVLA dataset-preparation entry point."""

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
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def gui_url(monkeypatch):
    from lerobot.gui import server as gui_server
    from lerobot.gui.api import datasets as datasets_api

    # Keep this browser test away from the operator's persisted datasets.
    monkeypatch.setattr(datasets_api, "_read_opened", lambda: [])
    monkeypatch.setattr(datasets_api, "_read_sources", lambda: [])

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(gui_server.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import requests

    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if requests.get(url, timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        server.should_exit = True
        pytest.fail("GUI server did not become ready within 15 seconds")

    yield url

    server.should_exit = True
    thread.join(timeout=10)


def test_prepare_button_tracks_source_availability(gui_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.route("**/api/datasets/sources", lambda route: route.fulfill(json=[]))
        page.goto(gui_url)
        page.wait_for_function("typeof switchTab === 'function'")
        page.evaluate("switchTab('preprocess')")
        page.wait_for_selector("#prep-source-select option", state="attached")

        assert page.locator("#prep-start-btn").is_disabled()
        assert page.input_value("#prep-output-repo") == ""

        page.unroute("**/api/datasets/sources")
        page.route(
            "**/api/datasets/sources",
            lambda route: route.fulfill(json=[{"path": "/tmp/hvla-browser-source"}]),
        )
        page.route(
            "**/api/datasets/sources/*/datasets",
            lambda route: route.fulfill(
                json=[
                    {
                        "root": "/tmp/hvla-browser-source/demo/source",
                        "name": "demo/source",
                        "total_episodes": 2,
                        "fps": 30,
                    }
                ]
            ),
        )
        page.evaluate("refreshPrepSources()")
        page.wait_for_function("document.getElementById('prep-source-select').value !== ''")

        assert page.locator("#prep-start-btn").is_enabled()
        assert page.input_value("#prep-output-repo") == "demo/source_hvla224"
        browser.close()
