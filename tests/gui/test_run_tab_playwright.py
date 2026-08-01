# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Playwright smoke for the Run tab's launch-mode state.

The Dataset selector is the record/teleop mode switch: "None (pure teleop)"
launches ``lerobot-teleoperate``, anything else launches ``lerobot-record``.
That makes its value safety-relevant — if anything but the user moves it, the
next Launch silently runs the wrong tool.

That happened: ``refreshRunDatasetSelects()`` (called whenever the dataset
list changes, including right after a run ends) used to reset a "+ New
dataset..." selection back to "None" when it couldn't find the typed name
among opened datasets. It never could — repo_ids are date-stamped at
creation — so an operator who aborted a take and relaunched got a teleop
session with no recording, no phases and no audio, and nothing saying why.
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
    """The full GUI app in a uvicorn thread, Run tab open."""
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
        pg.wait_for_selector("#run-teleop-record-dataset", timeout=10_000)
        yield pg
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


def test_new_dataset_selection_survives_a_dataset_list_refresh(page):
    """The exact failure: select New dataset, refresh, selection must hold."""
    page.select_option("#run-teleop-record-dataset", value="__new__")
    page.wait_for_selector("#run-teleop-new-dataset-name", state="visible", timeout=5_000)
    page.fill("#run-teleop-new-dataset-name", "gggg")

    # What the frontend runs after a recording ends (or any dataset change).
    page.evaluate("refreshRunDatasetSelects()")

    assert page.input_value("#run-teleop-record-dataset") == "__new__", (
        "dataset selection was reset — the next Launch would silently run teleoperate instead of record"
    )
    assert page.input_value("#run-teleop-new-dataset-name") == "gggg", (
        "the typed dataset name was lost across the refresh"
    )


def test_none_selection_also_survives(page):
    """The default must hold too — refresh never moves the mode switch at all."""
    assert page.input_value("#run-teleop-record-dataset") == ""
    page.evaluate("refreshRunDatasetSelects()")
    assert page.input_value("#run-teleop-record-dataset") == ""
