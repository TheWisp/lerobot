# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The transfers tray must render a model transfer in the model namespace (#87).

A screenshot cannot show a link's target, and the history section is collapsed
by default — so the two places the namespace was wrong are exactly the two a
visual check misses.

The live card reads `repo_type` from the in-memory job; the history card reads
the worker's durable JSON, which omitted it. `hubRepoUrl(id, undefined)` then
fell back to `/datasets/`, so a past model transfer linked to a URL that does
not exist. Both paths are asserted here against stubbed responses, which is
what makes the two distinguishable.
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

MODEL_JOB = {
    "job_id": "j-model",
    "dataset_id": "/runs/act_demo",
    "direction": "upload",
    "repo_id": "me/act-demo",
    "repo_type": "model",
    "status": "done",
    "stage": "done",
    "milestone": "Upload complete",
    "started_at": 1_760_000_000,
    "finished_at": 1_760_000_100,
    "files_total": 3,
    "files_done_estimate": 3,
    "bytes_total": 1234,
    "bytes_done_estimate": 1234,
}
DATASET_JOB = {**MODEL_JOB, "job_id": "j-data", "repo_id": "me/pusht", "repo_type": "dataset"}


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
    threading.Thread(target=server.run, daemon=True).start()
    import requests

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20
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
        pg = browser.new_page(viewport={"width": 1400, "height": 900})
        pg.goto(base)
        pg.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        yield pg
        browser.close()
    server.should_exit = True


def _stub(page, jobs: list[dict], history: list[dict]) -> None:
    page.evaluate(
        "([jobs, history]) => { window.fetch = (u) => {"
        "  const url = String(u);"
        "  const body = url.includes('/hub/jobs') ? {jobs}"
        "    : url.includes('/hub/history') ? {transfers: history}"
        "    : {};"
        "  return Promise.resolve(new Response(JSON.stringify(body), {status: 200}));"
        "}; }",
        [jobs, history],
    )


def _hrefs(page) -> list[str]:
    return page.evaluate(
        "Array.from(document.querySelectorAll('#transfers-popover a[href]')).map(a => a.href)"
    )


def test_a_live_model_transfer_links_into_the_model_namespace(page):
    """Models sit at the Hub root; datasets under /datasets. A card cannot be
    checked by eye — the text is the repo id either way."""
    _stub(page, [MODEL_JOB], [])
    page.evaluate("Transfers.openPopover(); Transfers.refreshNow();")
    page.wait_for_function(
        "document.querySelector('#transfers-list').innerText.includes('me/act-demo')", timeout=10_000
    )
    hrefs = [h for h in _hrefs(page) if "act-demo" in h]
    assert hrefs, "the card must link to the repo"
    assert all("/datasets/" not in h for h in hrefs), f"model linked into the dataset namespace: {hrefs}"
    assert any(h.rstrip("/").endswith("me/act-demo") for h in hrefs), hrefs


def test_a_live_dataset_transfer_still_links_under_datasets(page):
    """The fix must not push datasets into the model namespace."""
    _stub(page, [DATASET_JOB], [])
    page.evaluate("Transfers.openPopover(); Transfers.refreshNow();")
    page.wait_for_function(
        "document.querySelector('#transfers-list').innerText.includes('me/pusht')", timeout=10_000
    )
    hrefs = [h for h in _hrefs(page) if "pusht" in h]
    assert hrefs and all("/datasets/me/pusht" in h for h in hrefs), hrefs


def test_a_past_model_transfer_in_the_history_links_correctly(page):
    """The history renders from the worker's durable record, not the live job.

    That record omitted `repo_type`, so this is the card that linked to a
    dataset URL for a model — and it is behind a "Show" toggle, which is why no
    screenshot caught it.
    """
    _stub(page, [], [{**MODEL_JOB, "outcome": "done", "ended_at": 1_760_000_100}])
    page.evaluate("Transfers.openPopover(); Transfers.refreshNow();")
    page.wait_for_timeout(600)
    page.evaluate("Transfers.toggleHistory();")
    page.wait_for_function(
        "document.querySelector('#transfers-history-list').innerText.includes('me/act-demo')",
        timeout=10_000,
    )
    hrefs = [h for h in _hrefs(page) if "act-demo" in h]
    assert hrefs, "the history card must link to the repo"
    assert all("/datasets/" not in h for h in hrefs), (
        f"a past model transfer linked into the dataset namespace: {hrefs}"
    )


def test_a_record_without_repo_type_is_treated_as_a_dataset(page):
    """Records written before this change carry no repo_type.

    Defaulting them to dataset keeps every existing history entry correct; only
    model transfers written from now on carry the field.
    """
    legacy = {k: v for k, v in MODEL_JOB.items() if k != "repo_type"}
    legacy["repo_id"] = "me/legacy-ds"
    _stub(page, [legacy], [])
    page.evaluate("Transfers.openPopover(); Transfers.refreshNow();")
    page.wait_for_function(
        "document.querySelector('#transfers-list').innerText.includes('me/legacy-ds')", timeout=10_000
    )
    hrefs = [h for h in _hrefs(page) if "legacy-ds" in h]
    assert hrefs and all("/datasets/me/legacy-ds" in h for h in hrefs), hrefs
