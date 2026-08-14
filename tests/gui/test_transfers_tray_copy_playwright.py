# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The tray's destructive-action copy, pinned against the rendered DOM.

Two defects shipped here, both in the wording rather than the mechanism, and
both invisible to the endpoint tests because nothing was broken server-side.

First, ✕ promised "the draft PR is kept so Retry still works" — naming a
button that ✕ itself removes, since Retry lives on the card and history
entries render no actions. Second, one sentence covered every finished card,
so a *download* — which has no PR and never did — was described in terms of
a draft PR being kept.

Both are properties of which text the render picks for which job, so that is
what these assert: real jobs in the registry, a real browser, the ``title``
attributes as a user would hover them. A test reading the strings out of
``app.js`` would have passed with both bugs present.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from lerobot.gui import hub_jobs  # noqa: E402

pytestmark = pytest.mark.requires_playwright


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _finished_job(*, direction: str, pr_num: int | None) -> hub_jobs.HubJobState:
    job = hub_jobs.make_job(
        dataset_id=f"/local/{direction}{pr_num}",
        direction=direction,
        repo_id=f"u/{direction}{pr_num}",
    )
    job.status = "failed"
    job.error = "Network error"
    job.error_class = "network"
    job.pr_num = pr_num
    job.started_at = time.time() - 60
    job.finished_at = time.time()
    return job


@pytest.fixture
def tray():
    """The tray, open, with one finished card of each kind behind it."""
    from lerobot.gui import server as gui_server_mod
    from lerobot.gui.api import datasets as datasets_api

    jobs = {
        "upload_with_pr": _finished_job(direction="upload", pr_num=7),
        "upload_no_pr": _finished_job(direction="upload", pr_num=None),
        "download": _finished_job(direction="download", pr_num=None),
    }

    port = _free_port()
    config = uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

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

    # Only now: AppState is built in the startup event, so it does not exist
    # until the server is actually serving.
    state = datasets_api._app_state
    state.hub_jobs.clear()
    for job in jobs.values():
        state.hub_jobs[job.job_id] = job

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(base_url)
        page.click("#transfers-indicator")
        page.wait_for_selector("#transfers-popover .transfer-card", timeout=10_000)
        try:
            yield page, jobs
        finally:
            browser.close()
            server.should_exit = True
            state.hub_jobs.clear()


def _clear_tooltip(page, job) -> str:
    """The ✕ button's hover text on this job's card."""
    sel = f"#transfers-popover .transfer-card:has(a[title='{job.repo_id}']) .hide-btn"
    page.wait_for_selector(sel, timeout=10_000)
    return page.get_attribute(sel, "title") or ""


class TestClearCopyMatchesWhatClearingDoes:
    def test_every_variant_promises_nothing_is_deleted(self, tray):
        """The one thing ✕ must say, on every card, is that it destroys nothing.

        It sits next to Discard, which does destroy something.
        """
        page, jobs = tray
        for job in jobs.values():
            assert "Nothing is deleted" in _clear_tooltip(page, job), job.repo_id

    def test_no_variant_names_the_retry_button(self, tray):
        """✕ removes the card, and Retry is on the card.

        The original text said "the draft PR is kept so Retry still works",
        which described a route that stops existing the moment it is acted on.
        """
        page, jobs = tray
        for job in jobs.values():
            assert "Retry" not in _clear_tooltip(page, job), (
                f"{job.repo_id}: ✕ must not name a button it is about to remove"
            )

    def test_only_a_transfer_that_has_a_pr_talks_about_one(self, tray):
        """A download has no draft PR and never did."""
        page, jobs = tray
        assert "draft PR" in _clear_tooltip(page, jobs["upload_with_pr"])
        assert "PR" not in _clear_tooltip(page, jobs["download"])
        assert "PR" not in _clear_tooltip(page, jobs["upload_no_pr"])

    def test_the_pr_variant_says_what_resuming_needs_to_match(self, tray):
        """ "Resumes" alone invites "any later upload continues this one".

        What continues is this dataset to this repo, because that is what the
        PR lookup keys on.
        """
        text = _clear_tooltip(page=tray[0], job=tray[1]["upload_with_pr"])
        assert "this dataset to this repo" in text


class TestDiscardIsOfferedOnlyWhereItHasSomethingToDestroy:
    def _discard(self, page, job):
        cards = f"#transfers-popover .transfer-card:has(a[title='{job.repo_id}'])"
        return page.query_selector(f"{cards} .transfer-action-btn.danger")

    def test_offered_for_an_upload_holding_a_draft_pr(self, tray):
        page, jobs = tray
        btn = self._discard(page, jobs["upload_with_pr"])
        assert btn is not None
        title = btn.get_attribute("title") or ""
        assert "draft PR" in title and "local files are untouched" in title, (
            "Discard must name what it destroys and what it spares"
        )

    def test_not_offered_where_it_would_only_duplicate_the_clear(self, tray):
        """On a download, or an upload with no PR, it has nothing to close —
        it would be a second button doing ✕'s job under a name implying more.
        """
        page, jobs = tray
        assert self._discard(page, jobs["download"]) is None
        assert self._discard(page, jobs["upload_no_pr"]) is None
