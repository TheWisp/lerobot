# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Edge cases in the Hub transfer dialog and tray, driven through the real page.

Adding model transfers put a second repo type through client code that had only
ever carried one, and the parts that broke were the parts no test watched: the
backend dataset paths are covered in depth by ``test_hub_endpoints.py``, while
nothing asserted which endpoint the tray's Retry builds, which namespace the
dialog queries, or what the repo field is pre-filled with. All four defects
below lived in that gap, and one of them broke *datasets*.

Each test is a sequence rather than a single call, because that is what made
these invisible: every one of them needs a prior interaction, a debounce to
elapse, or a specific job shape before the wrong branch is taken. Where a path
is shared between repo types the dataset case is asserted beside the model one,
so a future model-side change cannot quietly take datasets with it.

Requests are answered by a stub installed over ``window.fetch`` that also
records what was asked for -- the URL a click produces is the assertion, and it
is invisible to a screenshot.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.requires_playwright

RUN_PATH = "/runs/act_pick_place"

# Answers keyed by URL fragment. Anything unmatched returns {}, which is enough
# for the render paths under test and keeps the stub from encoding a contract.
_STUB = """
([answers, username]) => {
    window.__calls = [];
    window.fetch = (u, opts) => {
        const url = String(u);
        window.__calls.push({url, opts: opts || null});
        let body = {};
        for (const [frag, ans] of answers) {
            if (url.includes(frag)) { body = ans; break; }
        }
        return Promise.resolve(new Response(JSON.stringify(body), {status: 200}));
    };
    if (username !== null) { window.hfUser = username; }
}
"""


def _install(page, answers: list[tuple[str, dict]], username: str | None = None) -> None:
    page.evaluate(_STUB, [answers, username])


def _calls(page, fragment: str) -> list[str]:
    return page.evaluate("(f) => window.__calls.map(c => c.url).filter(u => u.includes(f))", fragment)


def _repo_info_types(page) -> list[str]:
    """The `repo_type` of every repo-info request, in order."""
    return page.evaluate(
        "() => window.__calls.map(c => c.url)"
        ".filter(u => u.includes('/hub/repo-info'))"
        ".map(u => new URL(u, location.origin).searchParams.get('repo_type') || 'dataset')"
    )


REPO_EXISTS = {
    "exists": True,
    "files": 5,
    "total_size_mb": 1,
    "private": True,
    "downloads": 0,
    "sha": "abc",
    "last_modified": "2026-07-01T00:00:00Z",
}

# Answers repo-info according to the namespace it was asked about, which is what
# the Hub does: a dataset id looked up as a model is simply not there.
_STUB_BY_NAMESPACE = """
([datasetExists, modelExists, present]) => {
    window.__calls = [];
    window.fetch = (u) => {
        const url = String(u);
        window.__calls.push({url, opts: null});
        let body = {};
        if (url.includes('/hub/repo-info')) {
            const type = new URL(url, location.origin).searchParams.get('repo_type') || 'dataset';
            const ok = type === 'model' ? modelExists : datasetExists;
            body = ok ? present : {exists: false};
        } else if (url.includes('/run-mtime')) {
            body = {mtime: 1800000000};
        } else if (url.includes('/hub/diff')) {
            body = {modified: [], in_sync: true};
        }
        return Promise.resolve(new Response(JSON.stringify(body), {status: 200}));
    };
}
"""


def _install_by_namespace(page, *, dataset_repo_exists: bool, model_repo_exists: bool) -> None:
    page.evaluate(_STUB_BY_NAMESPACE, [dataset_repo_exists, model_repo_exists, REPO_EXISTS])


class TestTheModelDialogDoesNotFollowDatasetsAround:
    """A model interaction must not change what a later dataset dialog asks for.

    The repo kind used to be read from the context menu's `_folderContextIsModelRun`
    global, which a model action latched true and only a later right-click cleared.
    `open-sync` always opens with a null dataset id, so every incomplete-cache
    repair dialog after any model interaction queried the model namespace, found
    nothing, and disabled its own download button -- for a dataset that is on the
    Hub. That is a regression in dataset behaviour introduced by a model feature.
    """

    def test_a_dataset_repair_dialog_after_a_model_action_still_asks_about_a_dataset(self, gui_page):
        page = gui_page
        _install(page, [("/hub/repo-info", REPO_EXISTS), ("/hub/diff", {"modified": [], "in_sync": True})])

        page.evaluate(f"modelHubAction('upload', {RUN_PATH!r})")
        page.evaluate("closeHubModal()")
        page.evaluate(
            "openHubModal(null, 'open-sync',"
            " {body: {repo_id: 'u/pusht'}, detail: {code: 'incomplete_local_cache', repo_id: 'u/pusht'}})"
        )
        page.wait_for_timeout(700)  # past the 400 ms repo-info debounce

        assert page.evaluate("_hubRepoType") == "dataset"
        assert _repo_info_types(page)[-1] == "dataset"

    def test_that_dialog_can_still_start_its_download(self, gui_page):
        """The user-visible half: the execute button stays usable.

        Asserting the namespace alone would not show the cost. Models and
        datasets are separate id spaces on the Hub, so a dataset repo looked up
        as a model comes back absent -- and the open-sync branch reacts to
        `exists:false` by disabling the button and saying there is nothing to
        download. The stub here answers per namespace for that reason; one that
        says "exists" to every query cannot tell the two paths apart.
        """
        page = gui_page
        _install_by_namespace(page, dataset_repo_exists=True, model_repo_exists=False)

        page.evaluate(f"modelHubAction('download', {RUN_PATH!r})")
        page.evaluate("closeHubModal()")
        page.evaluate(
            "openHubModal(null, 'open-sync',"
            " {body: {repo_id: 'u/pusht'}, detail: {code: 'incomplete_local_cache', repo_id: 'u/pusht'}})"
        )
        page.wait_for_timeout(700)

        assert page.evaluate("document.getElementById('hub-execute-btn').disabled") is False

    def test_a_model_run_still_asks_about_a_model(self, gui_page):
        """The other direction: routing by argument must not lose the model case."""
        page = gui_page
        _install(page, [("/hub/repo-info", REPO_EXISTS), ("/run-mtime", {"mtime": 1_760_000_000})])

        page.evaluate(f"modelHubAction('upload', {RUN_PATH!r})")
        page.wait_for_timeout(700)

        assert page.evaluate("_hubRepoType") == "model"
        assert _repo_info_types(page)[-1] == "model"


class TestTheFreshnessVerdictSurvives:
    """The model comparison must still be on screen after the debounce fires.

    `fetchHubRepoInfo` ended with an unconditional `fetchHubDiff()`. For a run
    path that route 404s, the handler throws on the error body's missing
    `.modified`, and its catch blanks the shared status line -- so the verdict
    appeared and was erased ~400 ms later. A test that reads the line
    immediately passes while the feature is blank in every real use.
    """

    def test_the_verdict_is_still_there_after_the_debounce(self, gui_page):
        page = gui_page
        _install(page, [("/hub/repo-info", REPO_EXISTS), ("/run-mtime", {"mtime": 1_800_000_000})])

        page.evaluate(f"modelHubAction('upload', {RUN_PATH!r})")
        page.wait_for_function(
            "document.getElementById('hub-status').innerText.includes('newer')", timeout=5_000
        )
        page.wait_for_timeout(900)  # more than the 400 ms debounce

        assert "newer" in page.evaluate("document.getElementById('hub-status').innerText")

    def test_no_dataset_diff_is_requested_for_a_model(self, gui_page):
        """The route that blanked it must not be called at all for a run path."""
        page = gui_page
        _install(page, [("/hub/repo-info", REPO_EXISTS), ("/run-mtime", {"mtime": 1_800_000_000})])

        page.evaluate(f"modelHubAction('upload', {RUN_PATH!r})")
        page.wait_for_timeout(900)

        assert _calls(page, "/hub/diff") == []

    def test_editing_the_repo_id_recomputes_the_verdict(self, gui_page):
        """Typing must refresh the comparison, not clear it.

        `fetchModelFreshness` was reachable only from `openHubModal`, so once the
        input's own handler had blanked the line nothing could bring it back.
        """
        page = gui_page
        _install(page, [("/hub/repo-info", REPO_EXISTS), ("/run-mtime", {"mtime": 1_800_000_000})])

        page.evaluate(f"modelHubAction('upload', {RUN_PATH!r})")
        page.wait_for_timeout(700)
        page.evaluate("() => { window.__calls.length = 0; }")

        page.fill("#hub-repo-input", "someone/else")
        page.dispatch_event("#hub-repo-input", "input")
        page.wait_for_timeout(900)

        assert "newer" in page.evaluate("document.getElementById('hub-status').innerText")
        assert _calls(page, "/run-mtime"), "freshness was not recomputed after the edit"

    def test_a_dataset_still_gets_its_file_diff(self, gui_page):
        """Regression guard on the shared tail: datasets must keep the diff."""
        page = gui_page
        _install(page, [("/hub/repo-info", REPO_EXISTS), ("/hub/diff", {"modified": [], "in_sync": True})])

        page.evaluate(
            "datasets['/data/pusht'] ="
            " {repo_id: 'u/pusht', total_episodes: 3, total_frames: 9, root: '/data/pusht'}"
        )
        page.evaluate("hubUploadDataset('/data/pusht', 'dataset')")
        page.wait_for_timeout(900)

        assert _calls(page, "/hub/diff"), "the dataset diff request disappeared"


class TestRetryReachesTheRightEndpoint:
    """The tray renders Retry on every terminal card and its copy says to use it.

    A model job's id is a run directory, which the dataset route rejects with
    404 -- so Retry was unreachable for every model transfer while the UI
    actively pointed at it.
    """

    def _job(self, repo_type: str | None) -> dict:
        job = {
            "job_id": "j1",
            "dataset_id": RUN_PATH if repo_type == "model" else "/data/pusht",
            "direction": "upload",
            "repo_id": "u/thing",
            "repo_type": repo_type,
            "status": "failed",
            "stage": "failed",
            "error": "network",
        }
        if repo_type is None:
            del job["repo_type"]
        return job

    def _arm(self, page, repo_type: str | None) -> None:
        """Load the job through the real polling path, then retry it.

        `_jobs` is private to the tray module, so the job has to arrive the way
        one does in use -- via `/hub/jobs`. That also means the card is rendered
        from it, which is what puts the Retry button on screen in the first place.
        """
        _install(
            page,
            [
                ("/hub/jobs", {"jobs": [self._job(repo_type)]}),
                ("/hub/history", {"transfers": []}),
            ],
        )
        page.evaluate("() => Transfers.refreshNow()")
        page.wait_for_timeout(400)
        page.evaluate("() => { window.__calls.length = 0; }")
        page.evaluate("() => Transfers.retry('j1')")
        page.wait_for_timeout(400)

    def test_a_failed_model_transfer_retries_through_the_model_route(self, gui_page):
        page = gui_page
        self._arm(page, "model")

        posted = _calls(page, "/hub/upload")
        assert posted and posted[-1].endswith("/api/models/hub/upload"), posted

    def test_the_run_path_travels_in_the_body(self, gui_page):
        """The model route takes the path in the body, so the id has to move there."""
        page = gui_page
        self._arm(page, "model")

        body = page.evaluate(
            "() => JSON.parse(window.__calls.filter(c => c.url.includes('/hub/upload')).pop().opts.body)"
        )
        assert body["path"] == RUN_PATH
        assert body["repo_id"] == "u/thing"

    def test_a_failed_dataset_transfer_still_retries_through_the_dataset_route(self, gui_page):
        page = gui_page
        self._arm(page, "dataset")

        posted = _calls(page, "/hub/upload")
        assert posted and "/api/datasets/" in posted[-1], posted

    def test_a_record_without_a_repo_type_is_treated_as_a_dataset(self, gui_page):
        """History written before model transfers existed carries no repo_type."""
        page = gui_page
        self._arm(page, None)

        posted = _calls(page, "/hub/upload")
        assert posted and "/api/datasets/" in posted[-1], posted


class TestTheSuggestedRepoIdNamesTheRealOwner:
    """`me/<run>` is not a harmless placeholder.

    It passes the pre-flight whoami -- which only checks that *someone* is
    logged in -- and fails minutes later inside the worker on `create_repo`,
    where `classify_error` reports it as an expired token. The owner was read
    from `window.hfUser`, which nothing ever assigned.
    """

    def test_the_logged_in_user_owns_the_suggestion(self, gui_page):
        page = gui_page
        page.evaluate("window.hfUser = undefined")
        _install(page, [("/hub/auth-status", {"logged_in": True, "username": "thewisp"})])
        page.evaluate("checkHubAuth()")
        page.wait_for_function("window.hfUser === 'thewisp'", timeout=5_000)

        assert page.evaluate(f"defaultModelRepoId({RUN_PATH!r})") == "thewisp/act_pick_place"

    def test_the_dialog_is_pre_filled_with_it(self, gui_page):
        page = gui_page
        page.evaluate("window.hfUser = undefined")
        _install(
            page,
            [
                ("/hub/auth-status", {"logged_in": True, "username": "thewisp"}),
                ("/hub/repo-info", REPO_EXISTS),
                ("/run-mtime", {"mtime": 1_800_000_000}),
            ],
        )
        page.evaluate("checkHubAuth()")
        page.wait_for_function("window.hfUser === 'thewisp'", timeout=5_000)
        page.evaluate(f"modelHubAction('upload', {RUN_PATH!r})")

        assert page.input_value("#hub-repo-input") == "thewisp/act_pick_place"

    def test_logged_out_falls_back_rather_than_naming_nobody(self, gui_page):
        page = gui_page
        _install(page, [("/hub/auth-status", {"logged_in": False, "username": None})])
        page.evaluate("checkHubAuth()")
        page.wait_for_function("window.hfUser === null", timeout=5_000)

        assert page.evaluate(f"defaultModelRepoId({RUN_PATH!r})") == "me/act_pick_place"
