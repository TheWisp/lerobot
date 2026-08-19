# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A model run must be able to reach the Hub through the GUI (#87).

Two gates hid it, and each one alone made the feature look absent rather than
unimplemented: the context menu showed Upload/Download only for
`datasets[path]`, which a model run never satisfies, and `openHubModal`
returned early for the same reason — so even a forced click opened nothing.

These drive the real markup rather than calling the handlers, because both
gates were in the wiring rather than the logic.
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
def page(tmp_path):
    from lerobot.gui import server as gui_server_mod

    # The layout `_scan_training_run` recognises: `checkpoints/<step>/
    # pretrained_model/`. A flat `pretrained_model/` is not a training run, so
    # the tree lists nothing and no detail pane renders.
    run = tmp_path / "outputs" / "act_pick_place"
    ckpt = run / "checkpoints" / "000100"
    (ckpt / "pretrained_model").mkdir(parents=True)
    (ckpt / "pretrained_model" / "config.json").write_text('{"type": "act"}')
    (ckpt / "training_state").mkdir()
    (ckpt / "training_state" / "training_step.json").write_text('{"step": 100}')

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
        pg = browser.new_page()
        import requests as _rq

        _rq.post(f"{base}/api/models/sources", json={"path": str(run.parent)}, timeout=10)
        pg.goto(base)
        pg.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        yield pg, run
        browser.close()
    server.should_exit = True


def _visible(page, ids: list[str]) -> dict[str, bool]:
    return page.evaluate(
        "ids => Object.fromEntries(ids.map(i => [i, document.getElementById(i).style.display !== 'none']))",
        ids,
    )


HUB_ITEMS = ["folder-ctx-hub-upload", "folder-ctx-hub-download"]


def test_a_model_run_is_offered_hub_transfers(page):
    """The gate was `isOpenedDataset`, which a model run never satisfies."""
    pg, run = page
    ev = "{preventDefault(){},stopPropagation(){},clientX:20,clientY:20}"
    pg.evaluate(f"showFolderContextMenu({ev}, {str(run)!r}, true)")
    assert _visible(pg, HUB_ITEMS) == dict.fromkeys(HUB_ITEMS, True)


def test_a_source_folder_is_still_not(page):
    """Widening the gate must not offer a transfer for something untransferable."""
    pg, run = page
    ev = "{preventDefault(){},stopPropagation(){},clientX:20,clientY:20}"
    pg.evaluate(f"showFolderContextMenu({ev}, {str(run.parent)!r})")
    assert _visible(pg, HUB_ITEMS) == dict.fromkeys(HUB_ITEMS, False)


def test_the_modal_opens_for_a_model_run_and_targets_the_model_route(page):
    """`openHubModal` returned early for anything absent from `datasets`, so the
    menu item opened nothing at all."""
    pg, run = page
    ev = "{preventDefault(){},stopPropagation(){},clientX:20,clientY:20}"
    pg.evaluate(f"showFolderContextMenu({ev}, {str(run)!r}, true)")
    pg.evaluate("folderContextAction('hub-upload')")

    assert pg.is_visible("#hub-modal-overlay"), "the modal must actually open"
    assert pg.evaluate("_hubRepoType") == "model"
    assert pg.input_value("#hub-repo-input").endswith("/act_pick_place"), "suggested repo id"
    assert "model checkpoint" in pg.inner_text("#hub-local-info")

    # The request must go to the model route, carrying the run path in the body.
    # Record only the transfer POST: the modal also issues GETs with no options.
    pg.evaluate(
        "window._sent = null; window.fetch = (u, o) => {"
        "  if (o && o.body) window._sent = {url: String(u), body: JSON.parse(o.body)};"
        '  return Promise.resolve(new Response(\'{"job_id":"x"}\', {status: 200}));'
        "};"
    )
    pg.evaluate("document.getElementById('hub-repo-input').value = 'me/act-pick'")
    pg.click("#hub-execute-btn")
    pg.wait_for_function("window._sent != null", timeout=10_000)
    sent = pg.evaluate("window._sent")

    assert sent["url"].endswith("/api/models/hub/upload"), sent["url"]
    assert sent["body"]["path"] == str(run)
    assert sent["body"]["repo_id"] == "me/act-pick"


@pytest.mark.parametrize(
    ("local_epoch", "remote_iso", "expected"),
    [
        (1_760_000_000, "2020-01-01T00:00:00.000Z", "Local is newer"),
        (1_000_000_000, "2030-01-01T00:00:00.000Z", "Hub is newer"),
        (1_760_000_000, "2025-10-09T00:00:00.000Z", "Same date"),
    ],
)
def test_a_model_repo_is_compared_by_date_not_by_file_list(page, local_epoch, remote_iso, expected):
    """A checkpoint has no episode or shard layout to diff.

    The dataset comparison counts modified, local-only and remote-only files
    against a known layout; for a model the answerable question is simply which
    side was written more recently, so that is what the dialog says.
    """
    pg, run = page
    pg.evaluate(
        "([mtime, last]) => { window.fetch = (u, o) => {"
        "  const url = String(u);"
        "  if (url.includes('/api/models/run-mtime'))"
        "    return Promise.resolve(new Response(JSON.stringify({mtime}), {status: 200}));"
        "  if (url.includes('/hub/repo-info'))"
        "    return Promise.resolve(new Response("
        "      JSON.stringify({exists: true, repo_id: 'me/x', last_modified: last}), {status: 200}));"
        "  return Promise.resolve(new Response('{}', {status: 200}));"
        "}; }",
        [local_epoch, remote_iso],
    )
    ev = "{preventDefault(){},stopPropagation(){},clientX:20,clientY:20}"
    pg.evaluate(f"showFolderContextMenu({ev}, {str(run)!r}, true)")
    pg.evaluate("folderContextAction('hub-upload')")

    pg.wait_for_function(
        f"document.getElementById('hub-status').innerText.includes({expected!r})", timeout=15_000
    )
    status = pg.inner_text("#hub-status")
    assert expected in status, status
    assert "modified" not in status, f"a file-list comparison must not appear for a model: {status!r}"


def test_a_model_repo_that_is_not_on_the_hub_yet_says_nothing_about_dates(page):
    """First upload: there is no remote date, so there is nothing to compare."""
    pg, run = page
    pg.evaluate(
        "window.fetch = (u) => Promise.resolve(new Response("
        "  String(u).includes('/hub/repo-info')"
        "    ? JSON.stringify({exists: false, repo_id: 'me/x'})"
        "    : JSON.stringify({mtime: 1760000000}), {status: 200}));"
    )
    ev = "{preventDefault(){},stopPropagation(){},clientX:20,clientY:20}"
    pg.evaluate(f"showFolderContextMenu({ev}, {str(run)!r}, true)")
    pg.evaluate("folderContextAction('hub-upload')")
    pg.wait_for_timeout(1200)
    assert "newer" not in pg.inner_text("#hub-status")


def test_starting_an_upload_says_upload_not_download(page):
    """`closeHubModal()` clears `_hubAction`, and the verb was read after it.

    Every upload therefore announced "Download started" — for datasets as much
    as models, since the toast is shared. Visible only in a capture of a real
    transfer; no assertion had ever looked at the toast.
    """
    pg, run = page
    pg.evaluate(
        "window.fetch = (u, o) => Promise.resolve(new Response("
        "  String(u).includes('/hub/upload') ? '{\"job_id\":\"j\"}' : '{}', {status: 200}));"
    )
    ev = "{preventDefault(){},stopPropagation(){},clientX:20,clientY:20}"
    pg.evaluate(f"showFolderContextMenu({ev}, {str(run)!r}, true)")
    pg.evaluate("folderContextAction('hub-upload')")
    pg.evaluate("document.getElementById('hub-repo-input').value = 'me/act-pick'")
    pg.click("#hub-execute-btn")

    pg.wait_for_function(
        "document.body.innerText.includes('Upload started')"
        " || document.body.innerText.includes('Download started')",
        timeout=10_000,
    )
    body = pg.inner_text("body")
    assert "Upload started" in body, "an upload must not announce a download"
    assert "Download started" not in body


def _open_run_detail(pg, run):
    """Select the run so its detail header renders.

    The tab's model list loads asynchronously; selecting before it arrives
    finds no run and leaves the pane empty.
    """
    pg.evaluate("switchTab('model')")
    pg.wait_for_function(f"document.body.innerText.includes({run.name!r})", timeout=15_000)
    pg.evaluate(f"selectModelRun({str(run)!r})")
    pg.wait_for_selector(".model-detail-header", timeout=15_000)


def test_the_detail_header_offers_the_same_hub_actions_as_the_menu(page):
    """A capability reachable only by right-click is one most people never find.

    The detail pane is where you land after selecting a run; the context menu
    is the shortcut. They must offer the same actions for the same object, or
    the pane quietly under-reports what the run can do.
    """
    pg, run = page
    _open_run_detail(pg, run)

    header = pg.inner_text(".model-detail-header")
    assert "Hub" in header, f"the header must offer the Hub actions: {header!r}"

    pg.locator(".model-detail-header .hub-menu-wrap button").click()
    pg.wait_for_selector(".hub-menu:not([hidden])", timeout=5_000)
    items = pg.eval_on_selector_all(
        ".hub-menu:not([hidden]) .hub-menu-item", "els => els.map(e => e.innerText)"
    )

    ev = "{preventDefault(){},stopPropagation(){},clientX:20,clientY:20}"
    pg.evaluate(f"showFolderContextMenu({ev}, {str(run)!r}, true)")
    menu_items = pg.evaluate(
        "['folder-ctx-hub-upload', 'folder-ctx-hub-download']"
        ".filter(i => document.getElementById(i).style.display !== 'none')"
        ".map(i => document.getElementById(i).innerText)"
    )
    assert sorted(items) == sorted(menu_items), f"header offers {items}, context menu offers {menu_items}"


def test_the_header_dropdown_opens_the_same_dialog_as_the_menu(page):
    """Two entry points, one dialog — or they drift into different behaviour."""
    pg, run = page
    _open_run_detail(pg, run)
    pg.locator(".model-detail-header .hub-menu-wrap button").click()
    pg.wait_for_selector(".hub-menu:not([hidden])", timeout=5_000)
    pg.locator(".model-detail-header .hub-menu-item").first.click()

    assert pg.is_visible("#hub-modal-overlay"), "the dropdown must open the transfer dialog"
    assert pg.evaluate("_hubRepoType") == "model", "and treat the run as a model repo"
    assert "model checkpoint" in pg.inner_text("#hub-local-info")


def test_the_dropdown_closes_on_an_outside_click(page):
    """An anchored menu left open over the card is worse than no menu."""
    pg, run = page
    _open_run_detail(pg, run)
    pg.locator(".model-detail-header .hub-menu-wrap button").click()
    pg.wait_for_selector(".hub-menu:not([hidden])", timeout=5_000)

    pg.click(".model-detail-header h2")
    pg.wait_for_function("document.querySelector('.hub-menu:not([hidden])') == null", timeout=5_000)
