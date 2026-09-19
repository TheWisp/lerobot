# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Who is offered a Hub transfer, across every node type that shares the menu.

The gate used to ask whether the server currently held the dataset in memory.
That is process-local state: a GUI restart empties it for anything the user does
not re-open, so Upload and Download vanished from a dataset sitting in the tree
with its episode count rendered beside it. The action read as missing rather
than unavailable, and the endpoint behind it failed too.

The gate now asks what the node IS. These cover each path that reaches the same
menu — a dataset in the tree, an opened dataset, a model run, and a source
folder — for both actions, so widening it cannot quietly offer a transfer for
something untransferable.
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

HUB_ITEMS = ["folder-ctx-hub-upload", "folder-ctx-hub-download"]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_minimal_dataset(root, repo_id: str) -> None:
    import numpy as np

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=10, features=features, root=str(root))
    for _ in range(2):
        ds.add_frame(
            {
                "observation.state": np.zeros(2, np.float32),
                "action": np.zeros(2, np.float32),
                "task": "t",
            }
        )
    ds.save_episode()
    ds.finalize()


@pytest.fixture
def page(tmp_path, monkeypatch):
    from lerobot.gui import server as gui_server_mod
    from lerobot.gui.api import datasets as datasets_api

    # Keep the browser hermetic: never read or write the developer's real
    # ~/.config state, and never restore a dataset from a previous session.
    monkeypatch.setenv("LEROBOT_GUI_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(datasets_api, "_read_opened", lambda: [])

    ds_root = tmp_path / "cache" / "owner" / "name"
    ds_root.parent.mkdir(parents=True)
    _write_minimal_dataset(ds_root, "owner/name")

    run = tmp_path / "outputs" / "act_pick_place"
    ckpt = run / "checkpoints" / "000100"
    (ckpt / "pretrained_model").mkdir(parents=True)
    (ckpt / "pretrained_model" / "config.json").write_text('{"type": "act"}')
    (ckpt / "training_state").mkdir()
    (ckpt / "training_state" / "training_step.json").write_text('{"step": 100}')

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
    )
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

    requests.post(f"{base}/api/models/sources", json={"path": str(run.parent)}, timeout=10)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.goto(base)
        pg.wait_for_function("typeof showFolderContextMenu === 'function'", timeout=15_000)
        yield pg, ds_root, run, base
        browser.close()
    server.should_exit = True


def _visible(pg, ids: list[str]) -> dict[str, bool]:
    return pg.evaluate(
        "ids => Object.fromEntries(ids.map(i => [i, document.getElementById(i).style.display !== 'none']))",
        ids,
    )


def _open_menu(pg, path, *, is_model_run=False, is_dataset=False):
    ev = "{preventDefault(){},stopPropagation(){},clientX:20,clientY:20}"
    pg.evaluate(
        f"showFolderContextMenu({ev}, {str(path)!r}, {str(is_model_run).lower()}, {str(is_dataset).lower()})"
    )


def test_a_dataset_in_the_tree_is_offered_both_transfers(page):
    """The defect: a dataset the server has not loaded still offers its actions."""
    pg, ds_root, _run, _base = page
    assert pg.evaluate("Object.keys(datasets).length") == 0, "precondition: nothing is open"

    _open_menu(pg, ds_root, is_dataset=True)

    assert _visible(pg, HUB_ITEMS) == dict.fromkeys(HUB_ITEMS, True)


def test_an_opened_dataset_is_still_offered_both_transfers(page):
    """The path that always worked must keep working."""
    pg, ds_root, _run, _base = page
    pg.evaluate(f"openDataset({str(ds_root)!r})")
    pg.wait_for_function("Object.keys(datasets).length === 1", timeout=15_000)

    _open_menu(pg, ds_root, is_dataset=True)

    assert _visible(pg, HUB_ITEMS) == dict.fromkeys(HUB_ITEMS, True)


def test_a_model_run_is_still_offered_both_transfers(page):
    """Model runs reach the same menu and must not be affected."""
    pg, _ds_root, run, _base = page

    _open_menu(pg, run, is_model_run=True)

    assert _visible(pg, HUB_ITEMS) == dict.fromkeys(HUB_ITEMS, True)


def test_a_source_folder_is_offered_neither(page):
    """Widening the gate must not offer a transfer for something untransferable."""
    pg, ds_root, _run, _base = page

    _open_menu(pg, ds_root.parent)

    assert _visible(pg, HUB_ITEMS) == dict.fromkeys(HUB_ITEMS, False)


# ── The error path: a 500 must say what it was ─────────────────────────────
#
# The Hub modal parsed every response body as JSON. An unhandled server fault
# returns plain text, so `JSON.parse` threw and the operator saw
# `Unexpected token 'I', "Internal S"... is not valid JSON` — the parser's
# complaint about the word "Internal", with the actual failure nowhere.
#
# Driven in a real page rather than a stubbed module: app.js does not evaluate
# standalone, and a harness that half-loads it would be a flaky test of nothing.


def _parse(pg, status: int, body: str):
    """Run the real hubParseResponse against a fake Response, in the real page."""
    return pg.evaluate(
        """async ([status, body]) => {
            const res = { status, text: async () => body };
            const el = { textContent: '' };
            const btn = { disabled: true };
            const out = await hubParseResponse(res, el, btn);
            return {
                sentinel: out === HUB_RESPONSE_NOT_JSON,
                value: out === HUB_RESPONSE_NOT_JSON ? null : out,
                status: el.textContent,
                reenabled: btn.disabled === false,
            };
        }""",
        [status, body],
    )


def test_a_json_body_is_returned_as_before(page):
    pg, _ds, _run, _base = page

    got = _parse(pg, 409, '{"detail": {"job_id": "abc"}}')

    assert got["sentinel"] is False
    assert got["value"] == {"detail": {"job_id": "abc"}}
    assert got["status"] == "", "a parseable body reports nothing itself"


def test_a_plain_text_fault_is_reported_verbatim(page):
    """The defect: this used to surface as a JSON syntax error about 'I'."""
    pg, _ds, _run, _base = page

    got = _parse(pg, 500, "Internal Server Error")

    assert got["sentinel"] is True, "the caller must be told to stop, not parse"
    assert "500" in got["status"]
    assert "Internal Server Error" in got["status"]
    assert got["reenabled"], "the button is released so the operator can retry"


def test_an_empty_body_is_not_an_error(page):
    """A 204-shaped reply is legal and must not read as a server fault."""
    pg, _ds, _run, _base = page

    got = _parse(pg, 200, "")

    assert got["sentinel"] is False
    assert got["value"] == {}


def test_a_long_html_error_page_is_truncated_to_its_first_line(page):
    """Some proxies answer with an HTML page; the status line must stay readable."""
    pg, _ds, _run, _base = page

    got = _parse(pg, 502, "<html>\n<head><title>502 Bad Gateway</title></head>\n" + "x" * 5000)

    assert got["sentinel"] is True
    assert len(got["status"]) < 260, got["status"]
    assert "502" in got["status"]


# ── The modal must open for what the menu offers ───────────────────────────
#
# Making the menu item visible is not enough. `openHubModal` returned early when
# the dataset was absent from the client-side `datasets` map, so a click on a
# tree dataset did nothing at all — no modal, no error, no clue. An inert
# control is worse than a hidden one: the hidden one at least tells the truth.


def test_upload_opens_the_modal_for_a_dataset_that_is_not_open(page):
    pg, ds_root, _run, _base = page
    assert pg.evaluate("Object.keys(datasets).length") == 0, "precondition: nothing is open"

    pg.evaluate(f"hubUploadDataset({str(ds_root)!r}, 'dataset')")
    pg.wait_for_selector("#hub-repo-input", state="visible", timeout=10_000)

    assert pg.evaluate("document.getElementById('hub-modal-overlay').style.display") != "none"
    # Prefilled with the same <owner>/<name> the server derives from a path.
    assert pg.input_value("#hub-repo-input") == "owner/name"


def test_download_opens_the_modal_for_a_dataset_that_is_not_open(page):
    pg, ds_root, _run, _base = page

    pg.evaluate(f"hubDownloadDataset({str(ds_root)!r}, 'dataset')")
    pg.wait_for_selector("#hub-repo-input", state="visible", timeout=10_000)

    assert pg.input_value("#hub-repo-input") == "owner/name"


def test_an_opened_dataset_still_prefills_its_own_repo_id(page):
    """The registry's repo id wins over anything derived from the path."""
    pg, ds_root, _run, _base = page
    pg.evaluate(f"openDataset({str(ds_root)!r})")
    pg.wait_for_function("Object.keys(datasets).length === 1", timeout=15_000)

    pg.evaluate(f"hubUploadDataset({str(ds_root)!r}, 'dataset')")
    pg.wait_for_selector("#hub-repo-input", state="visible", timeout=10_000)

    assert pg.input_value("#hub-repo-input") == "owner/name"


def test_a_model_run_still_opens_its_modal(page):
    """Model runs never had a client-side record and must keep working."""
    pg, _ds_root, run, _base = page

    pg.evaluate(f"hubUploadDataset({str(run)!r}, 'model')")
    pg.wait_for_selector("#hub-repo-input", state="visible", timeout=10_000)

    assert pg.input_value("#hub-repo-input").endswith("/act_pick_place")


def test_the_model_context_menu_reaches_the_model_modal(page):
    """A model run is reachable from two places, and they must agree.

    The Model tab has its own `Hub ▾` menu; the tree's folder menu offers the
    same actions on a model node. Both end at `openHubModal(path, 'upload',
    {repoType: 'model'})`, but only one of them was ever driven end to end, so
    this goes through the menu dispatcher rather than trusting that they match.
    """
    pg, _ds_root, run, _base = page
    ev = "{preventDefault(){},stopPropagation(){},clientX:20,clientY:20}"
    pg.evaluate(f"showFolderContextMenu({ev}, {str(run)!r}, true, false)")
    pg.wait_for_timeout(300)
    assert pg.evaluate("document.getElementById('folder-ctx-hub-upload').style.display !== 'none'"), (
        "a model run must be offered the transfer"
    )

    pg.evaluate("folderContextAction('hub-upload')")
    pg.wait_for_selector("#hub-repo-input", state="visible", timeout=10_000)

    # A model repo, not a dataset one — the kind decides which route the submit
    # takes (/api/models/hub/* vs /api/datasets/{id}/hub/*).
    assert pg.input_value("#hub-repo-input").endswith("/act_pick_place")
    assert pg.evaluate("_hubRepoType") == "model"


def test_the_dataset_context_menu_reaches_the_dataset_modal(page):
    """The mirror of the above: the same dispatcher must not send a dataset
    down the model route."""
    pg, ds_root, _run, _base = page
    ev = "{preventDefault(){},stopPropagation(){},clientX:20,clientY:20}"
    pg.evaluate(f"showFolderContextMenu({ev}, {str(ds_root)!r}, false, true)")
    pg.wait_for_timeout(300)

    pg.evaluate("folderContextAction('hub-upload')")
    pg.wait_for_selector("#hub-repo-input", state="visible", timeout=10_000)

    assert pg.input_value("#hub-repo-input") == "owner/name"
    assert pg.evaluate("_hubRepoType") == "dataset"
