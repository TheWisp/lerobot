# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Duplicate and delete reach the tree, and only offer themselves on datasets.

The context menu is shared by source folders, datasets and model runs. Copying
or deleting a source folder is not something these routes can do, and offering
it would be a dead menu item on the one action that cannot be undone.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright

REPO_ID = "test/dup_ui"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _gui(hf_home: Path):
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

    with patch("lerobot.gui.api.datasets.HF_LEROBOT_HOME", hf_home), sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.goto(base)
        pg.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        pg.evaluate("switchTab('data')")
        yield pg
        browser.close()
    server.should_exit = True


@pytest.fixture
def gui(tmp_path, lerobot_dataset_factory):
    hf_home = tmp_path / "hf_home"
    root = hf_home / REPO_ID
    root.mkdir(parents=True)
    lerobot_dataset_factory(
        root=root,
        repo_id=REPO_ID,
        total_episodes=2,
        total_frames=20,
        total_tasks=1,
        use_videos=False,
        camera_features={},
    )
    with _gui(hf_home) as pg:
        yield pg, root


def _menu_state(page) -> dict[str, bool]:
    return page.evaluate(
        "Object.fromEntries(['folder-ctx-duplicate','folder-ctx-delete']"
        ".map(id => [id, document.getElementById(id).style.display !== 'none']))"
    )


def test_menu_offers_copy_and_delete_on_a_dataset_only(gui):
    """A source folder shares this menu and must not offer either action."""
    page, root = gui
    ev = "{preventDefault(){},stopPropagation(){},clientX:10,clientY:10}"

    page.evaluate(f"showFolderContextMenu({ev}, {str(root)!r}, false, true)")
    assert _menu_state(page) == {"folder-ctx-duplicate": True, "folder-ctx-delete": True}

    page.evaluate(f"showFolderContextMenu({ev}, {str(root.parent)!r})")
    assert _menu_state(page) == {"folder-ctx-duplicate": False, "folder-ctx-delete": False}

    # A model run reuses the same menu via isModelRun.
    page.evaluate(f"showFolderContextMenu({ev}, '/tmp/some/run', true)")
    assert _menu_state(page) == {"folder-ctx-duplicate": False, "folder-ctx-delete": False}


def test_right_clicking_the_opened_tree_row_offers_both(gui):
    """Dispatch a real contextmenu event rather than calling the handler.

    Calling `showFolderContextMenu` directly proves the handler's logic and
    nothing about the markup that invokes it — which is where the arguments
    landed on the wrong call and left the opened tree without either entry.
    """
    page, root = gui
    page.evaluate(f"openDataset({str(root)!r})")
    page.wait_for_function(f"window.datasets?.[{str(root)!r}] != null", timeout=20_000)
    page.wait_for_selector(".tree-header", timeout=10_000)
    page.dispatch_event(".tree-header", "contextmenu")
    assert _menu_state(page) == {"folder-ctx-duplicate": True, "folder-ctx-delete": True}, (
        "right-clicking an opened dataset must offer Duplicate and Delete"
    )


def test_duplicating_an_open_dataset_opens_the_copy(gui):
    """Duplicating something you are working on opens the copy to work on."""
    page, root = gui
    page.on("dialog", lambda d: d.accept("copied_here"))
    page.evaluate(f"openDataset({str(root)!r})")
    page.wait_for_function(f"window.datasets?.[{str(root)!r}] != null", timeout=20_000)

    page.evaluate(f"duplicateDatasetAt({str(root)!r})")
    page.wait_for_function(f"window.datasets?.[{str(root.parent / 'copied_here')!r}] != null", timeout=30_000)
    assert (root.parent / "copied_here" / "meta" / "info.json").is_file()
    assert (root / "meta" / "info.json").is_file(), "the original must survive"


def test_duplicating_a_scanned_dataset_leaves_the_copy_closed(gui):
    """Duplicating from the Sources list is browsing, not working.

    Opening is not free — it builds a LeRobotDataset, caches and a lock — so
    copying three datasets while browsing should not open three.
    """
    page, root = gui
    page.on("dialog", lambda d: d.accept("just_a_copy"))
    assert page.evaluate(f"window.datasets?.[{str(root)!r}] == null"), "source must start closed"

    page.evaluate(f"duplicateDatasetAt({str(root)!r})")
    copy = root.parent / "just_a_copy"
    deadline = time.monotonic() + 30
    while not (copy / "meta" / "info.json").exists() and time.monotonic() < deadline:
        page.wait_for_timeout(200)
    assert (copy / "meta" / "info.json").is_file(), "the copy must still be written"
    page.wait_for_timeout(1500)
    assert page.evaluate(f"window.datasets?.[{str(copy)!r}] == null"), (
        "a copy made while browsing must not be opened"
    )


def test_an_in_flight_copy_shows_in_the_tree_and_clears_after(gui):
    """The copy is visible where its result will appear, not only in a status line."""
    page, root = gui
    page.on("dialog", lambda d: d.accept("slow_copy"))
    # Ensure expanded rather than toggle — the default source starts expanded,
    # so toggling would collapse it and hide the row under test.
    src = str(root.parent.parent)
    page.evaluate(f"if (!expandedSources.has({src!r})) toggleSource({src!r})")
    page.wait_for_timeout(800)

    # Hold the request open so the placeholder can be observed.
    page.evaluate(
        "window._realFetch = window.fetch;"
        "window.fetch = (u, o) => String(u).includes('/duplicate')"
        "  ? new Promise(r => { window._release = () => r(window._realFetch(u, o)); })"
        "  : window._realFetch(u, o);"
    )
    # `void` so evaluate does not await the call: the promise is deliberately
    # held open below, and awaiting it here would block until the release.
    page.evaluate(f"void duplicateDatasetAt({str(root)!r})")
    page.wait_for_selector(".source-dataset.copying", timeout=10_000)
    assert "slow_copy" in page.inner_text(".source-dataset.copying")

    page.evaluate("window._release()")
    page.wait_for_function("document.querySelector('.source-dataset.copying') == null", timeout=30_000)
    assert (root.parent / "slow_copy" / "meta" / "info.json").is_file()


def test_delete_needs_confirmation_and_then_removes_the_files(gui):
    """Dismissing the confirm must leave the dataset alone."""
    page, root = gui

    accept = False

    def on_dialog(d):
        d.accept() if accept else d.dismiss()

    page.on("dialog", on_dialog)
    page.evaluate(f"deleteDatasetFilesAt({str(root)!r})")
    page.wait_for_timeout(500)
    assert root.is_dir(), "a dismissed confirm must not delete anything"

    accept = True
    # Open it first: deleting an *opened* dataset is the case where the UI has
    # to catch up, and a delete that succeeds on disk while the tree keeps a
    # dead entry is the failure this covers.
    page.evaluate(f"openDataset({str(root)!r})")
    page.wait_for_function(f"window.datasets?.[{str(root)!r}] != null", timeout=20_000)

    page.evaluate(f"deleteDatasetFilesAt({str(root)!r})")
    deadline = time.monotonic() + 20
    while root.exists() and time.monotonic() < deadline:
        page.wait_for_timeout(200)
    assert not root.exists(), "accepting the confirm must delete the files"

    # The UI half. A throw anywhere after the request lands leaves the files
    # gone, the status claiming failure, and the tree still listing a dataset
    # whose directory no longer exists.
    page.wait_for_function(f"window.datasets?.[{str(root)!r}] == null", timeout=10_000)
    status = page.evaluate("document.getElementById('status')?.innerText || ''")
    assert "fail" not in status.lower(), f"delete succeeded but reported: {status!r}"
    assert root.name not in page.evaluate("document.getElementById('tree-container').innerText")
    # The Inspector describes whatever it last rendered; after a delete that is
    # a directory which no longer exists.
    inspector = page.evaluate("document.getElementById('inspector-body')?.innerText || ''")
    assert root.name not in inspector, f"Inspector still describes the deleted dataset: {inspector!r}"


def test_suggested_copy_name_is_derived_from_the_folder(gui):
    page, _root = gui
    assert page.evaluate("duplicateNameFor('/a/b/pick_place')") == "pick_place_copy"
    assert page.evaluate("duplicateNameFor('/a/b/pick_place/')") == "pick_place_copy"
    assert page.evaluate("duplicateNameFor('')") == "dataset_copy"
