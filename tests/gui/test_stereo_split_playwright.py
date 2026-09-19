# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The stereo-split migration path, driven through the browser.

The unit tests cover the split arithmetic and the job round-trip. What they
cannot see is whether an operator can reach any of it: the entry is hidden
unless the dataset is opened, the modal has to read feature shapes the client
copy does not carry, and the whole path exists only until the OpenArm2 back
catalogue is converted. A regression here reads as "the feature is missing"
rather than as a failure.

Screenshots are written to ``tmp_path`` when SPLIT_SHOT_DIR is set, which is how
the PR's evidence is produced; the assertions run either way.
"""

from __future__ import annotations

import os
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

REPO_ID = "test/stereo_ui"
STEREO = {
    # Even width, so it can split; the wrist camera is deliberately odd-one-out
    # so the modal has to choose rather than offer everything.
    "top": {"shape": (64, 128, 3), "names": ["height", "width", "channels"]},
    "wrist": {"shape": (64, 96, 3), "names": ["height", "width", "channels"]},
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _shot(page, name: str) -> None:
    out = os.environ.get("SPLIT_SHOT_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(out) / name), full_page=False)


@contextmanager
def _gui(hf_home: Path):
    from lerobot.gui import server as gui_server_mod

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

    with patch("lerobot.gui.api.datasets.HF_LEROBOT_HOME", hf_home), sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1280, "height": 800})
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
        use_videos=True,
        camera_features=STEREO,
    )
    with _gui(hf_home) as pg:
        yield pg, root


_EV = "{preventDefault(){},stopPropagation(){},clientX:300,clientY:210}"


def _entry(page) -> dict:
    return page.evaluate(
        "(() => { const e = document.getElementById('folder-ctx-split-stereo');"
        " return {text: e ? e.textContent.trim() : null,"
        "         shown: e ? e.style.display !== 'none' : false}; })()"
    )


def test_entry_is_marked_temporary_and_only_offered_on_an_opened_dataset(gui):
    """The label carries [TEMP], and a source folder is not offered the action.

    The marking is the only thing telling a reader this entry is a migration aid
    rather than a standing capability, and the menu is shared with source folders
    and model runs, which cannot be converted.
    """
    page, root = gui

    page.evaluate(f"showFolderContextMenu({_EV}, {str(root.parent)!r})")
    assert _entry(page)["shown"] is False, "a source folder must not offer the conversion"

    page.evaluate(f"openDataset({str(root)!r})")
    page.wait_for_function(f"window.datasets?.[{str(root)!r}] != null", timeout=20_000)
    page.evaluate(f"showFolderContextMenu({_EV}, {str(root)!r}, false, true)")

    state = _entry(page)
    _shot(page, "stereo-1-context-menu.png")
    assert state["shown"] is True, "an opened dataset must offer the conversion"
    assert state["text"].startswith("[TEMP]"), f"entry must be marked temporary, got {state['text']!r}"


def test_modal_names_the_eyes_a_camera_would_split_into(gui):
    """The modal is where feature shapes are read, so it is where an operator
    finds out whether a camera can split at all — and into what."""
    page, root = gui
    page.evaluate(f"openDataset({str(root)!r})")
    page.wait_for_function(f"window.datasets?.[{str(root)!r}] != null", timeout=20_000)

    page.evaluate(f"openSplitStereoModal({str(root)!r})")
    page.wait_for_function("document.body.innerText.includes('Split stereo camera')", timeout=20_000)
    page.wait_for_function("document.body.innerText.includes('top_l')", timeout=20_000)
    text = page.evaluate("document.body.innerText")
    _shot(page, "stereo-2-split-modal.png")

    assert "top_l" in text and "top_r" in text, "the modal must name the resulting channels"
    assert "128" in text, "the modal must show the source width it is halving"
    assert "not modified" in text, "the modal must say the source dataset survives"
