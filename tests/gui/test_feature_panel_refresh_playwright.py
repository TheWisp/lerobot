# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A mask save has to become visible without reloading the page.

The feature panel caches two things it cannot see through: the dataset schema
(which gains `observation.masks.*` on a first adopt, and carries the treatment
each lane names) and a per-episode series payload. Both are keyed by dataset
and episode, so a write made by something other than the panel — a mask save,
an effects apply — leaves the keys valid and the contents stale. Reported from
the field as "after saving, it did not refresh the feature display of the
episode, and it took me a refresh".

`FeatureEditing.refreshFromServer` is the way back: re-read the dataset, drop
the caches, re-render. What is pinned here is the schema half — a vocabulary
the panel never saw written becomes visible without a reload, which is the
half the report was about (a saved object with no lane).

Two things are deliberately not covered. The series half — stale per-frame
presence spans — would need the episode's parquet rewritten mid-test rather
than its metadata, and the lane labels this asserts on come from the schema, so
it would not be caught here; removing the series-cache drop leaves this test
green. And the wiring, that the save handler and the effects apply both call
this, is one line at each call site and is left to review. Both are stated
rather than implied, because a test that looks like it covers the whole
mechanism is worse than one that says which half it holds.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from tests.datasets.test_saved_masks_training import masked_dataset_root  # noqa: E402,F401

pytestmark = pytest.mark.requires_playwright

NEW_LABEL = "wooden block"


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
        pg = browser.new_page(viewport={"width": 1600, "height": 1000})
        pg.goto(base_url)
        pg.wait_for_function("typeof openDataset === 'function'", timeout=15_000)
        yield pg
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


def _lane_labels(pg) -> list[str]:
    return pg.evaluate(
        "() => [...document.querySelectorAll('.row-mask-name')].map(e => e.textContent.trim())"
    )


def _add_label_on_disk(root, label: str) -> None:
    """Do to the metadata what a mask save does: append to the vocabulary.

    Appending is the safe half of the save's contract — stored label ids keep
    their meaning — which is why it is the change worth simulating here.
    """
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for name, feat in info["features"].items():
        if name.startswith("observation.masks.") and "mask_labels" in feat:
            feat["mask_labels"] = [*feat["mask_labels"], label]
            feat.setdefault("mask_treatments", {})[label] = {"key": "none", "params": {}}
    info_path.write_text(json.dumps(info, indent=2))


def test_a_write_the_panel_did_not_make_shows_without_a_reload(masked_dataset_root, page):  # noqa: F811
    root, _repo_id = masked_dataset_root
    ds_id = str(root)

    page.evaluate("(ds) => openDataset(ds)", ds_id)
    page.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
    length = page.evaluate(
        "async (ds) => (await (await fetch(`/api/datasets/${encodeURIComponent(ds)}/episodes`)).json())[0].length",
        ds_id,
    )
    page.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [ds_id, length])
    page.wait_for_function("() => document.querySelectorAll('.row-mask-name').length > 0", timeout=60_000)
    before = _lane_labels(page)
    assert before, "no mask lanes rendered, so this test could not tell a refresh from a no-op"
    assert not any(NEW_LABEL in x for x in before)

    _add_label_on_disk(root, NEW_LABEL)

    # Without the refresh the panel keeps serving its cached schema and series,
    # and this is exactly where the reload used to be required.
    page.evaluate("async (ds) => await window.FeatureEditing.refreshFromServer(ds)", ds_id)
    page.wait_for_timeout(2000)
    after = _lane_labels(page)

    assert any(NEW_LABEL in x for x in after), (
        f"the panel still shows the pre-write vocabulary after a refresh: {after}"
    )
    assert len(after) > len(before)
