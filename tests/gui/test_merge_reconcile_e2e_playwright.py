# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A reconciled merge driven through the browser, start to finish.

Everything else about this feature is tested a layer at a time: the dialog
against a hand-written payload, the endpoint against a stubbed merge, the
merge against real datasets. Each of those can pass while the assembled thing
does not, so this drives the actual flow -- open two datasets that disagree,
open the merge dialog, let the real validate call populate it, tick the box,
merge -- and then reads the dataset off disk to see it worked.
"""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.requires_playwright

from lerobot.datasets.dataset_tools import add_features_inplace  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

FLAG = "quality.human_flags"
LABELLED_VALUE = 7


@pytest.fixture
def merge_ui(tmp_path, monkeypatch, lerobot_dataset_factory, request):
    """A GUI serving one source that holds a labelled and an unlabelled dataset."""
    from lerobot.gui.api import datasets as datasets_api

    source = tmp_path / "source"
    source.mkdir()

    labelled = lerobot_dataset_factory(
        root=source / "labelled", repo_id="e2e/labelled", total_episodes=3, total_frames=30
    )
    plain = lerobot_dataset_factory(
        root=source / "plain", repo_id="e2e/plain", total_episodes=2, total_frames=20
    )
    add_features_inplace(labelled, {FLAG: (LABELLED_VALUE, {"dtype": "int64", "shape": (1,), "names": None})})

    # Keep the tree off the developer's real cache and real source registry.
    empty_home = tmp_path / "empty_hf_home"
    empty_home.mkdir()
    monkeypatch.setattr(datasets_api, "HF_LEROBOT_HOME", empty_home)
    sources_file = tmp_path / "dataset_sources.json"
    sources_file.write_text(
        json.dumps({"sources": [{"path": str(source), "removable": True, "expanded": True}]})
    )
    monkeypatch.setattr(datasets_api, "SOURCES_FILE", sources_file)

    # Booted only after the redirects above are in place.
    page = request.getfixturevalue("gui_page")
    return page, labelled, plain


def test_ticking_reconcile_in_the_dialog_merges_and_fills_the_column(merge_ui):
    page, labelled, plain = merge_ui
    labelled_frames, plain_frames = labelled.num_frames, plain.num_frames

    page.evaluate("switchTab('data')")
    # Both must be OPEN: the dialog's target list and the endpoint both work
    # off the datasets the GUI currently holds.
    for ds in (plain, labelled):
        page.evaluate("root => openDataset(root)", str(ds.root))
    page.wait_for_function("() => Object.keys(datasets).length >= 2", timeout=20_000)

    # The GUI keys open datasets by their path on disk, not by repo_id.
    ids = page.evaluate("() => Object.keys(datasets)")
    source_id, target_id = str(plain.root), str(labelled.root)
    assert {source_id, target_id} <= set(ids), f"GUI holds {ids}"

    # Merge the unlabelled one INTO the labelled one, which is the direction
    # that forces a fill on the side that was never labelled.
    page.evaluate("id => openMergeModal(id)", source_id)
    page.select_option("#merge-target-select", target_id)

    # The offer must come from a real /validate round trip, not from us.
    page.wait_for_selector("#merge-reconcile-row", state="visible", timeout=20_000)
    assert "Force merge" in page.locator("#merge-execute-btn").inner_text()

    page.check("#merge-reconcile")
    import os

    if os.environ.get("MERGE_E2E_SHOTS"):
        page.screenshot(path=os.environ["MERGE_E2E_SHOTS"] + "/e1-dialog.png")
    assert "reconcile" in page.locator("#merge-execute-btn").inner_text().lower()

    # A successful merge closes the dialog and toasts; #merge-status only ever
    # carries a failure, so waiting on it would hang on the happy path.
    page.on("dialog", lambda d: d.accept())
    failures = []
    page.on(
        "response",
        lambda r: failures.append((r.status, r.url)) if "merge-into" in r.url and r.status >= 400 else None,
    )
    page.click("#merge-execute-btn")

    page.wait_for_function(
        "() => document.getElementById('merge-modal-overlay').style.display === 'none'",
        timeout=60_000,
    )
    assert not failures, f"merge request failed: {failures}"
    assert "Merge Complete" in page.locator("body").inner_text()

    # The operator's own view has to catch up, not just the disk: the dialog
    # closes on success and the tree is refreshed asynchronously afterwards.
    page.wait_for_function(
        "id => datasets[id] && datasets[id].total_episodes === 5",
        arg=target_id,
        timeout=30_000,
    )
    if os.environ.get("MERGE_E2E_SHOTS"):
        page.screenshot(path=os.environ["MERGE_E2E_SHOTS"] + "/e2-complete.png")

    # What the operator was promised, read back off disk.
    merged = LeRobotDataset(labelled.repo_id, root=labelled.root)
    assert merged.num_frames == labelled_frames + plain_frames
    values = [int(torch.as_tensor(merged[i][FLAG]).flatten()[0]) for i in range(merged.num_frames)]
    assert sorted(set(values)) == [0, LABELLED_VALUE], f"unexpected values: {sorted(set(values))}"
    assert values.count(LABELLED_VALUE) == labelled_frames
    assert values.count(0) == plain_frames
