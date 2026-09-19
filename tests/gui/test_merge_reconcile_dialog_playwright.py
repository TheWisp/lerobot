# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The reconcile offer appears only where reconciliation can help.

Before this, a feature mismatch left one way forward: "Force merge (skip
validation)". Reconciling and forcing are different decisions -- one makes the
two schemas agree and then validates, the other stops checking -- so the
checkbox has to be visible when the first is possible, absent when it is not,
and must never leave the run on the force path once ticked.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.requires_playwright

# Shaped exactly as ``_validate_merge_compat`` builds them, keys included --
# a fixture that drifts from the validator tests a dialog nobody will ever
# open. ``test_the_fixtures_match_what_the_validator_emits`` holds them to it.
FEATURE_MISMATCH = {
    "compatible": False,
    "mismatches": [
        {
            "field": "features",
            "target_only": ["quality.human_flags"],
            "source_only": [],
            "shared_diff": {},
        }
    ],
}
FPS_MISMATCH = {
    "compatible": False,
    "mismatches": [{"field": "fps", "target": 30, "source": 60}],
}


def _render(page, validation):
    """Open the merge dialog and hand its renderer a validate response.

    The dataset selection ahead of this is skipped -- it needs two datasets on
    disk with differing schemas -- but everything from the validate response
    onward is the real path: the real overlay, the real renderer, the real
    controls. The payload is shaped as ``_validate_merge_compat`` builds it,
    and a guard below keeps it that way.
    """
    page.evaluate("() => { document.getElementById('merge-modal-overlay').style.display = 'flex'; }")
    page.evaluate("v => _renderMergeDiff(v)", validation)


def _visible(page, sel):
    return page.eval_on_selector(sel, "el => getComputedStyle(el).display !== 'none'")


def test_a_feature_mismatch_offers_reconciliation(gui_page):
    _render(gui_page, FEATURE_MISMATCH)
    assert _visible(gui_page, "#merge-reconcile-row")
    assert gui_page.locator("#merge-reconcile").is_checked() is False
    assert "Force merge" in gui_page.locator("#merge-execute-btn").inner_text()


def test_a_non_feature_mismatch_does_not(gui_page):
    """Reconciliation cannot settle a differing fps, so offering it would
    promise something the merge will still refuse to do."""
    _render(gui_page, FPS_MISMATCH)
    assert not _visible(gui_page, "#merge-reconcile-row")


def test_a_compatible_pair_does_not(gui_page):
    _render(gui_page, {"compatible": True, "mismatches": []})
    assert not _visible(gui_page, "#merge-reconcile-row")


def test_ticking_the_box_takes_the_run_off_the_force_path(gui_page):
    """The button is the only thing telling the operator which of the two
    decisions they are about to make."""
    _render(gui_page, FEATURE_MISMATCH)
    gui_page.check("#merge-reconcile")
    label = gui_page.locator("#merge-execute-btn").inner_text()
    assert "reconcile" in label.lower()
    assert "force" not in label.lower()

    gui_page.uncheck("#merge-reconcile")
    assert "Force merge" in gui_page.locator("#merge-execute-btn").inner_text()


def test_a_tick_does_not_follow_the_dialog_to_the_next_target(gui_page):
    """Re-validating against a compatible target must clear the offer, or a
    box ticked for one pair silently reshapes a different one."""
    _render(gui_page, FEATURE_MISMATCH)
    gui_page.check("#merge-reconcile")
    _render(gui_page, {"compatible": True, "mismatches": []})
    assert gui_page.locator("#merge-reconcile").is_checked() is False
    assert not _visible(gui_page, "#merge-reconcile-row")


def test_the_fixtures_match_what_the_validator_emits():
    """The dialog tests above drive a hand-written validation payload. If the
    validator's shape moves, they would keep passing against a payload the GUI
    never receives -- which is how a dialog test outlives the dialog."""
    import inspect

    from lerobot.gui.api import edits

    src = inspect.getsource(edits._validate_merge_compat)
    for key in ("target_only", "source_only", "shared_diff"):
        assert f'"{key}"' in src, f"features mismatch no longer carries {key}"
    assert '"field": "features"' in src
    assert '{"field": "fps", "target": ' in src, "fps mismatch keys changed"
