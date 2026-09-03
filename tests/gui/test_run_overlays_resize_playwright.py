# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The Run tab's Overlays panel resizes, clamps, and remembers its width.

The width is carried on a CSS custom property rather than an inline ``width``
so the collapsed state can keep overriding it. That indirection is why these
assertions read ``--run-overlays-width`` instead of the element's box: a test
that measured the box would also pass while collapsed, which is the one case
the property exists to protect.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.requires_playwright

MIN_W, MAX_W, DEFAULT_W = 220, 600, 320
STORAGE_KEY = "run.overlaysPanelWidth"


def _open_run_tab(page):
    page.evaluate("switchTab('run')")
    page.wait_for_selector("#run-overlays-resize", state="attached", timeout=10_000)


def _width(page) -> float:
    return page.evaluate(
        "() => parseFloat(getComputedStyle(document.getElementById('overlays-panel-run'))"
        ".getPropertyValue('--run-overlays-width'))"
    )


def _drag(page, dx: float):
    """Drag the handle by ``dx`` px. Leftward (negative) widens the panel."""
    box = page.locator("#run-overlays-resize").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + dx, box["y"] + box["height"] / 2, steps=5)
    page.mouse.up()


def test_a_drag_actually_changes_the_width(gui_page):
    """The complement to the clamp tests: a panel that never moves would satisfy
    'stays within bounds' while being completely broken."""
    _open_run_tab(gui_page)
    before = _width(gui_page)
    _drag(gui_page, -60)
    after = _width(gui_page)
    assert after != before, f"width did not move at all (stayed {before}px)"
    assert MIN_W <= after <= MAX_W


def test_dragging_past_the_maximum_clamps(gui_page):
    _open_run_tab(gui_page)
    _drag(gui_page, -2000)
    assert _width(gui_page) == MAX_W


def test_dragging_past_the_minimum_clamps(gui_page):
    _open_run_tab(gui_page)
    _drag(gui_page, 2000)
    assert _width(gui_page) == MIN_W


def test_the_chosen_width_survives_a_reload(gui_page):
    _open_run_tab(gui_page)
    _drag(gui_page, -2000)
    stored = gui_page.evaluate(f"() => localStorage.getItem({STORAGE_KEY!r})")
    assert stored == str(MAX_W), f"width was not persisted; localStorage held {stored!r}"

    gui_page.reload()
    gui_page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
    _open_run_tab(gui_page)
    assert _width(gui_page) == MAX_W, "a remembered width was not applied on load"


def test_a_stored_width_outside_the_bounds_is_ignored(gui_page):
    """A hand-edited or stale entry must not widen the panel past what the drag
    itself would allow.

    An out-of-range value is rejected by leaving the custom property unset, so
    the panel falls back to the ``var(--run-overlays-width, 320px)`` default.
    That is measured on the rendered box rather than the property, because an
    unset property reads as empty and would make any comparison vacuously true.
    """
    gui_page.evaluate(f"() => localStorage.setItem({STORAGE_KEY!r}, '5000')")
    gui_page.reload()
    gui_page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
    _open_run_tab(gui_page)
    assert gui_page.locator("#overlays-panel-run").bounding_box()["width"] == DEFAULT_W
