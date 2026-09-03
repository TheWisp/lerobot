# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Both side panels drag, clamp, and remember their width.

The Data tab's Inspector and the Run tab's Overlays panel share one
implementation, so both are driven here against the same expectations. That is
the point of the shared helper: the Inspector's behaviour is what the Run panel
was brought up to match, and a change to either must move both or neither.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.requires_playwright

MIN_W, MAX_W, DEFAULT_W = 220, 600, 320

# tab, resize-handle id, panel id, localStorage key
PANELS = [
    pytest.param("data", "inspector-resize", "inspector", "featureEditing.inspectorWidth", id="inspector"),
    pytest.param(
        "run", "run-overlays-resize", "overlays-panel-run", "run.overlaysPanelWidth", id="run-overlays"
    ),
]


def _open(page, tab: str, handle_id: str):
    page.evaluate(f"switchTab('{tab}')")
    page.wait_for_selector(f"#{handle_id}", state="attached", timeout=10_000)


def _width(page, panel_id: str) -> float:
    return page.locator(f"#{panel_id}").bounding_box()["width"]


def _drag(page, handle_id: str, dx: float):
    """Drag the handle by ``dx`` px. Leftward (negative) widens the panel."""
    box = page.locator(f"#{handle_id}").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + dx, box["y"] + box["height"] / 2, steps=5)
    page.mouse.up()


@pytest.mark.parametrize("tab,handle,panel,key", PANELS)
def test_a_drag_actually_changes_the_width(gui_page, tab, handle, panel, key):
    """The complement to the clamp tests: a panel that never moves would satisfy
    'stays within bounds' while being completely broken."""
    _open(gui_page, tab, handle)
    before = _width(gui_page, panel)
    _drag(gui_page, handle, -60)
    after = _width(gui_page, panel)
    assert after != before, f"width did not move at all (stayed {before}px)"
    assert MIN_W <= after <= MAX_W


@pytest.mark.parametrize("tab,handle,panel,key", PANELS)
def test_dragging_past_the_maximum_clamps(gui_page, tab, handle, panel, key):
    _open(gui_page, tab, handle)
    _drag(gui_page, handle, -2000)
    assert _width(gui_page, panel) == MAX_W


@pytest.mark.parametrize("tab,handle,panel,key", PANELS)
def test_dragging_past_the_minimum_clamps(gui_page, tab, handle, panel, key):
    _open(gui_page, tab, handle)
    _drag(gui_page, handle, 2000)
    assert _width(gui_page, panel) == MIN_W


@pytest.mark.parametrize("tab,handle,panel,key", PANELS)
def test_the_chosen_width_survives_a_reload(gui_page, tab, handle, panel, key):
    _open(gui_page, tab, handle)
    _drag(gui_page, handle, -2000)
    stored = gui_page.evaluate(f"() => localStorage.getItem({key!r})")
    assert stored == str(MAX_W), f"width was not persisted; localStorage held {stored!r}"

    gui_page.reload()
    gui_page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
    _open(gui_page, tab, handle)
    assert _width(gui_page, panel) == MAX_W, "a remembered width was not applied on load"


@pytest.mark.parametrize("tab,handle,panel,key", PANELS)
def test_a_stored_width_outside_the_bounds_is_ignored(gui_page, tab, handle, panel, key):
    """A hand-edited or stale entry did not come from this control, so it is
    ignored rather than clamped -- the panel keeps its 320px default."""
    gui_page.evaluate(f"() => localStorage.setItem({key!r}, '5000')")
    gui_page.reload()
    gui_page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
    _open(gui_page, tab, handle)
    assert _width(gui_page, panel) == DEFAULT_W


def test_the_run_panel_sizes_through_a_custom_property(gui_page):
    """Not an implementation detail: `.overlays-panel-run.collapsed` sets
    `width: auto`, which an inline `style.width` would beat. Carrying the width
    on the custom property is what lets collapsing still win."""
    _open(gui_page, "run", "run-overlays-resize")
    _drag(gui_page, "run-overlays-resize", -2000)
    inline, prop = gui_page.evaluate(
        "() => { const p = document.getElementById('overlays-panel-run');"
        " return [p.style.width, p.style.getPropertyValue('--run-overlays-width')]; }"
    )
    assert prop.strip() == f"{MAX_W}px"
    assert inline == "", f"width leaked onto inline style.width ({inline!r}), which beats the collapsed rule"
