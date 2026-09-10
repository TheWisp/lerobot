# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The horizontal seams: Sources/Opened, Inspector/Overlays, cameras/timeline.

All three run through one implementation (``setupVerticalResize`` in
feature_editing.js), so they are driven here against the same expectations --
the same reason the two side panels share ``test_panel_resize_playwright.py``.

The property that matters most is the upper bound. The camera grid had none: it
could be dragged past the bottom of the window, and the timeline underneath --
which does not shrink -- was simply clipped away by ``.main``'s
``overflow: hidden``. Feature rows and the "+ Add feature" button vanished with
no scrollbar to reach them, which is the failure these tests pin down.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.requires_playwright

# handle id, pane id, the pane that yields, localStorage key, the pane's floor
SEAMS = [
    pytest.param(
        "sources-opened-resize",
        "data-sources-section",
        "opened-section",
        "featureEditing.sourcesHeight",
        80,
        id="sources-opened",
    ),
    pytest.param(
        "inspector-overlays-resize",
        "overlays-panel",
        "inspector-body",
        "featureEditing.overlaysHeight",
        90,
        id="inspector-overlays",
    ),
]


OVERFLOW = (
    "() => {"
    "  const main = document.querySelector('#tab-data .main');"
    "  const stack = document.querySelector('#tab-data .timeline-stack');"
    "  return stack.getBoundingClientRect().bottom - main.getBoundingClientRect().bottom;"
    "}"
)


def _height(page, element_id: str) -> float:
    return page.locator(f"#{element_id}").bounding_box()["height"]


def _settles_inside_the_pane(page, timeout: int = 5_000):
    """Wait for the timeline to sit inside the pane that clips it.

    Waiting on the condition rather than sleeping a fixed interval and then
    reading once: the layout settles when it settles, and a fixed sleep is both
    a floor on the runtime and a flake on a slower machine.
    """
    page.wait_for_function(f"({OVERFLOW})() <= 1", timeout=timeout)


def _press(page, handle, handle_id: str, attempts: int = 6) -> dict:
    """Press the handle, and confirm the press landed on it.

    A seam is a few pixels tall and the sidebar under it is still settling while
    sources scan, so a press aimed at a box read even a frame earlier can land
    on the section instead. That produces no movement and no stored height,
    which reads downstream as the feature being broken rather than as the
    gesture never having started -- so the press is verified here, against the
    class the handler adds, and re-aimed if it missed.
    """
    for _ in range(attempts):
        handle.hover()
        box = handle.bounding_box()
        page.mouse.down()
        if "dragging" in (handle.get_attribute("class") or ""):
            return box
        page.mouse.up()
        page.wait_for_timeout(150)
    raise AssertionError(f"could not land a press on #{handle_id} in {attempts} attempts")


def _drag(page, handle_id: str, dy: float):
    """Press the handle and drag it ``dy`` px down (negative is up)."""
    handle = page.locator(f"#{handle_id}")
    box = _press(page, handle, handle_id)
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y + dy, steps=6)
    page.mouse.up()


@pytest.mark.parametrize("handle,pane,yields,key,floor", SEAMS)
def test_a_drag_moves_the_pane(gui_page, handle, pane, yields, key, floor):
    """A seam that never moved would satisfy every clamp assertion below while
    being completely broken."""
    before = _height(gui_page, pane)
    # The Overlays panel grows upward, so the sign that enlarges it is the
    # opposite of the Sources list's; both directions are covered by the clamp
    # tests, and this one only has to observe movement.
    _drag(gui_page, handle, 120)
    assert _height(gui_page, pane) != before, f"#{pane} did not move at all (stayed {before}px)"


@pytest.mark.parametrize("handle,pane,yields,key,floor", SEAMS)
def test_the_pane_that_yields_keeps_a_floor(gui_page, handle, pane, yields, key, floor):
    """Dragging to the end of the sidebar must not leave the other section as a
    title bar with nothing under it."""
    for direction in (2000, -2000):
        _drag(gui_page, handle, direction)
        assert _height(gui_page, yields) >= 100, (
            f"#{yields} was squeezed to {_height(gui_page, yields)}px by dragging {direction}"
        )
        assert _height(gui_page, pane) >= floor - 1, (
            f"#{pane} went below its own floor at {_height(gui_page, pane)}px"
        )


@pytest.mark.parametrize("handle,pane,yields,key,floor", SEAMS)
def test_the_chosen_height_survives_a_reload(gui_page, handle, pane, yields, key, floor):
    _drag(gui_page, handle, 120)
    chosen = _height(gui_page, pane)
    stored = gui_page.evaluate(f"() => localStorage.getItem({key!r})")
    assert stored is not None, "the height was not persisted"

    gui_page.reload()
    gui_page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
    gui_page.wait_for_selector(f"#{handle}", state="attached", timeout=10_000)
    assert abs(_height(gui_page, pane) - chosen) <= 1, "a remembered height was not applied on load"


def test_the_overlays_handle_goes_away_when_the_panel_collapses(gui_page):
    """Collapsed, the panel is its header and nothing else. A handle under it
    would be a control with nothing left to size."""
    assert gui_page.locator("#inspector-overlays-resize").is_visible()
    gui_page.click("#overlays-caret")
    assert not gui_page.locator("#inspector-overlays-resize").is_visible()
    gui_page.click("#overlays-caret")
    assert gui_page.locator("#inspector-overlays-resize").is_visible()


def test_the_camera_grid_cannot_be_dragged_over_the_timeline(gui_page):
    """The bug this clamp exists for: the timeline does not shrink, so an
    unbounded grid pushed it off the bottom of a container that clips."""
    gui_page.wait_for_selector("#cameras-timeline-resize", state="attached", timeout=10_000)
    _drag(gui_page, "cameras-timeline-resize", 3000)
    _settles_inside_the_pane(gui_page)


def test_a_grid_height_from_a_populated_session_is_re_clamped_when_rows_appear(gui_page):
    """The bound moves without the window moving. The timeline is empty until a
    dataset is open, so a height remembered from a session that had one is
    restored against a bound that is briefly far too generous -- and nothing
    would correct it, because the window never changed size."""
    gui_page.wait_for_selector("#cameras-timeline-resize", state="attached", timeout=10_000)
    gui_page.evaluate("() => localStorage.setItem('featureEditing.cameraGridHeight', '5000')")
    gui_page.reload()
    gui_page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
    gui_page.wait_for_selector("#cameras-timeline-resize", state="attached", timeout=10_000)

    # Grow the stack the way opening a dataset does, without needing one on disk.
    gui_page.evaluate(
        "() => { const rows = document.getElementById('feature-rows');"
        "  for (let i = 0; i < 8; i++) {"
        "    const r = document.createElement('div');"
        "    r.className = 'feature-row'; r.textContent = 'row ' + i; rows.appendChild(r);"
        "  } }"
    )
    _settles_inside_the_pane(gui_page)


def test_a_dragged_pane_survives_a_trip_to_another_tab(gui_page):
    """The Data tab is `display: none` while another tab is up, so everything in
    it measures zero. Anything that recomputed a bound from those measurements
    concluded there was no room and pinned every dragged pane to its floor --
    for the rest of the session, since nothing re-read the stored height."""
    _drag(gui_page, "sources-opened-resize", 160)
    dragged = _height(gui_page, "data-sources-section")
    assert dragged > 150, f"the drag did not take: {dragged}px"

    gui_page.evaluate("switchTab('model')")
    gui_page.wait_for_timeout(300)
    gui_page.evaluate("switchTab('data')")
    gui_page.wait_for_timeout(300)

    assert abs(_height(gui_page, "data-sources-section") - dragged) <= 1, (
        "the pane collapsed while the tab was hidden"
    )


def test_a_window_resize_from_another_tab_does_not_destroy_the_chosen_height(gui_page):
    """The same zero-measurement, reached without a ResizeObserver.

    A shorter window legitimately squeezes the pane -- once the pane that yields
    is at its floor, the sized one is what gives. What must not happen is the
    chosen height being *forgotten*: restoring the window has to restore it.
    """
    before = gui_page.viewport_size
    _drag(gui_page, "sources-opened-resize", 160)
    dragged = _height(gui_page, "data-sources-section")

    gui_page.evaluate("switchTab('run')")
    gui_page.set_viewport_size({"width": before["width"], "height": before["height"] - 200})
    gui_page.wait_for_timeout(200)
    gui_page.evaluate("switchTab('data')")
    gui_page.set_viewport_size(before)
    gui_page.wait_for_timeout(300)

    assert abs(_height(gui_page, "data-sources-section") - dragged) <= 1, (
        "the height chosen by the drag did not come back with the window"
    )


def test_a_fractional_height_is_remembered_as_itself(gui_page):
    """The height was read back out of a serialized `flex` string with a regex
    that matched the digits after the decimal point -- `0 1 123.5px` gave 5 --
    so a drag landing off a whole pixel was stored as a number the pane could
    never have been."""
    gui_page.evaluate(
        "() => { const p = document.getElementById('overlays-panel');"
        "  p.classList.add('sized'); p.style.setProperty('--pane-h', '123.5px'); }"
    )
    _drag(gui_page, "inspector-overlays-resize", -40)
    stored = int(gui_page.evaluate("() => localStorage.getItem('featureEditing.overlaysHeight')"))
    shown = _height(gui_page, "overlays-panel")
    assert abs(stored - shown) <= 1, f"stored {stored}px for a pane showing {shown}px"


def test_a_shorter_window_pulls_the_camera_grid_back(gui_page):
    """A height chosen on a tall window is not still valid on a short one, and
    nothing else would notice: the timeline is `flex-shrink: 0`."""
    gui_page.wait_for_selector("#cameras-timeline-resize", state="attached", timeout=10_000)
    _drag(gui_page, "cameras-timeline-resize", 3000)

    gui_page.set_viewport_size({"width": 1400, "height": 620})
    _settles_inside_the_pane(gui_page)
