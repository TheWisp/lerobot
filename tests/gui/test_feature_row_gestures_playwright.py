# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The feature row's own gestures, pinned where nothing pinned them.

These are characterization tests: they describe behaviour that predates the
lane-editing work and are expected to pass unchanged on the base branch. They
exist because that work modified `wireFeatureRowTrack` and `renderFeatureRow`,
and neither the trim guard, the scrub-without-selecting escape hatch, the
Escape key, nor the value-edit pending band had a test standing behind them --
so a change to those functions could have altered any of them silently.

Written to be boring on purpose: every wait is on an observed condition rather
than a duration, every coordinate is measured after the render that precedes
it, and every wait is bounded by a condition the page can report.

The selection is read from the band on screen rather than from the module's
own accessor, because that accessor is added by the change under test and a
characterization test has to be runnable against the branch the behaviour
predates -- otherwise it cannot show that the behaviour predates it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("playwright.sync_api")
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

pytestmark = pytest.mark.requires_playwright

FRAMES = 60
FPS = 10
ROW = '.row-track[data-feature="reward"]'


@pytest.fixture
def gestures_root(tmp_path) -> Path:
    """A synthesised episode with one editable scalar and one bitset column."""
    root = tmp_path / "gestures"
    ds = LeRobotDataset.create(
        repo_id="tests/gestures",
        fps=FPS,
        root=root,
        features={
            "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            "reward": {"dtype": "float32", "shape": (1,), "names": None},
            "quality": {"dtype": "int64", "shape": (1,), "names": None, "flags": ["blurry"]},
        },
        use_videos=False,
    )
    for frame in range(FRAMES):
        # Both columns must VARY within the episode. A column that is uniform
        # across every frame is detected as per-episode broadcast and is
        # deliberately kept off the timeline -- it becomes an Inspector card
        # instead -- so a constant fixture would leave these tests waiting for
        # a row that is correctly not there.
        ds.add_frame(
            {
                "action": torch.zeros(2, dtype=torch.float32),
                "observation.state": torch.zeros(2, dtype=torch.float32),
                "reward": torch.tensor([float(frame % 5)], dtype=torch.float32),
                "quality": torch.tensor([1 if 20 <= frame < 40 else 0], dtype=torch.int64),
                "task": "gestures",
            }
        )
    ds.save_episode()
    ds.finalize()
    return root


@pytest.fixture
def page(gestures_root, gui_page):
    pg = gui_page
    pg.wait_for_function("typeof openDataset === 'function'", timeout=15_000)
    ds_id = str(gestures_root)
    pg.evaluate("(ds) => openDataset(ds)", ds_id)
    pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
    pg.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [ds_id, FRAMES])
    pg.wait_for_function(f"() => document.querySelector('{ROW}')", timeout=60_000)
    # The row is only usable once its series has arrived and it has been drawn
    # from it; waiting on the SVG rather than on a duration is what keeps the
    # coordinates below meaningful.
    pg.wait_for_function(f"() => document.querySelector('{ROW} svg, {ROW} .row-flag-name')", timeout=60_000)
    pg.ds_id = ds_id
    return pg


def _box(pg):
    """Measured fresh every time: staging and selecting both re-render the row
    list, so a box captured earlier points at a node that no longer exists."""
    return pg.evaluate(
        """(sel) => { const r = document.querySelector(sel).getBoundingClientRect();
             return {x: r.x, y: r.y, w: r.width, h: r.height}; }""",
        ROW,
    )


def _x(pg, frame: int) -> float:
    b = _box(pg)
    return b["x"] + b["w"] * ((frame + 0.5) / FRAMES)


def _y(pg) -> float:
    return _box(pg)["y"] + _box(pg)["h"] * 0.5


def _selection(pg):
    """The selected range in frames, read from the band on screen.

    Deliberately not `_internals.currentSelection()`: that accessor is added by
    the change under test, and a characterization test has to be runnable
    against the branch this behaviour predates or it cannot show the behaviour
    predates it. The band is also the thing the operator actually sees.
    """
    return pg.evaluate(
        """(frames) => {
            const el = document.querySelector('.row-selection');
            if (!el) return null;
            const track = el.closest('.row-track').getBoundingClientRect();
            const r = el.getBoundingClientRect();
            const f = (v) => Math.round((v * frames) / track.width);
            return {frameFrom: f(r.x - track.x), frameTo: f(r.x - track.x + r.width)};
        }""",
        FRAMES,
    )


def _drag(pg, frm: int, to: int) -> None:
    y = _y(pg)
    pg.mouse.move(_x(pg, frm), y)
    pg.mouse.down()
    pg.mouse.move(_x(pg, to), y, steps=8)
    pg.mouse.up()
    pg.wait_for_function(
        """([a, b, frames]) => {
            const el = document.querySelector('.row-selection');
            if (!el) return false;
            const t = el.closest('.row-track').getBoundingClientRect();
            const r = el.getBoundingClientRect();
            const f = (v) => Math.round((v * frames) / t.width);
            return f(r.x - t.x) === a && f(r.x - t.x + r.width) === b;
        }""",
        arg=[min(frm, to), max(frm, to) + 1, FRAMES],
        timeout=10_000,
    )


def test_a_click_outside_the_trim_envelope_selects_nothing(page):
    """The row rejects presses outside `[trim_from, trim_to)`.

    The envelope is set by dragging the real trim handle rather than by poking
    the globals: `window.trimStart` / `window.trimEnd` are getter-only views of
    app.js's own bindings, so an assignment from a test is silently ignored and
    the test would pass against an envelope that never moved. Where the handle
    lands in frames is then read back rather than assumed, so the assertions do
    not depend on the pixel-to-frame rounding of whatever width the row got.
    """
    tl = page.evaluate(
        """() => { const r = document.getElementById('timeline').getBoundingClientRect();
             return {x: r.x, y: r.y, w: r.width, h: r.height}; }"""
    )
    handle = page.evaluate(
        """() => { const r = document.getElementById('trim-handle-left').getBoundingClientRect();
             return {x: r.x + r.width / 2, y: r.y + r.height / 2}; }"""
    )
    page.mouse.move(handle["x"], handle["y"])
    page.mouse.down()
    page.mouse.move(tl["x"] + tl["w"] * 0.25, handle["y"], steps=12)
    page.mouse.up()
    # The trim landing is the condition, not a duration.
    page.wait_for_function("() => window.trimStart > 2", timeout=10_000)
    page.wait_for_function("() => document.querySelectorAll('.row-trim-dim').length > 0", timeout=10_000)

    trim_from = page.evaluate("() => window.trimStart")
    page.keyboard.press("Escape")
    page.wait_for_function("() => document.querySelectorAll('.row-selection').length === 0", timeout=10_000)

    # The band is drawn synchronously in the row's mousedown handler, so
    # by the time the click round-trip returns there is nothing left to wait
    # for -- and waiting a fixed time to observe an absence is what makes this
    # kind of test flaky.
    page.mouse.click(_x(page, trim_from - 3), _y(page))
    assert _selection(page) is None, f"a click at {trim_from - 3}, before the trim start, selected"

    page.mouse.click(_x(page, trim_from + 5), _y(page))
    page.wait_for_function("() => document.querySelectorAll('.row-selection').length > 0", timeout=10_000)
    sel = _selection(page)
    assert sel is not None and sel["frameFrom"] == trim_from + 5, (
        f"a click inside the envelope gave {sel}; the two nulls above would then "
        "mean the row was inert rather than the guard firing"
    )


def test_dragging_the_playhead_thumb_scrubs_without_reselecting(page):
    """The documented navigate-without-select escape hatch: the scrubber moves
    the playhead and must leave the selection alone, which is what makes it
    possible to look around without losing the range you are editing."""
    _drag(page, 5, 25)
    before = _selection(page)
    assert before is not None

    tl = page.evaluate(
        """() => { const r = document.getElementById('timeline').getBoundingClientRect();
             return {x: r.x, y: r.y, w: r.width, h: r.height}; }"""
    )
    thumb = page.evaluate(
        """() => { const r = document.getElementById('timeline-scrubber').getBoundingClientRect();
             return {x: r.x + r.width / 2, y: r.y + r.height / 2}; }"""
    )
    frame_before = page.evaluate("() => window.currentFrame")
    page.mouse.move(thumb["x"], thumb["y"])
    page.mouse.down()
    page.mouse.move(tl["x"] + tl["w"] * 0.8, thumb["y"], steps=10)
    page.mouse.up()
    # Wait for the scrub to land rather than for a duration: the assertion
    # below is only meaningful once the playhead has actually moved, and a
    # fixed sleep would make this test's outcome depend on machine speed.
    page.wait_for_function("(f) => window.currentFrame !== f", arg=frame_before, timeout=10_000)

    after = _selection(page)
    assert after is not None, "scrubbing cleared the selection"
    assert (after["frameFrom"], after["frameTo"]) == (before["frameFrom"], before["frameTo"]), (
        f"scrubbing moved the selection from "
        f"{(before['frameFrom'], before['frameTo'])} to {(after['frameFrom'], after['frameTo'])}"
    )


def test_escape_clears_the_selection_and_the_band(page):
    """Escape is the way out of a selection, and the band must go with it --
    the model and the screen agreeing is the property the lane gesture now
    depends on."""
    _drag(page, 10, 30)
    assert _selection(page) is not None
    page.wait_for_function("() => document.querySelectorAll('.row-selection').length > 0", timeout=10_000)

    page.keyboard.press("Escape")
    page.wait_for_function("() => document.querySelectorAll('.row-selection').length === 0", timeout=10_000)
    assert _selection(page) is None


def test_a_staged_value_edit_is_visible_under_show_pending_edits(page):
    """The `feature_set` band -- the branch that existed before the flag and
    mask ones were added beside it. Three near-identical branches now switch on
    edit type in one function, so the oldest of them needs a test of its own."""
    _drag(page, 12, 28)
    page.evaluate(
        """async ([ds]) => {
            await fetch('/api/edits/feature-set', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({dataset_id: ds, episode_index: 0, feature: 'reward',
                                      frame_from: 12, frame_to: 28, value: 1.0}),
            });
            await window.refreshPendingEdits();
        }""",
        [page.ds_id],
    )
    page.wait_for_function(
        "() => (window.pendingEdits || []).some(e => e.edit_type === 'feature_set')", timeout=10_000
    )

    page.evaluate(
        """() => { const cb = document.getElementById('show-pending-edits-toggle');
             cb.checked = true; window.onShowPendingEditsToggle(); }"""
    )
    page.wait_for_function(
        """(sel) => document.querySelectorAll(
             `${sel} .row-pending-overlay:not(.flag-pending):not(.mask-pending)`).length === 1""",
        arg=ROW,
        timeout=10_000,
    )

    span = page.evaluate(
        """(sel) => { const t = document.querySelector(sel);
             const el = t.querySelector('.row-pending-overlay:not(.flag-pending):not(.mask-pending)');
             const tr = t.getBoundingClientRect(), r = el.getBoundingClientRect();
             const f = (v) => Math.round(v * 60 / tr.width);
             return [f(r.x - tr.x), f(r.x - tr.x + r.width)]; }""",
        ROW,
    )
    assert span == [12, 28], f"the band covered {span}, not the staged 12-28"


def test_the_origin_row_only_moves_the_inspector_highlight(page):
    """What the selection's `focusRow` is allowed to do, stated as a test.

    The row a drag starts on decides which Inspector card is highlighted and
    nothing else. It used to decide which lane could be edited, which made the
    selection band a lie on every other row -- a range dragged here left the
    others looking selected and inert. Pinning the rule keeps it from quietly
    becoming a gate again.
    """
    rows = page.evaluate(
        """() => [...document.querySelectorAll('.row-track')]
             .map(t => t.getAttribute('data-feature')).filter(Boolean)"""
    )
    # `reward` and `quality` both have a row AND a card; recorded features have
    # a row and no card, and highlight nothing, which is still no difference.
    carded = [r for r in rows if r in ("reward", "quality")]
    assert len(carded) >= 2, f"fixture should give two carded rows, got {rows}"

    def drag_on(feature, frm, to):
        b = page.evaluate(
            """(f) => { const r = document.querySelector(`.row-track[data-feature="${f}"]`)
                 .getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height}; }""",
            feature,
        )
        y = b["y"] + b["h"] * 0.5

        def xf(n):
            return b["x"] + b["w"] * ((n + 0.5) / FRAMES)

        page.mouse.move(xf(frm), y)
        page.mouse.down()
        page.mouse.move(xf(to), y, steps=8)
        page.mouse.up()
        page.wait_for_function(
            "(f) => { const c = document.querySelector('#inspector-body .feature-card.focused');"
            "  return !!c && c.getAttribute('data-feature') === f; }",
            arg=feature,
            timeout=10_000,
        )

    for feature in carded:
        drag_on(feature, 5, 25)
        focused = page.evaluate(
            "() => [...document.querySelectorAll('#inspector-body .feature-card.focused')]"
            ".map(c => c.getAttribute('data-feature'))"
        )
        assert focused == [feature], f"dragging on {feature} highlighted {focused}"

        # The selection itself is identical whichever row it came from.
        sel = _selection(page)
        assert (sel["frameFrom"], sel["frameTo"]) == (5, 26), (
            f"a selection dragged on {feature} came out as {sel}"
        )
        drawn = page.evaluate("() => document.querySelectorAll('.row-selection').length")
        assert drawn == len(rows), (
            f"the band is drawn on {drawn} of {len(rows)} rows; it claims to cover them all"
        )
