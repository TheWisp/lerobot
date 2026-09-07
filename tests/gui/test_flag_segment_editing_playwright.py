# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Clicking a flag lane, in a real browser.

The flag row and the mask row now share their lane geometry and their click
gesture (``timeline_lanes.js``), and the parts that broke on the mask row are
exactly the parts no unit test can reach: the row's own ``mousedown`` seeks and
replaces the selection before any bubble-phase handler runs, and staging
re-renders the row, replacing the node between press and release. This file
drives the flag row through the same ground.

What is specific to flags is the two-state toggle. A mask lane's click has a
third state to refuse -- an absent stretch carries nothing to mute -- so its
tests never covered "a run that is not there is exactly what you want to
create", which for flags is the ordinary case: every run is actionable, and the
direction comes from the run under the pointer rather than from the selection.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("playwright.sync_api")
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

pytestmark = pytest.mark.requires_playwright

FLAGS = ["blurry", "fumble", "occluded"]
FUMBLE = FLAGS.index("fumble")
FRAMES = 60
FPS = 10
ROW = '.row-track[data-feature="quality"]'

# `fumble` carries frames 20..40 and nothing else, so lane 1 reads
# clear [0,20) / set [20,40) / clear [40,60): a run to set, a run to clear, and
# a boundary for a selection to straddle.
SET_FROM, SET_TO = 20, 40


@pytest.fixture
def flagged_root(tmp_path) -> Path:
    """A synthetic episode with one bitset column. Synthesised, never a real
    dataset: these tests stage edits against it."""
    root = tmp_path / "flagdemo"
    ds = LeRobotDataset.create(
        repo_id="tests/flagdemo",
        fps=FPS,
        root=root,
        features={
            "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            "quality": {"dtype": "int64", "shape": (1,), "names": None, "flags": FLAGS},
        },
        use_videos=False,
    )
    for frame in range(FRAMES):
        carried = (1 << FUMBLE) if SET_FROM <= frame < SET_TO else 0
        ds.add_frame(
            {
                "action": torch.zeros(2, dtype=torch.float32),
                "observation.state": torch.zeros(2, dtype=torch.float32),
                "quality": torch.tensor([carried], dtype=torch.int64),
                "task": "flagging",
            }
        )
    ds.save_episode()
    ds.finalize()
    return root


@pytest.fixture
def page(flagged_root, gui_page):
    """The shared booted page, with the demo dataset open on episode 0.

    `gui_page` rather than a twelfth copy of the boot sequence: the copies had
    already drifted, and several never join the server thread — a leaked server
    outlives the monkeypatch that redirected its config and writes the
    developer's real `opened_datasets.json`.
    """
    pg = gui_page
    pg.wait_for_function("typeof openDataset === 'function'", timeout=15_000)
    ds_id = str(flagged_root)
    pg.evaluate("(ds) => openDataset(ds)", ds_id)
    pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
    pg.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [ds_id, FRAMES])
    pg.wait_for_function(f"() => document.querySelector('{ROW}')", timeout=60_000)
    return pg


def _track_box(pg):
    return pg.evaluate(
        """(sel) => {
            const r = document.querySelector(sel).getBoundingClientRect();
            return {x: r.x, y: r.y, w: r.width, h: r.height};
        }""",
        ROW,
    )


def _point(box, frame: int, lane: int) -> tuple[float, float]:
    """The centre of `frame`'s bar in `lane`.

    Read from the page rather than restated here: the lane band is
    ``timeline_lanes.js``'s to define, and a test that recomputed it would keep
    passing after the drawing and the hit test drifted apart -- the one defect
    this geometry exists to prevent.
    """
    return (box["x"] + box["w"] * ((frame + 0.5) / FRAMES), box["y"] + box["h"] * (lane / 100))


def _lane_mid_pct(pg, lane: int) -> float:
    return pg.evaluate("(l) => window.TimelineLanes.geometry(3).mid(l)", lane)


def _select(pg, frm: int, to: int) -> None:
    """Drag-select a frame range, the way an operator does.

    Along the row's top margin, which belongs to no lane: dragging across a
    lane while a selection already covers it would be claimed as a toggle,
    which is the gesture's own rule (a press that acts on a selection can never
    also make one).
    """
    box = _track_box(pg)
    y = box["y"] + box["h"] * 0.04
    pg.mouse.move(box["x"] + box["w"] * ((frm + 0.5) / FRAMES), y)
    pg.mouse.down()
    pg.mouse.move(box["x"] + box["w"] * ((to - 0.5) / FRAMES), y, steps=8)
    pg.mouse.up()
    pg.wait_for_timeout(200)


def _click_frame(pg, frame: int, lane: int) -> None:
    """Click a frame in a lane, re-measuring first: the row list reflows when
    an edit is staged, so coordinates taken before a click point elsewhere
    after it."""
    x, y = _point(_track_box(pg), frame, _lane_mid_pct(pg, lane))
    pg.mouse.click(x, y)
    pg.wait_for_timeout(800)


def _pending(pg):
    return pg.evaluate("() => (window.pendingEdits || []).filter(e => e.edit_type === 'feature_bits')")


def _one(pg):
    edits = _pending(pg)
    assert len(edits) == 1, f"expected one staged edit, got {edits}"
    p = edits[0]["params"]
    return p["set_bits"], p["clear_bits"], (p["frame_from"], p["frame_to"])


def test_clicking_a_clear_run_sets_the_flag_over_the_selection(page):
    """The ordinary case, and the one a mask lane has no equivalent of: the run
    under the pointer carries nothing, and that is exactly what is being
    created."""
    _select(page, 0, SET_FROM)
    _click_frame(page, 10, FUMBLE)

    set_bits, clear_bits, span = _one(page)
    assert set_bits == 1 << FUMBLE, "clicking an unset run did not set the flag"
    assert clear_bits == 0
    assert span == (0, SET_FROM), (
        f"the edit covered {span}, not the selection — the row's mousedown "
        "replaced the selection before the click ran"
    )


def test_clicking_a_set_run_clears_it(page):
    """Direction comes from the run, so the same gesture reverses here."""
    _select(page, SET_FROM, SET_TO)
    _click_frame(page, 30, FUMBLE)

    set_bits, clear_bits, span = _one(page)
    assert (set_bits, clear_bits) == (0, 1 << FUMBLE), "clicking a set run did not clear it"
    assert span == (SET_FROM, SET_TO)


def test_a_selection_straddling_a_boundary_edits_only_the_run_clicked(page):
    """The reported behaviour, and the reason the scope is positional rather
    than the whole selection: with `[....XXX]` selected, pointing at the dots
    fills them in and leaves the X's alone."""
    _select(page, 10, 30)
    _click_frame(page, 15, FUMBLE)

    set_bits, clear_bits, span = _one(page)
    assert (set_bits, clear_bits) == (1 << FUMBLE, 0)
    assert span == (10, SET_FROM), (
        f"the edit covered {span}; a selection straddling the boundary must stop at it, "
        "or clicking the unset half also rewrites the set half"
    )


def test_the_same_selection_clears_when_the_set_half_is_clicked(page):
    """The other half of the same selection, giving the opposite edit — which
    is what a single answer for the whole range could not express."""
    _select(page, 10, 30)
    _click_frame(page, 25, FUMBLE)

    set_bits, clear_bits, span = _one(page)
    assert (set_bits, clear_bits) == (0, 1 << FUMBLE)
    assert span == (SET_FROM, 30)


def test_each_flag_has_its_own_lane(page):
    """Lanes are independent, and the hit test must agree with the drawing:
    aiming at `occluded` must not edit `fumble` on the lane above it."""
    _select(page, 0, SET_FROM)
    _click_frame(page, 10, FLAGS.index("occluded"))

    set_bits, _, _ = _one(page)
    assert set_bits == 1 << FLAGS.index("occluded"), (
        f"clicking the 'occluded' lane staged set_bits={set_bits}; the click landed on another lane"
    )


def test_clicking_outside_the_selection_stages_nothing(page):
    """A run overlapping the selection is not the same as a pointer inside it."""
    _select(page, 0, SET_FROM)
    _click_frame(page, 50, FUMBLE)
    assert _pending(page) == [], "a click outside the selection edited the range"


def test_a_bare_click_selects_before_it_edits(page):
    """Clicking the row is how you seek, and it leaves that frame selected.

    The first click only selects -- a press with nothing selected can never
    also edit, which is the rule that keeps a seek from becoming an edit. The
    second click acts on the frame the first one visibly selected.
    """
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)

    _click_frame(page, 10, FUMBLE)
    assert _pending(page) == [], "the first click, with nothing selected, staged an edit"

    _click_frame(page, 10, FUMBLE)
    staged = _pending(page)
    assert len(staged) == 1, f"the second click should act on the selected frame: {staged}"
    assert (staged[0]["params"]["frame_from"], staged[0]["params"]["frame_to"]) == (10, 11)


def test_a_second_click_undoes_the_first_and_empties_the_queue(page):
    """Two things at once. The hit test must read the MERGED view, or the run
    still reads as clear and the second click re-stages the same set. And the
    queue records the intended end state, so a round trip back to where the
    dataset started leaves nothing to save."""
    _select(page, 0, SET_FROM)
    _click_frame(page, 10, FUMBLE)
    assert _pending(page), "nothing staged, so this test could not tell a fix from a no-op"

    _click_frame(page, 10, FUMBLE)
    assert _pending(page) == [], (
        "clicking a run twice left something staged; the round trip nets to no change"
    )


def test_a_staged_flag_edit_is_visible_under_show_pending_edits(page):
    """Flag edits stage as `feature_bits`, and the overlay asked only for
    `feature_set` — so a row whose only staged edits were flag edits drew
    nothing in the one view that exists to show what is staged."""
    _select(page, 0, SET_FROM)
    _click_frame(page, 10, FUMBLE)
    assert _pending(page), "nothing staged, so this test could not tell a fix from a no-op"

    page.evaluate(
        """() => {
            const cb = document.getElementById('show-pending-edits-toggle');
            cb.checked = true;
            window.onShowPendingEditsToggle();
        }"""
    )
    page.wait_for_timeout(500)
    n = page.evaluate(f"() => document.querySelectorAll('{ROW} .flag-pending').length")
    assert n == 1, f"expected one pending band on the edited lane, got {n}"


# ── The click preview ───────────────────────────────────────────────────────
# The gesture acts on the run under the pointer, and nothing on screen says
# where that run ends. Without a preview the operator learns what a click meant
# only after it happened -- which is why the original implementation shipped
# one, and why its CSS sat unused in style.css after the JS was lost.


def _preview(pg):
    """The preview band's direction, tag and span in frames, or None."""
    return pg.evaluate(
        """([sel, frames]) => {
            const track = document.querySelector(sel);
            const el = track.querySelector('.lane-preview');
            if (!el) return null;
            const t = track.getBoundingClientRect(), r = el.getBoundingClientRect();
            const round = (v) => Math.round(v * frames / t.width);
            return {
                dir: el.classList.contains('lane-preview-set') ? 'set' : 'clear',
                tag: el.textContent.trim(),
                from: round(r.x - t.x),
                to: round(r.x - t.x + r.width),
                armed: track.classList.contains('lane-armed'),
            };
        }""",
        [ROW, FRAMES],
    )


def _hover(pg, frame: int, lane: int) -> None:
    x, y = _point(_track_box(pg), frame, _lane_mid_pct(pg, lane))
    pg.mouse.move(x, y)
    pg.wait_for_timeout(250)


def test_hovering_a_clear_run_previews_the_flag_being_set(page):
    _select(page, 0, SET_FROM)
    _hover(page, 10, FUMBLE)

    p = _preview(page)
    assert p is not None, "no preview band appeared over the run under the pointer"
    assert p["dir"] == "set"
    assert p["tag"] == "+ fumble", f"the tag read {p['tag']!r}"
    assert p["armed"], "the track is not marked armed, so the cursor stays a plain arrow"


def test_hovering_a_set_run_previews_the_flag_being_cleared(page):
    _select(page, SET_FROM, SET_TO)
    _hover(page, 30, FUMBLE)

    p = _preview(page)
    assert p is not None
    assert p["dir"] == "clear"
    # A true minus sign, not a hyphen: at 9px the hyphen does not read as an
    # operator beside the "+" it pairs with.
    assert p["tag"] == "− fumble", f"the tag read {p['tag']!r}"


def test_the_preview_covers_the_run_not_the_whole_selection(page):
    """The assertion the whole preview exists for. The original implementation
    toggled the entire selection and drew the band across it; this one stops at
    the boundary, and the band has to say so or it promises the wrong edit."""
    _select(page, 10, 30)
    _hover(page, 15, FUMBLE)

    p = _preview(page)
    assert p is not None
    assert (p["from"], p["to"]) == (10, SET_FROM), (
        f"the band covered frames {p['from']}–{p['to']}, but the click stages 10–{SET_FROM}; "
        "a band across the whole selection promises an edit that will not happen"
    )

    # The other half of the same selection: the opposite direction, the
    # complementary span, no re-selection in between.
    _hover(page, 25, FUMBLE)
    p = _preview(page)
    assert (p["dir"], p["from"], p["to"]) == ("clear", SET_FROM, 30)


def test_an_empty_lane_previews_a_set_over_the_whole_selection(page):
    """`occluded` is carried by no frame, so the run is the whole episode and
    the band is the selection — the one case where those coincide."""
    _select(page, 30, 50)
    _hover(page, 40, FLAGS.index("occluded"))

    p = _preview(page)
    assert p is not None
    assert (p["dir"], p["tag"], p["from"], p["to"]) == ("set", "+ occluded", 30, 50)


def test_no_preview_until_something_is_selected(page):
    """With nothing selected there is nothing to preview, so no band.

    A one-frame selection does get one: it is editable, and the band is what
    says so. It was refused once, back when a single press could both create
    that selection and commit on it -- deferring the decision split those into
    two gestures, so the band there advertises an edit the next click will
    genuinely make.
    """
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    _hover(page, 10, FUMBLE)
    assert _preview(page) is None, "a band appeared with no selection to act on"

    _click_frame(page, 10, FUMBLE)  # selects [10, 11)
    _hover(page, 10, FUMBLE)
    band = _preview(page)
    assert band is not None, "the frame the click selected offered no band"
    assert (band["from"], band["to"]) == (10, 11), f"the band covered {band['from']}-{band['to']}"


def test_moving_outside_the_selection_clears_the_preview(page):
    """The mask row's reported defect in the same shape: an affordance that
    survives the pointer leaving the selection sits on the track eating the
    clicks meant to re-select."""
    _select(page, 0, SET_FROM)
    _hover(page, 10, FUMBLE)
    assert _preview(page) is not None, "nothing to clear, so this could not tell a fix from a no-op"

    for frame in (25, 45, 70):
        _hover(page, frame, FUMBLE)
    assert _preview(page) is None, "the band is still on the track outside the selection"


# ── Defects found in review, each pinned by the case that exposed it ────────


def test_dragging_inside_a_selection_reselects_rather_than_doing_nothing(page):
    """Every flag run is actionable, so claiming the press outright made the
    whole lane band dead to dragging: the row was never told a press had
    happened, and the claim was then discarded as a drag. Narrowing a range
    from inside it did nothing at all."""
    _select(page, 0, 50)
    box = _track_box(page)
    y = box["y"] + box["h"] * (_lane_mid_pct(page, FUMBLE) / 100)
    page.mouse.move(box["x"] + box["w"] * (10.5 / FRAMES), y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["w"] * (29.5 / FRAMES), y, steps=10)
    page.mouse.up()
    page.wait_for_timeout(500)

    sel = page.evaluate("() => window.FeatureEditing._internals.currentSelection()")
    assert sel is not None, "the drag left no selection at all"
    assert (sel["frameFrom"], sel["frameTo"]) == (10, 30), (
        f"dragging across a lane gave {sel['frameFrom']}–{sel['frameTo']}, not 10–30"
    )
    assert _pending(page) == [], "a drag staged an edit; it is a re-selection, not a click"


def test_a_click_keeps_the_selection_so_the_next_click_can_use_it(page):
    """The press now reaches the row, which seeks and replaces the selection
    with one frame. The commit puts the range back, or the second toggle in a
    range would have nothing to act on."""
    _select(page, 0, SET_FROM)
    _click_frame(page, 10, FUMBLE)
    assert _pending(page), "nothing staged, so this could not tell a fix from a no-op"

    sel = page.evaluate("() => window.FeatureEditing._internals.currentSelection()")
    assert (sel["frameFrom"], sel["frameTo"]) == (0, SET_FROM), (
        f"the selection came back as {sel['frameFrom']}–{sel['frameTo']}; the seek ate it"
    )


def test_clearing_the_selection_mid_press_stages_nothing_and_does_not_throw(page):
    """Escape is the cancel gesture, so a press it interrupts must edit nothing.

    Both halves are asserted, and the staging half is the one that matters: an
    earlier version of this test checked only that the release did not throw,
    and passed while the release staged an edit over the range Escape had just
    removed from the screen. `_invariant` reports through `console.error`,
    which Playwright's `pageerror` hook does not see, so nothing failed.
    """
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    _select(page, 0, SET_FROM)
    x, y = _point(_track_box(page), 10, _lane_mid_pct(page, FUMBLE))
    page.mouse.move(x, y)
    page.mouse.down()
    page.keyboard.press("Escape")
    page.wait_for_function("() => document.querySelectorAll('.row-selection').length === 0", timeout=10_000)
    page.mouse.up()
    page.wait_for_timeout(600)

    assert not errors, f"the release threw: {errors}"
    assert _pending(page) == [], (
        f"the release staged an edit over the range Escape cancelled: {[e['params'] for e in _pending(page)]}"
    )
    violations = page.evaluate("() => window.FeatureEditing.invariantViolations()")
    assert violations == [], (
        f"the edit reached the guard at all; Escape should have dropped the press: {violations}"
    )


def test_a_right_click_while_holding_abandons_the_gesture(page):
    """The release handler took ANY mouseup, so the right button's release
    resolved a claim the left button had not let go of — an edit fired on a
    right-click, and the real release then did nothing.

    Reaching for another button mid-press abandons the gesture rather than
    deferring it: a context menu between press and release makes the release a
    different act from the one that was started."""
    _select(page, 0, SET_FROM)
    x, y = _point(_track_box(page), 10, _lane_mid_pct(page, FUMBLE))
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.down(button="right")
    page.mouse.up(button="right")
    page.wait_for_timeout(400)
    assert _pending(page) == [], "the right-button release committed the left button's claim"

    page.mouse.up()
    page.wait_for_timeout(600)
    assert _pending(page) == [], "the abandoned gesture fired on the left release"

    # The abandoned press still reached the row, which sought a frame and left a
    # one-frame selection behind — the same thing any click on the row does.
    # So the recovery is an ordinary re-select, and the gesture is not dead.
    _select(page, 0, SET_FROM)
    _click_frame(page, 10, FUMBLE)
    assert _pending(page), "a clean click after an abandoned one stages nothing"


# ── The Inspector checkbox ──────────────────────────────────────────────────
# The path this feature shipped with, and the one `stageFlagEdit` was written
# for. It had no test of its own, which is how a change to that function --
# a new optional range argument, and a rewritten status message -- could have
# altered it without anything noticing. These pin what it does.

CARD = "#inspector-body .feature-card[data-feature='quality']"


def _box(pg, flag: str):
    return pg.locator(f"{CARD} input[data-widget='flag'][data-flag='{flag}']")


def _status(pg) -> str:
    return pg.evaluate("() => (document.getElementById('status-text') || {}).textContent || ''")


def test_the_inspector_checkbox_stages_over_the_whole_selection(page):
    """Unlike a lane click, the checkbox is not positional: it acts on the
    selection, whatever runs that spans."""
    _select(page, 5, 25)
    page.wait_for_selector(CARD, timeout=20_000)
    _box(page, "fumble").check()
    page.wait_for_timeout(800)

    set_bits, clear_bits, span = _one(page)
    assert (set_bits, clear_bits) == (1 << FUMBLE, 0)
    assert span == (5, 25), f"the checkbox staged {span}, not the selection"


def test_the_inspector_checkbox_unticks_over_the_selection(page):
    """`fumble` is carried over 20-40, so unticking inside that range clears."""
    _select(page, 25, 35)
    page.wait_for_selector(CARD, timeout=20_000)
    box = _box(page, "fumble")
    assert box.is_checked(), "the box should start ticked where every frame carries the flag"
    box.uncheck()
    page.wait_for_timeout(800)

    set_bits, clear_bits, span = _one(page)
    assert (set_bits, clear_bits) == (0, 1 << FUMBLE)
    assert span == (25, 35)


def test_a_partly_carried_flag_reads_as_neither_ticked_nor_unticked(page):
    """Frames 10-30 straddle the boundary at 20, so the flag is on some frames
    and off others -- a two-state box would have to lie about one of them."""
    _select(page, 10, 30)
    page.wait_for_selector(CARD, timeout=20_000)
    state = page.evaluate(
        """(card) => { const b = document.querySelector(
             `${card} input[data-widget='flag'][data-flag='fumble']`);
             return {checked: b.checked, indeterminate: b.indeterminate}; }""",
        CARD,
    )
    assert state["indeterminate"], f"a mixed range rendered as a plain {state['checked']}"


def test_the_checkbox_reports_when_nothing_changed(page):
    """The toast used to read the response's `pending`, which counts the whole
    column across episodes -- so it said "staged" when this edit had changed
    nothing. Ticking a flag a range already carries is exactly that case."""
    _select(page, SET_FROM, SET_TO)
    page.wait_for_selector(CARD, timeout=20_000)
    page.evaluate(
        """(card) => { const b = document.querySelector(
             `${card} input[data-widget='flag'][data-flag='fumble']`);
             b.checked = false; b.click(); b.checked = true; b.click(); }""",
        CARD,
    )
    page.wait_for_timeout(1200)
    # Re-ticking a range that already carries the flag stages nothing.
    trailing = [e for e in _pending(page) if e["params"]["set_bits"] == 1 << FUMBLE]
    assert trailing == [] or _status(page).endswith("nothing to change"), (
        f"staged={trailing}, status={_status(page)!r}"
    )
