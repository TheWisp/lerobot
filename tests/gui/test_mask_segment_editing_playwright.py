# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Clicking a mask segment, in a real browser.

Every defect this file covers was reported from the running GUI after the unit
tests were green, because all three live in the DOM and none of them is a
function of its arguments:

* the row's own ``mousedown`` seeks and replaces the selection, and it is
  registered first -- so a toggle bound on ``click`` acted on the single frame
  that handler had just selected, and looked like "clicking reselects";
* the delete ``x`` followed the cursor, which is a target you cannot aim at and
  which says nothing about which segment it would act on;
* "show pending edits" drew nothing for a mask row, because that overlay is
  gated on ``editable`` and mask rows are deliberately not.

The lesson is the file's reason to exist: `maskSegments` and the merge helpers
were unit-tested and correct throughout. What was broken was event ordering,
layout and a render gate -- none reachable without driving the page.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright

H, W = 64, 96
FRAMES = 60
CAM = "observation.images.top"
LABELS = ["ball", "tray"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def masked_root(tmp_path):
    """A dataset whose one lane has detected / disabled / absent runs.

    Structure matters more than realism: the click must have a solid segment to
    mute, a muted one to unmute, and a gap that refuses both.
    """
    import random

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt, write_episode

    random.seed(0)
    np.random.seed(0)
    root = tmp_path / "segdemo"
    ds = LeRobotDataset.create(
        repo_id="tests/segdemo",
        fps=30,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            CAM: {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
        },
        use_videos=True,
    )
    img = np.full((H, W, 3), 60, np.uint8)
    for _ in range(FRAMES):
        ds.add_frame(
            {
                "observation.state": np.zeros(2, np.float32),
                "action": np.zeros(2, np.float32),
                "task": "seg",
                CAM: img,
            }
        )
    ds.save_episode()
    ds.finalize()

    ds = LeRobotDataset("tests/segdemo", root=root)
    # A treatment, or enabling and disabling composite identically and the
    # tile test below cannot tell a fix from a no-op.
    adopt(ds, [CAM], LABELS, (H, W), treatments={"tray": {"key": "tint", "params": {"color": [255, 0, 0]}}})
    blob = np.zeros((H, W), bool)
    blob[10:40, 10:60] = True
    # ball: detected [0,20)  disabled [20,40)  absent [40,60)
    per_frame, muted = [], []
    for f in range(FRAMES):
        if f < 40:
            per_frame.append({"ball": blob, "tray": blob})
            muted.append(["ball"] if f >= 20 else [])
        else:
            per_frame.append({"tray": blob})
            muted.append([])
    write_episode(ds, 0, CAM, per_frame, disabled_per_frame=muted)
    return root


@pytest.fixture
def page(masked_root):
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
        ds_id = str(masked_root)
        pg.evaluate("(ds) => openDataset(ds)", ds_id)
        pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
        pg.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [ds_id, FRAMES])
        pg.wait_for_function(
            "() => document.querySelector('.row-track[data-feature=\"masks.top\"]')", timeout=60_000
        )
        pg.ds_id = ds_id
        yield pg
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


def _track_box(pg):
    return pg.evaluate(
        """() => {
            const t = document.querySelector('.row-track[data-feature="masks.top"]');
            const r = t.getBoundingClientRect();
            return {x: r.x, y: r.y, w: r.width, h: r.height};
        }"""
    )


def _point(box, frame: int, lane: int) -> tuple[float, float]:
    """Centre of `frame` in `lane`. Lanes occupy 10%..90% of the row height."""
    lane_h = 80 / len(LABELS)
    return (
        box["x"] + box["w"] * ((frame + 0.5) / FRAMES),
        box["y"] + box["h"] * ((10 + lane * lane_h + lane_h * 0.4) / 100),
    )


def _select(pg, frm: int, to: int) -> None:
    """Drag-select a frame range, the way an operator does."""
    box = _track_box(pg)
    x0, y = _point(box, frm, 0)
    x1, _ = _point(box, to - 1, 0)
    pg.mouse.move(x0, y)
    pg.mouse.down()
    pg.mouse.move(x1, y, steps=8)
    pg.mouse.up()


def _click_frame(pg, frame: int, lane: int = 0) -> None:
    """Click a frame in a lane, re-measuring first.

    The row list reflows when an edit is staged, so coordinates captured before
    a click point somewhere else after it -- which cost a debugging round when
    a second click silently landed on a different row.
    """
    x, y = _point(_track_box(pg), frame, lane)
    pg.mouse.click(x, y)
    pg.wait_for_timeout(800)


def _pending(pg):
    return pg.evaluate("() => (window.pendingEdits || []).filter(e => e.edit_type === 'mask_range')")


def test_clicking_a_segment_stages_a_toggle_over_the_selection(page):
    """The reported defect: the click landed, the selection had already been
    replaced by the row's own mousedown, and the edit covered one frame."""
    _select(page, 0, 20)
    _click_frame(page, 10)

    edits = _pending(page)
    assert len(edits) == 1, f"expected one staged edit, got {edits}"
    e = edits[0]["params"]
    assert e["label"] == "ball"
    assert e["action"] == "disable", "a detected segment must mute, not unmute"
    assert (e["from_frame"], e["to_frame"]) == (0, 20), (
        f"the edit covered {e['from_frame']}–{e['to_frame']}, not the selection — "
        "the row's mousedown replaced the selection before the click ran"
    )


def test_clicking_a_muted_segment_unmutes_it(page):
    """Direction comes from the segment, so the same gesture reverses here."""
    _select(page, 20, 40)
    _click_frame(page, 30)

    edits = _pending(page)
    assert len(edits) == 1, edits
    assert edits[0]["params"]["action"] == "enable"


def test_clicking_an_absent_stretch_stages_nothing(page):
    """Nothing can enable a mask that does not exist."""
    _select(page, 40, 60)
    _click_frame(page, 50)
    assert _pending(page) == []


def test_a_second_click_undoes_the_first_and_empties_the_queue(page):
    """Two things at once. The hit-test must read the MERGED view, or the
    segment still looks detected and the second click re-stages a disable. And
    the queue records the intended end state, so a round trip back to where the
    dataset started leaves nothing to save."""
    _select(page, 0, 20)
    _click_frame(page, 10)
    assert [e["params"]["action"] for e in _pending(page)] == ["disable"]

    _click_frame(page, 10)
    assert _pending(page) == [], (
        "clicking a segment twice left something staged; the round trip nets to no change"
    )


def _kill_x(pg):
    return pg.evaluate(
        "() => { const b = document.querySelector('.mask-seg-kill');"
        " return b ? b.getBoundingClientRect().x : null; }"
    )


def _hover_near_edge(pg, seg_to: int, lane: int = 0) -> None:
    """Reach for a segment's trailing edge, where delete lives."""
    box = _track_box(pg)
    x = box["x"] + box["w"] * (seg_to / FRAMES) - 8
    _, y = _point(box, seg_to - 1, lane)
    pg.mouse.move(x, y)
    pg.wait_for_timeout(400)


def test_the_middle_of_a_segment_offers_no_delete(page):
    """Reported: the x covered the bar, so an ordinary click deleted instead of
    toggling, and re-selecting was impossible. Deleting is a deliberate reach
    for the edge; the rest of the segment belongs to the toggle."""
    _select(page, 0, 20)
    page.mouse.move(*_point(_track_box(page), 10, 0))
    page.wait_for_timeout(400)
    assert _kill_x(page) is None, "the x covers the middle of the segment, where a click means toggle"


def test_the_delete_x_appears_at_the_segment_edge_and_stays_put(page):
    """It is pinned to the segment, so it does not move while you aim at it."""
    _select(page, 0, 20)
    _hover_near_edge(page, 20)
    first = _kill_x(page)
    assert first is not None, "reaching for the segment's edge offered no delete"

    _hover_near_edge(page, 20)
    second = _kill_x(page)
    assert second is not None
    assert abs(second - first) < 2, f"the x moved {abs(second - first):.0f}px; it is following the cursor"


def test_a_staged_mask_edit_is_visible_under_show_pending_edits(page):
    """The overlay is gated on `editable`, and mask rows are deliberately not —
    so a staged segment edit drew nothing in the one view meant to show it."""
    _select(page, 0, 20)
    _click_frame(page, 10)
    assert _pending(page), "nothing staged, so this test could not tell a fix from a no-op"

    page.evaluate(
        """() => {
            const cb = document.getElementById('show-pending-edits-toggle');
            cb.checked = true;
            window.onShowPendingEditsToggle();
        }"""
    )
    page.wait_for_timeout(500)
    n = page.evaluate(
        "() => document.querySelectorAll('.row-track[data-feature=\"masks.top\"] .mask-pending').length"
    )
    assert n >= 1, "the staged mask edit is invisible with 'show pending edits' on"


def test_moving_off_the_selection_hides_the_delete_x(page):
    """Reported: the x survived the pointer leaving the selection, so it sat
    over the track swallowing clicks — every attempt to re-select deleted
    something instead."""
    _select(page, 0, 20)
    _hover_near_edge(page, 20)
    assert page.evaluate("() => !!document.querySelector('.mask-seg-kill')"), (
        "no x appeared, so this test could not tell it hiding from it never showing"
    )

    # Straight out of the selection, the way you would to re-select. This path
    # crosses the button, which is what let it survive.
    for frame in (22, 30, 45):
        bx, by = _point(_track_box(page), frame, 0)
        page.mouse.move(bx, by)
        page.wait_for_timeout(150)
    page.wait_for_timeout(400)

    assert not page.evaluate("() => !!document.querySelector('.mask-seg-kill')"), (
        "the x is still on the track outside the selection, where it will eat the next click"
    )


def test_the_x_stays_reachable_while_the_pointer_is_on_it(page):
    """The complement: hiding it too eagerly makes it impossible to click, and
    'it disappears when I reach for it' is the same bug wearing a hat."""
    _select(page, 0, 20)
    _hover_near_edge(page, 20)
    box = page.evaluate(
        "() => { const b = document.querySelector('.mask-seg-kill');"
        " const r = b.getBoundingClientRect(); return {x: r.x + r.width/2, y: r.y + r.height/2}; }"
    )
    page.mouse.move(box["x"], box["y"])
    page.wait_for_timeout(400)
    assert page.evaluate("() => !!document.querySelector('.mask-seg-kill')"), (
        "the x vanished when the pointer reached it"
    )


def test_a_click_with_no_selection_selects_and_stages_nothing(page):
    """Reported as "clicking outside the range triggers it".

    The row's mousedown creates a single-frame selection. If the click handler
    then acts on whatever selection it finds, that freshly-made one frame is
    what it edits — so a click meant to START a selection silently staged a
    one-frame toggle. The capture guard only claims the gesture when a usable
    selection already exists, so the click that MADE the selection must not
    also act on it.
    """
    _click_frame(page, 10)
    assert _pending(page) == [], "a click with no prior selection staged an edit; it should only select"


def test_a_click_after_that_click_acts_on_the_whole_selection(page):
    """The complement: having selected, the next click must act on the range —
    otherwise 'select then click' would never work and the fix above would have
    made the feature unusable rather than correct."""
    _select(page, 0, 20)
    _click_frame(page, 10)
    pend = _pending(page)
    assert len(pend) == 1, pend
    assert (pend[0]["params"]["from_frame"], pend[0]["params"]["to_frame"]) == (0, 20), (
        f"acted on {pend[0]['params']} rather than the selection"
    )


def test_a_toggle_leaves_the_selection_intact(page):
    """Half the clicks landed on one frame because the selection did not
    survive the re-render that staging triggers, so the NEXT click fell back to
    the make-a-selection path."""
    _select(page, 0, 20)
    _click_frame(page, 10)
    _click_frame(page, 10)
    assert _pending(page) == [], "the second click did not undo the first"

    _click_frame(page, 10)
    pend = _pending(page)
    assert len(pend) == 1, pend
    assert (pend[0]["params"]["from_frame"], pend[0]["params"]["to_frame"]) == (0, 20), (
        f"the selection was lost between clicks; edit covered {pend[0]['params']}"
    )


def test_a_click_stages_nothing_until_something_is_selected(page):
    """The shape behind the reported one-frame edits.

    Originally guarded by refusing any selection narrower than two frames, on
    the grounds that a bare click leaves a one-frame selection and acting on it
    would make every seek an edit. That was true while the press was claimed at
    mousedown -- one gesture both created the selection and could commit on it.
    Deferring the decision split them in two, so the rule that survives is the
    narrower one asserted here: a press with nothing selected only selects.
    """
    page.keyboard.press("Escape")
    page.wait_for_function("() => document.querySelectorAll('.row-selection').length === 0", timeout=10_000)

    _click_frame(page, 10)
    assert _pending(page) == [], f"a click with no selection staged: {[e['params'] for e in _pending(page)]}"
    sel = page.evaluate("() => window.FeatureEditing._internals.currentSelection()")
    assert (sel["frameFrom"], sel["frameTo"]) == (10, 11), f"the click selected {sel}"


def test_a_one_frame_selection_can_be_toggled(page):
    """A click leaves one frame selected, and that frame is editable.

    Refusing it made the lane disagree with the Inspector's controls, which
    have never imposed a minimum width -- so the same one-frame range could be
    flagged from the panel and not from the bar.
    """
    page.keyboard.press("Escape")
    page.wait_for_function("() => document.querySelectorAll('.row-selection').length === 0", timeout=10_000)
    _click_frame(page, 10)  # selects frame 10 only
    assert _pending(page) == []

    _hover(page, 10)
    band = _preview(page)
    assert band is not None, "a one-frame selection offered no preview"
    assert (band["from"], band["to"]) == (10, 11), f"the band covered {band['from']}-{band['to']}"

    _click_frame(page, 10)  # now act on it
    staged = _pending(page)
    assert [e["params"]["action"] for e in staged] == ["disable"], staged
    assert (staged[0]["params"]["from_frame"], staged[0]["params"]["to_frame"]) == (10, 11), (
        f"the edit covered {staged[0]['params']['from_frame']}-{staged[0]['params']['to_frame']}, "
        "not the single selected frame"
    )


def test_a_toggle_survives_a_re_render_between_press_and_release(page):
    """The race behind "half the time it does nothing".

    Staging re-renders the row, which REPLACES the track node. When that lands
    between mousedown and mouseup the browser has no common ancestor to fire
    `click` on, so a toggle bound to `click` never runs -- and whether it lands
    there depends on how fast the panel refreshes, which is why it looked
    random.

    Playwright's synthetic clicks are too fast to hit that window on their own,
    so this forces it: press, re-render, release. A toggle that survives this
    is one that does not depend on `click` being synthesised.
    """
    _select(page, 0, 20)
    box = _track_box(page)
    x, y = _point(box, 10, 0)

    page.mouse.move(x, y)
    page.mouse.down()
    # Exactly what staging does to the DOM, while the button is still down.
    page.evaluate("() => window.FeatureEditing.renderFeatureRows()")
    page.wait_for_timeout(100)
    page.mouse.up()
    page.wait_for_timeout(800)

    pend = _pending(page)
    assert len(pend) == 1, f"the toggle was lost when the row re-rendered mid-gesture: {pend}"
    assert (pend[0]["params"]["from_frame"], pend[0]["params"]["to_frame"]) == (0, 20)


def test_disabled_draws_hollow_and_detected_draws_filled(page):
    """Reported as "I still don't see any visual difference for disabled".

    The first version used a 4-unit hatch pattern inside a stretched
    `viewBox="0 0 100 100"`, at half opacity. It rendered — but a lane is a few
    pixels tall, so a texture that size and an opacity step are not a difference
    anyone can see. Filled versus hollow survives at any lane height, and the
    distinction has to survive because the bar is also the control.
    """
    rects = page.evaluate(
        """() => [...document.querySelectorAll('.row-track[data-feature="masks.top"] rect.mask-seg')]
            .map(r => ({state: r.getAttribute('data-state'), fill: r.getAttribute('fill'),
                        stroke: r.getAttribute('stroke')}))"""
    )
    by_state = {r["state"]: r for r in rects}
    assert "detected" in by_state and "disabled" in by_state, (
        f"need both states on screen to compare them: {rects}"
    )
    assert by_state["detected"]["fill"] != "none", "a detected segment must read as filled"
    assert by_state["disabled"]["fill"] == "none", "a disabled segment must read as hollow"
    assert by_state["disabled"]["stroke"] not in (None, "none"), "a hollow bar needs an outline to exist"
    assert by_state["detected"]["fill"] != by_state["disabled"]["fill"], (
        "the two states are drawn the same way"
    )


def test_the_camera_tile_requests_the_composite_when_masks_are_stored(page):
    """Reported as "disabling makes no visual difference in the cameras".

    The frame endpoint composites only when asked, `masks.js` decided when the
    tiles should show the recipe, and nothing carried that decision into the
    URL -- `compositedActive()` was an export with no consumer. So the tiles
    served stored pixels always, and neither a treatment nor a muted label
    could ever change what you see.
    """
    page.wait_for_function("() => window.MaskOverlay && window.loadAllFrames", timeout=30_000)
    # Stub the PRODUCER of the decision. `masks.js` flipping this flag when
    # saved masks appear is covered elsewhere and depends on a poll; what was
    # missing, and what this pins, is that anything CONSUMES it.
    page.evaluate("() => { window.MaskOverlay.compositedActive = () => true; }")
    page.evaluate("() => window.loadAllFrames(10)")
    page.wait_for_timeout(1200)

    # The tiles are painted from windows; the last window URL says what they asked for.
    srcs = [page.evaluate("() => window.__windowPlayer && window.__windowPlayer.lastWindowUrl()") or ""]
    assert srcs[0], "no window was asked for, so this test could not tell a fix from a no-op"
    assert any("masks=composited" in s for s in srcs), (
        f"the tiles are asking for stored pixels while compositing is active: {srcs}"
    )


def test_one_click_never_edits_and_a_second_one_does(page):
    """The reported defect, and the rule that replaced it.

    From the server's own log, when this was broken:

        MASK_RANGE_STAGE ... action=disable frames=[16,17)
        MASK_RANGE_STAGE ... action=disable frames=[18,19)
        MASK_RANGE_STAGE ... action=disable frames=[21,22)

    That was ONE click editing. The press collapsed the selection to the frame
    it landed on and the commit then read that fresh selection, so a click
    outside the range toggled a frame nobody had chosen. It was first guarded
    by refusing any selection under two frames wide, which also refused the
    deliberate case.

    Deferring the decision removed the need for that guard: the lane only takes
    a press when a usable selection already exists AND contains the pointer, so
    one gesture can no longer both create a selection and commit on it. What is
    asserted here is the pair -- a first click only selects, and a second click
    on the frame it visibly selected is a separate, deliberate act that edits.
    """
    _select(page, 0, 10)

    for frame in (30, 32):
        _click_frame(page, frame, lane=1)
        spans = [(e["params"]["from_frame"], e["params"]["to_frame"]) for e in _pending(page)]
        assert spans == [], f"the first click at {frame} staged {spans}"
        sel = page.evaluate("() => window.FeatureEditing._internals.currentSelection()")
        assert (sel["frameFrom"], sel["frameTo"]) == (frame, frame + 1), (
            f"the click at {frame} selected {sel['frameFrom']}-{sel['frameTo']}"
        )

        _click_frame(page, frame, lane=1)
        spans = [(e["params"]["from_frame"], e["params"]["to_frame"]) for e in _pending(page)]
        assert spans == [(frame, frame + 1)], (
            f"the second click at {frame} should edit that frame; staged {spans}"
        )

        # Leave nothing behind for the next iteration.
        _click_frame(page, frame, lane=1)
        assert _pending(page) == [], "a third click should undo the second"


def test_saving_a_mask_edit_changes_the_frame_url(page):
    """The camera tile showed the pre-edit picture after a save.

    Three things had to be true and none was: the tile has to ASK for the
    composite (`compositedActive()` was an export with no consumer), the URL
    has to change when the rows change (the browser answers an unchanged URL
    from its own cache), and the save has to bump that version (nothing did).
    Each was invisible to a test of the layer below -- the endpoint composited
    correctly throughout.

    What is pinned here is the version moving on save, which is the half that
    was missing in code. The pixel-level end-to-end -- disable changes the
    served bytes, enable restores them exactly, delete changes them -- was run
    against a real server on a freshly built dataset and passes; it is NOT
    asserted here, because in this harness the served bytes do not change and I
    have not isolated why. Stated rather than quietly dropped: this test would
    stay green if the tile went back to serving stale pixels for some other
    reason.
    """
    page.wait_for_function("() => window.MaskOverlay && window.applyEdits", timeout=30_000)
    page.evaluate("() => { window.MaskOverlay.compositedActive = () => true; window.confirm = () => true; }")
    page.evaluate("() => window.loadAllFrames(5)")
    page.wait_for_timeout(1200)

    src = "() => window.__windowPlayer.lastWindowUrl()"
    before = page.evaluate(src)
    assert "masks=composited" in before, before

    page.evaluate(
        """async ([ds]) => {
            await fetch('/api/edits/mask-range', {method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({dataset_id: ds, episode_index: 0,
                camera: 'observation.images.top', label: 'tray',
                from_frame: 0, to_frame: 40, action: 'disable'})});
            await window.applyEdits();
        }""",
        [page.ds_id],
    )
    page.wait_for_timeout(2500)
    page.evaluate("() => window.loadAllFrames(5)")
    page.wait_for_timeout(1200)

    after = page.evaluate(src)
    assert after != before, (
        f"the frame URL did not change after a save, so the browser serves its cache: {after}"
    )


# ── the Inspector's dataset tier ────────────────────────────────────────────


def _treat_rows(pg):
    """The selected treatment is the filled button, not a `<select>` value.

    The control is a flat exclusive row: `tint` carries a colour, which a
    dropdown can neither show nor pick.
    """
    return pg.evaluate(
        """() => [...document.querySelectorAll('.ds-treat-row')].map(r => {
            const sel = r.querySelector('.ds-treat-btn.sel');
            return { name: r.querySelector('.ds-treat-name').textContent,
                     value: sel ? sel.getAttribute('data-key') : null }; })"""
    )


def _pick_treatment(pg, label: str, key: str) -> None:
    """Click a treatment the way an operator does."""
    pg.evaluate(
        """([label, key]) => {
            const b = document.querySelector(
                `.ds-treat[data-label="${label}"] .ds-treat-btn[data-key="${key}"]`);
            if (!b) throw new Error(`no ${key} button for ${label}`);
            b.click();
        }""",
        [label, key],
    )


def test_the_inspector_has_a_dataset_tier_keyed_by_label_name(page):
    """The Inspector rendered episode and selection scopes only, so nothing
    dataset-scoped had a home once an episode was open -- the summary was an
    empty state.

    Keyed by NAME, not by column: every camera shares one vocabulary, so a
    section per column would ask the same question two or three times.
    """
    rows = _treat_rows(page)
    assert rows, "no dataset section rendered"
    names = [r["name"] for r in rows]
    assert "tray" in names and "ball" in names, names
    assert "background" in names, "the background is a region too and needs its own row"
    assert not any(n.startswith("masks.") for n in names), (
        f"rows are keyed by column rather than by label: {names}"
    )
    assert len(names) == len(set(names)), f"a label appears twice, once per camera: {names}"


def test_editing_a_treatment_commits_in_place_not_on_the_bottom_bar(page):
    """Config commits next to itself; the timeline's bottom bar is for frame
    data. Routing a dataset-wide write through a bar labelled for the timeline
    is what made the previous panel ambiguous about scope."""
    _pick_treatment(page, "tray", "blur")
    page.wait_for_timeout(500)
    assert page.evaluate(
        "() => { const a = document.querySelector('.ds-treat-actions');"
        " return !!a && a.style.display !== 'none'; }"
    ), "editing a treatment raised no in-place save/cancel"

    page.evaluate("() => document.querySelector('.ds-treat-save').click()")
    page.wait_for_timeout(3000)

    assert page.evaluate("() => (window.pendingEdits || []).length") == 0, (
        "a treatment edit was left in the timeline's pending queue"
    )
    assert [r for r in _treat_rows(page) if r["name"] == "tray"][0]["value"] == "blur", (
        "the saved treatment did not survive the refresh"
    )


def test_cancelling_a_treatment_edit_restores_the_stored_value(page):
    before = [r for r in _treat_rows(page) if r["name"] == "tray"][0]["value"]
    # Both guards below exist because "restored to what it was" is equally true
    # of a control that never changes: the read must find a real value, and the
    # click must actually stage a different one, or the restore proves nothing.
    assert before, "no treatment is displayed as selected; the restore check would be vacuous"
    _pick_treatment(page, "tray", "random")
    page.wait_for_timeout(400)
    staged = [r for r in _treat_rows(page) if r["name"] == "tray"][0]["value"]
    assert staged == "random", f"the click staged nothing (value {staged!r}); cancel has nothing to undo"
    page.evaluate("() => document.querySelector('.ds-treat-cancel').click()")
    page.wait_for_timeout(400)
    assert [r for r in _treat_rows(page) if r["name"] == "tray"][0]["value"] == before


def test_picking_a_segmenter_seeds_the_rows_from_the_stored_vocabulary(page):
    """Turning a segmenter on should carry on with what the dataset already
    tracks, not start from nothing.

    Safe as a default only because of the write rule: re-running a stored label
    cannot overwrite what is stored, so seeding is idempotent. Whatever the
    operator has already typed wins — their prompts are theirs.
    """
    names = lambda: page.evaluate(  # noqa: E731
        "() => [...document.querySelectorAll('#overlays-panel .overlays-obj-name')].map(i => i.value)"
    )
    page.wait_for_timeout(2000)
    page.evaluate(
        """() => { const p = document.querySelector('#overlays-panel .overlays-picker');
            if (!p) return; p.value = 'sam3_track';
            p.dispatchEvent(new Event('change', {bubbles: true})); }"""
    )
    page.wait_for_timeout(2500)
    seeded = names()
    assert seeded, "picking a segmenter produced no rows at all"
    assert "tray" in seeded and "ball" in seeded, (
        f"the rows were not seeded from the stored vocabulary: {seeded}"
    )


def test_apply_is_offered_and_gated_on_a_named_object(page):
    """Apply is a MODE: ticking it arms the run, and playing writes the frames.

    Only its presence and its gate are pinned here — the run itself needs SAM3
    loaded (2.19 s, 2.55 GiB) and a played episode. What the mode does once armed
    is covered without a browser: the worker's mask channel in
    `test_apply_mask_channel.py`, the frame attribution in `test_apply_drain.py`,
    the write rule in `apply_run_filter.test.js`, and the pending edit it builds
    in `test_mask_run_edits.py`.

    The gate: arming with no object named is refused. A mode that cannot do
    anything is worse than a refusal — the operator plays a whole episode
    expecting masks and gets none.
    """
    page.evaluate(
        """() => { const p = document.querySelector('#overlays-panel .overlays-picker');
            if (!p) return; p.value = 'sam3_track';
            p.dispatchEvent(new Event('change', {bubbles: true})); }"""
    )
    page.wait_for_timeout(2500)
    cb = page.query_selector(".overlays-apply-cb")
    assert cb is not None, "the Apply control is not offered on the data tab"

    # With every row cleared it must refuse: a run with nothing to look for
    # would burn the episode's segmentation time to write nothing.
    page.evaluate(
        """() => { document.querySelectorAll('#overlays-panel .overlays-obj-name')
            .forEach(i => { i.value = ''; i.dispatchEvent(new Event('input', {bubbles: true})); }); }"""
    )
    page.wait_for_timeout(800)
    page.evaluate("() => document.querySelector('.overlays-apply-cb').click()")
    page.wait_for_timeout(1200)
    assert page.evaluate("() => document.querySelector('.overlays-apply-cb').checked") is False, (
        "Apply started a run with no object named"
    )


# ── The click preview ───────────────────────────────────────────────────────
# The same band the flag row draws, on the row with a third state. What is
# specific here is that an absent stretch must offer nothing: producing a mask
# needs the model and a segmentation pass, so a band there would advertise an
# edit the click cannot make.


def _preview(pg):
    return pg.evaluate(
        """(frames) => {
            const track = document.querySelector('.row-track[data-feature="masks.top"]');
            const el = track.querySelector('.lane-preview');
            if (!el) return null;
            const t = track.getBoundingClientRect(), r = el.getBoundingClientRect();
            const f = (v) => Math.round(v * frames / t.width);
            return {
                dir: el.classList.contains('lane-preview-set') ? 'set' : 'clear',
                tag: el.textContent.trim(),
                from: f(r.x - t.x),
                to: f(r.x - t.x + r.width),
                armed: track.classList.contains('lane-armed'),
            };
        }""",
        FRAMES,
    )


def _hover(pg, frame: int, lane: int = 0) -> None:
    pg.mouse.move(*_point(_track_box(pg), frame, lane))
    pg.wait_for_timeout(300)


def test_hovering_a_detected_segment_previews_it_being_muted(page):
    """Detected means it reaches training; the click withholds it. Drawn as an
    outline over the band going away rather than as a second filled band, which
    would read as "will be enabled"."""
    _select(page, 0, 20)
    _hover(page, 10)

    p = _preview(page)
    assert p is not None, "no band appeared over the segment under the pointer"
    assert p["dir"] == "clear"
    assert p["tag"] == "− ball", f"the tag read {p['tag']!r}"
    assert p["armed"]


def test_hovering_a_muted_segment_previews_it_being_restored(page):
    """The opposite direction from the opposite state, with no control to set."""
    _select(page, 20, 40)
    _hover(page, 30)

    p = _preview(page)
    assert p is not None
    assert (p["dir"], p["tag"]) == ("set", "+ ball")


def test_an_absent_stretch_offers_no_preview(page):
    """Nothing can enable a mask that was never stored, so nothing may promise
    to. The absence of the band is the whole message."""
    _select(page, 40, 60)
    _hover(page, 50)
    assert _preview(page) is None, "a band appeared over frames with no stored mask"


def test_the_band_stops_at_the_state_boundary(page):
    """A selection spanning detected and muted frames previews only the run
    under the pointer, in the direction that run implies."""
    _select(page, 10, 30)
    _hover(page, 15)
    p = _preview(page)
    assert (p["dir"], p["from"], p["to"]) == ("clear", 10, 20), p

    _hover(page, 25)
    p = _preview(page)
    assert (p["dir"], p["from"], p["to"]) == ("set", 20, 30), p


def test_the_delete_footprint_shows_the_x_and_not_a_toggle_band(page):
    """One affordance at a time. Over the delete button's own footprint the
    press deletes, so a toggle band there would promise the wrong edit; away
    from it the band is what speaks."""
    _select(page, 0, 20)
    _hover_near_edge(page, 20)
    at_edge = page.evaluate(
        """() => ({
            kill: !!document.querySelector('.mask-seg-kill'),
            band: !!document.querySelector('.lane-preview'),
        })"""
    )
    assert at_edge["kill"], "reaching for the segment's edge offered no delete"
    assert not at_edge["band"], "a toggle band is drawn where the press deletes"

    _hover(page, 10)
    assert _preview(page) is not None, "away from the edge the band should be back"


def test_the_delete_button_does_not_take_the_pointer(page):
    """It advertises what the press will do; it does not intercept it. A button
    that handled its own press had to stop that press reaching the row, and
    that is what swallowed drags started on top of it."""
    _select(page, 0, 20)
    _hover_near_edge(page, 20)
    inert = page.evaluate("() => getComputedStyle(document.querySelector('.mask-seg-kill')).pointerEvents")
    assert inert == "none", f"the delete button still takes the pointer (pointer-events: {inert})"


def test_a_drag_starting_on_the_delete_button_still_reselects(page):
    """Reported from a real session: about half of all drags did nothing.

    The delete button is pinned to the trailing edge of the selection -- which
    is exactly where the pointer already is after inspecting a segment's end --
    and it took the mousedown for itself. The row never saw the press, so there
    was no seek, no drag and no new selection: the row silently kept the
    previous range, and hovering the newly-dragged area showed no band because
    the pointer was outside the selection that was actually still current.
    """
    _select(page, 0, 20)
    _hover_near_edge(page, 20)
    kill = page.evaluate(
        """() => { const k = document.querySelector('.mask-seg-kill');
             if (!k) return null;
             const r = k.getBoundingClientRect();
             return {x: r.x + r.width / 2, y: r.y + r.height / 2}; }"""
    )
    assert kill is not None, "no delete button appeared, so this could not exercise the defect"

    box = _track_box(page)
    end_x = box["x"] + box["w"] * ((35 + 0.5) / FRAMES)
    page.mouse.move(kill["x"], kill["y"])
    page.mouse.down()
    page.mouse.move(end_x, kill["y"], steps=12)
    page.mouse.up()
    page.wait_for_timeout(500)

    sel = page.evaluate("() => window.FeatureEditing._internals.currentSelection()")
    assert sel is not None, "the drag left no selection"
    assert (sel["frameFrom"], sel["frameTo"]) != (0, 20), (
        "the drag was swallowed by the delete button -- the row still holds the old selection"
    )
    assert sel["frameTo"] > 30, f"the drag ended at {sel['frameTo']}, so it never tracked the pointer"
    assert _pending(page) == [], "a drag off the delete button staged an edit"


def test_clicking_the_delete_x_stages_only_a_delete(page):
    """The × is a child of the track, and the lane gesture used to claim the
    press in the capture phase before the button's own guard could run — so one
    press posted a toggle at mouseup and then a delete on click. The suite
    asserted where the × appeared and never clicked it."""
    _select(page, 0, 20)
    _hover_near_edge(page, 20)
    assert page.evaluate("() => !!document.querySelector('.mask-seg-kill')"), (
        "no × appeared, so this test could not tell one edit from two"
    )
    box = page.evaluate(
        """() => { const r = document.querySelector('.mask-seg-kill').getBoundingClientRect();
             return {x: r.x + r.width / 2, y: r.y + r.height / 2}; }"""
    )
    page.mouse.click(box["x"], box["y"])
    page.wait_for_timeout(1000)

    actions = [e["params"]["action"] for e in _pending(page)]
    assert actions == ["delete"], f"one press on the × staged {actions}"


def test_dragging_inside_a_selection_reselects_on_a_mask_row(page):
    """The mask row let presses through only where a lane was absent; inside a
    detected run the drag was swallowed the same way the flag row's was."""
    _select(page, 0, 40)
    box = _track_box(page)
    y = box["y"] + box["h"] * (10 + (80 / len(LABELS)) * 0.4) / 100
    page.mouse.move(box["x"] + box["w"] * (5.5 / FRAMES), y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["w"] * (24.5 / FRAMES), y, steps=10)
    page.mouse.up()
    page.wait_for_timeout(500)

    sel = page.evaluate("() => window.FeatureEditing._internals.currentSelection()")
    assert sel is not None and (sel["frameFrom"], sel["frameTo"]) == (5, 25), (
        f"dragging across a detected run gave {sel}, not 5–25"
    )
    assert _pending(page) == [], "a drag staged an edit"


# ── The runtime invariants ──────────────────────────────────────────────────


def test_a_swallowed_press_is_reported_rather_than_ignored(page):
    """The guard for the defect class this row keeps producing.

    Rather than reintroducing the delete button's press-eating, this installs
    an equivalent culprit: a button inside the track that takes the mousedown
    for itself. The lane gesture defers to it (a control inside a lane owns its
    own presses) and it stops propagation, so the row never learns a press
    happened -- and the row keeps the selection it already had, which on screen
    is indistinguishable from a drag that worked. That silence is the whole
    problem; the invariant turns it into a named console error.
    """
    _select(page, 0, 20)
    page.evaluate("() => window.FeatureEditing.clearInvariantViolations()")
    page.evaluate(
        """() => {
            const track = document.querySelector('.row-track[data-feature="masks.top"]');
            const rogue = document.createElement('button');
            rogue.id = 'rogue-swallower';
            rogue.style.cssText =
                'position:absolute; left:0; top:0; width:100%; height:100%; z-index:99; opacity:0;';
            rogue.addEventListener('mousedown', (e) => e.stopPropagation());
            track.appendChild(rogue);
        }"""
    )
    box = _track_box(page)
    page.mouse.click(box["x"] + box["w"] * 0.5, box["y"] + box["h"] * 0.5)
    page.wait_for_timeout(400)

    violations = page.evaluate("() => window.FeatureEditing.invariantViolations()")
    assert violations, "a press swallowed inside the track was not reported at all"
    assert "swallowed" in violations[0]["message"], violations[0]["message"]


def test_an_ordinary_press_reports_no_violation(page):
    """The invariant has to be quiet in normal use, or it is noise nobody reads
    and the one that matters is lost in it."""
    page.evaluate("() => window.FeatureEditing.clearInvariantViolations()")
    _select(page, 0, 20)
    _click_frame(page, 10)
    _hover(page, 12)
    box = _track_box(page)
    # Including the row's margins, where no lane claims the press.
    page.mouse.click(box["x"] + box["w"] * 0.5, box["y"] + box["h"] * 0.02)
    page.wait_for_timeout(400)

    assert page.evaluate("() => window.FeatureEditing.invariantViolations()") == []


def test_an_edit_outside_the_visible_selection_is_reported(page):
    """The second invariant: what gets staged must lie inside the band on
    screen. A commit reading a range the row never showed as chosen is the
    shape of every silent mis-edit here -- a stale snapshot, a selection
    replaced between press and release -- and a wrong range renders exactly
    like a right one, so only an assertion can tell them apart."""
    _select(page, 0, 20)
    out = page.evaluate(
        """() => {
            const FE = window.FeatureEditing;
            FE.clearInvariantViolations();
            const sel = FE._internals.currentSelection();
            const inside = FE._internals.assertEditWithinSelection(
                {from: sel.frameFrom, to: sel.frameTo}, sel);
            const quiet = FE.invariantViolations().length;
            FE._internals.assertEditWithinSelection(
                {from: sel.frameFrom, to: sel.frameTo + 5}, sel);
            return {inside, quiet, violations: FE.invariantViolations()};
        }"""
    )
    assert out["inside"] is True, "an edit matching the selection was reported as a violation"
    assert out["quiet"] == 0, "the in-range case was noisy"
    assert out["violations"], "an edit reaching past the visible selection was not reported"
    assert "outside the selection" in out["violations"][0]["message"], out["violations"][0]["message"]


def test_the_selection_survives_the_press_intact(page):
    """Reported after a hands-on pass: pressing inside the selection made the
    range visibly collapse to one frame and spring back on release.

    The press used to decide immediately -- the row sought a frame and re-selected --
    and a release without travel then put the old range back. Both outcomes
    were started and one was undone, which is what the flicker was. Nothing is
    decided now until the gesture says which it is, so the range is untouched
    for the whole press.
    """
    _select(page, 0, 20)
    box = _track_box(page)
    x, y = _point(box, 10, 0)
    page.mouse.move(x, y)
    page.mouse.down()
    page.wait_for_timeout(250)

    held = page.evaluate("() => window.FeatureEditing._internals.currentSelection()")
    assert held is not None, "the selection vanished while the button was held"
    assert (held["frameFrom"], held["frameTo"]) == (0, 20), (
        f"the selection collapsed to {held['frameFrom']}-{held['frameTo']} under the press"
    )
    assert _pending(page) == [], "the press staged an edit before it was released"

    page.mouse.up()
    page.wait_for_timeout(700)
    assert [e["params"]["action"] for e in _pending(page)] == ["disable"], (
        "releasing without travel should be the click"
    )


def test_the_playhead_is_not_moved_by_a_press_that_becomes_a_click(page):
    """A seek is part of starting a selection, not part of clicking a run. The
    press used to seek unconditionally and could not take it back."""
    _select(page, 0, 20)
    page.evaluate("() => window.loadAllFrames(50)")
    page.wait_for_function("() => window.currentFrame === 50", timeout=10_000)

    x, y = _point(_track_box(page), 10, 0)
    page.mouse.click(x, y)
    page.wait_for_timeout(700)

    assert _pending(page), "the click staged nothing, so this proves nothing about the seek"
    assert page.evaluate("() => window.currentFrame") == 50, (
        "clicking a run moved the playhead; the seek belongs to selecting, not to clicking"
    )


def _drag_on_row(pg, feature, frm, to):
    """Drag a range on a named row, at its vertical centre."""
    b = pg.evaluate(
        """(f) => { const r = document.querySelector(`.row-track[data-feature="${f}"]`)
             .getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height}; }""",
        feature,
    )
    y = b["y"] + b["h"] * 0.5

    def xf(f):
        return b["x"] + b["w"] * ((f + 0.5) / FRAMES)

    pg.mouse.move(xf(frm), y)
    pg.mouse.down()
    pg.mouse.move(xf(to), y, steps=8)
    pg.mouse.up()
    pg.wait_for_timeout(350)


def _visible_rows(pg):
    return pg.evaluate(
        """() => [...document.querySelectorAll('.row-track')]
             .map(t => t.getAttribute('data-feature')).filter(Boolean)"""
    )


def test_every_row_can_start_a_selection_the_lane_then_acts_on(page):
    """Reported as "it works about half the time", reproduced at 100%.

    A selection is a vertical slice -- a frame range -- and its band is painted
    on every row. The lane gesture used to require that the drag had *started*
    on this row, so a range dragged anywhere else left the mask row looking
    selected and completely inert: no preview, no delete affordance, and a
    press that fell through to making a new selection. Which row a drag started
    on is not something an operator tracks, which is why the failure looked
    random.

    Generic on purpose: every row on the timeline is tried as the origin, so a
    future scope-by-row cannot pass by leaving one pair working.
    """
    rows = _visible_rows(page)
    assert len(rows) >= 2, f"need more than one row to vary the origin, got {rows}"

    for origin in rows:
        page.keyboard.press("Escape")
        page.wait_for_function(
            "() => document.querySelectorAll('.row-selection').length === 0", timeout=10_000
        )
        _drag_on_row(page, origin, 0, 19)

        sel = page.evaluate("() => window.FeatureEditing._internals.currentSelection()")
        assert sel["focusRow"] == origin, f"the drag did not originate on {origin}"
        drawn = page.evaluate("() => document.querySelectorAll('.row-selection').length")
        assert drawn == len(rows), (
            f"the band is drawn on {drawn} of {len(rows)} rows; it claims to cover them all"
        )

        _hover(page, 10)
        assert _preview(page) is not None, (
            f"a selection originating on '{origin}' left the mask lane inert, "
            "though its band is drawn across every row"
        )


def test_the_lane_acts_on_a_selection_made_on_any_other_row(page):
    """The other half: the gesture must not merely light up, it must edit."""
    others = [r for r in _visible_rows(page) if not r.startswith("masks.")]
    assert others, "no non-mask row on screen, so this cannot exercise the defect"

    _drag_on_row(page, others[0], 0, 19)
    _click_frame(page, 10)

    assert [e["params"]["action"] for e in _pending(page)] == ["disable"], (
        "clicking the mask lane did not stage the toggle"
    )
    after = page.evaluate("() => window.FeatureEditing._internals.currentSelection()")
    assert (after["frameFrom"], after["frameTo"]) == (0, 20), (
        f"the click replaced the selection with {after['frameFrom']}-{after['frameTo']} "
        "instead of acting on it"
    )


def test_the_delete_button_is_drawn_only_where_pressing_deletes(page):
    """The × was drawn by a 28px proximity rule and acted on by a
    min(16, segPx/2) one, so on a narrow run the operator saw a delete button,
    pressed it, and got a mute instead. One rule now decides both.

    Swept across the whole segment rather than sampled at one point, because a
    single hover is what let the two rules disagree unnoticed.
    """
    _select(page, 0, 20)
    box = _track_box(page)
    disagreements = []
    for frame in range(0, 20):
        for offset in (0.2, 0.5, 0.8):
            x = box["x"] + box["w"] * ((frame + offset) / FRAMES)
            y = box["y"] + box["h"] * (10 + (80 / len(LABELS)) * 0.4) / 100
            page.mouse.move(x, y)
            page.wait_for_timeout(30)
            state = page.evaluate(
                """() => {
                    const k = document.querySelector('.mask-seg-kill');
                    const b = document.querySelector('.lane-preview');
                    return {kill: !!k, band: !!b};
                }"""
            )
            # Exactly one affordance at a time: the × means this press deletes,
            # the band means it toggles. Both or neither is the disagreement.
            if state["kill"] == state["band"]:
                disagreements.append((frame, offset, state))

    assert not disagreements, (
        f"the × and the toggle band co-occurred or both vanished at "
        f"{disagreements[:6]} ({len(disagreements)} points)"
    )


def test_the_delete_button_and_the_press_agree_on_a_one_frame_run(page):
    """The narrow-run case, asserted as the rule rather than as a pixel count.

    Whether a one-frame run carries a delete region depends on how wide a frame
    is in the row, which varies with the viewport and the episode length -- an
    earlier version of this test asserted "no button on one frame" and failed
    on a fixture where a frame is 18px and the region is legitimately 9px. What
    must hold at every density is that the button is drawn exactly where a
    press deletes.
    """
    page.keyboard.press("Escape")
    page.wait_for_function("() => document.querySelectorAll('.row-selection').length === 0", timeout=10_000)
    _click_frame(page, 10)  # selects exactly frame 10, so the run clips to one

    box = _track_box(page)
    for offset in (0.1, 0.5, 0.9):
        x = box["x"] + box["w"] * ((10 + offset) / FRAMES)
        page.mouse.move(x, box["y"] + box["h"] * (10 + (80 / len(LABELS)) * 0.4) / 100)
        page.wait_for_timeout(60)
        state = page.evaluate(
            """() => ({kill: !!document.querySelector('.mask-seg-kill'),
                       band: !!document.querySelector('.lane-preview')})"""
        )
        assert state["kill"] != state["band"], (
            f"at offset {offset} on a one-frame run the × and the toggle band "
            f"disagree about what a press does: {state}"
        )
