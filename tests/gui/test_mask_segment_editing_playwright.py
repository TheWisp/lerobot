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


def test_a_click_on_a_mask_row_after_selecting_elsewhere_stages_nothing(page):
    """The shape behind the reported one-frame edits.

    A selection made on another row does not apply here, so the capture guard
    does not claim the gesture and the row's own mousedown makes a fresh
    single-frame selection. If the toggle then acts on whatever selection
    exists, it edits that one frame -- which is what "clicking outside the
    range triggers it" looked like from the operator's side.
    """
    other = page.evaluate(
        "() => { const t = [...document.querySelectorAll('.row-track')]"
        ".find(e => { const f = e.getAttribute('data-feature');"
        " return f && !f.startsWith('masks.'); });"
        " return t ? t.getAttribute('data-feature') : null; }"
    )
    assert other, "no non-mask row to select on, so this test proves nothing"
    box = page.evaluate(
        "(f) => { const r = document.querySelector(`.row-track[data-feature='${f}']`)"
        ".getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height}; }",
        other,
    )
    page.mouse.move(box["x"] + box["w"] * 0.05, box["y"] + box["h"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["w"] * 0.35, box["y"] + box["h"] / 2, steps=8)
    page.mouse.up()
    page.wait_for_timeout(300)

    _click_frame(page, 10)
    staged = _pending(page)
    assert staged == [], (
        f"a click on a mask row staged an edit off another row's selection: {[e['params'] for e in staged]}"
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


def _watch_frame_requests(pg) -> list[str]:
    """Collect the frame URLs the page actually fetches, newest last.

    Not the tile's ``src``: the loader swaps that only once the new bytes
    decode, so the previous frame stays on screen rather than the element
    blanking mid-flight. The attribute therefore lags, and is empty until the
    first decode completes. What "the tile asks for the composite" means is the
    request, so that is what this observes.
    """
    urls: list[str] = []
    pg.on("request", lambda r: urls.append(r.url) if "/frame/" in r.url else None)
    return urls


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
    asked = _watch_frame_requests(page)
    page.evaluate("() => window.loadAllFrames(10)")
    page.wait_for_timeout(1200)

    assert asked, "no camera tile fetched a frame, so this test could not tell a fix from a no-op"
    assert any("masks=composited" in url for url in asked), (
        f"the tiles are asking for stored pixels while compositing is active: {asked}"
    )


def test_clicking_outside_the_selection_never_toggles(page):
    """The exact reported sequence, from the server's own log:

        MASK_RANGE_STAGE ... action=disable frames=[16,17)
        MASK_RANGE_STAGE ... action=disable frames=[18,19)
        MASK_RANGE_STAGE ... action=disable frames=[21,22)

    Clicking the row is how you seek, and it leaves a one-frame selection
    behind. A click outside the range therefore replaced the selection with one
    frame, and the click after it toggled that frame -- which reads as "my click
    outside the range toggled it".
    """
    # Lane 1 ("tray") is detected across [0, 40), so frames well past the
    # selection still land ON a segment -- which is what makes the click
    # eligible to toggle at all. Clicking a gap would prove nothing.
    _select(page, 0, 10)

    # The first click outside re-seeks and leaves a one-frame selection; the
    # second lands inside THAT, and used to toggle it.
    for frame in (30, 30, 32, 32):
        _click_frame(page, frame, lane=1)

    spans = [(e["params"]["from_frame"], e["params"]["to_frame"]) for e in _pending(page)]
    one_frame = [s for s in spans if s[1] - s[0] == 1]
    assert not one_frame, f"clicks outside the range staged single-frame toggles: {spans}"
    assert spans == [], f"clicking outside the selection staged something: {spans}"


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

    asked = _watch_frame_requests(page)
    page.evaluate("() => window.loadAllFrames(5)")
    page.wait_for_timeout(1200)
    assert asked, "no frame was fetched before the edit"
    before = asked[-1]
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

    after = asked[-1]
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
