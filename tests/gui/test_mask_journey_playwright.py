# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Whole journeys through the mask flow, checked before AND after each action.

The unit tests here are good at "this function returns the right thing" and
were all green while four defects were reported from the running GUI in one
session. Every one of them lived in the space BETWEEN actions -- a value
computed and then discarded by its caller, a save that rewrote something the
action before it had set, a layer that kept drawing after another took over.
None is visible from either end alone.

So these tests state an invariant, do one thing, and check the invariant again.
What is on disk before an edit is staged must be what is on disk after; a save
must change exactly the thing it is about and nothing else; and the playhead
must never move backwards while playing.
"""

from __future__ import annotations

import json
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
MASK_KEY = "masks.top"
LABELS = ["ball", "tray"]
TINT = {"key": "tint", "params": {"color": [255, 0, 0]}}
BLUR = {"key": "blur", "params": {}}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def masked_root(tmp_path):
    """A masked episode with a treatment AND a background already stored."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt, write_episode

    np.random.seed(0)
    root = tmp_path / "journey"
    ds = LeRobotDataset.create(
        repo_id="tests/journey",
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
                "task": "journey",
                CAM: img,
            }
        )
    ds.save_episode()
    ds.finalize()

    ds = LeRobotDataset("tests/journey", root=root)
    adopt(ds, [CAM], LABELS, (H, W), treatments={"tray": TINT}, background=BLUR)
    blob = np.zeros((H, W), bool)
    blob[10:40, 10:60] = True
    write_episode(ds, 0, CAM, [{"ball": blob, "tray": blob} for _ in range(FRAMES)])
    return root


@pytest.fixture
def page(masked_root, tmp_path, monkeypatch):
    from lerobot.gui import process_jobs as jobs_mod, server as gui_server_mod
    from lerobot.gui.api import process as process_mod

    # The episode-masks save spawns a worker that writes job state. Left alone
    # it lands in the developer's real ~/.cache; both the module that defines
    # the path and the one that imported the name have to be redirected, or the
    # bound copy keeps the original.
    jobs_dir = tmp_path / "process_jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(jobs_mod, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(process_mod, "JOBS_DIR", jobs_dir, raising=False)

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
            f"() => document.querySelector('.row-track[data-feature=\"{MASK_KEY}\"]')", timeout=60_000
        )
        pg.ds_id = ds_id
        pg.root = masked_root
        yield pg
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


def _recipe(root) -> dict:
    """The recipe as it is ON DISK -- the only thing a later session will read."""
    info = json.loads((root / "meta" / "info.json").read_text())
    return next(v for v in info["features"].values() if v.get("mask_encoding") == "coco_rle")


def _settled(pg, expression: str, seconds: float = 10.0):
    """Await a page promise, but never forever. page.evaluate has no timeout of
    its own, so a promise that never settles hangs the test with no name --
    which is how the superseded-frame bug surfaced only as CI stopping short."""
    ms = int(seconds * 1000)
    message = json.dumps(f"{expression} did not settle within {seconds:g}s")
    return pg.evaluate(
        f"() => Promise.race([Promise.resolve({expression}),"
        f" new Promise((_, reject) => setTimeout(() => reject(new Error({message})), {ms}))])"
    )


def _frame_requests(pg) -> list[str]:
    urls: list[str] = []
    pg.on("request", lambda r: urls.append(r.url) if "/frame/" in r.url else None)
    return urls


def _frame_numbers(urls: list[str]) -> list[int]:
    out = []
    for u in urls:
        head = u.split("/frame/", 1)[1]
        num = head.split("?", 1)[0]
        if num.isdigit():
            out.append(int(num))
    return out


# ── staging is not writing ──────────────────────────────────────────────────


def test_staging_a_treatment_leaves_the_stored_recipe_alone(page):
    """A staged edit is a promise about a future write, not a write."""
    before = _recipe(page.root)
    assert before["mask_treatments"]["tray"] == TINT, "fixture changed"

    page.evaluate(
        """async ([ds]) => {
            await fetch('/api/edits/mask-treatments', {method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({dataset_id: ds,
                treatments: {tray: {key: 'blur', params: {}}}, background: {key: 'none', params: {}}})});
        }""",
        [page.ds_id],
    )
    page.wait_for_timeout(300)

    assert _recipe(page.root) == before, "a staged treatment reached disk before Save"


def test_saving_lowers_the_staged_treatment_and_nothing_else(page):
    """The complement: staging must not be a way of never writing."""
    before = _recipe(page.root)

    staged = page.evaluate(
        """async ([ds]) => {
            window.Dialogs.confirm = async () => true;   // Save asks; this test is not about the asking
            await fetch('/api/edits/mask-treatments', {method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({dataset_id: ds,
                treatments: {tray: {key: 'blur', params: {}}}, background: {key: 'none', params: {}}})});
            const r = await fetch('/api/edits');
            return (await r.json()).edits.filter((e) => e.edit_type === 'mask_treatments').length;
        }""",
        [page.ds_id],
    )
    assert staged == 1, f"the treatment edit did not stage, so Save has nothing to lower: {staged}"

    page.evaluate("async () => { await window.applyEdits(); }")
    page.wait_for_timeout(2500)

    after = _recipe(page.root)
    assert after["mask_treatments"]["tray"]["key"] == "blur", "Save did not lower the staged treatment"
    # Everything the edit was NOT about is untouched.
    assert after["mask_labels"] == before["mask_labels"], "the vocabulary moved"
    assert after["mask_size"] == before["mask_size"], "the mask size changed"


# ── the transport ───────────────────────────────────────────────────────────


def test_the_playhead_never_moves_backwards_while_playing(page):
    """Reported as "playback is broken" after a save and turning the segmenter
    off, with the server log showing frames served in DESCENDING order."""
    _settled(page, "loadAllFrames(0)")
    page.wait_for_timeout(400)
    seen = _frame_requests(page)

    # `isPlaying` is module-scope in app.js and is NOT on window, so reading it
    # gives undefined and "toggle if not playing" toggles blindly -- which is how
    # this test could have paused instead of playing and proved nothing. The
    # button's own label is the observable, and it is what the operator reads.
    playing = "() => (document.getElementById('play-btn')?.textContent || '').includes('Pause')"
    page.evaluate(f"() => {{ if (!({playing})()) togglePlay(); }}")
    assert page.evaluate(playing), "Play did not start, so the frames below are not playback"
    page.wait_for_timeout(2500)
    page.evaluate(f"() => {{ if (({playing})()) togglePlay(); }}")

    frames = _frame_numbers(seen)
    assert len(frames) >= 3, f"playback fetched almost nothing, so this proves little: {frames}"
    # Looping is the one legitimate way back: the last frames wrap to the first.
    # Anything else -- and in particular the long descending sweep that was
    # reported -- is the playhead running backwards.
    steps = list(zip(frames, frames[1:], strict=False))
    wraps = [(a, b) for a, b in steps if b < a and a >= FRAMES * 0.5 and b <= FRAMES * 0.5]
    backwards = [(a, b) for a, b in steps if b < a and (a, b) not in wraps]
    assert not backwards, f"the playhead went backwards while playing: {backwards} (all: {frames})"
    assert len(wraps) <= 2, f"the episode wrapped {len(wraps)} times in 2.5 s: {wraps}"


def test_a_frame_request_that_is_superseded_still_settles(page):
    """masks.js refreshes the tiles, without awaiting, whenever the composite
    mode changes. Landing while an earlier request for the same tile is in
    flight, it abandons that request's src -- which fires neither load nor
    error -- and the earlier caller used to wait forever. That froze playback
    mid-play, and hung the playhead test above whenever the mode flipped after
    it had asked for a frame.

    Both requests are issued in one task, so no load can slip between them: the
    first is superseded every time, not only when the timing is unlucky."""
    settled = page.evaluate(
        """() => new Promise((done) => {
            const first = loadAllFrames(0);
            loadAllFrames(1);
            first.then(() => done(true));
            setTimeout(() => done(false), 5000);
        })"""
    )
    assert settled, "a superseded frame request never settled, so anything awaiting it waits forever"


def test_playback_keeps_going_when_the_tiles_are_refreshed_mid_play(page):
    """The user-facing form of the superseded-request bug. masks.js refreshes
    the tiles, unawaited, whenever a treatment is staged or the composite mode
    changes. One landing while playLoop's frame was still loading orphaned the
    promise playLoop was awaiting, and playback stopped on that frame for good
    with the button still reading Pause.

    The refresh is the one a treatment edit makes, fired for a stretch while
    playing so that some land mid-load; then the playhead is sampled."""
    playing = "() => (document.getElementById('play-btn')?.textContent || '').includes('Pause')"
    page.evaluate(f"() => {{ if (!({playing})()) togglePlay(); }}")
    assert page.evaluate(playing), "Play did not start"
    page.evaluate(
        """() => new Promise((done) => {
            const until = performance.now() + 1500;
            (function tick() {
                window.MaskOverlay.stagedTreatmentsChanged();
                if (performance.now() < until) setTimeout(tick, 7); else done();
            })();
        })"""
    )
    frames = []
    for _ in range(20):
        frames.append(page.evaluate("() => window.currentFrame"))
        page.wait_for_timeout(100)
    still_playing = page.evaluate(playing)
    page.evaluate(f"() => {{ if (({playing})()) togglePlay(); }}")

    assert still_playing, "playback stopped on its own"
    assert len(set(frames)) > 1, (
        f"playback froze on frame {frames[0] + 1} with the button still reading Pause"
    )


def test_playback_still_composites_after_a_write(page):
    """A write invalidates the frame caches the composite was being served from.
    The tiles must come back asking for the composite, not silently revert to
    stored pixels -- which looks exactly like the masks having been lost.

    Driven with a treatment save, because that is a write this harness can
    actually complete: the segmentation save runs in a subprocess that needs a
    model, so asserting on its effects here would assert on nothing.
    """
    page.evaluate(
        "() => { window.MaskOverlay.compositedActive = () => true;"
        "        window.Dialogs.confirm = async () => true; }"
    )
    before = _recipe(page.root)

    page.evaluate(
        """async ([ds]) => {
            await fetch('/api/edits/mask-treatments', {method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({dataset_id: ds,
                treatments: {tray: {key: 'blur', params: {}}}, background: {key: 'blur', params: {}}})});
            await window.applyEdits();
        }""",
        [page.ds_id],
    )
    page.wait_for_timeout(2500)
    assert _recipe(page.root) != before, "the write did not land, so this proves nothing"

    asked = _frame_requests(page)
    page.evaluate("() => loadAllFrames(5)")
    page.wait_for_timeout(1000)

    assert asked, "no frame was fetched after the write"
    assert any("masks=composited" in u for u in asked), (
        f"the tiles stopped asking for the composite after a write: {asked[-3:]}"
    )


# ── the two mechanisms, and only the two ────────────────────────────────────
#
# The design names exactly two ways to add masks: Apply while playing, which
# commits through the bottom bar, and the Inspector's dataset-wide filler --
# "This is the whole-dataset path, and the only one". The overlay panel had
# grown one of each anyway, which is why two save buttons lit up after a run.


def test_the_overlay_panel_offers_no_way_to_write(page):
    """The panel is the live query and has no scope of its own, which is the
    same reason treatments were moved out of it."""
    found = page.evaluate(
        """() => ({
            episodeSave: !!document.getElementById('ovl-save-masks'),
            datasetApply: !!document.querySelector('.overlays-process'),
            applyCheckbox: !!document.querySelector('.overlays-apply-cb'),
        })"""
    )
    assert not found["episodeSave"], "the panel still hosts an episode-scoped save"
    assert not found["datasetApply"], "the panel still hosts a dataset-wide apply"


def test_the_panel_still_reports_what_is_saved(page):
    """Removing the write must not remove the read: how much of this episode
    already carries masks is context for the query above it."""
    page.wait_for_timeout(1500)
    assert page.evaluate("() => !!document.getElementById('ovl-save-masks-hint')"), (
        "the coverage line went with the button"
    )


def test_the_filler_confirms_before_it_runs(page):
    """It is the only dataset-wide mechanism, so it is also the confirmation for
    one: what it runs over, with what, and what it will not touch."""
    opened = page.evaluate(
        """() => {
            const b = document.querySelector('.ds-fill-gaps');
            if (!b) return null;
            b.click();
            return true;
        }"""
    )
    if not opened:
        pytest.skip("the Inspector's dataset tier is not rendered in this state")
    page.wait_for_selector(".fg-modal", timeout=10_000)

    text = page.evaluate("() => document.querySelector('.fg-summary')?.textContent || ''")
    assert "Runs over" in text, f"the dialog does not say what it runs over: {text!r}"
    assert "absent" in text, "the dialog does not say it only fills gaps"

    buttons = page.evaluate(
        """() => ({
            cancel: !!document.querySelector('.fg-cancel'),
            ok: document.querySelector('.fg-run')?.textContent || '',
        })"""
    )
    assert buttons["cancel"], "no Cancel"
    assert buttons["ok"].strip() == "OK", f"the confirming button reads {buttons['ok']!r}"


def test_ok_is_unavailable_until_a_label_is_ticked(page):
    """A run over no labels is a job that walks every episode to do nothing."""
    if not page.evaluate(
        "() => { const b = document.querySelector('.ds-fill-gaps'); if (!b) return false; b.click(); return true; }"
    ):
        pytest.skip("the Inspector's dataset tier is not rendered in this state")
    page.wait_for_selector(".fg-modal", timeout=10_000)

    state = page.evaluate(
        """() => {
            const boxes = [...document.querySelectorAll('.fg-rows input')];
            boxes.forEach((c) => { if (c.checked) { c.checked = false; c.dispatchEvent(new Event('change')); } });
            const off = document.querySelector('.fg-run').disabled;
            if (boxes[0]) { boxes[0].checked = true; boxes[0].dispatchEvent(new Event('change')); }
            return { off, on: document.querySelector('.fg-run').disabled, n: boxes.length };
        }"""
    )
    assert state["n"] > 0, "no labels offered, so this proves nothing"
    assert state["off"], "OK was available with nothing ticked"
    assert not state["on"], "OK stayed unavailable after ticking a label"


def test_cancelling_the_filler_writes_nothing(page):
    """The complement: a confirmation that runs anyway is not a confirmation."""
    before = _recipe(page.root)
    if not page.evaluate(
        "() => { const b = document.querySelector('.ds-fill-gaps'); if (!b) return false; b.click(); return true; }"
    ):
        pytest.skip("the Inspector's dataset tier is not rendered in this state")
    page.wait_for_selector(".fg-modal", timeout=10_000)
    page.evaluate("() => document.querySelector('.fg-cancel').click()")
    page.wait_for_timeout(1200)

    assert page.evaluate("() => !document.querySelector('.fg-modal')"), "Cancel left the dialog open"
    assert _recipe(page.root) == before, "Cancel wrote to the dataset"


def test_the_fillers_request_is_accepted_by_the_endpoint(page):
    """The body is built in JS and validated by a pydantic model in Python, and
    nothing links the two: a missing field is a 422 at runtime and nothing
    earlier. That happened -- `episode` is required even when `episodes` is
    given, the client sent only the list, and the whole-dataset path answered
    every click with "Save masks failed".

    Asserts only that the request was ACCEPTED. What the job then does needs a
    segmentation model, which this harness has no business loading.
    """
    seen: list[tuple[int, str]] = []
    page.on(
        "response",
        lambda r: seen.append((r.status, r.url)) if "/process/episode-masks" in r.url else None,
    )
    if not page.evaluate(
        "() => { const b = document.querySelector('.ds-fill-gaps'); if (!b) return false; b.click(); return true; }"
    ):
        pytest.skip("the Inspector's dataset tier is not rendered in this state")
    page.wait_for_selector(".fg-modal", timeout=10_000)
    page.evaluate(
        """() => {
            const c = document.querySelector('.fg-rows input');
            if (c && !c.checked) { c.checked = true; c.dispatchEvent(new Event('change')); }
            document.querySelector('.fg-run').click();
        }"""
    )
    page.wait_for_timeout(3000)

    assert seen, "the filler never reached the endpoint"
    status = seen[0][0]
    assert status != 422, "the endpoint rejected the body the client built (missing a required field)"
    assert status < 400, f"the filler's request was refused: {status}"


# ── the treatment control ───────────────────────────────────────────────────
#
# It was a flat row of icon buttons in the overlay panel, deleted with that
# panel for a SCOPE reason -- a treatment is dataset-wide and the panel had
# none. What replaced it in the Inspector was a <select>, which cannot show or
# pick the colour a `tint` carries, so the tint colour became uneditable
# anywhere in the UI. The control is back, in the tier that does have the scope.


def _treat_keys(pg) -> dict:
    return dict(
        pg.evaluate(
            """() => [...document.querySelectorAll('.ds-treat')].map(w => {
                const b = w.querySelector('.ds-treat-btn.sel');
                return [w.dataset.label, b && b.dataset.key];
            })"""
        )
    )


def test_the_treatment_control_is_flat_and_shows_the_current_choice(page):
    st = page.evaluate(
        """() => ({
            widgets: document.querySelectorAll('.ds-treat').length,
            selects: document.querySelectorAll('.ds-treat-select').length,
            chips: document.querySelectorAll('.ds-tint-chip').length,
        })"""
    )
    assert st["widgets"] >= 2, f"no treatment widgets rendered: {st}"
    assert st["selects"] == 0, "a dropdown is still there"
    assert st["chips"] == st["widgets"], "every row should carry a tint swatch"

    keys = _treat_keys(page)
    assert all(keys.values()), f"a row shows no current treatment: {keys}"
    assert keys.get("__background__") == "blur", (
        f"the stored background does not read back on the control: {keys}"
    )


def test_a_tint_colour_can_be_picked_and_reaches_disk(page):
    """The capability a <select> could not offer at all. Round trip, not just
    the click: what is stored has to be the colour that was chosen."""
    before = _recipe(page.root)["mask_treatments"].get("tray")

    page.evaluate(
        """() => {
            window.Dialogs.confirm = async () => true;
            const w = document.querySelector('.ds-treat[data-label="tray"]');
            w.querySelector('.ds-treat-btn[data-key="tint"]').click();
        }"""
    )
    page.wait_for_timeout(500)
    pop = page.evaluate(
        """() => {
            const p = document.querySelector('.ds-tint-pop');
            return p && p.style.display !== 'none'
                ? {swatches: p.querySelectorAll('.ds-tint-sw').length, custom: !!p.querySelector('input[type=color]')}
                : null;
        }"""
    )
    assert pop, "clicking tint did not open the colour picker"
    assert pop["swatches"] >= 8 and pop["custom"], f"the picker offers no way to choose: {pop}"

    page.evaluate("() => document.querySelectorAll('.ds-tint-sw')[0].click()")
    page.wait_for_timeout(400)
    page.evaluate("() => document.querySelector('.ds-treat-save').click()")
    page.wait_for_timeout(3000)

    after = _recipe(page.root)["mask_treatments"].get("tray")
    assert after != before, f"the picked colour never reached disk: {before} -> {after}"
    assert after["key"] == "tint" and after["params"]["color"] == [239, 68, 68], (
        f"a different colour was stored than the one picked: {after}"
    )


def test_the_treatment_controls_hold_still_while_a_save_is_on_its_way(page):
    """A save takes a moment on a busy server. A treatment picked in that
    moment was not in the save, and was cleared when the save returned, so
    it silently never happened. Now nothing in the card can be pressed until
    the save has landed. The save's request is held here, so the moment
    lasts as long as the test needs it to."""
    # Holds the save and nothing after it, and is never removed: removing a
    # page's last route switches request interception off, and a request the
    # page sends in that moment -- the apply that follows the save -- is
    # paused and never let through.
    held = []
    page.route(
        "**/api/edits/mask-treatments",
        lambda route: route.continue_() if held else held.append(route),
    )
    page.evaluate(
        """() => {
            window.Dialogs.confirm = async () => true;
            document.querySelector('.ds-treat[data-label="__background__"] .ds-treat-btn[data-key="none"]').click();
            document.querySelector('.ds-treat-save').click();
        }"""
    )
    deadline = time.monotonic() + 30
    while not held and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert held, "the save never reached the server"

    card = page.evaluate(
        """() => {
            const buttons = [...document.querySelectorAll('.ds-treatments button')];
            return {count: buttons.length,
                    pressable: buttons.filter((b) => !b.disabled).map((b) => b.textContent.trim()),
                    save: document.querySelector('.ds-treat-save')?.textContent.trim()};
        }"""
    )
    assert card["count"] and not card["pressable"], f"pressable while the save is on its way: {card}"
    assert card["save"] == "Saving…", card
    page.evaluate(
        """() => document.querySelector(
            '.ds-treat[data-label="__background__"] .ds-treat-btn[data-key="tint"]').click()"""
    )
    assert _treat_keys(page).get("__background__") == "none", (
        "a pick got through while the save was on its way"
    )

    held[0].continue_()
    page.wait_for_function(
        "() => [...document.querySelectorAll('.ds-treatments button')].every((b) => !b.disabled)",
        timeout=30_000,
    )
    assert _recipe(page.root)["mask_background"]["key"] == "none", "the save did not land"


def test_the_dataset_control_does_not_reach_the_run_tab(page):
    """It is dataset-scoped. The Run tab's panel is the live query and must not
    grow a dataset-wide control by sharing a renderer with the Inspector."""
    page.evaluate("() => switchTab('run')")
    page.wait_for_timeout(2000)
    leaked = page.evaluate("() => document.querySelectorAll('#overlays-body .ds-treat').length")
    assert leaked == 0, f"the dataset tier's control appeared in the Run tab: {leaked}"

    page.evaluate("() => switchTab('data')")
    page.wait_for_timeout(1500)
    assert page.evaluate("() => document.querySelectorAll('.ds-treat').length >= 2"), (
        "the control did not survive a tab round trip"
    )


def test_the_dataset_tier_carries_the_dataset_s_own_facts(page):
    """The tier is the home for dataset scope, so what dataset you are looking at
    belongs in it. Those facts used to live only in the Inspector's EMPTY state,
    which is replaced the moment an episode is selected -- so they left the
    screen exactly when you started working.
    """
    facts = page.evaluate(
        """() => {
            const card = document.querySelector('.ds-facts');
            if (!card) return null;
            const keys = [...card.querySelectorAll('.ds-fact-key')].map(e => e.textContent.trim());
            const vals = [...card.querySelectorAll('.ds-fact-val')].map(e => e.textContent.trim());
            return Object.fromEntries(keys.map((k, i) => [k, vals[i]]));
        }"""
    )
    assert facts, "the dataset tier shows no facts"
    for key in ("repo", "episodes", "frames", "fps", "cameras"):
        assert key in facts, f"{key} missing from the dataset facts: {facts}"
    assert facts["episodes"] == "1", f"episode count is wrong: {facts}"
    assert facts["frames"] == str(FRAMES), f"frame count is wrong: {facts}"
    assert facts["cameras"] == "1", f"camera count is wrong: {facts}"


def test_the_facts_survive_selecting_an_episode(page):
    """The regression that motivates this: they were an empty state."""
    assert page.evaluate("() => !!document.querySelector('.ds-facts')"), "no facts to begin with"
    page.evaluate("() => loadAllFrames(10)")
    page.wait_for_timeout(800)
    assert page.evaluate("() => !!document.querySelector('.ds-facts')"), (
        "the dataset's facts disappeared once a frame was selected"
    )
