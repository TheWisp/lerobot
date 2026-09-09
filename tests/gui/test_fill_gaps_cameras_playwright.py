# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The fill-gaps dialog and the job it starts must name the same cameras.

Reported from the rig: "despite my settings to right_wrist and top_l, it only
used left_wrist and found nothing". The job's own record agreed -- its progress
file read ``"coverage": {"masks.left_wrist": 0}`` over 3282 frames and its
worker log said ``mask pass decode: ... for 1 camera(s)`` -- on a dataset with
four cameras.

Nothing caught it because the only tests that reached this dialog replaced the
overlay panel with a stub (``window.Overlays.dataQuery = () => ({objects: ...})``)
that carries no cameras at all, and none of them clicked OK. The camera list the
request would send was never observed by anything.

So these drive the real panel: pick a segmenter, click the camera buttons, open
the dialog from the Inspector's own button, read what it says it will run over,
click OK, and read the body of the request that goes out.

The dataset is shaped like the rig's -- three cameras with masks stored on ONE
of them -- so the panel's default (the masked camera) and the operator's pick
are different sets, and a fallback to either is visible in the assertion rather
than hidden behind a coincidence.
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

H, W = 48, 64
FRAMES = 12
EPISODES = 2
MASKED = "observation.images.top"
PICKED = ["observation.images.left_wrist", "observation.images.right_wrist"]
CAMS = [MASKED, *PICKED]
LABELS = ["ball", "tray"]
# One label carries a stored treatment, so "the panel seeded from the recipe"
# is checkable against something other than the default.
STORED_TREATMENT = {"key": "blur", "params": {}}
PANEL = "#overlays-panel"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def dataset_root(tmp_path):
    """Three cameras, masks stored on exactly one of them."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt, write_episode

    root = tmp_path / "fillgaps"
    feats = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    for cam in CAMS:
        feats[cam] = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id="tests/fillgaps", fps=10, root=root, features=feats, use_videos=True)
    img = np.full((H, W, 3), 90, np.uint8)
    for _ in range(EPISODES):
        for _ in range(FRAMES):
            ds.add_frame(
                {
                    "observation.state": np.zeros(2, np.float32),
                    "action": np.zeros(2, np.float32),
                    "task": "fillgaps",
                    **dict.fromkeys(CAMS, img),
                }
            )
        ds.save_episode()
    ds.finalize()

    ds = LeRobotDataset("tests/fillgaps", root=root)
    adopt(ds, [MASKED], LABELS, (H, W), treatments={"tray": STORED_TREATMENT})
    blob = np.zeros((H, W), bool)
    blob[8:30, 8:40] = True
    write_episode(ds, 0, MASKED, [{"ball": blob, "tray": blob} for _ in range(FRAMES)])
    return root


@pytest.fixture
def page(dataset_root, tmp_path, monkeypatch):
    from lerobot.gui import process_jobs as jobs_mod, server as gui_server_mod
    from lerobot.gui.api import process as process_mod

    jobs_dir = tmp_path / "process_jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(jobs_mod, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(process_mod, "JOBS_DIR", jobs_dir, raising=False)

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import requests

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if requests.get(base, timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        server.should_exit = True
        pytest.fail("GUI server did not come up")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1600, "height": 1000})
        _stub_the_gpu(pg)
        pg.sent = []
        _capture_the_job(pg)
        pg.goto(base)
        pg.wait_for_function("typeof openDataset === 'function'", timeout=15_000)
        ds_id = str(dataset_root)
        pg.evaluate("(ds) => openDataset(ds)", ds_id)
        pg.wait_for_function("(ds) => window.datasets && window.datasets[ds]", arg=ds_id, timeout=60_000)
        pg.evaluate("([ds, n]) => selectEpisode(ds, 0, n)", [ds_id, FRAMES])
        pg.wait_for_function("() => !!document.querySelector('.ds-fill-gaps')", timeout=30_000)
        pg.ds_id = ds_id
        yield pg
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


def _stub_the_gpu(pg):
    """Picking a segmenter configures the data publisher, which would load SAM3.
    None of these tests are about the model -- they are about which cameras the
    panel says it is working on."""
    pg.route(
        "**/api/overlays/data/**",
        lambda route: route.fulfill(status=200, content_type="application/json", body="{}"),
    )


def _capture_the_job(pg):
    """Hold the episode-masks request instead of running it: what this file is
    about is the body, and a real run would spawn a worker."""

    def take(route):
        pg.sent.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"job_id": "captured", "status": "started"}),
        )

    pg.route("**/api/process/episode-masks", take)
    pg.route(
        "**/api/process/jobs",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"jobs": [{"job_id": "captured", "status": "complete", "coverage": {}}]}),
        ),
    )


def _pick_segmenter(pg):
    pg.evaluate(
        "(sel) => { const p = document.querySelector(sel + ' .overlays-picker');"
        " p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); }",
        PANEL,
    )
    pg.wait_for_function(
        "(sel) => document.querySelectorAll(sel + ' .overlays-cam-btn').length > 0",
        arg=PANEL,
        timeout=20_000,
    )


def _cam_state(pg) -> dict:
    """What each of the three readings of "which cameras" says right now."""
    return pg.evaluate(
        """(sel) => ({
            // the buttons, as the operator sees them
            on: [...document.querySelectorAll(sel + ' .overlays-cam-btn.on')].map(b => b.dataset.cam),
            // the panel's own state
            selected: (window.Overlays.dataQuery() || {}).cameras,
        })""",
        PANEL,
    )


def _click_cams(pg, wanted: list[str]):
    """Drive the buttons until the selection is exactly `wanted`, the way an
    operator would -- one click per camera that disagrees."""
    pg.evaluate(
        """([sel, want]) => {
            const set = new Set(want);
            for (const b of document.querySelectorAll(sel + ' .overlays-cam-btn')) {
                const on = b.classList.contains('on');
                if (on !== set.has(b.dataset.cam)) b.click();
            }
        }""",
        [PANEL, wanted],
    )
    pg.wait_for_timeout(200)


def _open_dialog(pg):
    pg.evaluate("() => document.querySelector('.ds-fill-gaps').click()")
    pg.wait_for_selector(".fg-modal", timeout=20_000)


def _dialog_cameras(pg) -> list[str]:
    """The cameras the dialog has selected, as the operator sees them."""
    cams = pg.evaluate("() => [...document.querySelectorAll('.fg-cam.on')].map(b => b.dataset.cam)")
    assert pg.evaluate("() => document.querySelectorAll('.fg-cam').length") > 0, (
        "the dialog offers no camera control"
    )
    return [c.split(".")[-1] for c in cams]


def _dialog_click_cam(pg, cam):
    """Toggle one of the dialog's own camera buttons."""
    pg.evaluate('(c) => document.querySelector(`.fg-cam[data-cam="${c}"]`).click()', cam)
    pg.wait_for_timeout(150)


def _ok(pg):
    """Tick every label the dialog offers, then OK -- the operator's gesture.
    OK is disabled with nothing ticked, so a click alone is a no-op and would
    make every assertion below vacuous."""
    ticked = pg.evaluate(
        """() => {
            const rows = [...document.querySelectorAll('.fg-rows input')];
            for (const c of rows) if (!c.checked) c.click();
            return rows.length;
        }"""
    )
    assert ticked, "the dialog offered no labels to fill"
    assert pg.evaluate("() => document.querySelector('.fg-run').disabled") is False, (
        "OK is still disabled with every label ticked"
    )
    pg.evaluate("() => { window.confirm = () => true; document.querySelector('.fg-run').click(); }")


# ── the contract ────────────────────────────────────────────────────────────


def test_the_data_tab_shortcuts_stand_down_while_the_dialog_is_up(page):
    """A confirmation must own the keyboard.

    The app's data-tab shortcuts are bound to the document and exempt only
    INPUT and SELECT. The dialog's camera chips are buttons, so clicking one --
    the whole point of the picker -- parks focus on a BUTTON and every shortcut
    became live again behind the modal: Space started playback, the arrows moved
    to another episode, and Delete staged a delete of the episode underneath,
    all while the operator was reading a confirmation about a different run.
    """
    pg = page
    _pick_segmenter(pg)
    _open_dialog(pg)
    cams = _dialog_cameras(pg)
    assert cams, "no camera chip to focus, so this asserts nothing"
    pg.evaluate(
        """() => {
            window.__fired = [];
            for (const n of ['togglePlay', 'navigateEpisode', 'deleteCurrentEpisode']) {
                const real = window[n];
                if (typeof real === 'function') window[n] = (...a) => window.__fired.push(n);
            }
        }"""
    )
    # Focus lands on a chip exactly as a click leaves it.
    pg.evaluate('() => document.querySelector(".fg-cam").focus()')
    for key in (" ", "ArrowDown", "Delete"):
        pg.keyboard.press("Space" if key == " " else key)
        pg.wait_for_timeout(60)
    fired = pg.evaluate("() => window.__fired")
    assert fired == [], f"the shortcuts fired behind the open dialog: {fired}"
    assert pg.evaluate("() => !!document.querySelector('.fg-modal')"), (
        "the dialog closed on its own, so the keys above proved nothing"
    )

    # The complement, or "nothing ever happens" would satisfy the assertion
    # above: with the dialog gone the very same key must reach the app again.
    pg.evaluate("() => document.querySelector('.fg-cancel').click()")
    pg.wait_for_selector(".fg-modal", state="detached", timeout=10_000)
    pg.evaluate("() => document.body.focus()")
    pg.keyboard.press("Space")
    pg.wait_for_timeout(60)
    assert pg.evaluate("() => window.__fired") == ["togglePlay"], (
        "the shortcut did not come back after the dialog closed; the guard is too wide"
    )


def test_the_dialog_names_the_cameras_the_job_is_given(page):
    """One list, two readings: the dialog's own buttons and the request body.
    The reported run had them disagree, and the operator only ever sees the
    first. The choice is made in the dialog, so that is where it is made here.
    """
    pg = page
    _pick_segmenter(pg)

    _open_dialog(pg)
    _dialog_click_cam(pg, MASKED)  # narrow it to something smaller than "all"
    named = _dialog_cameras(pg)
    assert sorted(named) == sorted(c.split(".")[-1] for c in PICKED), (
        f"unticking {MASKED.split('.')[-1]} left {named}"
    )
    _ok(pg)
    pg.wait_for_timeout(1500)

    assert len(pg.sent) == 1, f"expected exactly one episode-masks request, got {pg.sent}"
    sent = pg.sent[0]["cameras"]
    assert sorted(sent) == sorted(PICKED), f"the job was given {sent}, the dialog named {named}"
    assert sorted(named) == sorted(c.split(".")[-1] for c in sent), (
        f"the dialog named {named} and the request carried {sent}"
    )


def test_every_episode_is_in_the_request(page):
    """The dialog says "across N episodes", so the request has to carry them --
    a filler that ran one episode would still satisfy the camera assertions
    above."""
    pg = page
    _pick_segmenter(pg)
    _open_dialog(pg)
    _ok(pg)
    pg.wait_for_timeout(1500)

    assert len(pg.sent) == 1, pg.sent
    assert sorted(pg.sent[0]["episodes"]) == list(range(EPISODES)), pg.sent[0]


# ── the state the reported run was actually in ──────────────────────────────
#
# The three above pass on the code that shipped the defect: while the segmenter
# is on, the panel's buttons are in the document and a scrape of them agrees
# with the panel's own state, so every reading of "which cameras" gives the same
# answer and nothing can disagree.
#
# Turning the segmenter off takes the camera control out of the DOM -- the data
# panel's control surface for "no model" is empty -- while the panel keeps the
# selection in its closure. That is what a scrape could not survive: it found no
# buttons and fell back to every camera in the dataset. The dialog is unaffected
# either way now, because it asks the dataset rather than the panel.


def _unpick_segmenter(pg):
    pg.evaluate(
        "(sel) => { const p = document.querySelector(sel + ' .overlays-picker');"
        " p.value = ''; p.dispatchEvent(new Event('change', {bubbles: true})); }",
        PANEL,
    )
    pg.wait_for_function(
        "(sel) => document.querySelectorAll(sel + ' .overlays-cam-btn').length === 0",
        arg=PANEL,
        timeout=10_000,
    )


def test_the_dialog_and_the_job_agree_with_the_segmenter_off(page):
    """A dataset-wide fill does not need a running preview, and the Inspector's
    button is offered whether or not one is on. With it off there are no camera
    buttons in the page at all, so anything that answered "which cameras" by
    looking at the document found nothing -- and fell through to a guess. The
    dialog does not ask the document: it offers the dataset's cameras."""
    pg = page
    _pick_segmenter(pg)
    _click_cams(pg, PICKED)
    _unpick_segmenter(pg)

    assert pg.evaluate(f"() => document.querySelectorAll('{PANEL} .overlays-cam-btn').length") == 0, (
        "the panel still has camera buttons, so the scrape path is not under test"
    )

    _open_dialog(pg)
    named = _dialog_cameras(pg)
    every = sorted(c.split(".")[-1] for c in CAMS)
    assert named and sorted(named) == every, f"the dialog offered {named}, not every camera {every}"
    _ok(pg)
    pg.wait_for_timeout(1500)

    assert len(pg.sent) == 1, f"expected exactly one episode-masks request, got {pg.sent}"
    sent = pg.sent[0]["cameras"]
    assert sorted(c.split(".")[-1] for c in sent) == sorted(named), (
        f"the dialog named {named} and the job was given {sent}"
    )


RUN_ONLY_CAM = "observation.images.wrist_run"
RUN_PANEL = "#overlays-panel-run"


def _teleop_cameras(pg, cams):
    """Report a live obs stream, so the Run tab's panel has cameras to render.

    Its list comes from the teleop stream rather than from the dataset, which is
    why it can hold a camera the open dataset does not even have.
    """
    pg.route(
        "**/api/run/obs-stream/meta",
        lambda r: r.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"available": True, "image_keys": dict.fromkeys(cams, [3, 4, 3])}),
        ),
    )


def _pick_run_segmenter(pg):
    pg.evaluate("() => switchTab('run')")
    pg.wait_for_timeout(400)
    pg.evaluate(
        "(sel) => { const p = document.querySelector(sel + ' .overlays-picker');"
        " p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); }",
        RUN_PANEL,
    )
    pg.wait_for_function(
        "(s) => document.querySelectorAll(s + ' .overlays-cam-btn').length > 0",
        arg=RUN_PANEL,
        timeout=20_000,
    )
    pg.evaluate("() => switchTab('data')")
    pg.wait_for_timeout(400)


def test_the_run_tabs_cameras_do_not_reach_a_data_tab_job(page):
    """The Run tab is a different tab doing a different job on a live stream.
    Its camera is not a camera of this dataset, so it can never be one of "every
    camera" -- but with the segmenter left ON here the Data panel's buttons are
    also in the page, and a scrape used to sweep up both."""
    pg = page
    _teleop_cameras(pg, [RUN_ONLY_CAM])
    _pick_run_segmenter(pg)
    _pick_segmenter(pg)

    # The other panel is holding a camera of its own, in the page, lit.
    lit = pg.evaluate("() => [...document.querySelectorAll('.overlays-cam-btn.on')].map(b => b.dataset.cam)")
    assert RUN_ONLY_CAM in lit, f"the run panel is not holding a camera, so nothing is under test: {lit}"

    _open_dialog(pg)
    named = _dialog_cameras(pg)
    _ok(pg)
    pg.wait_for_timeout(1500)

    assert len(pg.sent) == 1, f"expected exactly one episode-masks request, got {pg.sent}"
    sent = pg.sent[0]["cameras"]
    assert RUN_ONLY_CAM not in sent, f"the run tab's camera was handed to a data-tab job: {sent}"
    assert RUN_ONLY_CAM.split(".")[-1] not in named, f"the dialog offered it: {named}"
    assert sorted(sent) == sorted(CAMS), f"the dialog named {named} and the job was given {sent}"


# ── the whole gesture, with the segmenter left on ───────────────────────────
#
# The sequence as it is actually performed: turn SAM3 on, type what to look
# for, choose the cameras, leave it running, and start the dataset-wide pass.
# Nothing here turns the segmenter off, and nothing visits another tab. What
# the pass runs on must be what the DIALOG showed -- every camera, less any the
# operator unticked -- whatever the panel happens to hold.


@pytest.fixture
def fresh_root(tmp_path):
    """Four cameras and no masks anywhere — the state a first pass is for, and
    the state in which the panel's typed objects ARE the label set."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cams = [MASKED, *PICKED, "observation.images.top_r"]
    root = tmp_path / "fresh"
    feats = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    for cam in cams:
        feats[cam] = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id="tests/fresh", fps=10, root=root, features=feats, use_videos=True)
    img = np.full((H, W, 3), 80, np.uint8)
    for _ in range(EPISODES):
        for _ in range(FRAMES):
            ds.add_frame(
                {
                    "observation.state": np.zeros(2, np.float32),
                    "action": np.zeros(2, np.float32),
                    "task": "fresh",
                    **dict.fromkeys(cams, img),
                }
            )
        ds.save_episode()
    ds.finalize()
    return root


def _type_objects(pg, names):
    """Type what to look for into the panel's object rows, as the operator does."""
    for i, name in enumerate(names):
        if i:
            pg.evaluate("(s) => document.querySelector(s + ' .overlays-add-obj').click()", PANEL)
            pg.wait_for_timeout(250)
        pg.evaluate(
            """([s, i, name]) => { const row = document.querySelectorAll(s + ' .overlays-obj-name')[i];
                row.value = name; row.dispatchEvent(new Event('input', {bubbles: true})); }""",
            [PANEL, i, name],
        )
    pg.wait_for_timeout(900)


def test_a_dataset_wide_pass_runs_every_camera_and_the_panels_labels(page, fresh_root):
    """SAM3 on, labels typed, cameras changed, segmenter left on, then the
    dataset-wide pass. The two halves come from different places and this is
    where that is pinned: the LABELS are the panel's, the CAMERAS are the
    dataset's. The panel is deliberately set to a smaller camera set, so a pass
    that inherited it would be visible here."""
    pg = page
    typed = ["yellow ball", "black holder"]

    pg.evaluate("(d) => openDataset(d)", str(fresh_root))
    pg.wait_for_function("(d) => window.datasets && window.datasets[d]", arg=str(fresh_root), timeout=60_000)
    pg.evaluate("([d, n]) => selectEpisode(d, 0, n)", [str(fresh_root), FRAMES])
    pg.wait_for_timeout(600)

    _pick_segmenter(pg)
    _type_objects(pg, typed)
    _click_cams(pg, PICKED)

    panel = pg.evaluate("() => window.Overlays.dataQuery()")
    assert sorted(panel["cameras"]) == sorted(PICKED), f"the panel did not take the cameras: {panel}"
    assert [o["name"] for o in panel["objects"]] == typed, f"the panel did not take the labels: {panel}"
    # Still running: this is not the segmenter-off path.
    assert pg.evaluate(f"() => document.querySelectorAll('{PANEL} .overlays-cam-btn').length") > 0

    pg.wait_for_function("() => !!document.querySelector('.ds-fill-gaps')", timeout=30_000)
    _open_dialog(pg)
    named = _dialog_cameras(pg)
    every = sorted(
        c.split(".")[-1] for c in pg.evaluate("() => window.datasets[window.currentDataset].camera_keys")
    )
    assert sorted(named) == every, f"the dialog offered {named}, not every camera {every}"
    _ok(pg)
    pg.wait_for_timeout(1500)

    assert len(pg.sent) == 1, f"expected exactly one episode-masks request, got {pg.sent}"
    body = pg.sent[0]
    assert sorted(c.split(".")[-1] for c in body["cameras"]) == every, (
        f"the pass runs {body['cameras']}, not every camera {every}"
    )
    assert [o["name"] for o in body["objects"]] == typed, (
        f"the pass looks for {[o['name'] for o in body['objects']]}, the panel was set to {typed}"
    )
    assert sorted(body["episodes"]) == list(range(EPISODES)), body["episodes"]


# ── every combination of what changes the answer ────────────────────────────
#
# Three things decide which code path resolves the cameras and the labels, and
# they are independent, so they are enumerated rather than sampled:
#
#   masks     the dataset already stores some, or none at all. Decides where the
#             label set comes from (the stored vocabulary, or what is typed in
#             the panel) and what the panel defaults its cameras to.
#   cameras   left at the panel's default, or changed by the operator.
#   segmenter still running when the pass is started, or turned off first.
#             Decides whether the panel's camera buttons are in the page at all.
#
# The invariant is the same in all eight: what the dialog shows is what the job
# is given, and what the dialog shows is the dataset's cameras -- never the
# panel's selection, which for a fill is the set with nothing to add. Neither
# turning the segmenter off nor changing the panel may move the answer.

COMBOS = [
    (masks, changed, running)
    for masks in (True, False)
    for changed in (True, False)
    for running in (True, False)
]


@pytest.mark.parametrize("has_masks,changed,segmenter_running", COMBOS)
def test_the_pass_runs_the_dialogs_cameras_whatever_the_panel_holds(
    page, fresh_root, has_masks, changed, segmenter_running
):
    pg = page
    typed = ["yellow ball", "black holder"]

    if not has_masks:
        pg.evaluate("(d) => openDataset(d)", str(fresh_root))
        pg.wait_for_function(
            "(d) => window.datasets && window.datasets[d]", arg=str(fresh_root), timeout=60_000
        )
        pg.evaluate("([d, n]) => selectEpisode(d, 0, n)", [str(fresh_root), FRAMES])
        pg.wait_for_timeout(600)

    _pick_segmenter(pg)
    if not has_masks:
        # With nothing stored there is no vocabulary, so the panel's objects are
        # the label set and the dialog has nothing else to offer.
        _type_objects(pg, typed)
    if changed:
        _click_cams(pg, PICKED)

    # What the panel is set to, read before anything can disturb it. This is the
    # answer every reading below has to match, whatever the default happens to be.
    panel = pg.evaluate("() => window.Overlays.dataQuery()")
    every = pg.evaluate("(d) => window.datasets[d].camera_keys", pg.evaluate("() => window.currentDataset"))
    # A dataset-wide pass takes every camera, whatever the panel is set to. The
    # panel's own state is read here so the claim is not vacuous: in the
    # `changed` half it holds a strictly smaller set, and the dialog still shows
    # all of them.
    expected = sorted(every)
    assert expected, f"no cameras resolved at all: {panel}"
    if changed:
        assert sorted(panel["cameras"]) == sorted(PICKED), f"the operator's pick did not take: {panel}"
        assert sorted(panel["cameras"]) != expected, (
            "the panel holds every camera anyway, so this case proves nothing"
        )

    if not segmenter_running:
        pg.evaluate(
            "(sel) => { const p = document.querySelector(sel + ' .overlays-picker');"
            " p.value = ''; p.dispatchEvent(new Event('change', {bubbles: true})); }",
            PANEL,
        )
        pg.wait_for_function(
            "(s) => document.querySelectorAll(s + ' .overlays-cam-btn').length === 0",
            arg=PANEL,
            timeout=10_000,
        )

    pg.wait_for_function("() => !!document.querySelector('.ds-fill-gaps')", timeout=30_000)
    _open_dialog(pg)
    shown = sorted(_dialog_cameras(pg))
    offered = pg.evaluate(
        "() => [...document.querySelectorAll('.fg-rows input')].map(c => c.getAttribute('data-label'))"
    )
    _ok(pg)
    pg.wait_for_timeout(1500)

    assert len(pg.sent) == 1, f"expected exactly one episode-masks request, got {pg.sent}"
    body = pg.sent[0]
    short = sorted(c.split(".")[-1] for c in body["cameras"])

    assert shown == sorted(c.split(".")[-1] for c in expected), (
        f"the dialog showed {shown}, expected every camera {expected}"
    )
    assert short == shown, f"the dialog showed {shown} and the job was given {short}"
    assert sorted(o["name"] for o in body["objects"]) == sorted(offered), (
        f"the job looks for {[o['name'] for o in body['objects']]}, the dialog offered {offered}"
    )
    # The label set comes from the right place for the dataset's state.
    assert sorted(offered) == sorted(LABELS if has_masks else typed), (
        f"labels came from the wrong source: {offered}"
    )
    assert sorted(body["episodes"]) == list(range(EPISODES)), body["episodes"]


def test_the_dialog_offers_every_camera_whatever_the_panel_shows(page):
    """The reported run, stated as a rule.

    The panel resolves a default when the operator has not picked: the cameras
    that already carry masks, which is right for a live preview and is the one
    set a fill has nothing to add to. The dialog does not inherit it, or
    anything else the panel holds -- "fill the gaps" means the dataset's gaps,
    so every camera is offered and ticked, and narrowing is the operator's to do
    in the dialog they are looking at.
    """
    pg = page
    _pick_segmenter(pg)  # no camera clicked: whatever appears is the panel's guess

    panel = pg.evaluate("() => window.Overlays.dataQuery()")
    assert panel["cameras"] == [MASKED], f"the panel no longer guesses the masked camera: {panel}"

    _open_dialog(pg)
    shown = sorted(_dialog_cameras(pg))
    _ok(pg)
    pg.wait_for_timeout(1500)

    every = sorted(c.split(".")[-1] for c in CAMS)
    assert shown == every, f"the dialog offered {shown}, not every camera {every}"
    assert len(pg.sent) == 1, pg.sent
    assert sorted(c.split(".")[-1] for c in pg.sent[0]["cameras"]) == every, pg.sent[0]["cameras"]


def test_the_panels_pick_does_not_narrow_the_dialog_either(page):
    """The complement, and the half that changed: the panel's cameras are not an
    instruction to the dialog even when the operator DID choose them there. One
    rule, in one direction -- otherwise "every camera" is a claim that quietly
    depends on what the panel happens to hold."""
    pg = page
    _pick_segmenter(pg)
    _click_cams(pg, PICKED)

    _open_dialog(pg)
    shown = sorted(_dialog_cameras(pg))
    _ok(pg)
    pg.wait_for_timeout(1500)

    every = sorted(c.split(".")[-1] for c in CAMS)
    assert shown == every, f"the dialog inherited the panel's pick: {shown}"
    assert sorted(c.split(".")[-1] for c in pg.sent[0]["cameras"]) == every, pg.sent[0]["cameras"]


def test_unticking_in_the_dialog_narrows_the_job(page):
    """Narrowing is why the ticks exist at all -- a pass over four 720p cameras
    costs real time. Without this, "every camera" is satisfied by a dialog whose
    ticks do nothing."""
    pg = page
    _pick_segmenter(pg)

    _open_dialog(pg)
    drop = next(c for c in CAMS if c != MASKED)
    _dialog_click_cam(pg, drop)
    shown_on = sorted(_dialog_cameras(pg))
    _ok(pg)
    pg.wait_for_timeout(1500)

    want = sorted(c.split(".")[-1] for c in CAMS if c != drop)
    assert shown_on == want, f"unticking {drop.split(chr(46))[-1]} left {shown_on}"
    assert sorted(c.split(".")[-1] for c in pg.sent[0]["cameras"]) == want, pg.sent[0]["cameras"]


# ── what the dialog does when the ground moves under it ──────────────────────


def test_the_operator_is_told_why_the_server_refused(page):
    """A refusal the dialog delegates to the endpoint has to reach the operator.

    The dialog deliberately does not gate every impossible pass itself -- a
    dataset that declares no cameras is refused server-side, "which is where
    that belongs". FastAPI sends those as a bare string `detail`, and the client
    read only the object shape, so the reason was thrown away and the toast said
    "HTTP 400": the delegation quietly went nowhere.
    """
    pg = page
    pg.route(
        "**/api/process/episode-masks",
        lambda route: route.fulfill(
            status=400,
            content_type="application/json",
            body=json.dumps({"detail": "no cameras selected"}),
        ),
    )
    _pick_segmenter(pg)
    _open_dialog(pg)
    _ok(pg)
    pg.wait_for_function("() => document.body.innerText.includes('Save masks failed')", timeout=15_000)
    text = pg.evaluate("() => document.body.innerText")
    assert "no cameras selected" in text, f"the server's reason never reached the operator: {text[-400:]!r}"
    assert "HTTP 400" not in text, "the reason was replaced by the status code"


def test_a_dataset_switch_while_the_dialog_is_up_cancels_the_pass(page, fresh_root):
    """The dialog resolves its cameras, episodes and name for the dataset that
    was open when it was built; the job runner resolves the dataset from the
    current selection. If the tree moves in between, the two disagree and the
    pass would run this dataset's cameras against another one.

    The switch goes through the app's own `openDataset`, because
    `window.currentDataset` is a mirror that app.js rewrites from its own state
    on every playhead sync -- assigning it directly is undone before OK.
    """
    pg = page
    # No segmenter: this fixture's dataset already has a mask vocabulary, so the
    # dialog opens on its own. Picking one would seed a treatment-less object row
    # that makes the panel's own dataset-switch snapshot throw -- a fault on main,
    # in a file this branch does not touch, and not what this test is about.
    _open_dialog(pg)
    shown = _dialog_cameras(pg)
    # The operator opens another dataset while the dialog is still up.
    pg.evaluate("(d) => openDataset(d)", str(fresh_root))
    pg.wait_for_function("(d) => window.datasets && window.datasets[d]", arg=str(fresh_root), timeout=60_000)
    pg.evaluate("([d, n]) => selectEpisode(d, 0, n)", [str(fresh_root), FRAMES])
    pg.wait_for_function("(d) => window.currentDataset === d", arg=str(fresh_root), timeout=15_000)
    _ok(pg)
    pg.wait_for_timeout(1200)
    assert not pg.sent, f"a pass was started against a dataset the dialog never described: {pg.sent}"
    body = pg.evaluate("() => document.body.innerText")
    assert "Fill gaps cancelled" in body, f"the operator was not told it was cancelled: {body[-300:]!r}"
    # The complement: the dialog HAD named a real camera set, so the assertion
    # above cannot pass merely because the dialog was empty.
    assert shown, "the dialog named no cameras, so this proves nothing"


def test_the_dialog_does_not_throw_on_a_dataset_that_is_not_loaded_yet(page):
    """A dataset can be selected before its record has arrived. The dialog read
    `ds.camera_keys` unconditionally, so it threw on the way to being shown and
    the click simply did nothing, with the reason only in the console.

    The panel is given a named object first: without one the function returns
    earlier, for a different reason, and never reaches the line under test.
    """
    pg = page
    _pick_segmenter(pg)
    _type_objects(pg, LABELS[:1])
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    # The record disappears while the id stays selected.
    # Remove only this dataset's record, not the whole map: the overlay panel's
    # own refresh walks the map on a timer and throws if it is empty, which is a
    # pre-existing fault in a file this branch does not touch.
    pg.evaluate(
        "() => { const d = window.currentDataset;"
        " window.__keep = [d, window.datasets[d]]; delete window.datasets[d]; }"
    )
    pg.evaluate("() => document.querySelector('.ds-fill-gaps').click()")
    pg.wait_for_timeout(900)
    built = pg.query_selector(".fg-modal") is not None
    pg.evaluate("() => { window.datasets[window.__keep[0]] = window.__keep[1]; }")
    assert not errors, f"opening the dialog threw: {errors}"
    assert not built, "a dialog was built for a dataset whose record is not loaded"


def test_an_explicit_empty_camera_list_is_sent_as_empty_not_widened(page):
    """`[]` from a caller means "nothing chosen" and must reach the server as
    that, so the server can refuse it. Falling through to the panel and then to
    every camera is the substitution this whole path exists to remove -- and the
    server reads an empty list as "no filter", so a widened list would run the
    entire dataset.
    """
    pg = page
    _pick_segmenter(pg)
    pg.evaluate(
        """([eps, labels]) => window.OverlayStream.runMaskJob(null, eps, {
            confirmed: true, overwriteOk: true, cameras: [],
            objects: labels.map((n) => ({ name: n, sign: '+', treatment: { key: 'none' } })),
        })""",
        [[0], LABELS[:1]],
    )
    deadline = time.monotonic() + 15
    while not pg.sent and time.monotonic() < deadline:
        pg.wait_for_timeout(200)
    assert pg.sent, "no request was sent"
    assert pg.sent[-1]["cameras"] == [], (
        f"an empty selection was widened to {pg.sent[-1]['cameras']} before it left the page"
    )


def test_unticking_every_camera_withdraws_ok_rather_than_running_them_all(page):
    """The one state where the dialog must refuse rather than fall back.

    An empty camera set is exactly the input the server reads as "no filter",
    so a dialog that let it through would run the whole dataset -- the widening
    this change exists to remove. It is also the only note styled as a warning,
    because it is the only one the operator can act on.
    """
    pg = page
    _open_dialog(pg)
    pg.evaluate("() => { const c = document.querySelector('.fg-rows input'); if (!c.checked) c.click(); }")
    pg.wait_for_timeout(200)
    assert pg.evaluate("() => !document.querySelector('.fg-run').disabled"), (
        "OK was already refused before any camera was unticked; this proves nothing"
    )
    pg.evaluate("() => document.querySelectorAll('.fg-cam.on').forEach((b) => b.click())")
    pg.wait_for_timeout(250)
    state = pg.evaluate(
        """() => ({
            ok: !document.querySelector('.fg-run').disabled,
            title: document.querySelector('.fg-run').title,
            note: document.querySelector('.fg-cams-note').textContent,
            warn: document.querySelector('.fg-cams-note').classList.contains('warn'),
        })"""
    )
    assert state["ok"] is False, f"OK stayed available with no camera chosen: {state}"
    assert "camera" in state["title"].lower(), state
    assert state["note"] == "pick at least one camera", state
    assert state["warn"] is True, f"the one actionable note is not styled as one: {state}"

    # And it comes back, so the refusal is a state and not a dead end.
    pg.evaluate("() => document.querySelector('.fg-cam').click()")
    pg.wait_for_timeout(250)
    back = pg.evaluate(
        "() => ({ ok: !document.querySelector('.fg-run').disabled,"
        " note: document.querySelector('.fg-cams-note').textContent,"
        " warn: document.querySelector('.fg-cams-note').classList.contains('warn') })"
    )
    assert back == {"ok": True, "note": "", "warn": False}, back


@pytest.fixture
def cameraless_root(tmp_path):
    """A dataset with state and action but no camera at all."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "nocams"
    feats = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    ds = LeRobotDataset.create(repo_id="tests/nocams", fps=10, root=root, features=feats, use_videos=False)
    for _ in range(2):
        for _i in range(FRAMES):
            ds.add_frame(
                {
                    "observation.state": np.zeros(2, np.float32),
                    "action": np.zeros(2, np.float32),
                    "task": "nocams",
                }
            )
        ds.save_episode()
    ds.finalize()
    return root


def test_a_dataset_with_no_cameras_is_not_offered_segmentation(page, cameraless_root):
    """A mask column belongs to a camera, so a dataset that declares none has
    nothing to segment. It used to be offered the pass anyway, with a dialog
    that said so and an endpoint that refused it -- a gate the operator cannot
    satisfy. The offer is withheld instead, and the reason is on screen.
    """
    pg = page
    pg.evaluate("(d) => openDataset(d)", str(cameraless_root))
    pg.wait_for_function(
        "(d) => window.datasets && window.datasets[d]", arg=str(cameraless_root), timeout=60_000
    )
    pg.evaluate("([d, n]) => selectEpisode(d, 0, n)", [str(cameraless_root), FRAMES])
    # Naming an object is what used to make the button appear.
    _pick_segmenter_without_cameras(pg)
    _type_objects(pg, LABELS[:1])
    pg.wait_for_timeout(1200)
    assert pg.evaluate("(d) => (window.datasets[d].camera_keys || []).length", str(cameraless_root)) == 0
    assert not pg.query_selector(".ds-fill-gaps"), "a camera-less dataset was offered a segmentation pass"
    body = pg.evaluate("() => document.body.innerText")
    assert "declares no cameras" in body, f"the reason is not on screen: {body[-300:]!r}"


def _pick_segmenter_without_cameras(pg):
    """As `_pick_segmenter`, but without waiting for camera buttons that a
    camera-less dataset will never render."""
    pg.evaluate(
        "(sel) => { const p = document.querySelector(sel + ' .overlays-picker');"
        " if (p) { p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); } }",
        PANEL,
    )
    pg.wait_for_timeout(1000)


def test_the_title_counts_a_single_episode_in_the_singular(page, fresh_root):
    """`Fill gaps across 1 episodes` shipped in the recorded evidence: the line
    below it pluralised and the heading did not."""
    pg = page
    pg.evaluate("(d) => openDataset(d)", str(fresh_root))
    pg.wait_for_function("(d) => window.datasets && window.datasets[d]", arg=str(fresh_root), timeout=60_000)
    pg.evaluate("([d, n]) => selectEpisode(d, 0, n)", [str(fresh_root), FRAMES])
    pg.route(
        "**/masks/label-coverage",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"labels": [{"name": LABELS[0], "episodes": 1}], "total_episodes": 1}),
        ),
    )
    _pick_segmenter(pg)
    _type_objects(pg, LABELS[:1])
    pg.wait_for_function("() => !!document.querySelector('.ds-fill-gaps')", timeout=30_000)
    _open_dialog(pg)
    title = pg.evaluate("() => document.querySelector('.fg-modal h3').textContent")
    assert title == "Segment 1 episode", title


def test_the_heading_matches_the_button_that_opened_it(page, fresh_root):
    """Two entry points, two names. A dataset with no mask column is offered
    "Segment across all episodes…", because a column is what a camera needs
    before it can have gaps; a dataset that has one is offered "Fill gaps".
    The dialog used to be headed "Fill gaps" from both.
    """
    pg = page
    # No stored masks: the labels come from the panel, and the button says Segment.
    pg.evaluate("(d) => openDataset(d)", str(fresh_root))
    pg.wait_for_function("(d) => window.datasets && window.datasets[d]", arg=str(fresh_root), timeout=60_000)
    pg.evaluate("([d, n]) => selectEpisode(d, 0, n)", [str(fresh_root), FRAMES])
    _pick_segmenter(pg)
    _type_objects(pg, LABELS[:1])
    pg.wait_for_function("() => !!document.querySelector('.ds-fill-gaps')", timeout=30_000)
    button = pg.evaluate("() => document.querySelector('.ds-fill-gaps').textContent")
    _open_dialog(pg)
    heading = pg.evaluate("() => document.querySelector('.fg-modal h3').textContent")
    assert button.startswith("Segment"), button
    assert heading.startswith("Segment"), f"button says {button!r}, dialog says {heading!r}"
    pg.evaluate("() => document.querySelector('.fg-cancel').click()")

    # The fixture dataset DOES have a stored vocabulary: both say Fill gaps.
    pg.evaluate("(d) => openDataset(d)", pg.ds_id)
    pg.wait_for_function("(d) => window.datasets && window.datasets[d]", arg=pg.ds_id, timeout=60_000)
    pg.evaluate("([d, n]) => selectEpisode(d, 0, n)", [pg.ds_id, FRAMES])
    pg.wait_for_function("() => !!document.querySelector('.ds-fill-gaps')", timeout=30_000)
    button2 = pg.evaluate("() => document.querySelector('.ds-fill-gaps').textContent")
    _open_dialog(pg)
    heading2 = pg.evaluate("() => document.querySelector('.fg-modal h3').textContent")
    assert button2.startswith("Fill gaps"), button2
    assert heading2.startswith("Fill gaps"), f"button says {button2!r}, dialog says {heading2!r}"
