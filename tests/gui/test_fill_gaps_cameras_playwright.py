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
    """The camera names the dialog puts in front of the operator, as written."""
    line = pg.evaluate(
        """() => [...document.querySelectorAll('.fg-summary div')]
                 .map(d => d.textContent).find(t => t.includes('cameras:')) || ''"""
    )
    assert "cameras:" in line, f"the dialog does not say which cameras it will run over: {line!r}"
    return [s.strip() for s in line.split("cameras:", 1)[1].split(",") if s.strip()]


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


def test_the_dialog_names_the_cameras_the_job_is_given(page):
    """One list, three readings: the buttons, the dialog's summary, and the
    request body. The reported run had them disagree, and the operator only ever
    sees the first two."""
    pg = page
    _pick_segmenter(pg)
    _click_cams(pg, PICKED)

    state = _cam_state(pg)
    assert sorted(state["on"]) == sorted(PICKED), f"the buttons did not take the pick: {state}"
    assert sorted(state["selected"]) == sorted(PICKED), f"the panel's state did not follow: {state}"

    _open_dialog(pg)
    named = _dialog_cameras(pg)
    _ok(pg)
    pg.wait_for_timeout(1500)

    assert len(pg.sent) == 1, f"expected exactly one episode-masks request, got {pg.sent}"
    sent = pg.sent[0]["cameras"]
    assert sorted(sent) == sorted(PICKED), f"the job was given {sent}, but the operator picked {PICKED}"
    assert sorted(named) == sorted(c.split(".")[-1] for c in sent), (
        f"the dialog named {named} and the request carried {sent}"
    )


def test_the_default_is_the_masked_camera_and_is_sent_as_such(page):
    """The complement, without which the test above passes for a dataset whose
    every path happens to yield the same list: untouched, the panel defaults to
    the camera that already carries masks, and that is what goes out."""
    pg = page
    _pick_segmenter(pg)

    state = _cam_state(pg)
    assert state["on"] == [MASKED], f"the default is not the masked camera: {state}"

    _open_dialog(pg)
    named = _dialog_cameras(pg)
    _ok(pg)
    pg.wait_for_timeout(1500)

    assert len(pg.sent) == 1, f"expected exactly one episode-masks request, got {pg.sent}"
    assert pg.sent[0]["cameras"] == [MASKED], pg.sent[0]["cameras"]
    assert named == [MASKED.split(".")[-1]], named


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
# selection in its closure. From there the two readings part company: the dialog
# still asks the panel and names the operator's pick, and a scrape finds no
# buttons at all and falls back to every camera in the dataset.


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
    buttons to read, so anything that answers "which cameras" by looking at the
    document answers "all of them" -- while the dialog, which asks the panel,
    still says what the operator picked."""
    pg = page
    _pick_segmenter(pg)
    _click_cams(pg, PICKED)
    _unpick_segmenter(pg)

    kept = pg.evaluate("() => (window.Overlays.dataQuery() || {}).cameras")
    assert sorted(kept) == sorted(PICKED), f"the panel forgot the pick when the segmenter went off: {kept}"

    _open_dialog(pg)
    named = _dialog_cameras(pg)
    assert sorted(named) == sorted(c.split(".")[-1] for c in PICKED), (
        f"the dialog stopped naming the operator's pick: {named}"
    )
    _ok(pg)
    pg.wait_for_timeout(1500)

    assert len(pg.sent) == 1, f"expected exactly one episode-masks request, got {pg.sent}"
    sent = pg.sent[0]["cameras"]
    assert sorted(sent) == sorted(PICKED), f"the dialog named {named} and the job was given {sent}"


# ── the other panel ─────────────────────────────────────────────────────────
#
# There are two overlay panels in one document, one per tab, and both render
# `.overlays-cam-btn`. A scrape of the document therefore answers with the union
# of whatever both panels have lit, and the Run tab's panel keeps its buttons in
# the page after it has rendered them once -- it is hidden, not removed.
#
# So the segmenter being off is not the only way the two readings part company.
# With it ON in the Data tab, and the Run tab having been visited at any point,
# a job started from the Data tab is given the Run tab's cameras as well.


RUN_PANEL = "#overlays-panel-run"
RUN_ONLY_CAM = "observation.images.wrist_run"


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
    What it has selected is not an input to a dataset-wide fill started from the
    Data tab, and with the segmenter left ON here -- so the Data panel's own
    buttons are in the page and a scrape finds something -- it used to be
    included anyway."""
    pg = page
    _teleop_cameras(pg, [RUN_ONLY_CAM])
    _pick_run_segmenter(pg)

    _pick_segmenter(pg)
    _click_cams(pg, PICKED)
    assert sorted(_cam_state(pg)["selected"]) == sorted(PICKED), _cam_state(pg)
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
    assert sorted(sent) == sorted(PICKED), f"the dialog named {named} and the job was given {sent}"
