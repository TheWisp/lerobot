"""A dataset switch has to re-scope the overlay panel.

The panel is scoped to a dataset: it snapshots the outgoing one's config and
restores the incoming one's. That snapshot reads each object row's
`treatment.key`, and picking a segmenter on a dataset that already carries masks
seeds those rows from the stored recipe -- as a name and a sign, with no
treatment. So the first dataset switch after a pick threw, out of the last
statement of the switch handler, and the panel stayed scoped to the dataset
before it: its cameras, its objects and its background, against a dataset they
do not describe. A job started from the Inspector then ran the previous
dataset's cameras.

Nothing appears on screen when that happens, which is most of why it lasted. So
these drive it: two datasets whose masked cameras differ, a real switch, and the
panel read back afterwards.
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


# ── the switch between two datasets ─────────────────────────────────────────
#
# The panel is scoped to a dataset: it snapshots the outgoing one's config and
# restores the incoming one's. The snapshot reads each object row's
# `treatment.key`, and picking a segmenter on a dataset that already carries
# masks seeds those rows from the stored recipe -- as name and sign, with no
# treatment. So the first dataset switch after a pick threw, out of the last
# statement of the switch handler, and the panel stayed scoped to the dataset
# before it: its cameras, its objects, its background, now describing a
# dataset they do not belong to. The Inspector's filler then offered them, and
# a job started from it ran the previous dataset's cameras.


@pytest.fixture
def second_root(tmp_path):
    """Another dataset, masked on a DIFFERENT camera, so which one the panel is
    scoped to is visible in the answer rather than shared by both."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import adopt, write_episode

    other = PICKED[0]
    root = tmp_path / "fillgaps2"
    feats = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    for cam in CAMS:
        feats[cam] = {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id="tests/fillgaps2", fps=10, root=root, features=feats, use_videos=True)
    img = np.full((H, W, 3), 70, np.uint8)
    for _ in range(FRAMES):
        ds.add_frame(
            {
                "observation.state": np.zeros(2, np.float32),
                "action": np.zeros(2, np.float32),
                "task": "fillgaps2",
                **dict.fromkeys(CAMS, img),
            }
        )
    ds.save_episode()
    ds.finalize()
    ds = LeRobotDataset("tests/fillgaps2", root=root)
    adopt(ds, [other], LABELS, (H, W))
    blob = np.zeros((H, W), bool)
    blob[8:30, 8:40] = True
    write_episode(ds, 0, other, [{"ball": blob, "tray": blob} for _ in range(FRAMES)])
    return root


def test_switching_datasets_after_a_pick_rescopes_the_panel(page, second_root):
    """One dataset switch, with a segmenter picked first so the object rows are
    seeded from the stored recipe. The switch must complete and re-scope; the
    filler must then describe the dataset that is open."""
    pg = page
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))

    _pick_segmenter(pg)
    _click_cams(pg, PICKED)
    assert sorted(_cam_state(pg)["selected"]) == sorted(PICKED)

    other = str(second_root)
    pg.evaluate("(d) => openDataset(d)", other)
    pg.wait_for_function("(d) => window.datasets && window.datasets[d]", arg=other, timeout=60_000)
    pg.evaluate("([d, n]) => selectEpisode(d, 0, n)", [other, FRAMES])
    pg.wait_for_timeout(1200)

    assert not errors, f"the dataset switch threw: {errors}"
    assert pg.evaluate("() => window.currentDataset") == other, "the app did not follow the switch"

    # The panel is a new dataset's now: it must not still be answering with the
    # previous one's selection.
    _pick_segmenter(pg)
    scoped = _cam_state(pg)
    assert sorted(scoped["selected"]) != sorted(PICKED), (
        f"the panel carried the previous dataset's cameras into this one: {scoped}"
    )

    # And what the filler offers is that, not the dataset before it.
    _open_dialog(pg)
    named = _dialog_cameras(pg)
    _ok(pg)
    pg.wait_for_timeout(1500)
    assert len(pg.sent) == 1, f"expected exactly one episode-masks request, got {pg.sent}"
    sent = pg.sent[0]["cameras"]
    assert sorted(sent) == sorted(scoped["selected"]), f"dialog {named}, panel {scoped}, job {sent}"
    assert pg.sent[0]["source_id"] == other, (
        f"the job was started against the previous dataset: {pg.sent[0]['source_id']}"
    )


def test_picking_a_segmenter_seeds_the_rows_with_their_stored_treatment(page):
    """Why the row shape matters, stated as behaviour rather than as a shape.

    Picking a segmenter on a dataset that already carries masks seeds the
    panel's rows from the stored recipe -- the point being to carry on with
    what the dataset already says, not to start from nothing. The seeder
    returns whole rows, treatment included; the panel rebuilt them as a name
    and a sign, so the stored treatment was dropped on the way in and the row
    was left in a shape its readers do not admit. The dropped treatment is the
    visible half and this pins it; the shape is the half that threw.
    """
    pg = page
    # The panel seeds at the moment the segmenter is picked, from the recipe as
    # it stands then -- and the recipe arrives from the server. Picking before it
    # lands seeds nothing, which is a race this test would otherwise report as
    # the defect it is checking for. (It passed locally and failed on the slower
    # CI runner, which is the only reason it was visible at all.)
    pg.wait_for_function(
        "() => { const r = window.MaskOverlay && window.MaskOverlay.savedRecipe"
        "  && window.MaskOverlay.savedRecipe(); return !!(r && (r.labels || []).length); }",
        timeout=30_000,
    )
    _pick_segmenter(pg)
    rows = pg.evaluate("() => (window.Overlays.dataQuery() || {}).objects")
    seeded = {r["name"]: r.get("treatment") for r in rows}

    assert set(seeded) == set(LABELS), f"the panel did not seed from the stored vocabulary: {rows}"
    assert seeded["tray"] == STORED_TREATMENT, (
        f"the stored treatment was not carried into the panel: {seeded}"
    )
    # The complement: a label the recipe stores nothing for must not acquire
    # one, or "carried the treatment" is satisfied by handing every row the same.
    assert seeded["ball"] == {"key": "none", "params": {}}, seeded
