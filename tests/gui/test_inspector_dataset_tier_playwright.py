# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The decoded instruction must be readable without spending a timeline row.

``task_index`` is stored per frame, but one instruction per episode is the
format's contract — ``modify_tasks`` maps episode → one task and rewrites every
row from ``episode_index``. So the synthetic ``task`` feature is per-episode,
which in this frontend means "hidden from the timeline, shown in the
Inspector": a string feature that is constant for the whole episode renders as
a single full-width band, a whole row spent on one unchanging value.

Both halves are pinned because each was wrong in turn. The flag was first
inherited from ``_detect_per_episode_features``, which skips DEFAULT_FEATURES
and ``task_index`` among them — always False, so the band appeared anyway.
Declaring the flag then exposed two gaps behind it: the Inspector dropped the
card for being read-only, and once admitted it rendered before the feature
series arrived and read "no data in selection" forever, because only the rows
were re-rendered on load.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright

REPO_ID = "test/dataset_tier"
TASK = "assemble cylinder into ring"
EPISODES = 2
FRAMES = 20


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _gui_open_on(hf_home: Path, root: Path):
    """Serve the GUI, open ``root`` by absolute path, select episode 0."""
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
        server.should_exit = True
        pytest.fail("GUI server did not come up")

    # Opened by absolute path so the request never reaches the Hub, and patched
    # so the GUI's own source scan cannot see a real dataset directory either.
    key = str(root)
    with patch("lerobot.gui.api.datasets.HF_LEROBOT_HOME", Path(hf_home)), sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.goto(base_url)
        pg.wait_for_function("typeof switchTab === 'function'", timeout=10_000)
        pg.evaluate("switchTab('data')")
        pg.evaluate(f"openDataset({key!r})")
        pg.wait_for_function(f"window.datasets?.[{key!r}]?.features_schema?.task != null", timeout=15_000)
        # The factory does not split frames evenly, so take the real length —
        # a wrong one leaves window.totalFrames disagreeing with the series.
        ep_len = pg.evaluate(f"window.episodes[{key!r}][0].length")
        assert ep_len > 1, f"episode 0 must have frames to render, got {ep_len}"
        pg.evaluate(f"selectEpisode({key!r}, 0, {ep_len})")
        # The card is filled by the feature-series fetch, not by the synchronous
        # render that precedes it.
        pg.wait_for_function(
            "(() => { const c = document.querySelector("
            "'#inspector-body .feature-card[data-feature=\"task\"]');"
            " return !!c && !c.innerText.includes('no data in selection')"
            " && c.innerText.trim().length > 12; })()",
            timeout=15_000,
        )
        yield pg, key
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


def _build(factory, hf_home: Path, repo_id: str, **kw) -> Path:
    root = hf_home / repo_id
    root.mkdir(parents=True)
    factory(
        root=root,
        repo_id=repo_id,
        total_episodes=EPISODES,
        total_frames=FRAMES,
        use_videos=False,
        camera_features={},
        **kw,
    )
    return root


@pytest.fixture
def page(tmp_path, lerobot_dataset_factory):
    """One hermetic single-task dataset, open on episode 0."""
    import pandas as pd

    hf_home = tmp_path / "hf_home"
    # Named explicitly — the assertions look for this string, and the factory
    # would otherwise mint a generic "task_0".
    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task"))
    root = _build(lerobot_dataset_factory, hf_home, REPO_ID, total_tasks=1, tasks=tasks)
    with _gui_open_on(hf_home, root) as pk:
        yield pk


def test_the_inspector_has_a_dataset_section(page):
    """The Inspector renders one section per scope and had none for the dataset."""
    pg, _ = page
    titles = pg.evaluate(
        "() => [...document.querySelectorAll('#inspector-body .sel-title')].map(e => e.textContent.trim())"
    )
    assert "Dataset" in titles, f"no DATASET section: {titles}"
    assert titles.index("Dataset") == 0, f"dataset is the broadest scope and goes first: {titles}"


def test_the_section_says_which_dataset_and_how_big(page):
    pg, _ = page
    facts = pg.evaluate(
        """() => {
            const card = document.querySelector('#inspector-body .ds-facts');
            if (!card) return null;
            const k = [...card.querySelectorAll('.ds-fact-key')].map(e => e.textContent.trim());
            const v = [...card.querySelectorAll('.ds-fact-val')].map(e => e.textContent.trim());
            return Object.fromEntries(k.map((x, i) => [x, v[i]]));
        }"""
    )
    assert facts, "the dataset section carries no facts"
    assert facts["repo"] == REPO_ID, facts
    assert facts["episodes"] == str(EPISODES), facts
    assert facts["frames"] == str(FRAMES), facts  # the factory's FRAMES is the total


def test_the_facts_survive_selecting_a_frame(page):
    """The regression that motivates the section: they lived only in the
    Inspector's EMPTY state, which is replaced as soon as anything is selected."""
    pg, _ = page
    assert pg.evaluate("() => !!document.querySelector('#inspector-body .ds-facts')")
    pg.evaluate("() => loadAllFrames(5)")
    pg.wait_for_timeout(600)
    assert pg.evaluate("() => !!document.querySelector('#inspector-body .ds-facts')"), (
        "the dataset's facts disappeared once a frame was selected"
    )


def test_the_no_episode_state_shows_the_same_section(page):
    """One rendering of the dataset's facts, not two. The empty state used to
    spell them out in its own markup, so adding a fact meant editing two places
    and the panel had two ways to look."""
    pg, _ = page
    # Re-open the dataset with no episode selected: the Inspector falls back to
    # its no-selection state. Driven through the module's own hook, not an
    # internal function, so this exercises the path the app actually takes.
    pg.evaluate(
        "() => { window.currentEpisode = null; window.FeatureEditing.onDatasetOpened(window.currentDataset); }"
    )
    pg.wait_for_timeout(600)

    assert pg.evaluate("() => !!document.querySelector('#inspector-body .ds-facts')"), (
        "the no-episode state renders its own summary instead of the section"
    )
    assert (
        pg.evaluate(
            "() => [...document.querySelectorAll('#inspector-body .sel-title')].map(e => e.textContent.trim())"
        )[0]
        == "Dataset"
    )


def test_the_filler_gives_the_job_runner_somewhere_to_report(page):
    """A dataset-wide fill is the longest-running thing the GUI starts, and it
    reported nothing while it ran.

    The shared job runner reports progress into a button. The panel's own save
    hands it one; the Inspector's filler has no button, passed `null`, and every
    `if (btn)` update inside the runner was therefore skipped. What the operator
    saw was one status line set before the run began -- identical whether the job
    was working, finished, or wedged.

    Pinned at the seam rather than through a real job: the defect is that the
    filler hands the runner no sink, which is visible without segmenting
    anything. That `setStatus` is what the sink writes to is asserted too, so a
    callback that goes nowhere cannot pass this.
    """
    pg, _ = page
    captured = pg.evaluate(
        """() => {
            const seen = { opts: null, statuses: [] };
            const realStatus = window.setStatus;
            window.setStatus = (m) => { seen.statuses.push(m); };
            window.OverlayStream = window.OverlayStream || {};
            const realRun = window.OverlayStream.runMaskJob;
            window.OverlayStream.runMaskJob = (btn, eps, opts) => {
                seen.opts = { hasProgress: typeof (opts || {}).onProgress === 'function' };
                // Drive the sink the way the runner's poll loop does.
                if (opts && opts.onProgress) opts.onProgress('Filling… episode 2/7');
                return Promise.resolve();
            };
            return Promise.resolve(
                window.FeatureEditing._internals.runFillGaps(window.currentDataset, ['ring'], 7)
            ).then(() => {
                window.OverlayStream.runMaskJob = realRun;
                window.setStatus = realStatus;
                return seen;
            });
        }"""
    )
    assert captured["opts"] is not None, "the filler never reached the shared job runner"
    assert captured["opts"]["hasProgress"], (
        "the filler passed no progress callback; every update inside the runner is skipped"
    )
    assert "Filling… episode 2/7" in captured["statuses"], (
        f"progress from the runner did not reach the status line: {captured['statuses']}"
    )


# ── the first pass has to be reachable ───────────────────────────────────────
#
# A dataset with no mask column is the state EVERY dataset starts in, and it was
# the one state with no way forward: the Overlays panel has no write by design,
# apply-while-playing is refused until a column exists, and the Inspector's fill
# button was rendered only when a vocabulary already existed -- withheld exactly
# when it was the only way in. Found by driving a real 274-episode dataset.


def _name_objects(pg, names):
    """Stand in for the Overlays panel without loading a segmenter."""
    pg.evaluate(
        """(names) => {
            window.Overlays = window.Overlays || {};
            window.Overlays.dataQuery = () => ({ objects: names.map(n => ({name: n})) });
            window.FeatureEditing.onLiveObjectsChanged();
        }""",
        names,
    )


def test_the_first_pass_is_withheld_until_something_is_named(page):
    """The complement, and it must come first: with nothing named there is
    nothing to segment for, so the offer would be an action with no subject."""
    pg, _ = page
    _name_objects(pg, [])
    assert pg.evaluate("() => !!document.querySelector('.ds-fill-gaps')") is False, (
        "a pass was offered with no object named"
    )
    hint = pg.evaluate("() => (document.querySelector('.ds-treat-hint') || {}).textContent || ''")
    assert "Name an object" in hint, f"the empty state must say what to do next: {hint!r}"


def test_a_dataset_with_no_masks_offers_the_first_pass_once_named(page):
    """The fix: naming an object in the panel makes the dataset tier offer the
    pass that creates the column."""
    pg, _ = page
    _name_objects(pg, ["ball", "black holder"])
    assert pg.evaluate("() => !!document.querySelector('.ds-fill-gaps')"), (
        "no way to start a first pass on a dataset with no masks"
    )
    label = pg.evaluate("() => document.querySelector('.ds-fill-gaps').textContent")
    assert "Segment" in label, f"the button should say it segments, not that it fills gaps: {label!r}"


def test_the_dialog_seeds_its_labels_from_the_panel_when_nothing_is_stored(page):
    """There is no vocabulary to draw a menu from, so the panel's objects are the
    label set — and they are ticked, because the operator just typed them."""
    pg, _ = page
    _name_objects(pg, ["ball", "black holder"])
    pg.evaluate("() => document.querySelector('.ds-fill-gaps').click()")
    pg.wait_for_selector(".fg-modal", timeout=20000)
    rows = pg.evaluate(
        """() => [...document.querySelectorAll('.fg-rows input')].map(c => ({
            label: c.getAttribute('data-label'), checked: c.checked }))"""
    )
    assert [r["label"] for r in rows] == ["ball", "black holder"], rows
    assert all(r["checked"] for r in rows), f"seeded labels must be ticked: {rows}"
    assert pg.evaluate("() => document.querySelector('.fg-run').disabled") is False


def test_the_panel_tells_the_inspector_when_object_names_change():
    """The tests above drive `onLiveObjectsChanged` directly, so none of them
    would notice if nothing ever called it — which is exactly the state the fix
    started in: the offer appeared only when some unrelated action happened to
    re-render the Inspector.

    Checked at the source because the alternative needs a loaded segmenter to
    make the panel's name inputs exist at all, and this is a one-line wiring
    contract between two files.
    """
    import pathlib

    static = pathlib.Path(__file__).resolve().parents[2] / "src/lerobot/gui/static"
    panel = (static / "overlays.js").read_text()
    tier = (static / "feature_editing.js").read_text()
    assert "onLiveObjectsChanged" in tier, "feature_editing no longer exports the notifier"
    assert "FeatureEditing?.onLiveObjectsChanged?.()" in panel, (
        "overlays.js does not tell the Inspector that the named objects changed; "
        "the first-pass offer will not appear until something else re-renders it"
    )
