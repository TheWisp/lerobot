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
