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

REPO_ID = "test/task_display"
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


@pytest.fixture
def multi_task_page(tmp_path, lerobot_dataset_factory):
    """A dataset whose episode 0 carries two different tasks.

    The factory's own ``multi_task`` mode picks a task count per episode at
    random, which would make the assertion flaky. The indices are rewritten
    directly instead so the variation is guaranteed.
    """
    import pandas as pd

    hf_home = tmp_path / "hf_home"
    tasks = pd.DataFrame({"task_index": [0, 1]}, index=pd.Index([TASK, "grasp the ring"], name="task"))
    root = _build(lerobot_dataset_factory, hf_home, "test/multi_task", total_tasks=2, tasks=tasks)
    shards = sorted((root / "data").rglob("*.parquet"))
    assert shards, "factory wrote no data shards"
    for shard in shards:
        df = pd.read_parquet(shard)
        ep0 = df["episode_index"] == 0
        df.loc[ep0, "task_index"] = (df.loc[ep0, "frame_index"] % 2).astype("int64")
        df.to_parquet(shard, index=False)
    with _gui_open_on(hf_home, root) as pk:
        yield pk


def test_task_is_declared_per_episode(page):
    """The flag must not depend on the detector, which never reports task_index."""
    page, key = page
    ft = page.evaluate(f"window.datasets[{key!r}].features_schema.task")
    assert ft["is_per_episode"] is True, f"task must be per-episode, got {ft}"
    assert ft["dtype"] == "string", "the index must be decoded to a string"
    schema = page.evaluate(f"Object.keys(window.datasets[{key!r}].features_schema)")
    assert "task_index" not in schema, "the storage feature must stay hidden"


def test_task_does_not_occupy_a_timeline_row(page):
    """A value constant across the episode must not spend a row on a solid band."""
    page, _key = page
    assert page.query_selector('.feature-row[data-feature="task"]') is None, (
        "task took a timeline row; a per-episode string renders as one full-width band"
    )


def test_task_instruction_is_readable_in_the_inspector(page):
    """Hidden from the timeline, so the Inspector is the only place left."""
    page, _key = page
    card = page.query_selector('#inspector-body .feature-card[data-feature="task"]')
    assert card is not None, "task must appear in the Inspector once hidden from the timeline"
    text = card.inner_text()
    assert TASK in text, f"the instruction itself must be shown, got {text!r}"
    assert "no data in selection" not in text, (
        "card rendered before the feature series arrived and was never refreshed"
    )


def test_read_only_card_does_not_preview_an_edit(page):
    """`cardSummary` describes what an edit over the range would overwrite.

    On a card that cannot be edited there is no edit to preview, and the summary
    then restates the value shown directly beneath it, counts a range the user
    never selected, and says "uniform" against a "mixed" case the format cannot
    produce. The instruction should appear exactly once.
    """
    page, _key = page
    card = page.query_selector('#inspector-body .feature-card[data-feature="task"]')
    assert card.query_selector(".card-summary") is None, "read-only card rendered an edit-preview summary"
    text = card.inner_text()
    assert text.count(TASK) == 1, f"instruction shown {text.count(TASK)} times: {text!r}"
    assert "uniform" not in text, f"summary leaked into a read-only card: {text!r}"
    assert "frames)" not in text, f"range size restated from the section header: {text!r}"
    # The snapshot hint is suppressed when every frame holds the same value.
    # Without this assertion, dropping `&& !isUniform` stays green.
    assert "(frame" not in text, f"snapshot hint on an episode-wide value: {text!r}"


def test_hint_returns_when_the_value_actually_varies(multi_task_page):
    """Negative control for the suppression above.

    `action` cannot serve as this control — it is a vector, and the uniform check
    is gated on `!isVector`. This needs a read-only *scalar* whose frames differ,
    which is a dataset carrying more than one task inside one episode.
    """
    page, key = multi_task_page
    card = page.query_selector('#inspector-body .feature-card[data-feature="task"]')
    assert card is not None, "task card must still render when the value varies"
    text = card.inner_text()
    assert "(frame" in text, (
        f"a varying value must say which frame is shown; suppression is too broad: {text!r}"
    )


def test_a_non_per_episode_string_would_render_as_one_full_width_band(page):
    """Premise for the two tests above.

    Flipping the flag on the client schema and re-rendering reproduces what the
    inherited flag used to produce, so the assertions above are known to be
    guarding a real difference rather than passing vacuously.
    """
    page, key = page
    page.evaluate(
        f"window.datasets[{key!r}].features_schema.task.is_per_episode = false;"
        f"window.FeatureEditing.onEpisodeSelected({key!r}, 0);"
    )
    page.wait_for_selector('.feature-row[data-feature="task"]', timeout=10_000)
    widths = page.evaluate(
        "Array.from(document.querySelectorAll('.row-track[data-feature=\"task\"] rect'))"
        ".map(r => r.getAttribute('width'))"
    )
    assert widths == ["100%"], f"expected one band spanning the episode, got {widths}"
