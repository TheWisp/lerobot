# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The dataset tree must notice datasets that appear on disk.

The tree is a cache of a directory listing with nothing invalidating it:
``scanSource()`` fetches the listing only when a source is expanded, and
``renderSources()`` repaints that cache without re-fetching. So a dataset
created, renamed or removed outside the browser stayed invisible until the user
collapsed and re-expanded the source or reloaded the page — neither
discoverable, and the failure is quiet: the dataset looks like it was never
written.

Asserted against a real browser and a real directory, because the defect is
that a fetch does *not* happen; a test calling ``scanSource()`` directly would
pass with the bug fully present.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_dataset(source: Path, name: str, total_episodes: int = 1) -> Path:
    """A directory counts as a dataset when it holds meta/info.json."""
    root = source / name
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": total_episodes,
                "total_frames": total_episodes * 10,
                "fps": 30,
                "robot_type": "test",
            }
        )
    )
    return root


def _make_training_run(source: Path, name: str, step: int = 1000) -> Path:
    """A directory counts as a training run when it holds checkpoints/<step>/."""
    root = source / name / "checkpoints" / f"{step:06d}" / "pretrained_model"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"type": "act"}))
    return source / name


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The Data tab, with one source expanded and one dataset already in it."""
    from lerobot.gui import server as gui_server_mod
    from lerobot.gui.api import datasets as datasets_api

    source = tmp_path / "source"
    source.mkdir()
    _make_dataset(source, "already_there")
    collapsed_source = tmp_path / "collapsed_source"
    collapsed_source.mkdir()
    _make_dataset(collapsed_source, "basketball", total_episodes=7)

    # The tree always prepends a non-removable default source at
    # HF_LEROBOT_HOME. Left alone, every refresh in these tests walks the
    # developer's real dataset cache — 150 entries here — which makes the tests
    # slower, machine-dependent, and unable to assert anything about how many
    # scans a refresh issues.
    monkeypatch.setattr(datasets_api, "HF_LEROBOT_HOME", tmp_path / "empty_hf_home")
    (tmp_path / "empty_hf_home").mkdir()

    # Redirect the source registry: it lives in the real ~/.config otherwise,
    # and a test has no business writing the developer's configured sources.
    sources_file = tmp_path / "dataset_sources.json"
    sources_file.write_text(
        json.dumps(
            {
                "sources": [
                    {"path": str(source), "removable": True, "expanded": True},
                    {"path": str(collapsed_source), "removable": True, "expanded": False},
                ]
            }
        )
    )
    monkeypatch.setattr(datasets_api, "SOURCES_FILE", sources_file)

    # A model source too, so the model tab has something to go stale. It is the
    # tab with the worse staleness — its init is one-shot per page load — and
    # the one a user visits right after a training run finishes.
    from lerobot.gui.api import models as models_api

    model_source = tmp_path / "runs"
    model_source.mkdir()
    _make_training_run(model_source, "already_trained")
    model_sources_file = tmp_path / "model_sources.json"
    model_sources_file.write_text(
        json.dumps({"sources": [{"path": str(model_source), "removable": True, "expanded": True}]})
    )
    monkeypatch.setattr(models_api, "SOURCES_FILE", model_sources_file)

    port = _free_port()
    config = uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

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
        page = browser.new_page()
        page.goto(base_url)
        page.wait_for_selector("text=already_there", timeout=15_000)
        try:
            yield page, source, model_source
        finally:
            browser.close()
            server.should_exit = True


class TestTreeNoticesDatasetsWrittenOutsideTheBrowser:
    def test_a_new_dataset_is_invisible_until_something_rescans(self, tree):
        """The premise. Without this the next test could pass vacuously — if the
        tree happened to poll, it would prove nothing about the focus handler.
        """
        page, source, _ = tree
        _make_dataset(source, "written_while_you_were_away")
        page.wait_for_timeout(500)
        assert page.locator("text=written_while_you_were_away").count() == 0, (
            "the tree is expected to be a cache; if this fails the premise changed"
        )

    def test_returning_to_the_window_picks_it_up(self, tree):
        page, source, _ = tree
        _make_dataset(source, "written_while_you_were_away")
        page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
        page.wait_for_selector("text=written_while_you_were_away", timeout=10_000)

    def test_a_removed_dataset_disappears_too(self, tree):
        """Deletion is the same staleness in the other direction — a tree that
        only ever grows would still be lying after a dataset is cleaned up.
        """
        page, source, _ = tree
        for child in (source / "already_there" / "meta").iterdir():
            child.unlink()
        (source / "already_there" / "meta").rmdir()
        (source / "already_there").rmdir()
        page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
        page.wait_for_function("() => !document.body.innerText.includes('already_there')", timeout=10_000)

    def test_selecting_the_tab_rescans_it_too(self, tree):
        """Focus is not the only way a listing goes stale.

        Selecting a tab is the other, and for the model and robot tabs it is the
        more likely one: the event that made them stale — a training run
        finishing, a profile saved elsewhere — is what sends the user to that
        tab. Only the data tab used to refresh on selection.
        """
        page, source, _ = tree
        page.evaluate("switchTab('run')")
        _make_dataset(source, "written_while_on_another_tab")
        page.evaluate("switchTab('data')")
        page.wait_for_selector("text=written_while_on_another_tab", timeout=10_000)

    def test_the_model_tab_picks_up_a_finished_training_run(self, tree):
        """The case the fix exists for, on the tab with the worse staleness.

        `modelTabInit` is guarded by an `initialized` flag, so before this the
        model tree loaded once per page load and switching to it did nothing —
        a checkpoint written while the user was elsewhere stayed invisible until
        a reload, at exactly the moment they went looking for it.
        """
        page, _, model_source = tree
        page.evaluate("switchTab('model')")
        page.wait_for_selector("text=already_trained", timeout=15_000)

        page.evaluate("switchTab('data')")
        _make_training_run(model_source, "finished_while_you_were_away")

        page.evaluate("switchTab('model')")
        page.wait_for_selector("text=finished_while_you_were_away", timeout=15_000)

    def test_a_renamed_run_does_not_leave_a_stale_detail_card(self, tree):
        """The listing refreshing while the card does not is worse than neither.

        The sidebar shows the new name, the card shows the old one, and its
        Open Folder / Test on Robot buttons act on a path that no longer exists.
        Observed against the real model cache before this was fixed.
        """
        page, _, model_source = tree
        page.evaluate("switchTab('model')")
        page.wait_for_selector("text=already_trained", timeout=15_000)
        page.evaluate("path => selectModelRun(path)", str(model_source / "already_trained"))
        page.wait_for_timeout(800)
        assert page.locator("#model-detail").is_visible()

        (model_source / "already_trained").rename(model_source / "zz_run_after_rename")

        page.evaluate("switchTab('data')")
        page.evaluate("switchTab('model')")
        page.wait_for_selector("text=zz_run_after_rename", timeout=15_000)
        # Assert on what the user sees, not on innerText: a hidden node still
        # reports its text, so an innerText check passed while the card was
        # correctly hidden — and would equally have passed had it been left
        # visible with stale markup.
        page.wait_for_function(
            "() => document.getElementById('model-detail').style.display === 'none'"
            "&& document.getElementById('model-detail').innerHTML === ''",
            timeout=10_000,
        )

    def test_a_deleted_run_returns_the_panel_to_its_empty_state(self, tree):
        """Clearing has to restore the placeholder, not just hide the card —
        they are siblings, so hiding one without showing the other leaves a
        blank pane that looks broken rather than empty."""
        import shutil

        page, _, model_source = tree
        page.evaluate("switchTab('model')")
        page.wait_for_selector("text=already_trained", timeout=15_000)
        page.evaluate("path => selectModelRun(path)", str(model_source / "already_trained"))
        page.wait_for_timeout(800)

        shutil.rmtree(model_source / "already_trained")

        page.evaluate("switchTab('data')")
        page.evaluate("switchTab('model')")
        page.wait_for_function(
            "() => document.getElementById('model-empty')"
            "&& getComputedStyle(document.getElementById('model-empty')).display !== 'none'",
            timeout=10_000,
        )
        assert not page.locator("#model-detail").is_visible()

    def test_every_tab_that_lists_files_has_a_refresh(self, tree):
        """The table is the unification. If a fourth such tab is added without
        an entry, this fails rather than the tab silently going stale."""
        page, _, _ = tree
        listed = page.evaluate("Object.keys(REFRESH_BY_TAB).sort()")
        assert listed == ["data", "model", "robot"]
        for tab in listed:
            assert page.evaluate(f"typeof REFRESH_BY_TAB['{tab}'] === 'function'")

    def test_only_the_tab_on_screen_is_rescanned(self, tree):
        """The other trees are not rendered, so scanning for them is work
        nobody can see — and on a source holding hundreds of entries it is not
        free."""
        page, source, _ = tree
        page.evaluate("switchTab('run')")
        _make_dataset(source, "written_while_on_another_tab")
        seen: list[str] = []
        page.on("request", lambda r: seen.append(r.url))
        page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
        page.wait_for_timeout(300)
        assert not [u for u in seen if u.rstrip("/").endswith("/datasets")]

    def test_a_refresh_scans_each_expanded_source_once(self, tree):
        """Bounds the work by count, not by clock.

        A scan is a directory walk on the server — measured at 6ms over a
        150-dataset cache and 30ms over a 35-run output tree, per source. That
        is affordable once per source and not affordable N times, so the count
        is what needs pinning. A millisecond budget would fail on a slow disk
        and pass on a fast one while the code issued duplicate scans.

        Driven through the refresh entry point rather than a tab click, so the
        assertion is about the refresh and not about whatever else selecting a
        tab happens to trigger.
        """
        page, _, _ = tree
        scans: list[str] = []
        page.on(
            "request",
            lambda r: scans.append(r.url) if r.url.rstrip("/").endswith("/datasets") else None,
        )
        page.evaluate("window.refreshTabFromDisk('data')")
        page.wait_for_timeout(1200)
        expanded = page.evaluate("expandedSources.size")
        assert len(scans) == expanded, (
            f"{expanded} expanded source(s) should scan {expanded}x, got {len(scans)}"
        )

    def test_a_collapsed_source_is_not_scanned(self, tree):
        """The expensive part scales with sources, so the ones the user has
        folded away should cost nothing."""
        page, _, _ = tree
        page.evaluate("expandedSources.clear()")
        scans: list[str] = []
        page.on(
            "request",
            lambda r: scans.append(r.url) if r.url.rstrip("/").endswith("/datasets") else None,
        )
        page.evaluate("switchTab('run')")
        page.evaluate("switchTab('data')")
        page.wait_for_timeout(1200)
        assert scans == []

    def test_repeated_focus_events_do_not_hammer_the_endpoint(self, tree):
        """Focus fires on every alt-tab, and a listing over a source holding
        hundreds of datasets is not free."""
        page, _, _ = tree
        seen: list[str] = []
        page.on("request", lambda r: seen.append(r.url) if "/datasets" in r.url else None)
        for _ in range(10):
            page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
        page.wait_for_timeout(300)
        scans = [u for u in seen if u.rstrip("/").endswith("/datasets")]
        assert len(scans) <= 2, f"debounce did not hold: {len(scans)} scans for 10 events"


def test_dataset_browser_combines_search_favorites_and_sorting(tree):
    page, source, _ = tree
    alpha = _make_dataset(source, "alpha_dataset", total_episodes=3)
    _make_dataset(source, "beta_dataset", total_episodes=9)
    page.evaluate("window.refreshTabFromDisk('data')")
    page.wait_for_selector("text=beta_dataset", timeout=10_000)

    page.fill("#dataset-search", "source beta")
    assert page.locator(".source-dataset-name").all_inner_texts() == ["beta_dataset"]

    beta_row = page.locator(".source-dataset", has_text="beta_dataset")
    beta_row.locator(".source-dataset-favorite").click()
    assert page.locator("#dataset-favorite-count").inner_text() == "1"

    page.fill("#dataset-search", "")
    page.locator("#dataset-favorites-only").click()
    assert page.locator("#dataset-favorites-only").get_attribute("aria-pressed") == "true"
    assert page.locator(".source-dataset-name").all_inner_texts() == ["beta_dataset"]

    page.locator("#dataset-favorites-only").click()
    page.select_option("#dataset-sort", "episodes-desc")
    assert page.locator(".source-dataset-name").first.inner_text() == "beta_dataset"

    page.evaluate("root => rememberDatasetOpened(root)", str(alpha))
    page.select_option("#dataset-sort", "last-opened")
    assert page.locator(".source-dataset-name").first.inner_text() == "alpha_dataset"

    page.reload()
    page.wait_for_selector("text=alpha_dataset", timeout=15_000)
    assert page.locator("#dataset-sort").input_value() == "last-opened"
    assert page.locator("#dataset-favorite-count").inner_text() == "1"
    assert page.locator(".source-dataset-name").first.inner_text() == "alpha_dataset"
    assert page.locator(".source-dataset-favorite.active").count() == 1


def test_dataset_search_scans_and_reveals_a_collapsed_source(tree):
    page, source, _ = tree
    collapsed_source = source.parent / "collapsed_source"

    assert page.locator("text=basketball").count() == 0
    assert not page.evaluate("path => Object.hasOwn(sourceDatasets, path)", str(collapsed_source))

    page.fill("#dataset-search", "basketball")
    page.wait_for_selector("text=basketball", timeout=10_000)

    assert page.locator(".source-dataset-name").all_inner_texts() == ["basketball"]
    assert page.locator("#dataset-filter-summary").inner_text() == "1 of 2 datasets · 0 favorites"

    page.fill("#dataset-search", "")
    page.wait_for_function("() => !document.body.innerText.includes('basketball')", timeout=10_000)


def test_pending_copy_uses_the_same_filtered_count_denominator(tree):
    page, source, _ = tree
    destination = source / "copying_dataset"
    page.evaluate(
        "([destination, source]) => { pendingCopies.set(destination, {source}); renderSources(); }",
        [str(destination), str(source / "already_there")],
    )

    page.fill("#dataset-search", "source")
    header = page.locator(f'.source-folder-header[title="{source}"]')
    assert header.locator(".source-folder-count").inner_text() == "2/2"
