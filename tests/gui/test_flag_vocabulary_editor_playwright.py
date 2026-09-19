# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Naming a flag: in place, with the server's refusal where it can be acted on.

Driven through a real browser because the two defects here were only visible
that way. The vocabulary controls used ``window.prompt``, which cannot show the
row being renamed beside the rows it must not collide with, and has nowhere to
put an answer -- the server refuses a duplicate name with a 400, but by the time
that arrives the dialog is gone and the rejection went to a status line nobody
was looking at, so the edit read as accepted while nothing had changed on disk.

Every test here installs a ``dialog`` handler that fails on sight: a native
prompt reappearing is itself the regression, independent of what it then does.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest
import torch

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import (  # noqa: E402
    Error as PlaywrightError,
    sync_playwright,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

pytestmark = pytest.mark.requires_playwright

FLAGS = ["blurry", "fumble", "occluded"]
FRAMES = 40
FPS = 10
CARD = "#inspector-body .feature-card[data-feature='quality']"

FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    "quality": {"dtype": "int64", "shape": (1,), "names": None, "flags": FLAGS},
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_dataset(root: Path) -> None:
    """A synthetic episode where frames 12-19 carry `fumble`.

    Synthesised rather than pointed at anything real: these tests rename flags
    and write ``info.json``.
    """
    ds = LeRobotDataset.create(
        repo_id="test/flag-editor", fps=FPS, root=root, features=FEATURES, use_videos=False
    )
    for frame in range(FRAMES):
        value = float(frame)
        ds.add_frame(
            {
                "action": torch.tensor([value, value], dtype=torch.float32),
                "observation.state": torch.tensor([value, value], dtype=torch.float32),
                "quality": torch.tensor([0b10 if 12 <= frame < 20 else 0], dtype=torch.int64),
                "task": "flagging",
            }
        )
    ds.save_episode()
    ds.finalize()


@pytest.fixture
def gui(tmp_path, monkeypatch):
    """A browser on a real GUI with one flags-carrying dataset open.

    Yields ``(page, dataset_root)``. Function-scoped on purpose: each test
    rewrites the vocabulary on disk, so sharing one would make the order matter.
    """
    from lerobot.gui import server as gui_server_mod

    monkeypatch.setenv("LEROBOT_GUI_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "hf"))

    source = tmp_path / "source"
    dataset_root = source / "flag_demo"
    _build_dataset(dataset_root)

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
            if requests.get(f"{base_url}/", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        server.should_exit = True
        pytest.fail("GUI server did not become ready within 15s")

    requests.post(f"{base_url}/api/datasets/sources", json={"path": str(source)}, timeout=20)

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError as e:
            server.should_exit = True
            pytest.skip(f"chromium not available: {e}")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        # The regression, stated as a trap rather than as an assertion: any
        # native prompt or confirm anywhere in these flows fails the test.
        page.on("dialog", lambda d: pytest.fail(f"a native dialog appeared: {d.message!r}"))
        page.goto(base_url)
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.evaluate("path => openDataset(path)", str(dataset_root))
        page.wait_for_timeout(2000)
        page.evaluate(f"ds => selectEpisode(ds, 0, {FRAMES})", str(dataset_root))
        page.wait_for_function("window.currentDataset != null", timeout=30_000)
        page.wait_for_selector(CARD, timeout=20_000)
        _select_range(page, 8, 20)
        yield page, dataset_root
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


def _select_range(page, first: int, last: int) -> None:
    """Drag-select frames [first, last) on the quality row, with the real mouse.

    A precondition of everything below rather than scene-setting: the card's
    controls are disabled until a range exists, because applying a flag needs one.
    """
    track = page.locator("[data-feature='quality'][data-length]").first
    box = track.bounding_box()
    length = int(track.get_attribute("data-length"))

    def x(frame: int) -> float:
        return box["x"] + box["width"] * (frame + 0.5) / length

    y = box["y"] + box["height"] / 2
    page.mouse.move(x(first), y)
    page.mouse.down()
    page.mouse.move(x(last - 1), y, steps=8)
    page.mouse.up()
    page.wait_for_timeout(600)


def _open_rename(page, bit: int) -> None:
    row = page.locator(f"{CARD} .flag-row").nth(bit)
    row.hover()
    page.wait_for_timeout(150)
    row.locator(".flag-rename").click()
    page.wait_for_selector(f"{CARD} .flag-editor-input", timeout=5_000)


def _vocabulary_on_screen(page) -> list[str]:
    return page.evaluate(
        "() => [...document.querySelectorAll(\"[data-widget='flag']\")].map(b => b.dataset.flag)"
    )


def _vocabulary_on_disk(root: Path) -> list[str]:
    return json.loads((root / "meta" / "info.json").read_text())["features"]["quality"]["flags"]


def _error(page) -> str:
    return page.locator(f"{CARD} .flag-editor-error").inner_text()


# ── Renaming ──────────────────────────────────────────────────────────────────


def test_renaming_edits_the_row_in_place_seeded_with_the_current_name(gui):
    page, _ = gui
    _open_rename(page, bit=1)
    assert page.locator(f"{CARD} .flag-editor-input").input_value() == "fumble"
    # The row being edited is still one of the rows it must be unique against.
    assert page.locator(f"{CARD} .flag-row").count() == len(FLAGS)


def test_a_duplicate_rename_is_reported_and_nothing_is_written(gui):
    """The defect: the 400 went to a status line and the prompt was already gone,
    so a refused rename was indistinguishable from a successful one."""
    page, root = gui
    _open_rename(page, bit=1)
    page.locator(f"{CARD} .flag-editor-input").fill("blurry")
    page.locator(f"{CARD} .flag-editor-save").click()
    page.wait_for_timeout(800)

    assert "already has a flag" in _error(page)
    assert _vocabulary_on_disk(root) == FLAGS, "a refused rename must not reach disk"
    # The edited row is an input while the editor is open, so the checkboxes are
    # the *other* flags; what matters is that none of them moved.
    assert _vocabulary_on_screen(page) == ["blurry", "occluded"]
    assert page.locator(f"{CARD} .flag-row").count() == len(FLAGS)


def test_a_refused_rename_stays_open_holding_what_was_typed(gui):
    """Correctable, not lost: the whole reason the editor outlives the error."""
    page, root = gui
    _open_rename(page, bit=1)
    page.locator(f"{CARD} .flag-editor-input").fill("blurry")
    page.locator(f"{CARD} .flag-editor-save").click()
    page.wait_for_timeout(800)
    assert page.locator(f"{CARD} .flag-editor-input").input_value() == "blurry"

    page.locator(f"{CARD} .flag-editor-input").fill("slipped")
    page.locator(f"{CARD} .flag-editor-save").click()
    page.wait_for_timeout(1000)
    assert page.locator(f"{CARD} .flag-editor-input").count() == 0, "success must close the editor"
    assert _vocabulary_on_screen(page) == ["blurry", "slipped", "occluded"]
    assert _vocabulary_on_disk(root) == ["blurry", "slipped", "occluded"]


def test_a_rename_moves_no_bit(gui):
    """The summary count is the outside view of it: the same frames carry the
    flag afterwards, under the new name."""
    page, _ = gui
    before = page.locator(f"{CARD} .flag-summary, {CARD}").inner_text()
    assert "8/12" in before, before
    _open_rename(page, bit=1)
    page.locator(f"{CARD} .flag-editor-input").fill("slipped")
    page.locator(f"{CARD} .flag-editor-save").click()
    page.wait_for_timeout(1000)
    after = page.locator(CARD).inner_text()
    assert "slipped 8/12" in after, after


def test_escape_discards_the_rename(gui):
    page, root = gui
    _open_rename(page, bit=0)
    page.locator(f"{CARD} .flag-editor-input").fill("thrown away")
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)
    assert page.locator(f"{CARD} .flag-editor-input").count() == 0
    assert _vocabulary_on_screen(page) == FLAGS
    assert _vocabulary_on_disk(root) == FLAGS


# ── Adding ────────────────────────────────────────────────────────────────────


def test_adding_opens_a_blank_row_where_the_flag_will_be(gui):
    page, _ = gui
    page.locator(f"{CARD} .flag-add").click()
    page.wait_for_selector(f"{CARD} .flag-row-new", timeout=5_000)
    assert page.locator(f"{CARD} .flag-editor-input").input_value() == ""
    # A second blank row would let two names be typed against one bit.
    assert page.locator(f"{CARD} .flag-add").is_disabled()


def test_an_empty_name_is_refused_without_a_round_trip(gui):
    page, root = gui
    page.locator(f"{CARD} .flag-add").click()
    page.wait_for_selector(f"{CARD} .flag-row-new", timeout=5_000)
    page.locator(f"{CARD} .flag-editor-save").click()
    page.wait_for_timeout(400)
    assert "needs a name" in _error(page)
    assert _vocabulary_on_disk(root) == FLAGS


def test_a_duplicate_on_add_is_reported_then_correctable(gui):
    page, root = gui
    page.locator(f"{CARD} .flag-add").click()
    page.wait_for_selector(f"{CARD} .flag-row-new", timeout=5_000)
    page.locator(f"{CARD} .flag-editor-input").fill("blurry")
    page.locator(f"{CARD} .flag-editor-save").click()
    page.wait_for_timeout(800)
    assert "already has a flag" in _error(page)
    assert _vocabulary_on_disk(root) == FLAGS

    page.locator(f"{CARD} .flag-editor-input").fill("dropped")
    page.locator(f"{CARD} .flag-editor-save").click()
    page.wait_for_timeout(1000)
    assert _vocabulary_on_screen(page) == [*FLAGS, "dropped"]
    assert _vocabulary_on_disk(root) == [*FLAGS, "dropped"]


def test_cancelling_an_add_removes_the_row_and_re_enables_the_button(gui):
    page, root = gui
    page.locator(f"{CARD} .flag-add").click()
    page.wait_for_selector(f"{CARD} .flag-row-new", timeout=5_000)
    page.locator(f"{CARD} .flag-editor-cancel").click()
    page.wait_for_timeout(400)
    assert page.locator(f"{CARD} .flag-row-new").count() == 0
    assert not page.locator(f"{CARD} .flag-add").is_disabled()
    assert _vocabulary_on_disk(root) == FLAGS


# ── The timeline row ──────────────────────────────────────────────────────────


def test_a_flags_row_stays_close_to_the_height_of_an_ordinary_row(gui):
    """Lanes are sized to their 9px names, not to a comfortable band.

    A flags column is one column and should read as one row. Sized generously it
    stops doing that: three flags at the previous 18px-per-lane made this row
    74px against 36px for every row above it, so the column with the least data
    in it dominated the timeline.
    """
    page, _ = gui
    heights = page.evaluate(
        """() => Object.fromEntries([...document.querySelectorAll('.feature-row')].map(r => [
            r.querySelector('.row-label')?.innerText.split('\\n')[0] || '?',
            Math.round(r.getBoundingClientRect().height),
        ]))"""
    )
    ordinary = heights.get("action")
    assert ordinary, heights
    assert heights["quality"] <= ordinary + 3 * 4, (
        f"a {len(FLAGS)}-flag row is {heights['quality']}px against {ordinary}px ordinary: {heights}"
    )


def test_every_lane_name_is_readable_at_the_compacted_pitch(gui):
    """The other half of compactness: a pitch too tight for the names is not a
    saving, and nothing else in the row says which lane is which.

    Asserted on the *spacing between* lane names rather than on each name's own
    box. The box is deliberately smaller than the line it renders -- an
    unclipped overflow -- so its height says nothing about whether the glyphs
    are legible. What decides that is whether the next lane's text starts below
    this one's, which is the pitch.
    """
    page, _ = gui
    lanes = page.evaluate(
        """() => {
            const row = document.querySelector('.feature-row.flags-row');
            const track = row.querySelector('.row-track').getBoundingClientRect();
            return [...row.querySelectorAll('.row-flag-name')].map(el => {
                const r = el.getBoundingClientRect();
                return {
                    text: el.innerText.trim(),
                    top: r.top - track.top,
                    font: parseFloat(getComputedStyle(el).fontSize),
                    line: parseFloat(getComputedStyle(el).lineHeight),
                    inside: r.top >= track.top - 0.5 && r.bottom <= track.bottom + 0.5,
                };
            });
        }"""
    )
    assert [lane["text"] for lane in lanes] == FLAGS
    for lane in lanes:
        assert lane["inside"], f"{lane['text']} escapes the track: {lane}"
        assert lane["font"] >= 9, f"{lane['text']} is set below the 9px floor: {lane}"
        # The lane box is deliberately shorter than the line it renders, so the
        # line box is the only thing left that can crush the glyphs.
        assert lane["line"] >= lane["font"], f"{lane['text']} has a line box under its font: {lane}"
    pitches = [b["top"] - a["top"] for a, b in zip(lanes, lanes[1:], strict=False)]
    assert all(p >= lanes[0]["font"] for p in pitches), (
        f"lane names are closer together than their own glyphs are tall: {pitches}"
    )
