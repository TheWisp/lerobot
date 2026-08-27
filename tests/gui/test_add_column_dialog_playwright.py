# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Playwright coverage for the Add-column dialog's kind switching.

Driven through a real browser rather than asserted against the markup because
the defect this guards was purely a *rendering* one: every ``hidden`` toggle in
this dialog was inert, since

    .add-feature-dialog form > flag { display: block }

outranks the user-agent's ``[hidden] { display: none }`` (specificity 0,1,2
against 0,1,0). The JS was already setting ``.hidden`` correctly; the form
simply showed every field of every kind at once. Nothing short of computed
style catches that -- a DOM-level check on ``el.hidden`` passes while the
dialog is unusable.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import (  # noqa: E402
    Error as PlaywrightError,
    sync_playwright,
)

pytestmark = pytest.mark.requires_playwright

# Which rows each kind owns. Mirrors KIND_FIELDS in add_feature_dialog.js; the
# point of restating it is that a change to one without the other fails here.
ROWS = {
    "number": {"add-feature-dtype-row", "add-feature-shape-row", "add-feature-fill-row"},
    "text": {"add-feature-fill-row"},
    "flags": {"add-feature-flags-row"},
}
ALL_ROWS = set().union(*ROWS.values())


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def gui_page(tmp_path, monkeypatch):
    from lerobot.gui import server as gui_server_mod

    monkeypatch.setenv("LEROBOT_GUI_CONFIG_DIR", str(tmp_path / "config"))

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

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError as e:
            server.should_exit = True
            pytest.skip(f"chromium not available: {e}")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(base_url)
        page.wait_for_function("typeof switchTab === 'function'", timeout=10_000)
        # The dialog needs no open dataset to render; opening it directly keeps
        # this test about the form rather than about dataset loading.
        page.evaluate("document.getElementById('add-feature-dialog').showModal()")
        page.wait_for_selector("#add-feature-kind", timeout=10_000)
        yield page
        browser.close()

    server.should_exit = True
    thread.join(timeout=10)


def visible_rows(page) -> set[str]:
    """Rows actually painted, by computed style -- not by the `hidden` property."""
    return set(
        page.evaluate(
            """() => [...document.querySelectorAll('#add-feature-dialog form > label')]
                .filter(l => l.id && getComputedStyle(l).display !== 'none')
                .map(l => l.id)"""
        )
    )


def pick(page, kind: str) -> None:
    page.select_option("#add-feature-kind", kind)


def pretend_a_dataset_is_open(page) -> None:
    """Satisfy the submit handler's first guard without loading a dataset.

    ``window.currentDataset`` is a getter-only mirror of a module-scoped
    variable in app.js, so assigning to it is a silent no-op. The property is
    declared ``configurable``, so redefining it is the supported way to stand
    one in -- and these tests are about form validation, not dataset loading.
    """
    page.evaluate(
        """() => Object.defineProperty(window, 'currentDataset', {
            value: 'pretend/dataset', configurable: true,
        })"""
    )


@pytest.mark.parametrize("kind", sorted(ROWS))
def test_each_kind_shows_only_its_own_rows(gui_page, kind):
    pick(gui_page, kind)
    shown = visible_rows(gui_page) & ALL_ROWS
    assert shown == ROWS[kind]


def test_number_is_the_default_kind(gui_page):
    """Opening the dialog must not require a choice to reach a usable state."""
    assert visible_rows(gui_page) & ALL_ROWS == ROWS["number"]
    assert gui_page.input_value("#add-feature-kind") == "number"


def test_the_picker_reflects_the_selected_kind(gui_page):
    pick(gui_page, "flags")
    assert gui_page.input_value("#add-feature-kind") == "flags"


def test_every_kind_the_picker_offers_is_one_the_form_knows(gui_page):
    """A kind listed but not in KIND_FIELDS would show no fields at all, which
    reads as a broken dialog rather than as an unfinished one."""
    offered = gui_page.evaluate(
        """() => [...document.querySelectorAll('#add-feature-kind option')].map(o => o.value)"""
    )
    assert set(offered) == set(ROWS), offered


def test_switching_kinds_is_reversible(gui_page):
    """Switching away and back must restore the rows, not leave them hidden --
    the failure mode if visibility were applied once rather than recomputed."""
    before = visible_rows(gui_page) & ALL_ROWS
    pick(gui_page, "flags")
    pick(gui_page, "text")
    pick(gui_page, "number")
    assert visible_rows(gui_page) & ALL_ROWS == before


def test_the_flags_box_is_reachable_and_typable(gui_page):
    """A field that is displayed but zero-height or overlapped is still
    unusable, and Playwright's fill() refuses those."""
    pick(gui_page, "flags")
    gui_page.fill("#add-feature-form textarea[name='flags']", "blurry\nfumble")
    assert gui_page.input_value("#add-feature-form textarea[name='flags']") == "blurry\nfumble"


def test_an_empty_flag_list_is_refused_before_the_confirm(gui_page):
    """The operator should not be asked to confirm an irreversible rewrite that
    the server would then reject."""
    confirms = []
    gui_page.on("dialog", lambda d: (confirms.append(d.message), d.dismiss()))
    pick(gui_page, "flags")
    gui_page.fill("#add-feature-form input[name='name']", "quality")
    pretend_a_dataset_is_open(gui_page)
    gui_page.click("#add-feature-submit")
    assert confirms == [], "no confirm should be raised for an empty vocabulary"
    assert not gui_page.is_hidden("#add-feature-error")
    assert "at least one flag" in gui_page.inner_text("#add-feature-error")


def test_a_repeated_flag_is_refused_before_the_confirm(gui_page):
    confirms = []
    gui_page.on("dialog", lambda d: (confirms.append(d.message), d.dismiss()))
    pick(gui_page, "flags")
    gui_page.fill("#add-feature-form input[name='name']", "quality")
    gui_page.fill("#add-feature-form textarea[name='flags']", "blurry\nfumble\nblurry")
    pretend_a_dataset_is_open(gui_page)
    gui_page.click("#add-feature-submit")
    assert confirms == []
    assert "blurry" in gui_page.inner_text("#add-feature-error")
