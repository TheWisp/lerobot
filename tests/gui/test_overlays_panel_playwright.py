# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Overlays panel in a real browser — the wiring unit tests cannot reach.

The defect this pins: the "Process dataset…" gate lived only at the end of
``renderObjects()``, but typing an object name updates the model and calls
``renderAction()`` WITHOUT re-rendering the rows. So a user who named an object
(with the Background already defaulting to Random) saw the button stay greyed
out, its tooltip telling them to do the thing they had just done — the feature
looked broken until they happened to click a treatment button. No dataset, GPU
or overlay worker is needed: the gate is pure panel state.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.requires_playwright

NAME_SEL = '#overlays-panel .overlays-obj-name[data-i="0"]'
PROC_SEL = "#overlays-panel .overlays-process"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def overlays_gui_server(monkeypatch):
    from lerobot.gui import server as gui_server_mod
    from lerobot.gui.api import datasets as datasets_api

    # Keep the page hermetic: no user-persisted datasets restored on load.
    monkeypatch.setattr(datasets_api, "_read_opened", lambda: [])
    monkeypatch.setattr(datasets_api, "_read_sources", lambda: [])

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
            if requests.get(f"{base_url}/api/overlays/models", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        server.should_exit = True
        pytest.fail("GUI server did not become ready within 15 seconds")

    yield base_url

    server.should_exit = True
    thread.join(timeout=10)


def test_the_data_panel_offers_no_write_and_no_treatment(overlays_gui_server):
    """Replaces a test for the "Apply to all episodes…" button this panel had
    grown. The design names two ways to add masks and neither is here: the panel
    is the live query, so it neither writes nor edits dataset metadata.
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(overlays_gui_server, wait_until="networkidle")
            page.evaluate(
                "(() => { const p = document.querySelector('#overlays-panel .overlays-picker');"
                " p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); })()"
            )
            page.wait_for_selector("#overlays-panel .overlays-objrow", timeout=5000)

            for sel, what in (
                (".overlays-process", "a dataset-wide apply"),
                ("#ovl-save-masks", "an episode-scoped save"),
                (".ds-treat-btn", "a treatment control"),
            ):
                n = page.eval_on_selector_all(f"#overlays-panel {sel}", "e => e.length")
                assert n == 0, f"the data panel still offers {what} ({n} found)"
        finally:
            browser.close()


def test_the_live_panel_does_offer_treatments(overlays_gui_server):
    """The Run tab's overlay has no save at all, so a treatment there is a
    rendering knob for the preview and nothing is written. The scope argument
    that took this control out of the DATA panel never applied to it, and
    removing it here was a regression.
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            _stub_worker_endpoints(page)
            page.goto(overlays_gui_server, wait_until="networkidle")
            _pick(page, "live", "sam3_track")
            # By count, not visibility: the run panel is rendered but not shown
            # until its tab is, and this is about what it renders.
            page.wait_for_function(
                "() => document.querySelectorAll('#overlays-panel-run .overlays-objrow').length > 0",
                timeout=5000,
            )

            btns = page.eval_on_selector_all("#overlays-panel-run .ds-treat-btn", "e => e.length")
            assert btns > 0, "the live panel offers no treatment control"
            assert (
                page.eval_on_selector_all('#overlays-panel-run .ds-treat[data-bg="1"]', "e => e.length") == 1
            ), "the live panel has no Background row"
            assert (
                page.eval_on_selector_all(
                    '#overlays-panel-run .ds-treat-btn[data-key="tint"] .ds-tint-chip', "e => e.length"
                )
                > 0
            ), "tint offers no colour swatch"

            # And it must stay a preview: choosing one writes nothing.
            posts = []
            page.on("request", lambda r: posts.append(r.url) if r.method == "POST" else None)
            page.eval_on_selector('#overlays-panel-run .ds-treat-btn[data-key="blur"]', "b => b.click()")
            page.wait_for_timeout(600)
            wrote = [u for u in posts if "/edits/" in u or "/process/episode-masks" in u]
            assert not wrote, f"choosing a treatment in the live panel wrote something: {wrote}"
        finally:
            browser.close()


def test_last_row_clears_in_place_instead_of_being_immortal(overlays_gui_server):
    """The x used to be offered only with 2+ rows, so the panel's only row could
    never be cleared with one click — inconsistently, since every OTHER row was.
    Now the x is always there: with several rows it removes, on the last row it
    clears in place (a text row needs an input to type into)."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(overlays_gui_server, wait_until="networkidle")
            page.evaluate(
                "(() => { const p = document.querySelector('#overlays-panel .overlays-picker');"
                " p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); })()"
            )
            page.wait_for_selector(NAME_SEL, timeout=5000)

            rm_sel = '#overlays-panel .overlays-obj-rm[data-i="0"]'
            assert page.eval_on_selector(rm_sel, "e => e.title") == "clear", (
                "the single row must offer x, labelled as a clear"
            )

            page.click(NAME_SEL)
            page.type(NAME_SEL, "robot arm", delay=20)
            page.click(rm_sel)
            page.wait_for_function(
                f"() => {{ const i = document.querySelector('{NAME_SEL}'); return i && i.value === ''; }}",
                timeout=5000,
            )
            # Still exactly one (empty) row — cleared, not deleted.
            assert page.eval_on_selector_all("#overlays-panel .overlays-obj-name", "els => els.length") == 1

            # With a second row, the x removes rather than clears.
            page.click("#overlays-panel .overlays-add-obj")
            page.wait_for_function(
                "() => document.querySelectorAll('#overlays-panel .overlays-obj-name').length === 2",
                timeout=5000,
            )
            assert page.eval_on_selector(rm_sel, "e => e.title") == "remove"
            page.click('#overlays-panel .overlays-obj-rm[data-i="1"]')
            page.wait_for_function(
                "() => document.querySelectorAll('#overlays-panel .overlays-obj-name').length === 1",
                timeout=5000,
            )
        finally:
            browser.close()


def test_overlay_config_is_scoped_per_dataset(overlays_gui_server):
    """Datasets differ where episodes within one do not, so the panel's config is
    remembered PER DATASET and swapped on switch (app.js calls refreshCameras
    exactly then). The concrete defect this pins: the auto-opened process preview
    used to inherit the source's treatments and re-apply them onto already-treated
    pixels — with scoping, a never-seen dataset gets a fresh, inert config."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(overlays_gui_server, wait_until="networkidle")
            # window.currentDataset is a READ-ONLY getter mirroring app.js's internal
            # state (sibling scripts must not assign it) — a plain assignment is
            # silently ignored, which cost this test a debugging session. It is
            # configurable, so the test impersonates app.js by redefining it.
            page.evaluate(
                "window.__setDs = (v) => Object.defineProperty(window, 'currentDataset',"
                " { value: v, writable: true, configurable: true })"
            )
            # Opening the first dataset fires refreshCameras too (app.js does this on
            # every dataset change) — that is when dsA becomes the scope owner.
            page.evaluate("window.__setDs('/tmp/dsA'); window.Overlays.refreshCameras()")
            page.evaluate(
                "(() => { const p = document.querySelector('#overlays-panel .overlays-picker');"
                " p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); })()"
            )
            page.wait_for_selector(NAME_SEL, timeout=5000)

            # Author dsA's config: named object, instances = Largest. Treatment is
            # no longer part of this panel's config -- it is dataset-scoped and
            # lives in the Inspector -- so the scoping is shown with what remains.
            page.click(NAME_SEL)
            page.type(NAME_SEL, "ring", delay=15)
            page.evaluate(
                "(() => document.querySelector('#overlays-panel .overlays-multi"
                ' .overlays-seg-btn[data-multi="0"]\').click())()'
            )

            def state():
                return page.evaluate(
                    """(() => ({
                        name: document.querySelector('#overlays-panel .overlays-obj-name[data-i="0"]').value,
                        multi: document.querySelector('#overlays-panel .overlays-multi'
                                + ' .overlays-seg-btn.sel').dataset.multi,
                    }))()"""
                )

            assert state() == {"name": "ring", "multi": "0"}

            # Switch to a never-seen dataset: fresh, inert config (the preview case).
            page.evaluate("window.__setDs('/tmp/dsB'); window.Overlays.refreshCameras()")
            page.wait_for_function(f"() => document.querySelector('{NAME_SEL}').value === ''", timeout=5000)
            assert state() == {"name": "", "multi": "1"}

            # Author dsB differently, then bounce back and forth: each keeps its own.
            page.click(NAME_SEL)
            page.type(NAME_SEL, "cube", delay=15)
            page.evaluate("window.__setDs('/tmp/dsA'); window.Overlays.refreshCameras()")
            page.wait_for_function(
                f"() => document.querySelector('{NAME_SEL}').value === 'ring'", timeout=5000
            )
            assert state() == {"name": "ring", "multi": "0"}
            page.evaluate("window.__setDs('/tmp/dsB'); window.Overlays.refreshCameras()")
            page.wait_for_function(
                f"() => document.querySelector('{NAME_SEL}').value === 'cube'", timeout=5000
            )
        finally:
            browser.close()


# Which controls each tab renders, in order, for each processing step. Data mode is
# offered only segmenters, so policy_saliency is unreachable there.
CONTROL_SURFACE = {
    ("data", ""): [],
    # No "process" on the data panel: the design names two ways to add masks --
    # Apply while playing, and the Inspector's dataset-wide filler -- and the
    # whole-dataset button this panel had grown was neither.
    ("data", "sam3_track"): ["objects", "instances", "resolution", "cameras"],
    ("live", ""): [],
    ("live", "sam3_track"): ["objects", "box_method", "instances", "resolution", "cameras"],
    ("live", "policy_saliency"): ["select:method", "select:style", "slider", "cameras"],
}
PANEL_ROOT = {"data": "overlays-panel", "live": "overlays-panel-run"}

# Read the rendered body back as an ordered list of control keys.
READ_SURFACE = """(id) => {
    const body = document.querySelector('#' + id + ' .overlays-model-body');
    const key = (el) => {
        if (el.classList.contains('overlays-objrows')) return 'objects';
        if (el.classList.contains('overlays-boxmethod')) return 'box_method';
        if (el.classList.contains('overlays-multi')) return 'instances';
        if (el.classList.contains('overlays-res')) return 'resolution';
        if (el.classList.contains('overlays-cameras')) return 'cameras';
        if (el.classList.contains('overlays-process')) return 'process';
        if (el.classList.contains('overlays-slider')) return 'slider';
        if (el.classList.contains('overlays-select')) return 'select:' + el.dataset.key;
        return null;
    };
    return [...body.children].map(key).filter(Boolean);
}"""


def _stub_worker_endpoints(page):
    """Nothing in this file may reach a real worker or the aux-GPU slot: picking a
    segmenter on the RUN tab is enough to launch one (a live segmenter needs no typed
    word — the tiles are its input)."""
    ok = lambda route: route.fulfill(status=200, content_type="application/json", body="{}")  # noqa: E731
    page.route("**/api/overlays/live/**", ok)
    page.route("**/api/overlays/data/**", ok)


def _pick(page, mode, model):
    page.evaluate(
        "([id, m]) => { const p = document.querySelector('#' + id + ' .overlays-picker');"
        " p.value = m; p.dispatchEvent(new Event('change', {bubbles: true})); }",
        [PANEL_ROOT[mode], model],
    )


def test_panel_renders_the_declared_control_surface_for_every_step(overlays_gui_server):
    """Enumerates every (tab, step) the picker can reach and pins which controls render,
    in order. Visibility used to be a set of inline ternaries inside one HTML blob, and
    the run-tab-only "box read by" picker shipped on the data tab because one of them was
    missing; each control now declares a single `when` rule and this table is the oracle
    for all of them at once."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            _stub_worker_endpoints(page)
            page.goto(overlays_gui_server, wait_until="networkidle")
            for (mode, model), expected in CONTROL_SURFACE.items():
                _pick(page, mode, model)
                page.wait_for_timeout(200)
                got = page.evaluate(READ_SURFACE, PANEL_ROOT[mode])
                assert got == expected, f"{mode}/{model or 'none'}: {got} != {expected}"
                if not model:  # the off state says so instead of rendering an empty body
                    assert "Pick a processing step" in page.eval_on_selector(
                        f"#{PANEL_ROOT[mode]} .overlays-hint", "e => e.textContent"
                    )
        finally:
            browser.close()


def test_box_method_repaints_in_place_and_keeps_a_half_typed_name(overlays_gui_server):
    """The picker deliberately repaints its own selection rather than re-rendering the
    body: a re-render would destroy a name being typed into an object row."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            _stub_worker_endpoints(page)
            page.goto(overlays_gui_server, wait_until="networkidle")
            page.click('.tab[data-tab="run"]')  # the run panel is hidden on the Data tab
            _pick(page, "live", "sam3_track")
            name_sel = '#overlays-panel-run .overlays-obj-name[data-i="0"]'
            page.wait_for_selector("#overlays-panel-run .overlays-boxmethod", timeout=5000)

            page.click(name_sel)
            page.type(name_sel, "green ri", delay=15)
            page.click('#overlays-panel-run .overlays-boxmethod [data-boxm="exemplar"]')
            page.wait_for_function(
                "() => document.querySelector('#overlays-panel-run .overlays-boxmethod"
                ' .overlays-seg-btn.sel\').dataset.boxm === "exemplar"',
                timeout=5000,
            )
            assert page.eval_on_selector(name_sel, "e => e.value") == "green ri"
        finally:
            browser.close()


def test_data_panel_offers_no_gesture_controls(overlays_gui_server):
    """Click-to-segment is run-tab only, so the data panel must not advertise it. Gating the
    gesture is not enough: the 'box read by' picker and the hint that says to click a tile
    are rendered from the panel template, and both shipped on the data tab still offering a
    choice that had no effect there."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(overlays_gui_server, wait_until="networkidle")
            page.evaluate(
                "(() => { const p = document.querySelector('#overlays-panel .overlays-picker');"
                " p.value = 'sam3_track'; p.dispatchEvent(new Event('change', {bubbles: true})); })()"
            )
            page.wait_for_selector("#overlays-panel .overlays-objrows", timeout=5000)

            assert page.query_selector("#overlays-panel .overlays-boxmethod") is None, (
                "the box-method picker only affects a dragged box, which this tab cannot make"
            )
            hint = page.eval_on_selector("#overlays-panel .overlays-hint", "e => e.textContent")
            assert "click" not in hint.lower(), f"the hint promises a gesture this tab lacks: {hint!r}"
        finally:
            browser.close()
