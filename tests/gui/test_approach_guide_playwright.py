"""The Approach tab's guided row is the entry point of the teach, demo and act flow.

A remembered "show details" fold once killed it. The start-up code restored the
fold, the restore reached the demo editor's state before the script had declared
it, and the throw ended the script: the row stayed at its placeholder in every
browser that had opened the details before, while a fresh browser worked.
"""

from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError  # noqa: E402

pytestmark = pytest.mark.requires_playwright

PLACEHOLDER = "connecting to the server"


@pytest.mark.parametrize("details", ["0", "1"])
def test_the_guided_row_starts_whatever_fold_the_browser_remembers(gui_page, details):
    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    # The camera listing enumerates RealSense devices: on a rig it would reach a camera another server streams from.
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    page.evaluate(f"localStorage.setItem('ap-details', '{details}')")
    page.reload()
    page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
    page.click('button[data-tab="approach"]')
    try:
        page.wait_for_function(
            f"!document.getElementById('ap-guide-text').textContent.includes('{PLACEHOLDER}')",
            timeout=10_000,
        )
    except PlaywrightTimeoutError:
        pass
    assert errors == [], f"the page threw while starting: {errors}"
    assert PLACEHOLDER not in page.locator("#ap-guide-text").inner_text(), "the guided row never started"
    # The test server has no camera: the first step asks for one, and offers exactly one button.
    assert page.locator("#ap-guide-step").inner_text() == "Camera"
    assert page.locator("#ap-guide button:visible").count() == 1
    assert page.locator("#ap-subtabs").is_visible() == (details == "1"), "the remembered fold is restored"
