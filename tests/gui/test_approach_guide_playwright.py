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


def test_the_editor_marks_a_pregrasp_and_a_grasp_end_and_saves_them(gui_page):
    import numpy as np

    from lerobot.gui.api import pregrasp
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    samples = []
    for i in range(60):
        q = dict.fromkeys(MOTOR_NAMES, 0.0)
        q["shoulder_pan"], q["gripper"] = i * 0.5, (60.0 if i < 40 else 85.0)
        samples.append({"t": i / 30.0, "obs": dict(q), "cmd": dict(q)})
    demo = pregrasp._demo_from_samples("editor", "gamepad", samples, [], lambda q: np.eye(4), t0=1000.0)
    demo.intr = {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0, "width": 848, "height": 480}
    demo.frames = [(1000.0 + k / 10.0, np.full((48, 84, 3), 90 + k, np.uint8)) for k in range(20)]
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'demo')")
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="demo"]')
        page.wait_for_function("de.curve && de.curve.n === 60", timeout=10_000)

        def scrub(i: int) -> None:
            page.locator("#de-slider").fill(str(i))
            page.locator("#de-slider").dispatch_event("input")

        scrub(20)
        page.click("text=Add pre-grasp here")
        scrub(10)
        page.click("text=Set grasp end here")  # before the pre-grasp: refused with a reason, nothing added
        assert "after the last pre-grasp" in page.locator("#de-status").inner_text()
        scrub(45)
        page.click("text=Set grasp end here")
        page.click("#demo-editor >> text=Save")
        page.wait_for_function(
            "document.getElementById('de-status').textContent.startsWith('saved')", timeout=10_000
        )
        with pregrasp._state.lock:
            kps = list(pregrasp._state.demo.keypoints)
        assert kps == [{"t": demo.t[20], "kind": "pregrasp"}, {"t": demo.t[45], "kind": "grasp_end"}]
        rows = page.locator("#de-list").inner_text()
        assert "straight line from wherever the arm is" in rows and "replayed exactly as shown" in rows
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None
