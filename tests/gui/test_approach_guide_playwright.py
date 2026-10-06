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
    demo.taught = True  # recorded after a teach: unnamed marks follow that object
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


def test_the_editor_designates_an_object_by_a_click_and_shows_it_tracked(gui_page, tmp_path):
    import io
    import json
    import time

    import numpy as np
    import requests

    from lerobot.gui.api import pregrasp
    from tests.gui.test_stream_objects import INTR, write_stream

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    t0, n = time.time(), 30
    rec = write_stream(tmp_path / "recording", 15, t0=t0, hz=15.0)
    demo = pregrasp._Demo(
        name="stack",
        concept="demo",
        fps=30.0,
        t=np.arange(n) / 30.0,
        tips=np.tile(np.eye(4), (n, 1, 1)),
        grippers=np.zeros(n),
        q_obs=np.zeros((n, 7)),
        q_cmd=np.zeros((n, 7)),
        deltas=np.tile(np.eye(4), (n, 1, 1)),
        seen=np.zeros(n, dtype=bool),
        delta0=np.eye(4),
        t0=t0,
        intr=dict(INTR),
        recording=str(rec),
    )

    class FakeProc:
        def poll(self):
            return None

        def terminate(self):
            pass

    pregrasp._state.worker.proc = FakeProc()
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
        pregrasp._state.worker.pending.clear()
        pregrasp._state.worker.jobs.clear()
    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'demo')")
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="demo"]')
        page.wait_for_function(
            "de.curve && de.curve.n === 30 && document.getElementById('de-frame').naturalWidth === 848",
            timeout=10_000,
        )
        box = page.locator("#de-frame").bounding_box()
        page.locator("#de-frame").click(position={"x": box["width"] / 2, "y": box["height"] / 2})
        page.wait_for_function(
            "document.getElementById('de-obj-list').textContent.includes('tracking')", timeout=10_000
        )
        base = page.url.split("#")[0].rstrip("/")
        job = requests.get(base + "/api/pregrasp/worker/job", params={"wait": 0}, timeout=5).json()
        assert job["kind"] == "stream_object" and job["name"] == "object_1", "a click alone designates"
        assert abs(job["click"][0] - 424) <= 2 and abs(job["click"][1] - 240) <= 2, (
            "the click lands on the stream's own pixels"
        )
        buf = io.BytesIO()
        np.savez(
            buf,
            meta=json.dumps({"ok": True, "frames": 15}),
            deltas=np.tile(np.eye(4), (15, 1, 1)),
            seen=np.ones(15, dtype=bool),
            masks=np.zeros((15, 120, 212), dtype=bool),
            mask=np.zeros((480, 848), dtype=bool),
        )
        requests.post(
            base + "/api/pregrasp/worker/result", params={"id": job["id"]}, data=buf.getvalue(), timeout=5
        )
        page.wait_for_function(
            "document.getElementById('de-obj-list').textContent.includes('seen in 100%')", timeout=10_000
        )
        page.fill("#de-obj-name", "cube")
        page.locator("#de-frame").click(position={"x": box["width"] / 4, "y": box["height"] / 4})
        page.wait_for_function(
            "document.getElementById('de-obj-list').textContent.includes('cube')", timeout=10_000
        )
        job = requests.get(base + "/api/pregrasp/worker/job", params={"wait": 0}, timeout=5).json()
        assert job["kind"] == "stream_object" and job["name"] == "cube", (
            "a name typed before the click names it"
        )
        assert errors == [], f"the page threw: {errors}"
    finally:
        pregrasp._state.worker.proc = None
        with pregrasp._state.lock:
            pregrasp._state.demo = None
            pregrasp._state.worker.pending.clear()
            pregrasp._state.worker.jobs.clear()


def test_marks_on_a_demo_without_a_teach_follow_the_clicked_object(gui_page, tmp_path):
    """Marks saved unnamed on a demo recorded without a teach sent the act to "teach first"."""
    import pathlib
    import time

    from lerobot.gui.api import pregrasp
    from tests.gui.test_stream_objects import _demo_with_object

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )

    def camera_and_worker_up(route):
        # The guided row reaches the demo's steps only with a live camera and a ready worker.
        resp = route.fetch()
        body = resp.json()
        body["camera_live"] = True
        body["worker"] = {**body.get("worker", {}), "running": True, "ready": True}
        route.fulfill(response=resp, json=body)

    page.route("**/api/pregrasp/state", camera_and_worker_up)
    demo = _demo_with_object(tmp_path, time.time())
    demo.root = str(tmp_path / "demos" / "bound")
    pathlib.Path(demo.root).mkdir(parents=True)
    demo.keypoints = [
        {"t": float(demo.t[6]), "kind": "pregrasp"},
        {"t": float(demo.t[20]), "kind": "grasp_end"},
    ]
    with pregrasp._state.lock:
        pregrasp._state.demo, pregrasp._state.teach = demo, None
    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'demo')")
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.wait_for_function(
            "document.getElementById('ap-guide-text').textContent.includes('do not say which object')",
            timeout=10_000,
        )
        page.click('button.ap-subtab[data-sub="demo"]')
        page.wait_for_function(
            "document.getElementById('de-status').textContent.includes('now follow gamepad')", timeout=10_000
        )
        assert not page.locator("#de-marks-for-row").is_visible(), "one object leaves nothing to choose"
        page.click("#demo-editor >> text=Save")
        page.wait_for_function(
            "document.getElementById('de-status').textContent.startsWith('saved')", timeout=10_000
        )
        with pregrasp._state.lock:
            kps = list(pregrasp._state.demo.keypoints)
        assert [k.get("object") for k in kps] == ["gamepad", "gamepad"]
        page.wait_for_function(
            "document.getElementById('ap-guide-text').textContent.includes('click gamepad in the camera view')",
            timeout=10_000,
        )
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


@pytest.mark.parametrize("designated", [True, False])
def test_a_lost_track_holds_back_only_the_act_that_depends_on_it(gui_page, tmp_path, designated):
    """The row hid Act whenever the tracker had lost the object, even one the gripper had just covered. A designated
    object is found again at Act wherever it was last seen, so its lost track offers Act; an object taught before
    the demo depends on the track that began at its teach, so a lost one still asks for the object back."""
    import pathlib
    import time

    from lerobot.gui.api import pregrasp
    from tests.gui.test_stream_objects import _demo_with_object

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    ref = {"object": "gamepad", "ok": True, "inliers": 80, "turn_deg": 5.0, "reason": ""}

    def ready_but_lost(route):
        resp = route.fetch()
        body = resp.json()
        body["camera_live"] = True
        body["worker"] = {**body.get("worker", {}), "running": True, "ready": True}
        body["arm_connected"] = True
        body["teach"] = {"concept": "gamepad", "mode": "features", "ref": ref if designated else None}
        body["track"] = {**body.get("track", {}), "on": True, "fps": 5.0, "last": {"state": "lost"}}
        route.fulfill(response=resp, json=body)

    page.route("**/api/pregrasp/state", ready_but_lost)
    page.route(
        "**/api/jog/state",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body='{"connected": true, "mode": "cartesian"}'
        ),
    )
    demo = _demo_with_object(tmp_path, time.time())
    demo.root = str(tmp_path / "demos" / "bound")
    pathlib.Path(demo.root).mkdir(parents=True)
    named = {"object": "gamepad"} if designated else {}
    demo.keypoints = [
        {"t": float(demo.t[6]), "kind": "pregrasp", **named},
        {"t": float(demo.t[20]), "kind": "grasp_end", **named},
    ]
    demo.taught = not designated
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
    try:
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        step = "Act" if designated else "Track"
        page.wait_for_function(
            f"document.getElementById('ap-guide-step').textContent === '{step}'", timeout=10_000
        )
        text = page.locator("#ap-guide-text").inner_text()
        if designated:
            assert "Act finds it again where it was last seen" in text, text
            assert page.locator("#ap-guide button:visible", has_text="Act").count() == 1
        else:
            assert 'the tracker lost "gamepad"' in text, text
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_the_details_say_what_found_the_object_and_what_tracks_it(gui_page):
    """After a find the details said "taught by SAM3 + DINO … start tracking, then record a demo in Teach or load one":
    tracking starts by itself, the demo was loaded, and the pose then came from Point2Pose, not SAM3 + DINO."""
    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    teach = {
        "at": "23:48:02",
        "mode": "features",
        "concept": "object_1",
        "n_points": 140,
        "radius_mm": 40.0,
        "shape_class": "box",
        "face_planarity": 0.62,
        "face_usable": True,
        "ref": {"object": "object_1", "ok": True, "inliers": 115, "turn_deg": 12.0, "reason": ""},
    }
    tracked = {
        "at": "23:48:05",
        "ok": True,
        "mode": "features",
        "algo": "p2p",
        "n_inliers": 40,
        "n_matches": 42,
        "rms_m": 0.002,
        "scale": 1.0,
        "axis_source": "fit",
        "motion": {"rotation_deg": 3.0},
    }

    live = {"test": None}  # a find clears the live pose; the tracker's first view brings one

    def found(route):
        resp = route.fetch()
        body = resp.json()
        body.update(teach_pending=False, teach=teach, test=live["test"])
        route.fulfill(response=resp, json=body)

    page.route("**/api/pregrasp/state", found)
    page.reload()
    page.wait_for_function("typeof pgState === 'function'", timeout=15_000)
    page.evaluate("pgUI.awaiting = true; pgState()")
    page.wait_for_function(
        "document.getElementById('pg-status').textContent.includes('object_1')", timeout=10_000
    )
    status = page.locator("#pg-status").inner_text()
    assert (
        "SAM3 cut it out where you clicked" in status and "Point2Pose tracks it from this frame" in status
    ), status
    assert "start tracking" not in status and "record a demo" not in status, status
    live["test"] = tracked
    page.evaluate("pgState()")
    page.wait_for_function(
        "document.getElementById('pg-info').textContent.includes('23:48:05')", timeout=10_000
    )
    info = page.locator("#pg-info").inner_text()
    assert "tracked 23:48:05 by Point2Pose" in info, info
    assert errors == [], f"the page threw: {errors}"


def test_the_timeline_holds_still_while_its_label_changes(gui_page):
    """The time label grew by " · object hidden" on hidden samples and with longer sample numbers, and the slider beside
    it shrank to make room: during playback the whole timeline jumped back and forth."""
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
    for i in range(120):
        q = dict.fromkeys(MOTOR_NAMES, 0.0)
        samples.append({"t": i / 30.0, "obs": dict(q), "cmd": dict(q)})
    demo = pregrasp._demo_from_samples("timeline", "gamepad", samples, [], lambda q: np.eye(4), t0=1000.0)
    demo.taught = True
    demo.seen = np.arange(120) % 20 < 10  # seen for ten samples, hidden for the next ten
    demo.frames = [(1000.0 + k / 10.0, np.full((48, 84, 3), 90 + k, np.uint8)) for k in range(40)]
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'demo')")
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="demo"]')
        page.wait_for_function("de.curve !== null && de.curve.n === 120", timeout=10_000)
        widths, labels = {}, {}
        for i in (0, 5, 15, 99, 105, 115):  # seen and hidden, one to three digits
            page.evaluate(f"deSeek({i}, true)")
            widths[i] = page.evaluate("document.getElementById('de-slider').getBoundingClientRect().width")
            labels[i] = page.locator("#de-time").inner_text()
        assert len(set(widths.values())) == 1, f"the slider changed width with its label: {widths} {labels}"
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_the_fingertip_path_appears_when_the_arm_connects(gui_page):
    """The path is drawn through the connected arm's camera calibration. A demo opened while no arm was connected
    came without it, and the editor never asked again: the operator lost the path until reloading the page."""
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
    arm = {"on": False}

    def state(route):
        resp = route.fetch()
        body = resp.json()
        body["arm_connected"] = arm["on"]
        route.fulfill(response=resp, json=body)

    def curve(route):
        resp = route.fetch()
        body = resp.json()
        body["uv"] = [[100 + 5 * i, 300 - 2 * i] for i in range(body["n"])] if arm["on"] else None
        route.fulfill(response=resp, json=body)

    page.route("**/api/pregrasp/state", state)
    page.route("**/api/pregrasp/demo/curve", curve)
    samples = []
    for i in range(60):
        q = dict.fromkeys(MOTOR_NAMES, 0.0)
        samples.append({"t": i / 30.0, "obs": dict(q), "cmd": dict(q)})
    demo = pregrasp._demo_from_samples("path", "gamepad", samples, [], lambda q: np.eye(4), t0=1000.0)
    demo.taught = True
    demo.seen = np.ones(60, dtype=bool)  # no "object hidden" badge to paint
    demo.intr = {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0, "width": 848, "height": 480}
    demo.frames = [(1000.0 + k / 10.0, np.full((48, 84, 3), 90 + k, np.uint8)) for k in range(20)]
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
    painted = (
        "(() => { const cv = document.getElementById('de-over'); if (!cv.width) return 0;"
        " const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;"
        " let n = 0; for (let k = 3; k < d.length; k += 4) if (d[k] > 0) n++; return n; })()"
    )
    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'demo')")
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="demo"]')
        page.wait_for_function("de.curve !== null && de.curve.n === 60", timeout=10_000)
        page.evaluate("deSeek(30, true)")
        assert page.evaluate("de.curve.uv") is None and page.evaluate(painted) == 0, "no arm, no path"
        arm["on"] = True
        page.wait_for_function("de.curve.uv !== null", timeout=10_000)
        page.wait_for_function(f"{painted} > 0", timeout=10_000)
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None
