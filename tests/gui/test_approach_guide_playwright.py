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
    if not designated:
        demo.objects.clear()  # recorded after a teach and nothing clicked on it: the marks can only follow the teach
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
            assert text == "ready", text
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


def test_the_frame_the_act_reads_is_named_in_a_corner_badge(gui_page):
    """The label naming the frame the act reads the object's pose from was drawn beside the object, where the marks'
    labels covered it; the timeline's colours had no key."""
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

    def curve(route):
        resp = route.fetch()
        body = resp.json()
        body["pose_t"] = body["t"][30]
        route.fulfill(response=resp, json=body)

    page.route("**/api/pregrasp/demo/curve", curve)
    samples = []
    for i in range(60):
        q = dict.fromkeys(MOTOR_NAMES, 0.0)
        samples.append({"t": i / 30.0, "obs": dict(q), "cmd": dict(q)})
    demo = pregrasp._demo_from_samples("badge", "gamepad", samples, [], lambda q: np.eye(4), t0=1000.0)
    demo.taught = True
    demo.seen = np.ones(60, dtype=bool)
    demo.frames = [(1000.0 + k / 10.0, np.full((48, 84, 3), 90, np.uint8)) for k in range(20)]
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
    pink = (
        "(() => { const cv = document.getElementById('de-over'); if (!cv.width) return 0;"
        " const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data; let n = 0;"
        " for (let k = 0; k < d.length; k += 4) if (d[k] > 200 && d[k + 2] > 200 && d[k + 1] < 150 && d[k + 3] > 0) n++;"
        " return n; })()"
    )
    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'demo')")
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="demo"]')
        page.wait_for_function("de.curve !== null && de.curve.pose_t !== null", timeout=10_000)
        page.evaluate("deSeek(30, true)")
        page.wait_for_function(f"{pink} > 0", timeout=10_000)
        page.evaluate("deSeek(10, true)")
        page.wait_for_function(f"{pink} === 0", timeout=10_000)
        legend = page.locator("#de-legend").inner_text()
        assert (
            "pre-grasp" in legend
            and "pose: the act reads the object's pose here" in legend
            and "object hidden" in legend
        )
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


@pytest.mark.parametrize("inliers", [32, 310])
def test_the_guided_row_says_whether_the_find_is_weak_or_strong(gui_page, tmp_path, inliers):
    """A find matching 32 of the gamepad's 400 demo points put the grasp 20 degrees off; the row said only that finds
    are reliable up to about 30 degrees. It now says which this one is, and what to do about a weak one."""
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
    strong = inliers >= 60
    ref = {
        "object": "gamepad",
        "ok": True,
        "inliers": inliers,
        "turn_deg": 12.0,
        "reason": "",
        "card_points": 400,
        "strong": strong,
        "share": inliers / 400,
    }

    def ready(route):
        resp = route.fetch()
        body = resp.json()
        body["camera_live"] = True
        body["worker"] = {**body.get("worker", {}), "running": True, "ready": True}
        body["arm_connected"] = True
        body["teach"] = {"concept": "gamepad", "mode": "features", "ref": ref}
        body["track"] = {**body.get("track", {}), "on": True, "fps": 5.0, "last": {"state": "tracking"}}
        route.fulfill(response=resp, json=body)

    page.route("**/api/pregrasp/state", ready)
    page.route(
        "**/api/jog/state",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body='{"connected": true, "mode": "cartesian"}'
        ),
    )
    demo = _demo_with_object(tmp_path, time.time())
    demo.root = str(tmp_path / "demos" / "bound")
    pathlib.Path(demo.root).mkdir(parents=True)
    demo.keypoints = [
        {"t": float(demo.t[6]), "kind": "pregrasp", "object": "gamepad"},
        {"t": float(demo.t[20]), "kind": "grasp_end", "object": "gamepad"},
    ]
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
    try:
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.wait_for_function(
            "document.getElementById('ap-guide-step').textContent === 'Act'", timeout=10_000
        )
        text = page.locator("#ap-guide-text").inner_text()
        if strong:
            assert text == "ready", text
        else:
            assert text.startswith("gamepad: a weak find (32 of 400 points)"), text
            assert "turn it closer to how it lay in the demo" in text, text
        assert "reliable up to about 30" not in text, text
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_the_editor_binds_the_grasp_to_the_picked_object_and_the_place_to_the_one_it_goes_onto(
    gui_page, tmp_path
):
    """The marks' object was one choice for the whole demo. A place follows another object than the grasp: the
    editor chooses each stage's object on its own, and the place's list leaves out the object picked."""
    import time

    from lerobot.gui.api import pregrasp
    from tests.gui.test_stream_objects import _two_object_demo

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    demo, _box = _two_object_demo(tmp_path, time.time())
    with pregrasp._state.lock:
        pregrasp._state.demo, pregrasp._state.teach = demo, None
    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'demo')")
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="demo"]')
        page.wait_for_function(
            "de.curve && document.querySelectorAll('#de-place-for option').length === 1", timeout=10_000
        )
        assert page.locator("#de-marks-for").input_value() == "gamepad"
        assert page.locator("#de-place-for option").all_inner_texts() == ["box"], (
            "every object but the one picked"
        )

        def scrub(i: int) -> None:
            page.locator("#de-slider").fill(str(i))
            page.locator("#de-slider").dispatch_event("input")

        scrub(6)
        page.click("text=Add pre-grasp here")
        scrub(10)
        page.click("text=Add pre-place here")  # before the grasp end exists: refused with a reason
        assert "set the grasp end first" in page.locator("#de-status").inner_text()
        scrub(12)
        page.click("text=Set grasp end here")
        scrub(18)
        page.click("text=Add pre-place here")
        scrub(24)
        page.click("text=Set place end here")
        rows = page.locator("#de-list").inner_text()
        assert "pre-place" in rows and "carried with box" in rows and "release included" in rows
        page.click("#demo-editor >> text=Save")
        page.wait_for_function(
            "document.getElementById('de-status').textContent.startsWith('saved')", timeout=10_000
        )
        with pregrasp._state.lock:
            kps = list(pregrasp._state.demo.keypoints)
        assert [(k["kind"], k["object"]) for k in kps] == [
            ("pregrasp", "gamepad"),
            ("grasp_end", "gamepad"),
            ("preplace", "box"),
            ("place_end", "box"),
        ]
        # Picking the box instead: the grasp follows it, and the place, which cannot go onto it, follows the gamepad.
        page.select_option("#de-marks-for", "box")
        objects = page.evaluate("de.kps.map(k => k.kind + ':' + k.object)")
        assert objects == ["pregrasp:box", "grasp_end:box", "preplace:gamepad", "place_end:gamepad"], objects
        assert page.locator("#de-place-for option").all_inner_texts() == ["gamepad"]
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_the_place_lands_as_shown_until_the_operator_frees_its_turn_about_the_object(gui_page, tmp_path):
    """The landing is chosen beside the place's object: as shown by default; turned any way about that object's middle
    once the operator says so, which the server keeps with the demo and the editor shows again when it reopens."""
    import time

    from lerobot.gui.api import pregrasp
    from tests.gui.test_stream_objects import _two_object_demo

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    demo, _box = _two_object_demo(tmp_path, time.time())
    with pregrasp._state.lock:
        pregrasp._state.demo, pregrasp._state.teach = demo, None
    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'demo')")
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="demo"]')
        page.wait_for_function(
            "de.curve && document.querySelectorAll('#de-place-for option').length === 1", timeout=10_000
        )
        assert page.locator("#de-landing").input_value() == "exact", "no symmetry unless asked"
        page.select_option("#de-landing", "turn")
        page.wait_for_function(
            "document.getElementById('de-status').textContent.includes('turned any way about the middle of box')",
            timeout=10_000,
        )
        with pregrasp._state.lock:
            assert pregrasp._state.demo.landing == "turn"
        page.evaluate("deLoad(true)")
        page.wait_for_function("de.curve && de.curve.landing === 'turn'", timeout=10_000)
        assert page.locator("#de-landing").input_value() == "turn", "shown again as kept"
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_the_guided_row_finds_the_place_object_after_the_picked_one_without_a_track(gui_page, tmp_path):
    """With a place marked, Act needs the object it goes onto found too: once the picked object is found, the row
    asks for a click on the other one, and that click locates it instead of teaching it, which would have moved the
    live track off the object picked."""
    import json
    import pathlib
    import time

    from lerobot.gui.api import pregrasp
    from tests.gui.test_stream_objects import _two_object_demo

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    located: dict = {}

    def found_the_gamepad(route):
        resp = route.fetch()
        body = resp.json()
        body["camera_live"] = True
        body["worker"] = {**body.get("worker", {}), "running": True, "ready": True}
        body["arm_connected"] = True
        ref = {
            "object": "gamepad",
            "ok": True,
            "inliers": 150,
            "card_points": 400,
            "strong": True,
            "turn_deg": 5.0,
        }
        body["teach"] = {"concept": "gamepad", "mode": "features", "ref": ref}
        body["track"] = {**body.get("track", {}), "on": True, "fps": 5.0, "last": {"state": "tracking"}}
        body["located"] = dict(located)
        route.fulfill(response=resp, json=body)

    clicks: list[dict] = []

    def locate(route):
        clicks.append(json.loads(route.request.post_data))
        located["box"] = {
            "object": "box",
            "ok": True,
            "inliers": 160,
            "card_points": 400,
            "strong": True,
            "turn_deg": 12.0,
        }
        route.fulfill(status=200, content_type="application/json", body=json.dumps(located["box"]))

    page.route("**/api/pregrasp/state", found_the_gamepad)
    page.route("**/api/pregrasp/locate", locate)
    page.route(
        "**/api/jog/state",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body='{"connected": true, "mode": "cartesian"}'
        ),
    )
    demo, _box = _two_object_demo(tmp_path, time.time())
    demo.root = str(tmp_path / "demos" / "stack")
    pathlib.Path(demo.root).mkdir(parents=True)
    demo.keypoints = [
        {"t": float(demo.t[6]), "kind": "pregrasp", "object": "gamepad"},
        {"t": float(demo.t[12]), "kind": "grasp_end", "object": "gamepad"},
        {"t": float(demo.t[18]), "kind": "preplace", "object": "box"},
        {"t": float(demo.t[24]), "kind": "place_end", "object": "box"},
    ]
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
    try:
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.wait_for_function(
            "document.getElementById('ap-guide-text').textContent.includes('click box in the camera view')",
            timeout=10_000,
        )
        assert "the object to place onto" in page.locator("#ap-guide-text").inner_text()
        page.evaluate("pgTeachAt(500, 320)")
        page.wait_for_function(
            "document.getElementById('ap-guide-step').textContent === 'Act'", timeout=10_000
        )
        assert clicks == [{"click": [500, 320], "object": "box"}], clicks
        text = page.locator("#ap-guide-text").inner_text()
        assert text == "ready", text
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_each_objects_pose_frame_is_set_on_its_row_and_a_leftover_teach_is_no_pick(gui_page, tmp_path):
    """Two faults met on a stacking demo. Its marks followed "the object taught before the demo", a teach left from an
    earlier demo, because that was the Pick list's default; and the cube placed onto was in full view only while the
    arm was far, which only a waypoint at that moment could have read. A clicked object now leaves no such choice,
    and each object's pose frame is set on its own row."""
    import time

    from lerobot.gui.api import pregrasp
    from tests.gui.test_stream_objects import _two_object_demo

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    demo, _box = _two_object_demo(tmp_path, time.time())
    demo.taught = True  # recorded while an earlier demo's teach was still up
    demo.keypoints = [
        {"t": float(demo.t[6]), "kind": "pregrasp"},
        {"t": float(demo.t[12]), "kind": "grasp_end"},
    ]
    with pregrasp._state.lock:
        pregrasp._state.demo, pregrasp._state.teach = demo, None
    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'demo')")
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="demo"]')
        page.wait_for_function(
            "document.getElementById('de-status').textContent.includes('now follow gamepad')", timeout=10_000
        )
        assert page.locator("#de-marks-for option").all_inner_texts() == ["gamepad", "box"], (
            "no leftover teach"
        )
        box_row = page.locator("#de-obj-list tr", has_text="box")
        assert "pose read at 0.07 s (clicked)" in box_row.inner_text()
        page.locator("#de-slider").fill("3")
        page.locator("#de-slider").dispatch_event("input")
        box_row.locator("text=pose here").click()
        assert "box's pose is read at 0.10 s" in page.locator("#de-status").inner_text()
        assert (
            "pose read at 0.10 s (set)" in box_row.inner_text() and box_row.locator("text=reset").count() == 1
        )
        page.click("#demo-editor >> text=Save")
        page.wait_for_function(
            "document.getElementById('de-status').textContent.startsWith('saved')", timeout=10_000
        )
        with pregrasp._state.lock:
            kps = [(k["kind"], k.get("object")) for k in pregrasp._state.demo.keypoints]
        assert kps == [("pose", "box"), ("pregrasp", "gamepad"), ("grasp_end", "gamepad")], kps
        box_row.locator("text=reset").click()
        page.click("#demo-editor >> text=Save")
        page.wait_for_function(
            "document.getElementById('de-status').textContent.startsWith('saved')", timeout=10_000
        )
        with pregrasp._state.lock:
            assert not any(k["kind"] == "pose" for k in pregrasp._state.demo.keypoints)
        page.wait_for_function(
            "document.querySelector('#de-obj-list').textContent.includes('pose read at 0.07 s (clicked)')",
            timeout=10_000,
        )
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_an_act_that_ended_asks_for_no_verdict(gui_page, tmp_path):
    """After every act the row asked "What happened? lifted / missed / collided", which a place made meaningless and
    which the act's own recording answers. The row goes straight back to its next step."""
    import json
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
    ref = {
        "object": "gamepad",
        "ok": True,
        "inliers": 150,
        "card_points": 400,
        "strong": True,
        "turn_deg": 3.0,
    }

    def acted(route):
        resp = route.fetch()
        body = resp.json()
        body["camera_live"] = True
        body["worker"] = {**body.get("worker", {}), "running": True, "ready": True}
        body["arm_connected"] = True
        body["teach"] = {"concept": "gamepad", "mode": "features", "ref": ref}
        body["track"] = {**body.get("track", {}), "on": True, "fps": 5.0, "last": {"state": "tracking"}}
        body["act"] = {**body.get("act", {}), "on": False, "ok": True, "step": "done", "reason": ""}
        route.fulfill(response=resp, json=body)

    page.route("**/api/pregrasp/state", acted)
    page.route(
        "**/api/pregrasp/trials",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"rows": [{"result": "done", "verdict": None}]}),
        ),
    )
    page.route(
        "**/api/jog/state",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body='{"connected": true, "mode": "cartesian"}'
        ),
    )
    demo = _demo_with_object(tmp_path, time.time())
    demo.root = str(tmp_path / "demos" / "bound")
    pathlib.Path(demo.root).mkdir(parents=True)
    demo.keypoints = [
        {"t": float(demo.t[6]), "kind": "pregrasp", "object": "gamepad"},
        {"t": float(demo.t[20]), "kind": "grasp_end", "object": "gamepad"},
    ]
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
    try:
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.wait_for_function(
            "document.getElementById('ap-guide-step').textContent === 'Act'", timeout=10_000
        )
        page.wait_for_timeout(1500)  # a few more polls of the row
        assert page.locator("#ap-guide-step").inner_text() == "Act"
        assert "What happened" not in page.locator("#ap-guide-text").inner_text()
        assert page.locator("#ap-guide button", has_text="lifted").count() == 0
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_a_stopped_act_is_one_line_with_the_whole_reason_under_why(gui_page, tmp_path):
    """The Act row strung together the last act's whole reason, both finds, the tracker's rate and a description of
    the marks: a paragraph for an operator who needs one line. It says why the act stopped in a few words; the whole
    reason is one click away."""
    import pathlib
    import time

    from lerobot.gui.api import pregrasp
    from tests.gui.test_stream_objects import _demo_with_object

    reason = (
        "how the demo holds gamepad cannot be measured: at its grip, the demo never holds gamepad still in the "
        "gripper after the gripper closed on it, before the lift; while carried, gamepad is not visible enough"
    )
    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    ref = {
        "object": "gamepad",
        "ok": True,
        "inliers": 267,
        "card_points": 373,
        "strong": True,
        "turn_deg": 18.0,
    }

    def stopped(route):
        resp = route.fetch()
        body = resp.json()
        body["camera_live"] = True
        body["worker"] = {**body.get("worker", {}), "running": True, "ready": True}
        body["arm_connected"] = True
        body["teach"] = {"concept": "gamepad", "mode": "features", "ref": ref}
        body["track"] = {**body.get("track", {}), "on": True, "fps": 6.0, "last": {"state": "tracking"}}
        body["act"] = {**body.get("act", {}), "on": False, "ok": False, "step": "aborted", "reason": reason}
        route.fulfill(response=resp, json=body)

    page.route("**/api/pregrasp/state", stopped)
    page.route(
        "**/api/jog/state",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body='{"connected": true, "mode": "cartesian"}'
        ),
    )
    demo = _demo_with_object(tmp_path, time.time())
    demo.root = str(tmp_path / "demos" / "bound")
    pathlib.Path(demo.root).mkdir(parents=True)
    demo.keypoints = [
        {"t": float(demo.t[6]), "kind": "pregrasp", "object": "gamepad"},
        {"t": float(demo.t[20]), "kind": "grasp_end", "object": "gamepad"},
    ]
    with pregrasp._state.lock:
        pregrasp._state.demo = demo
    try:
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.wait_for_function(
            "document.getElementById('ap-guide-step').textContent === 'Act'", timeout=10_000
        )
        assert (
            page.locator("#ap-guide-text").inner_text()
            == "last act stopped: how the demo holds gamepad cannot be measured"
        )
        assert not page.locator("#ap-guide-why span").is_visible(), "the whole reason waits under why"
        page.click("#ap-guide-why summary")
        assert page.locator("#ap-guide-why span").inner_text() == reason
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_a_page_older_than_the_server_asks_to_be_reloaded_and_the_trials_ask_nothing(gui_page):
    """A server restart does not reload an open tab: after two fixes went live, the operator's tab still asked for a
    verdict after an act and still strung its paragraph together, because it ran the code it had loaded before them.
    The row now says so and offers a reload; and the trials table under details asks for no verdict either."""
    import json

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    served = {"version": None}

    def newer_server(route):
        resp = route.fetch()
        body = resp.json()
        served["version"] = body.get("page_version")
        body["page_version"] = "1" + (served["version"] or "")  # what a later index.html would name
        route.fulfill(response=resp, json=body)

    page.route("**/api/pregrasp/state", newer_server)
    rows = [
        {
            "at": "2026-10-07 13:13:16",
            "demo": "pick_place",
            "result": "done",
            "reason": "",
            "verdict": None,
            "place": {"hold_used": "grasp pose", "shift_mm": 4.1, "shift_deg": 1.1},
        }
    ]
    page.route(
        "**/api/pregrasp/trials",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({"rows": rows})
        ),
    )
    page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'act')")
    page.reload()
    page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
    page.click('button[data-tab="approach"]')
    page.wait_for_function(
        "document.getElementById('ap-guide-step').textContent === 'Reload'", timeout=10_000
    )
    assert served["version"] == page.evaluate("PG_PAGE_VERSION"), (
        "the real server names the version this page loaded"
    )
    assert page.locator("#ap-guide button:visible", has_text="Reload").count() == 1
    page.click('button.ap-subtab[data-sub="act"]')
    page.wait_for_function(
        "document.getElementById('pg-trials').textContent.includes('pick_place')", timeout=10_000
    )
    table = page.locator("#pg-trials")
    assert "grasp pose, corrected 4.1 mm 1.1°" in table.inner_text()
    assert table.locator("button").count() == 0, "no verdict to give"
    assert errors == [], f"the page threw: {errors}"


def test_an_injected_error_is_set_under_the_act_row_named_while_on_and_sent_with_act(gui_page, tmp_path):
    """The operator tests how the act absorbs a grasp aimed off: a fold under the Act row takes the error, keeps it
    across reloads, names it in its summary while it is on, and the Act button sends it."""
    import json
    import pathlib
    import time

    from lerobot.gui.api import pregrasp
    from tests.gui.test_stream_objects import _demo_with_object

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    sent: list[dict] = []
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    ref = {
        "object": "gamepad",
        "ok": True,
        "inliers": 267,
        "card_points": 373,
        "strong": True,
        "turn_deg": 3.0,
    }

    def ready(route):
        resp = route.fetch()
        body = resp.json()
        body["camera_live"] = True
        body["worker"] = {**body.get("worker", {}), "running": True, "ready": True}
        body["arm_connected"] = True
        body["teach"] = {"concept": "gamepad", "mode": "features", "ref": ref}
        body["track"] = {**body.get("track", {}), "on": True, "fps": 6.0, "last": {"state": "tracking"}}
        body["act"] = {**body.get("act", {}), "on": False, "ok": None, "step": "", "reason": ""}
        route.fulfill(response=resp, json=body)

    def act(route):
        sent.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(status=200, content_type="application/json", body='{"status": "acting", "n": 1}')

    page.route("**/api/pregrasp/state", ready)
    page.route("**/api/pregrasp/act", act)
    page.route(
        "**/api/jog/state",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body='{"connected": true, "mode": "cartesian"}'
        ),
    )
    demo = _demo_with_object(tmp_path, time.time())
    demo.root = str(tmp_path / "demos" / "inject")
    pathlib.Path(demo.root).mkdir(parents=True)
    demo.keypoints = [
        {"t": float(demo.t[6]), "kind": "pregrasp", "object": "gamepad"},
        {"t": float(demo.t[20]), "kind": "grasp_end", "object": "gamepad"},
    ]
    with pregrasp._state.lock:
        pregrasp._state.demo = demo

    def to_act_row():
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.wait_for_function(
            "document.getElementById('ap-guide-step').textContent === 'Act'", timeout=10_000
        )

    try:
        page.reload()
        page.evaluate("localStorage.removeItem('ap-inject'); localStorage.removeItem('ap-inject-open')")
        page.reload()
        to_act_row()
        summary = page.locator("#ap-inject-summary")
        assert summary.is_visible() and summary.inner_text() == "inject an error"
        page.click("#ap-guide-btn")
        page.wait_for_function("document.getElementById('ap-guide-btn').disabled === false")
        assert sent[-1]["inject"] is None and sent[-1]["correct_hold"] is True, "off unless set"

        summary.click()
        page.select_option("#ap-inject-at", "aim")
        page.fill("#ap-inject-dx_mm", "5")
        page.press("#ap-inject-dx_mm", "Tab")
        page.fill("#ap-inject-rz_deg", "10")
        page.press("#ap-inject-rz_deg", "Tab")
        page.uncheck("#ap-inject-hold")
        assert summary.inner_text() == "injecting a missed aim: x 5 mm, about z 10°; the hold not corrected"
        page.reload()
        to_act_row()
        assert (
            summary.inner_text() == "injecting a missed aim: x 5 mm, about z 10°; the hold not corrected"
        ), "kept across a reload, and named while on"
        assert page.locator("#ap-inject-dx_mm").is_visible(), "the fold stays open as it was left"
        page.click("#ap-guide-btn")
        page.wait_for_function("document.getElementById('ap-guide-btn').disabled === false")
        assert sent[-1]["inject"] == {
            "at": "aim",
            "dx_mm": 5,
            "dy_mm": 0,
            "dz_mm": 0,
            "rx_deg": 0,
            "ry_deg": 0,
            "rz_deg": 10,
        }
        assert sent[-1]["correct_hold"] is False
        page.click("#ap-inject button")
        assert summary.inner_text() == "inject an error"
        assert errors == [], f"the page threw: {errors}"
    finally:
        page.evaluate("localStorage.removeItem('ap-inject'); localStorage.removeItem('ap-inject-open')")
        with pregrasp._state.lock:
            pregrasp._state.demo = None


def test_the_camera_view_follows_the_live_camera_whoever_runs_the_tracker(gui_page):
    """With the camera live and the tracker off, the camera view kept the last find's picture: only tracking refreshed
    it, and only once the page itself had started tracking, so a tracker a script started never showed either. The
    operator saw a frozen camera (2026-10-09). The view now shows the camera's frames while the camera is live and
    nothing is tracked, the tracker's while the server tracks, nothing once the camera stops, and a find's picture
    for a while before the camera's frames return."""
    import time

    import cv2
    import numpy as np

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    rig = {"camera": True, "tracking": False}
    hits = {"frame.jpg": 0, "live.jpg": 0, "test.jpg": 0}
    _ok, jpeg = cv2.imencode(".jpg", np.full((480, 848, 3), 90, np.uint8))

    def state(route):
        resp = route.fetch()
        body = resp.json()
        body["camera_live"] = rig["camera"]
        body["worker"] = {**body.get("worker", {}), "running": True, "ready": True}
        body["track"] = {
            **body.get("track", {}),
            "on": rig["tracking"],
            "fps": 5.0,
            "last": {"state": "tracking" if rig["tracking"] else "stopped"},
        }
        route.fulfill(response=resp, json=body)

    def image(name):
        def serve(route):
            hits[name] += 1
            route.fulfill(status=200, content_type="image/jpeg", body=jpeg.tobytes())

        return serve

    page.route("**/api/pregrasp/state", state)
    for name in hits:
        page.route(f"**/api/pregrasp/**/{name}*", image(name))
        page.route(f"**/api/pregrasp/{name}*", image(name))

    def until(cond, timeout_s: float = 10.0) -> bool:
        t_end = time.time() + timeout_s
        while not cond():
            if time.time() > t_end:
                return False
            page.wait_for_timeout(100)
        return True

    def quiet(name: str, seconds: float) -> int:
        """How many more requests for ``name`` arrive over ``seconds``."""
        n = hits[name]
        page.wait_for_timeout(int(seconds * 1000))
        return hits[name] - n

    page.reload()
    page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
    page.click('button[data-tab="approach"]')
    assert until(lambda: hits["frame.jpg"] >= 5), f"no live camera frames without tracking: {hits}"

    rig["tracking"] = True  # started by someone else: the page pressed nothing
    assert until(lambda: hits["live.jpg"] >= 3), f"the tracker's frames never took the view: {hits}"
    assert quiet("frame.jpg", 1.5) <= 1, "the camera's frames stop while the tracker's show"

    rig["tracking"] = False
    n = hits["frame.jpg"]
    assert until(lambda: hits["frame.jpg"] >= n + 3), f"the camera's frames did not come back: {hits}"

    page.evaluate("pgShowResult('test')")
    assert quiet("frame.jpg", 2.0) <= 1, "a find's picture is held, not replaced at once"
    assert "test.jpg" in page.evaluate("document.getElementById('pg-frame').src")
    n = hits["frame.jpg"]
    assert until(lambda: hits["frame.jpg"] >= n + 3, timeout_s=8.0), (
        "the camera's frames return after the hold"
    )

    rig["camera"] = False
    page.wait_for_timeout(1500)  # the row's next poll sees the camera stopped
    assert quiet("frame.jpg", 1.5) == 0, "no camera, no requests for its frames"
    assert errors == [], f"the page threw: {errors}"


def test_the_groups_panel_switch_says_and_sets_whether_acts_use_the_borrowed_points(gui_page):
    """Whether acts run with the point groups, whose borrowed points carry an object no view places: a checkbox in
    the Groups panel that shows the server's setting after a reload and sets it when clicked."""
    from lerobot.gui.api import pregrasp

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    page.route("**/api/pregrasp/groups/stream*", lambda route: route.fulfill(status=204, body=""))
    with pregrasp._state.lock:
        before, pregrasp._state.groups_with_acts = pregrasp._state.groups_with_acts, False
    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'groups')")
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="groups"]')
        box = page.locator("#groups-with-acts")
        page.wait_for_function("document.getElementById('groups-status').textContent !== ''", timeout=10_000)
        assert not box.is_checked(), "off as the server has it"
        box.check()
        page.wait_for_function("document.getElementById('groups-with-acts').checked", timeout=10_000)
        deadline = 50
        while not pregrasp._state.groups_with_acts and deadline:
            page.wait_for_timeout(100)
            deadline -= 1
        assert pregrasp._state.groups_with_acts, "the click set the server's switch"
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="groups"]')
        page.wait_for_function("document.getElementById('groups-with-acts').checked", timeout=10_000)
        box.uncheck()
        deadline = 50
        while pregrasp._state.groups_with_acts and deadline:
            page.wait_for_timeout(100)
            deadline -= 1
        assert not pregrasp._state.groups_with_acts, "and clears it"
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.groups_with_acts = before


def test_the_depth_check_box_and_tolerance_say_and_set_the_servers_option(gui_page):
    """The depth check, on by default at 20 mm: a box and a tolerance beside the trust slider that show the server's
    setting after a reload and set it when changed."""
    from lerobot.gui.api import pregrasp

    page = gui_page
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route(
        "**/api/showservo/cameras",
        lambda route: route.fulfill(status=200, content_type="application/json", body="[]"),
    )
    with pregrasp._state.lock:
        before = (pregrasp._state.depth_check, pregrasp._state.depth_tol_m)
        pregrasp._state.depth_check, pregrasp._state.depth_tol_m = True, 0.02

    def open_setup():
        page.reload()
        page.wait_for_function("typeof switchTab === 'function'", timeout=15_000)
        page.click('button[data-tab="approach"]')
        page.click('button.ap-subtab[data-sub="setup"]')

    def until(cond):
        for _ in range(50):
            if cond():
                return True
            page.wait_for_timeout(100)
        return cond()

    try:
        page.evaluate("localStorage.setItem('ap-details', '1'); localStorage.setItem('ap-sub', 'setup')")
        open_setup()
        box, mm = page.locator("#pg-depth"), page.locator("#pg-depth-mm")
        box.wait_for(state="visible", timeout=10_000)
        page.wait_for_function("document.getElementById('pg-depth-mm').value === '20'", timeout=10_000)
        assert box.is_checked(), "on at 20 mm as the server has it"
        mm.fill("15")
        mm.dispatch_event("change")
        assert until(lambda: abs(pregrasp._state.depth_tol_m - 0.015) < 1e-9), (
            "the tolerance set the server's"
        )
        box.uncheck()
        assert until(lambda: not pregrasp._state.depth_check), "the box turned it off"
        open_setup()
        page.wait_for_function(
            "!document.getElementById('pg-depth').checked && document.getElementById('pg-depth-mm').value === '15'",
            timeout=10_000,
        )
        assert errors == [], f"the page threw: {errors}"
    finally:
        with pregrasp._state.lock:
            pregrasp._state.depth_check, pregrasp._state.depth_tol_m = before
