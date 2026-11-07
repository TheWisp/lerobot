"""Camera-only debugging must respect the Run owner and reject stale selections."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from lerobot.gui.api import flash_tracking, run


def test_other_run_cannot_receive_tracker_controls(monkeypatch):
    monkeypatch.setattr(run, "_active_command", "record")
    monkeypatch.setattr(run, "_active_process", SimpleNamespace(returncode=None))
    with pytest.raises(HTTPException) as error:
        asyncio.run(flash_tracking.control(flash_tracking.TrackingControl(action="freeze", id="test")))
    assert error.value.status_code == 409


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("shape", "depth_tolerance_m", float("nan")),
        ("observation", "mask_margin_px", -1),
        ("motion", "grid_step_px", 0),
    ],
)
def test_invalid_configuration_fails_before_camera_launch(section, field, value):
    from lerobot.policies.flash.surface_session_config import defaults

    settings = defaults()
    settings["config"][section][field] = value
    with pytest.raises(ValidationError):
        flash_tracking.TrackingStart(pipeline_settings=settings)


def test_retired_camera_engine_is_not_a_hidden_alternative():
    with pytest.raises(ValidationError):
        flash_tracking.TrackingStart(engine="legacy")
    with pytest.raises(ValidationError):
        flash_tracking.TrackingStart(fps=0)


def test_snapshot_keeps_camera_and_estimate_frame_identity_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(flash_tracking, "_directory", tmp_path)
    monkeypatch.setattr(run, "_active_command", None)
    (tmp_path / "preview.json").write_text(json.dumps({"frame_id": 31, "captured_s": 2.0}))
    (tmp_path / "state.json").write_text(
        json.dumps({"status": "fitted", "frame": {"frame_id": 27, "captured_s": 1.8}})
    )
    state = asyncio.run(flash_tracking.state())
    assert state["preview"]["frame_id"] == 31
    assert state["estimate"]["frame"]["frame_id"] == 27
    assert state["running"] is False


def test_click_command_preserves_frozen_frame_identity(monkeypatch):
    class Pipe:
        def __init__(self):
            self.data = b""

        def write(self, value):
            self.data += value

        async def drain(self):
            pass

    pipe = Pipe()
    monkeypatch.setattr(run, "_active_command", "flash_surface_live")
    monkeypatch.setattr(run, "_active_process", SimpleNamespace(returncode=None, stdin=pipe))
    asyncio.run(
        flash_tracking.control(
            flash_tracking.TrackingControl(action="select", id="click", point=(12, 15), frame_id=31)
        )
    )
    sent = json.loads(pipe.data)
    assert sent["point"] == [12, 15] and sent["frame_id"] == 31


def test_new_live_inspector_launches_shared_worker_and_passes_exact_settings(tmp_path, monkeypatch):
    from lerobot.policies.flash.surface_session_config import defaults

    settings = defaults()
    settings["shared_matching"]["backend"] = "lk"
    settings["config"]["shape"]["registration"]["backend"] = "open3d_plane"
    settings["surface_experiment"]["spacing_m"] = 0.004
    captured = {}

    async def launch(args, **kwargs):
        captured.update(args=args, **kwargs)

    monkeypatch.setattr(run, "_launch_lock", asyncio.Lock())
    monkeypatch.setattr(run, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(run, "_launch_subprocess", launch)
    monkeypatch.setattr("lerobot.gui.api.robot.release_preview_cameras", lambda: None)
    monkeypatch.setattr(flash_tracking, "_directory", None)
    monkeypatch.setattr(flash_tracking.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    req = flash_tracking.TrackingStart(pipeline_settings=settings)
    asyncio.run(flash_tracking.start(req))
    assert captured["args"][-2] == "lerobot.policies.flash.live_surface_worker"
    assert captured["config"]["pipeline_settings"] == settings
    assert json.loads((tmp_path / "config.json").read_text())["pipeline_settings"] == json.loads(
        json.dumps(settings)
    )
    settings["group_experiment"] = None
    with pytest.raises(ValidationError, match="requires group"):
        flash_tracking.TrackingStart(pipeline_settings=settings)


@pytest.mark.requires_playwright
def test_shared_live_view_freezes_clicked_frame_and_never_paints_new_preview_over_old_pose(gui_page):
    import base64

    import cv2
    import numpy as np

    from lerobot.gui.flash_live_scene import encode_rgbd_scene
    from lerobot.policies.flash.perception import RGBDFrame
    from lerobot.policies.flash.surface_registration import MATCHERS
    from lerobot.policies.flash.surface_session_config import defaults

    page = gui_page
    rgb = np.full((50, 100, 3), 70, np.uint8)
    _, png = cv2.imencode(".png", rgb)
    frozen = {
        "frame_id": 42,
        "width": 100,
        "height": 50,
        "image": "data:image/png;base64," + base64.b64encode(png).decode(),
    }
    state = {
        "running": True,
        "config": {"engine": "experiment", "pipeline_settings": defaults(), "method": "group_growing"},
        "estimate": {"status": "frozen", "frame": frozen},
        "preview": {**frozen, "frame_id": 99},
    }
    page.route("**/api/run/flash/tracking/state", lambda r: r.fulfill(json=state))
    page.route(
        "**/api/run/flash/tracking/shared-defaults",
        lambda r: r.fulfill(json={"settings": defaults(), "matchers": MATCHERS}),
    )
    controls = []

    def control(route):
        controls.append(route.request.post_data_json)
        route.fulfill(json={"id": "ok"})

    page.route("**/api/run/flash/tracking/control", control)
    page.evaluate("async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash');}")
    page.get_by_role("button", name="Live camera", exact=True).click()
    page.wait_for_function("flashLive.frame?.frozen === true")
    # A delayed experiment-list response must not replace the live view.
    page.route(
        "**/api/run/flash/experiments",
        lambda r: r.fulfill(json=[{"id": "later", "name": "Later", "experiment": "test"}]),
    )
    page.evaluate("async () => {flashExperiments.selected = null; await loadFlashExperiments();}")
    assert page.locator("#flash-live-panel").is_visible()
    assert page.evaluate("flashLive.visible")
    page.locator("#flash-live-image").click()
    page.wait_for_timeout(100)
    assert controls[-1]["frame_id"] == 42
    assert controls[-1]["point"] == pytest.approx([50, 25], abs=1)
    page.locator("#flash-live-name").fill("red object")
    page.get_by_role("button", name="Track by name", exact=True).click()
    assert controls[-1]["prompt"] == "red object" and "point" not in controls[-1]
    frame = RGBDFrame(
        43,
        rgb,
        np.full((50, 100), 1000, np.uint16),
        np.array([[100, 0, 50], [0, 100, 25], [0, 0, 1]]),
        0.001,
        np.eye(4),
        1.0,
        1.0,
        "test",
    )
    packet = encode_rgbd_scene(frame, {"frame": 43, "time_s": 1.0, "method": "group_growing", "objects": {}})
    page.route(
        "**/api/run/flash/tracking/scene",
        lambda r: r.fulfill(body=packet, content_type="application/octet-stream"),
    )
    state["estimate"] = {"status": "tracking", "frame_id": 43}
    page.wait_for_function("flashLive.processed === 43")
    assert page.evaluate("flashLive.frame.frame_id") == 43
    assert page.evaluate("flashLive.packet.frame") == 43
    # The visible image must be the processed packet's pixels despite preview 99.
    assert page.evaluate(
        "Array.from(document.getElementById('flash-live-image').getContext('2d').getImageData(20,20,1,1).data)"
    ) == [70, 70, 70, 255]
    page.locator("#flash-live-3d").click()
    page.wait_for_function("flashLive.viewer !== null")
    assert page.locator("#flash-live-scene").is_visible()


@pytest.mark.requires_playwright
def test_flash_has_its_own_tab_with_video_and_live_subtabs(gui_page):
    from lerobot.policies.flash.surface_registration import MATCHERS
    from lerobot.policies.flash.surface_session_config import defaults

    page = gui_page
    page.route("**/api/run/flash/experiments", lambda r: r.fulfill(json=[]))
    page.route("**/api/run/flash/tracking/state", lambda r: r.fulfill(json={"running": False}))
    page.route(
        "**/api/run/flash/tracking/cameras",
        lambda r: r.fulfill(
            json={"cameras": [{"id": "front", "name": "Front camera"}, {"id": "side", "name": "Side camera"}]}
        ),
    )
    page.route(
        "**/api/run/flash/tracking/shared-defaults",
        lambda r: r.fulfill(json={"settings": defaults(), "matchers": MATCHERS}),
    )
    page.route("**/api/run/flash/tracking/stop", lambda r: r.fulfill(json={"status": "stopped"}))

    page.locator('.tab[data-tab="flash"]').click()
    page.wait_for_function("document.getElementById('tab-flash').classList.contains('active')")
    assert page.locator("#flash-experiment-sidebar").is_visible()
    assert page.locator("#flash-experiment-view").is_visible()
    assert (
        page.locator("#flash-source-examples").evaluate("e => e.parentElement.parentElement.id")
        == "tab-flash"
    )
    assert page.locator("#flash-examples-workspace").is_visible()
    assert not page.locator("#flash-live-workspace").is_visible()
    assert not page.locator("#run-launch-btn").is_visible()
    assert page.locator('.workflow-btn[data-workflow="flash"]').count() == 0

    page.locator("#flash-source-live").click()
    page.wait_for_function("flashLive.visible && flashLive.settings !== null")
    assert page.locator("#flash-live-panel").is_visible()
    assert page.locator("#flash-live-sidebar").is_visible()
    assert not page.locator("#flash-examples-workspace").is_visible()
    assert not page.locator("#flash-experiment-sidebar").is_visible()
    for setting in ("camera", "method", "matching", "matcher", "config", "start", "stop"):
        assert page.locator(f"#flash-live-{setting}").evaluate("e => !!e.closest('#flash-live-sidebar')")
    page.wait_for_function("document.getElementById('flash-live-camera').options.length === 3")
    page.locator("#flash-live-camera").select_option("side")
    assert page.locator("#flash-live-camera").input_value() == "side"
    page.locator("#flash-source-examples").click()
    page.wait_for_function("!flashLive.visible")
    assert page.locator("#flash-experiment-sidebar").is_visible()
    assert not page.locator("#flash-live-sidebar").is_visible()

    # Saved links to the former Run workflow open the new top-level FLASH tab.
    page.locator('.tab[data-tab="run"]').click()
    page.evaluate("window.location.hash = '#/run/flash'")
    page.wait_for_function("document.getElementById('tab-flash').classList.contains('active')")


@pytest.mark.requires_playwright
def test_source_navigation_stops_camera_before_returning_to_examples(gui_page):
    from lerobot.policies.flash.surface_registration import MATCHERS
    from lerobot.policies.flash.surface_session_config import defaults

    page = gui_page
    errors, calls = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("**/api/run/flash/experiments", lambda r: r.fulfill(json=[]))
    page.route("**/api/run/flash/tracking/state", lambda r: r.fulfill(json={"running": False}))
    page.route(
        "**/api/run/flash/tracking/shared-defaults",
        lambda r: r.fulfill(json={"settings": defaults(), "matchers": MATCHERS}),
    )
    failure = [True]

    def stop(route):
        calls.append("stop")
        route.fulfill(
            status=500 if failure[0] else 200,
            json={"detail": "Camera stop failed"} if failure[0] else {"status": "stopped"},
        )

    page.route("**/api/run/flash/tracking/stop", stop)
    page.evaluate("async () => { switchTab('run'); await runTabInit(); await selectWorkflow('flash'); }")
    assert page.locator("#flash-source-examples").get_attribute("aria-pressed") == "true"
    assert page.get_by_role("button", name="Demo / rollout editor").count() == 0
    assert not page.get_by_role("button", name="Launch", exact=True).is_visible()
    page.get_by_role("button", name="Live camera", exact=True).click()
    page.wait_for_function("flashLive.visible && flashLive.settings !== null")
    assert page.locator("#flash-live-panel").is_visible()
    assert not page.locator("#flash-experiment-sidebar").is_visible()
    page.get_by_role("button", name="Video examples", exact=True).click()
    page.wait_for_function(
        "document.getElementById('flash-source-live').getAttribute('aria-pressed') === 'true'"
    )
    assert calls == ["stop"]
    failure[0] = False
    page.get_by_role("button", name="Video examples", exact=True).click()
    page.wait_for_function("!flashLive.visible")
    assert calls == ["stop", "stop"]
    assert page.locator("#flash-experiment-sidebar").is_visible()
    assert not page.locator("#flash-live-panel").is_visible()
    page.evaluate("async () => { await selectWorkflow('teleop'); }")
    assert page.get_by_role("button", name="Launch", exact=True).is_visible()
    assert errors == []
