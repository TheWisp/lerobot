"""Camera-only debugging must respect the Run owner and reject stale selections."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from lerobot.gui.api import flash_tracking, run
from lerobot.gui.flash_pipeline_fields import pipeline_fields


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    monkeypatch.setenv("LEROBOT_FLASH_HISTORY_DIR", str(tmp_path / "history"))


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
        flash_tracking.TrackingStart(method="conditioned_fixed")
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

    async def check_backend(backend):
        captured["checked_backend"] = backend

    monkeypatch.setattr(run, "_launch_lock", asyncio.Lock())
    monkeypatch.setattr(run, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(run, "_launch_subprocess", launch)
    monkeypatch.setattr(flash_tracking, "_check_registration_backend", check_backend)
    monkeypatch.setattr("lerobot.gui.api.robot.release_preview_cameras", lambda: None)
    monkeypatch.setattr(flash_tracking, "_directory", None)
    monkeypatch.setattr(flash_tracking.tempfile, "mkdtemp", lambda **_: str(tmp_path))
    req = flash_tracking.TrackingStart(pipeline_settings=settings)
    asyncio.run(flash_tracking.start(req))
    assert captured["checked_backend"] == "open3d_plane"
    assert captured["args"][-2] == "lerobot.policies.flash.live_surface_worker"
    assert captured["config"]["pipeline_settings"] == settings
    assert "method" not in captured["config"]
    assert json.loads((tmp_path / "config.json").read_text())["pipeline_settings"] == json.loads(
        json.dumps(settings)
    )
    settings["group_experiment"] = None
    with pytest.raises(ValidationError, match="requires group"):
        flash_tracking.TrackingStart(pipeline_settings=settings)


def test_optional_registration_is_checked_in_the_worker_python_before_camera_launch(monkeypatch):
    seen = []

    class Probe:
        returncode = 1

        async def communicate(self):
            return b"", b""

    async def spawn(*args, **kwargs):
        seen.append(args)
        return Probe()

    monkeypatch.setattr(flash_tracking, "_python", lambda: "/tracking/python")
    monkeypatch.setattr(flash_tracking.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(HTTPException, match="small_gicp is unavailable"):
        asyncio.run(flash_tracking._check_registration_backend("small_gicp"))
    assert seen == [
        (
            "/tracking/python",
            "-c",
            "import importlib.util, sys; sys.exit(importlib.util.find_spec(sys.argv[1]) is None)",
            "small_gicp",
        )
    ]
    asyncio.run(flash_tracking._check_registration_backend("visible_icp"))
    assert len(seen) == 1


@pytest.mark.requires_playwright
def test_shared_live_view_picks_displayed_frame_without_freeze_and_keeps_processed_image(gui_page):
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
    displayed = {
        "frame_id": 42,
        "width": 100,
        "height": 50,
        "image": "data:image/png;base64," + base64.b64encode(png).decode(),
    }
    state = {
        "running": True,
        "config": {"engine": "experiment", "pipeline_settings": defaults()},
        "estimate": {"status": "preview"},
        "preview": displayed,
    }
    page.route("**/api/run/flash/tracking/state", lambda r: r.fulfill(json=state))
    page.route(
        "**/api/run/flash/tracking/shared-defaults",
        lambda r: r.fulfill(json={"settings": defaults(), "matchers": MATCHERS, "fields": pipeline_fields()}),
    )
    controls = []

    def control(route):
        controls.append(route.request.post_data_json)
        route.fulfill(json={"id": "ok"})

    page.route("**/api/run/flash/tracking/control", control)
    page.evaluate("async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash');}")
    page.locator("#flash-source-live").click()
    page.wait_for_function("flashLive.frame?.frame_id === 42")
    assert page.locator("#flash-live-method").count() == 0
    # A delayed experiment-list response must not replace the live view.
    page.route(
        "**/api/run/flash/experiments",
        lambda r: r.fulfill(json=[{"id": "later", "name": "Later", "experiment": "test"}]),
    )
    page.evaluate("async () => {flashExperiments.selected = null; await loadFlashExperiments();}")
    assert page.locator("#flash-live-panel").is_visible()
    assert page.evaluate("flashLive.visible")
    page.locator("#flash-live-pick").click()
    assert page.locator("#flash-live-image").evaluate("e => getComputedStyle(e).cursor") == "crosshair"
    page.locator("#flash-live-image").click()
    page.wait_for_timeout(100)
    assert controls[-1]["frame_id"] == 42
    assert controls[-1]["point"] == pytest.approx([50, 25], abs=1)
    assert page.locator("#flash-live-pick").get_attribute("aria-pressed") != "true"
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
    packet = encode_rgbd_scene(
        frame,
        {
            "frame": 43,
            "time_s": 1.0,
            "method": "group_growing",
            "objects": {
                "block": {
                    "name": "block",
                    "status": "lost",
                    "reason": "Fit protrudes into measured empty space",
                    "pose": None,
                    "surface": [],
                    "axis_points": [],
                    "contours": [],
                    "group_tracks": {"group_id": 7},
                }
            },
            "scene_group_tracks": [
                {"id": 7, "ids": [1], "pixels": [[25, 25]], "links": [], "camera_xyz": [[0, 0, 1]]},
                {
                    "id": 9,
                    "ids": [2],
                    "pixels": [[75, 25]],
                    "links": [],
                    "camera_xyz": [[0.1, 0, 1]],
                    "predicted_pixels": [[80, 25]],
                    "predicted_camera_xyz": [[0.15, 0, 1]],
                },
            ],
        },
    )
    page.route(
        "**/api/run/flash/tracking/scene",
        lambda r: r.fulfill(body=packet, content_type="application/octet-stream"),
    )
    state["estimate"] = {"status": "tracking", "frame_id": 43}
    page.wait_for_function("flashLive.processed === 43")
    assert page.evaluate("flashLive.frame.frame_id") == 43
    assert page.evaluate("flashLive.packet.frame") == 43
    assert page.locator("#flash-live-group-count").inner_text() == "2 groups · 1 shown"
    # Isolated measured tracks remain in the group record, but are not drawn
    # as dangling wireframe dots. The saved video viewer uses this same helper.
    assert page.evaluate(
        "Array.from(document.getElementById('flash-live-image').getContext('2d').getImageData(25,25,1,1).data)"
    ) == [70, 70, 70, 255]
    page.locator('#flash-live-groups button[data-mode="all"]').click()
    assert page.locator("#flash-live-group-count").inner_text() == "2 groups · 2 shown"
    assert page.evaluate(
        "Array.from(document.getElementById('flash-live-image').getContext('2d').getImageData(80,25,1,1).data)"
    ) == [70, 70, 70, 255]
    assert (
        "block: lost — Fit protrudes into measured empty space"
        in page.locator("#flash-live-objects").inner_text()
    )
    # The visible image must be the processed packet's pixels despite preview 99.
    assert page.evaluate(
        "Array.from(document.getElementById('flash-live-image').getContext('2d').getImageData(20,20,1,1).data)"
    ) == [70, 70, 70, 255]
    page.locator('#flash-live-view-mode button[data-mode="3d"]').click()
    page.wait_for_function("flashLive.viewer !== null")
    assert page.locator("#flash-live-scene").is_visible()
    assert page.locator("#flash-live-camera-view").is_visible()
    page.locator('#flash-live-view-mode button[data-mode="2d"]').click()
    assert page.locator("#flash-live-image").is_visible()
    assert not page.locator("#flash-live-camera-view").is_visible()
    state["preview"] = {**displayed, "frame_id": 99}
    state["estimate"] = {"status": "error", "message": "small_gicp is unavailable"}
    page.wait_for_function("flashLive.packet === null && flashLive.frame?.frame_id === 99")
    assert "small_gicp is unavailable" in page.locator("#flash-live-status").inner_text()


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
        lambda r: r.fulfill(json={"settings": defaults(), "matchers": MATCHERS, "fields": pipeline_fields()}),
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
    for setting in ("camera", "matching", "matcher", "settings", "start", "stop"):
        assert page.locator(f"#flash-live-{setting}").evaluate("e => !!e.closest('#flash-live-sidebar')")
    assert page.locator("#flash-live-method").count() == 0
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
        lambda r: r.fulfill(json={"settings": defaults(), "matchers": MATCHERS, "fields": pipeline_fields()}),
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


def test_advanced_values_reach_actual_frame_preparation_and_solver():
    from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
    from lerobot.policies.flash.surface_session_config import defaults
    from tests.policies.flash.test_live_surface_session import SuppliedMasks, sequence

    values = defaults()
    values["config"]["shape"]["max_evaluations"] = 1
    values["config"]["observation"]["mask_margin_px"] = 6
    req = flash_tracking.TrackingStart(pipeline_settings=values)
    # Cross the launch JSON and actual worker constructor, not a mirror of form fields.
    loaded = json.loads(req.model_dump_json())["pipeline_settings"]
    frames = list(sequence())[:2]
    with LiveSurfaceSession(loaded, SuppliedMasks(frames)).engine as engine:
        row0, _ = engine.step(*frames[0])
        row1, _ = engine.step(*frames[1])
    detail0 = row0["objects"]["left"]["group_growing_details"]
    assert detail0["observation"]["clean_count"] == (56 - 12) * (45 - 12)
    assert row1["objects"]["left"]["group_growing_details"]["fit"]["evaluations"] <= 1


def test_history_api_survives_new_session_and_rejects_path_traversal(tmp_path):
    from lerobot.policies.flash.tracking_history import history_root
    from tests.policies.flash.test_tracking_history import new_history, submit

    history = new_history(history_root() / "recent")
    submit(history, 0, 1)
    history.close()
    saved = asyncio.run(flash_tracking.save_history())["id"]
    listing = asyncio.run(flash_tracking.history(saved))
    assert [r["frame"] for r in listing["frames"]] == [0, 1]
    assert listing["recordings"][0]["id"] == saved
    packet = asyncio.run(flash_tracking.history_scene(1, saved))
    assert packet.media_type == "application/octet-stream"
    with pytest.raises(HTTPException):
        asyncio.run(flash_tracking.history_scene(1, "../config"))
    with pytest.raises(HTTPException):
        asyncio.run(flash_tracking.delete_history("../config"))
    assert asyncio.run(flash_tracking.delete_history(saved))["status"] == "deleted"
    assert not (history_root() / saved).exists()
    assert asyncio.run(flash_tracking.delete_history("recent"))["status"] == "deleted"
    assert asyncio.run(flash_tracking.history("recent"))["frames"] == []


@pytest.mark.requires_playwright
def test_typed_pipeline_form_submits_to_real_start_schema(gui_page):
    from lerobot.policies.flash.surface_registration import MATCHERS
    from lerobot.policies.flash.surface_session_config import defaults

    page = gui_page
    page.route("**/api/run/flash/experiments", lambda r: r.fulfill(json=[]))
    page.route("**/api/run/flash/tracking/state", lambda r: r.fulfill(json={"running": False}))
    page.route(
        "**/api/run/flash/tracking/shared-defaults",
        lambda r: r.fulfill(json={"settings": defaults(), "matchers": MATCHERS, "fields": pipeline_fields()}),
    )
    page.route(
        "**/api/run/flash/tracking/cameras",
        lambda r: r.fulfill(json={"cameras": [{"id": "front", "name": "Front"}]}),
    )
    submitted = []

    def start(route):
        submitted.append(flash_tracking.TrackingStart(**route.request.post_data_json))
        route.fulfill(json={"status": "started"})

    page.route("**/api/run/flash/tracking/start", start)
    page.locator('.tab[data-tab="flash"]').click()
    page.locator("#flash-source-live").click()
    page.wait_for_function(
        "flashLive.settings !== null && !document.getElementById('flash-live-start').disabled"
    )
    page.locator("#flash-live-settings").click()
    dialog = page.locator("#flash-live-settings-dialog")
    assert dialog.is_visible()
    assert dialog.locator("textarea").count() == 0
    dialog.get_by_label("Total fitting iteration limit", exact=True).fill("7")
    dialog.get_by_label("Depth agreement tolerance (mm)", exact=True).fill("4")
    dialog.get_by_label("Use pose uncertainty", exact=True).uncheck()
    assert not dialog.get_by_label("Iterations per resolution", exact=True).is_visible()
    dialog.get_by_role("button", name="Close", exact=True).click()
    assert page.locator("#flash-live-history-mb").count() == 0
    page.get_by_label("Recent history (seconds)", exact=True).fill("90")
    page.locator("#flash-live-start").click()
    page.wait_for_function("flashLive.starting === null")
    assert submitted
    assert submitted[-1].history_seconds == 90
    assert "history_max_mb" not in submitted[-1].model_dump()
    settings = submitted[-1].pipeline_settings
    assert settings["config"]["shape"]["max_evaluations"] == 7
    assert settings["config"]["shape"]["depth_tolerance_m"] == 0.004
    assert settings["surface_experiment"]["uncertainty"]["enabled"] is False


@pytest.mark.requires_playwright
def test_history_scrub_save_and_reload_use_retained_evidence(gui_page):
    from lerobot.policies.flash.tracking_history import history_root, read_packet
    from tests.policies.flash.test_tracking_history import new_history, submit

    recent = history_root() / "recent"
    history = new_history(recent)
    submit(history, 0, 1, 2)
    history.close()
    page = gui_page
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("**/api/run/flash/experiments", lambda r: r.fulfill(json=[]))
    page.route("**/api/run/flash/tracking/state", lambda r: r.fulfill(json={"running": False}))
    page.route("**/api/run/flash/tracking/cameras", lambda r: r.fulfill(json={"cameras": []}))
    page.locator('.tab[data-tab="flash"]').click()
    page.locator("#flash-source-live").click()
    page.wait_for_function("flashLive.packet?.frame === 2")
    assert page.locator("#flash-live-playback").is_visible()
    # Scrubbing must not let an older, slower response repaint the newer selection.
    delayed = []
    page.route("**/history/scene?recording=recent&frame=0", lambda route: delayed.append(route))
    page.evaluate("void loadFlashHistory(0)")
    page.wait_for_timeout(100)
    assert delayed
    page.evaluate("void loadFlashHistory(1)")
    page.wait_for_function("flashLive.packet?.frame === 1")
    delayed[0].fulfill(body=read_packet(recent, 0), content_type="application/octet-stream")
    page.wait_for_timeout(100)
    assert page.evaluate("flashLive.packet.result.frame") == 1
    assert "test: lost" in page.locator("#flash-live-objects").inner_text()
    page.locator("#flash-live-save-history").click()
    page.wait_for_function("document.getElementById('flash-live-recording').value.startsWith('saved-')")
    saved = page.locator("#flash-live-recording").input_value()
    assert (history_root() / saved / "index.json").is_file()
    page.reload()
    page.locator('.tab[data-tab="flash"]').click()
    page.locator("#flash-source-live").click()
    page.wait_for_function("flashLive.historyFrames?.length === 3")
    page.locator('.flash-live-recording-row button[aria-current="false"]').last.click()
    page.wait_for_function("flashLive.packet?.frame === 2")
    page.locator('#flash-live-view-mode button[data-mode="3d"]').click()
    page.wait_for_function("flashLive.viewer !== null")
    assert page.locator("#flash-live-camera-view").is_visible()
    page.get_by_role("button", name="Delete recording", exact=False).click()
    page.wait_for_function("document.getElementById('flash-live-recording').value === 'recent'")
    assert not (history_root() / saved).exists()
    page.get_by_role("button", name="Clear recent history").click()
    page.wait_for_function("flashLive.historyFrames.length === 0")
    assert not list(recent.glob("*.bin.z"))
    assert errors == []
