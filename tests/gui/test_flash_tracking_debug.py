"""Camera-only debugging must respect the Run owner and reject stale selections."""

import asyncio
import importlib.util
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


def test_invalid_configuration_fails_before_camera_launch():
    with pytest.raises(ValidationError):
        flash_tracking.TrackingStart(shape={"depth_tolerance_m": float("nan")})
    with pytest.raises(ValidationError):
        flash_tracking.TrackingStart(fps=0)
    with pytest.raises(ValidationError):
        flash_tracking.TrackingStart(observation={"mask_margin_px": -1})
    with pytest.raises(ValidationError):
        flash_tracking.TrackingStart(motion={"grid_step_px": 0})


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
    monkeypatch.setattr(run, "_active_command", "flash_tracking")
    monkeypatch.setattr(run, "_active_process", SimpleNamespace(returncode=None, stdin=pipe))
    asyncio.run(
        flash_tracking.control(
            flash_tracking.TrackingControl(action="select", id="click", point=(12, 15), frame_id=31)
        )
    )
    sent = json.loads(pipe.data)
    assert sent["point"] == [12, 15] and sent["frame_id"] == 31


@pytest.mark.requires_playwright
@pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None or importlib.util.find_spec("uvicorn") is None,
    reason="GUI browser dependencies are not installed",
)
def test_live_debug_click_uses_painted_frozen_frame_not_latest_preview(gui_page):
    import base64

    import cv2
    import numpy as np

    page = gui_page
    _, png = cv2.imencode(".png", np.zeros((50, 100, 3), np.uint8))
    frame = {
        "frame_id": 42,
        "captured_s": 1.0,
        "width": 100,
        "height": 50,
        "image": "data:image/png;base64," + base64.b64encode(png).decode(),
    }
    packet = {
        "running": True,
        "server_time_s": 1.1,
        "preview": {**frame, "frame_id": 43},
        "estimate": {"status": "frozen", "frame": frame},
    }
    controls = []
    page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json=packet))

    def control(route):
        controls.append(route.request.post_data_json)
        route.fulfill(json={"id": "ack"})

    page.route("**/api/run/flash/tracking/control", control)
    page.evaluate("switchTab('run'); selectWorkflow('flash')")
    page.get_by_role("button", name="Live tracking · camera only →").click()
    page.wait_for_function("document.getElementById('flash-surface-estimate')?.dataset.frameId === '42'")
    # A click in the center of the scaled canvas must be converted back to
    # native image pixels, and bound to frame 42 even though preview is newer.
    page.locator("#flash-surface-estimate").click()
    page.wait_for_function(
        "document.getElementById('flash-tracking-message').textContent === 'Processing selection…'"
    )
    assert len(controls) == 1
    assert controls[0]["frame_id"] == 42
    assert controls[0]["point"] == pytest.approx([50, 25], abs=1)
    assert controls[0]["action"] == "select"


def test_debugger_exports_the_common_configuration_and_migrates_old_payloads():
    request = flash_tracking.TrackingStart(sam_resolution=448, observation={"mask_margin_px": 3})
    saved = request.model_dump()
    assert "observation" not in saved and "sam_resolution" not in saved
    assert saved["vision_pipeline"]["sam_resolution"] == 448
    assert saved["vision_pipeline"]["observation"]["mask_margin_px"] == 3
    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

    assert VisionPipelineConfig.from_dict(saved["vision_pipeline"]) == request.vision_pipeline
    with pytest.raises(ValidationError):
        flash_tracking.TrackingStart(vision_pipeline={"unknown_setting": 1})
    with pytest.raises(ValidationError, match="only once"):
        flash_tracking.TrackingStart(observation={}, vision_pipeline={"observation": {}})


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright not installed")
def test_object_inspection_removes_pose_but_scene_toggle_restores_it(gui_page):
    page = gui_page
    page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json={"running": False}))
    page.evaluate("switchTab('run'); selectWorkflow('flash')")
    page.get_by_role("button", name="Live tracking · camera only →").click()
    page.wait_for_selector("#flash-surface-cloud")
    result = page.evaluate("""() => {
        clearTimeout(flashTrackingTimer);
        const reference = [[0,0,1],[.1,0,1],[0,.1,1],[.07,.02,1.05]];
        const axes = [[.05,0,1],[0,.05,1],[0,0,1.05]];
        const frame = (points, axes, pose) => ({session_id:'test',estimate:{command_id:'selection',
            diagnostics:{reference_frame_id:1}, camera_from_reference:pose, reference_m:points,
            observed_m:points, anchors_m:points.slice(0,3), axes_m:axes}});
        const image = () => document.getElementById('flash-surface-cloud').toDataURL();
        const paint = packet => { flashTrackingSnapshot=packet; flashTrackingDrawCloud(); return image(); };
        const identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
        const before = paint(frame(reference,axes,identity));
        const move = p => [-p[1]+.05, p[0]+.03,p[2]];
        const moved = frame(reference.map(move),axes.map(move),[[0,-1,0,.05],[1,0,0,.03],[0,0,1,0],[0,0,0,1]]);
        const after = paint(moved);
        const framingBefore = JSON.stringify(flashTrackingFraming);
        moved.estimate.observed_m = moved.estimate.observed_m.slice(0,1);
        paint(moved);
        const unchangedFraming = framingBefore === JSON.stringify(flashTrackingFraming);
        moved.estimate.observed_m = moved.estimate.reference_m;
        document.getElementById('flash-surface-follow').click();
        const scene = image();
        document.getElementById('flash-surface-follow').click();
        flashTrackingYaw=.5; flashTrackingDrawCloud();
        const orbited = image();
        return {same:before===after, unchangedFraming, sceneChanged:scene!==after, orbitWorks:orbited!==after};
    }""")
    assert result == {"same": True, "unchangedFraming": True, "sceneChanged": True, "orbitWorks": True}


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright not installed")
def test_sampling_controls_reach_shared_configuration(gui_page):
    page = gui_page
    page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json={"running": False}))
    sent = []

    def launch(route):
        sent.append(route.request.post_data_json)
        route.fulfill(json={"status": "started"})

    page.route("**/api/run/flash/tracking/start", launch)
    page.evaluate("switchTab('run'); selectWorkflow('flash')")
    page.get_by_role("button", name="Live tracking · camera only →").click()
    page.wait_for_function("document.getElementById('flash-sampling-budget')?.value !== ''")
    page.locator("#flash-sampling-budget").fill("512")
    page.locator("#flash-sampling-min").fill("0.8")
    page.locator("#flash-sampling-across").fill("80")
    page.get_by_role("button", name="Open camera", exact=True).click()
    page.wait_for_timeout(100)
    assert len(sent) == 1
    config = flash_tracking.TrackingStart(**sent[0]).vision_pipeline
    assert config.shape.max_fit_points == 512
    assert config.observation.min_surface_spacing_m == pytest.approx(0.0008)
    assert config.observation.samples_across_object == 80
