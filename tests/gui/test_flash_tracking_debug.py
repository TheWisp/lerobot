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


def test_record_command_preserves_duration_across_gui_process_boundary(monkeypatch):
    class Pipe:
        data = b""

        def write(self, value):
            self.data += value

        async def drain(self):
            pass

    pipe = Pipe()
    monkeypatch.setattr(run, "_active_command", "flash_tracking")
    monkeypatch.setattr(run, "_active_process", SimpleNamespace(returncode=None, stdin=pipe))
    asyncio.run(
        flash_tracking.control(flash_tracking.TrackingControl(action="record", id="trace", duration_s=7.5))
    )
    sent = json.loads(pipe.data)
    assert sent["action"] == "record" and sent["duration_s"] == 7.5
    with pytest.raises(HTTPException) as error:
        asyncio.run(flash_tracking.control(flash_tracking.TrackingControl(action="record", id="missing")))
    assert error.value.status_code == 400
    with pytest.raises(ValidationError):
        flash_tracking.TrackingControl(action="record", id="too-long", duration_s=121)


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
    page.evaluate(
        "async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash'); await showFlashReview();}"
    )
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


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright not installed")
def test_live_debug_record_button_sends_bounded_duration(gui_page):
    page = gui_page
    packet = {
        "running": True,
        "server_time_s": 1.1,
        "estimate": {"status": "tracking", "tracks": [{"role": "object", "name": "red hand"}]},
    }
    controls = []
    page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json=packet))

    def control(route):
        controls.append(route.request.post_data_json)
        route.fulfill(json={"id": "ack"})

    page.route("**/api/run/flash/tracking/control", control)
    page.evaluate(
        "async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash'); await showFlashReview();}"
    )
    page.get_by_role("button", name="Live tracking · camera only →").click()
    page.locator("#flash-trace-seconds").fill("7")
    page.get_by_role("button", name="Record diagnostic trace").click()
    page.wait_for_function(
        "document.getElementById('flash-tracking-message').textContent.includes('Starting RGB-D')"
    )
    assert controls[-1]["action"] == "record" and controls[-1]["duration_s"] == 7


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright not installed")
def test_object_closeup_uses_estimate_frame_and_observed_mask(gui_page):
    import base64

    import cv2
    import numpy as np

    page = gui_page
    rgb = np.zeros((100, 200, 3), np.uint8)
    rgb[:] = (0, 0, 255)
    rgb[30:51, 110:131] = (255, 0, 0)
    _, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    frame = {
        "frame_id": 42,
        "captured_s": 1.0,
        "width": 200,
        "height": 100,
        "image": "data:image/png;base64," + base64.b64encode(png).decode(),
    }
    packet = {
        "running": True,
        "server_time_s": 1.1,
        "preview": {**frame, "frame_id": 43},
        "estimate": {
            "status": "fitted",
            "frame": frame,
            "contours": [[[110, 30], [130, 30], [130, 50], [110, 50]]],
        },
    }
    page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json=packet))
    page.evaluate(
        "async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash'); await showFlashReview();}"
    )
    page.get_by_role("button", name="Live tracking · camera only →").click()
    page.wait_for_function("document.getElementById('flash-surface-local')?.dataset.frameId === '42'")
    result = page.evaluate("""() => {
        const full = document.getElementById('flash-surface-estimate');
        const local = document.getElementById('flash-surface-local');
        const rgb = (canvas,x,y) => Array.from(canvas.getContext('2d').getImageData(x,y,1,1).data).slice(0,3);
        return {fullFrame:full.dataset.frameId, localFrame:local.dataset.frameId,
            fullCenter:rgb(full,100,50), localCenter:rgb(local,100,50),
            localBackground:rgb(local,10,10), caption:document.getElementById('flash-surface-local-caption').textContent,
            crop:flashTrackingObjectCrop(flashTrackingSnapshot.estimate)};
    }""")
    assert result["fullFrame"] == result["localFrame"] == "42"
    assert result["fullCenter"] == [0, 0, 255]
    assert result["localCenter"] == [255, 0, 0]
    assert result["localBackground"] == [0, 0, 255]
    assert "frame 42" in result["caption"]
    assert result["crop"]["source"] == "observed"


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright not installed")
def test_multi_object_list_changes_closeup_and_3d_without_changing_source_frame(gui_page):
    import base64

    import cv2
    import numpy as np

    page = gui_page
    rgb = np.zeros((100, 200, 3), np.uint8)
    rgb[:] = (0, 0, 255)
    rgb[20:61, 20:61] = (255, 0, 0)
    rgb[20:61, 140:181] = (0, 255, 0)
    _, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    frame = {
        "frame_id": 52,
        "captured_s": 1.0,
        "width": 200,
        "height": 100,
        "image": "data:image/png;base64," + base64.b64encode(png).decode(),
    }
    identity = np.eye(4).tolist()

    def obj(name, box, shape):
        x0, y0, x1, y1 = box
        return {
            "name": name,
            "status": "reference",
            "accepted": True,
            "contours": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]],
            "reference_px": [[(x0 + x1) // 2, (y0 + y1) // 2]],
            "anchors_px": [[x0 + 10, y0 + 10], [x0 + 15, y0 + 10], [x0 + 10, y0 + 15]],
            "axes_px": [[x0 + 20, y0 + 10], [x0 + 10, y0 + 20], [x0 + 20, y0 + 5]],
            "camera_from_reference": identity,
            "reference_m": shape,
            "observed_m": shape,
        }

    packet = {
        "running": True,
        "session_id": "multi-test",
        "server_time_s": 1.1,
        "preview": frame,
        "estimate": {
            "status": "tracking",
            "frame": frame,
            "tracks": [
                {"role": "object", "name": "red", "status": "reference"},
                {"role": "object_2", "name": "green", "status": "reference"},
            ],
            "objects": {
                "object": obj("red", (20, 20, 60, 60), [[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]]),
                "object_2": obj("green", (140, 20, 180, 60), [[0, 0, 1], [0.2, 0, 1], [0, 0.03, 1]]),
            },
        },
    }
    page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json=packet))
    page.evaluate(
        "async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash'); await showFlashReview();}"
    )
    page.get_by_role("button", name="Live tracking · camera only →").click()
    page.wait_for_function("document.getElementById('flash-surface-local')?.dataset.frameId === '52'")
    page.wait_for_function("""() => {
        const canvas = document.getElementById('flash-surface-estimate');
        const ctx = canvas.getContext('2d');
        return [40,160].every(x =>
            Array.from(ctx.getImageData(x,40,1,1).data).slice(0,3).every(channel => channel === 255));
    }""")
    before = page.evaluate("""() => ({
        color:Array.from(document.getElementById('flash-surface-local').getContext('2d').getImageData(60,70,1,1).data).slice(0,3),
        cloud:document.getElementById('flash-surface-cloud').toDataURL(),
        full:document.getElementById('flash-surface-estimate').dataset.frameId,
        fullImage:document.getElementById('flash-surface-estimate').toDataURL(),
    })""")
    stale_alpha = page.evaluate("""() => {
        document.querySelector('#flash-tracked-objects button[data-role="object_2"]').click();
        return document.getElementById('flash-surface-local').getContext('2d').getImageData(130,70,1,1).data[3];
    }""")
    assert stale_alpha == 0  # never label the previous object's crop as the new one
    page.wait_for_function("""() => flashTrackingFocusRole === 'object_2' &&
            document.getElementById('flash-surface-local').getContext('2d').getImageData(130,70,1,1).data[3] === 255""")
    after = page.evaluate("""() => ({
        color:Array.from(document.getElementById('flash-surface-local').getContext('2d').getImageData(130,70,1,1).data).slice(0,3),
        cloud:document.getElementById('flash-surface-cloud').toDataURL(),
        full:document.getElementById('flash-surface-estimate').dataset.frameId,
        fullImage:document.getElementById('flash-surface-estimate').toDataURL(),
        selected:document.querySelector('#flash-tracked-objects button[data-role="object_2"]').getAttribute('aria-pressed'),
    })""")
    assert before["color"][0] > before["color"][1] and before["color"][0] > before["color"][2]
    assert after["color"][1] > after["color"][0] and after["color"][1] > after["color"][2]
    assert before["full"] == after["full"] == "52"
    assert before["fullImage"] == after["fullImage"]
    assert before["cloud"] != after["cloud"]
    assert after["selected"] == "true"


def test_debugger_exports_the_common_configuration_and_migrates_old_payloads():
    request = flash_tracking.TrackingStart(sam_resolution=448, observation={"mask_margin_px": 3})
    saved = request.model_dump()
    assert "observation" not in saved and "sam_resolution" not in saved
    assert saved["vision_pipeline"]["sam_resolution"] == 448
    assert saved["vision_pipeline"]["observation"]["mask_margin_px"] == 3
    from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

    assert VisionPipelineConfig.from_dict(saved["vision_pipeline"]) == request.vision_pipeline
    from lerobot.policies.flash.observation_processing import default_processing_config

    assert saved["causal_tracking"] == default_processing_config()["causal_tracking"]
    assert saved["group_fallback"] is False
    assert flash_tracking.TrackingStart(group_fallback=True).model_dump()["group_fallback"] is True
    with pytest.raises(ValidationError, match="causal tracking keys differ"):
        flash_tracking.TrackingStart(group_fallback=True, causal_tracking={"history_frames": 8})
    with pytest.raises(ValidationError):
        flash_tracking.TrackingStart(vision_pipeline={"unknown_setting": 1})
    with pytest.raises(ValidationError, match="only once"):
        flash_tracking.TrackingStart(observation={}, vision_pipeline={"observation": {}})


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright not installed")
def test_live_detection_correction_toggle_reaches_the_worker_configuration(gui_page):
    page = gui_page
    submitted = []
    page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json={"running": False}))

    def capture(route):
        submitted.append(route.request.post_data_json)
        route.fulfill(json={"status": "starting"})

    page.route("**/api/run/flash/tracking/start", capture)
    page.evaluate(
        "async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash'); await showFlashReview();}"
    )
    page.get_by_role("button", name="Live tracking · camera only →").click()
    page.locator("#flash-detection-correction").wait_for()
    page.locator("#flash-detection-correction").check()
    page.get_by_role("button", name="Open camera").click()
    page.wait_for_function(
        "() => document.getElementById('flash-tracking-message').textContent.includes('Opening')"
    )
    assert len(submitted) == 1
    request = flash_tracking.TrackingStart.model_validate(submitted[0])
    assert request.vision_pipeline.detection_correction.enabled is True
    assert request.vision_pipeline.detection_correction.control(offline=False)["interval_frames"] == 60


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright not installed")
def test_live_group_toggle_reaches_the_worker_configuration(gui_page):
    page = gui_page
    submitted = []
    page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json={"running": False}))
    page.route(
        "**/api/run/flash/tracking/start",
        lambda route: (
            submitted.append(route.request.post_data_json),
            route.fulfill(json={"status": "starting"}),
        ),
    )
    page.evaluate(
        "async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash'); await showFlashReview();}"
    )
    page.get_by_role("button", name="Live tracking · camera only →").click()
    page.locator("#flash-group-fallback").check()
    page.get_by_role("button", name="Open camera").click()
    page.wait_for_function(
        "() => document.getElementById('flash-tracking-message').textContent.includes('Opening')"
    )
    assert flash_tracking.TrackingStart.model_validate(submitted[0]).group_fallback is True


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright not installed")
def test_group_pose_is_drawn_on_its_own_camera_frame(gui_page):
    import base64

    import cv2
    import numpy as np

    page = gui_page
    _, png = cv2.imencode(".png", np.zeros((80, 100, 3), np.uint8))
    image = "data:image/png;base64," + base64.b64encode(png).decode()

    def frame(index):
        return {"frame_id": index, "captured_s": 1.0, "width": 100, "height": 80, "image": image}

    config = flash_tracking.TrackingStart(group_fallback=True).model_dump()
    packet = {
        "running": True,
        "server_time_s": 1.1,
        "config": config,
        "preview": frame(43),
        "estimate": {
            "status": "tracking",
            "frame": frame(40),
            "objects": {"object": {"name": "object"}},
            "group_frame": frame(42),
            "group_objects": {
                "object": {"name": "object", "group_current_frame": True, "group_reference_px": [[40, 40]]}
            },
        },
    }
    page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json=packet))
    page.evaluate(
        "async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash'); await showFlashReview();}"
    )
    page.get_by_role("button", name="Live tracking · camera only →").click()
    page.wait_for_function("document.getElementById('flash-surface-group')?.dataset.frameId === '42'")
    result = page.evaluate("""() => ({
        direct: document.getElementById('flash-surface-estimate').dataset.frameId,
        group: document.getElementById('flash-surface-group').dataset.frameId,
        color: Array.from(document.getElementById('flash-surface-group').getContext('2d').getImageData(40,40,1,1).data).slice(0,3),
    })""")
    assert result == {"direct": "40", "group": "42", "color": [255, 191, 77]}


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright not installed")
def test_object_inspection_removes_pose_but_scene_toggle_restores_it(gui_page):
    page = gui_page
    page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json={"running": False}))
    page.evaluate(
        "async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash'); await showFlashReview();}"
    )
    page.get_by_role("button", name="Live tracking · camera only →").click()
    page.wait_for_selector("#flash-surface-cloud")
    page.wait_for_function("flashTrackingCloudViewer !== null")
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
        return {same:before===after, unchangedFraming, sceneChanged:scene!==after};
    }""")
    assert result == {"same": True, "unchangedFraming": True, "sceneChanged": True}
    before = page.evaluate("() => flashTrackingCloudViewer.camera.position.toArray()")
    canvas = page.locator("#flash-surface-cloud")
    box = canvas.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + 80, box["y"] + box["height"] / 2 + 40)
    page.mouse.up()
    after = page.evaluate("() => flashTrackingCloudViewer.camera.position.toArray()")
    assert before != after
    assert page.evaluate(
        "() => sessionStorage.getItem(`flashSurfaceView:${flashTrackingCloudViewer.identity}`) !== null"
    )
    target_before = page.evaluate("() => flashTrackingCloudViewer.controls.target.toArray()")
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down(button="right")
    page.mouse.move(box["x"] + box["width"] / 2 + 50, box["y"] + box["height"] / 2 + 20)
    page.mouse.up(button="right")
    target_after = page.evaluate("() => flashTrackingCloudViewer.controls.target.toArray()")
    assert target_before != target_after
    distance_before = page.evaluate("""() => flashTrackingCloudViewer.camera.position.distanceTo(
        flashTrackingCloudViewer.controls.target)""")
    page.mouse.wheel(0, -300)
    distance_after = page.evaluate("""() => flashTrackingCloudViewer.camera.position.distanceTo(
        flashTrackingCloudViewer.controls.target)""")
    assert distance_after < distance_before
    page.get_by_role("button", name="Reset 3D view").click()
    reset_target = page.evaluate("() => flashTrackingCloudViewer.controls.target.toArray()")
    assert reset_target != target_after
    assert page.evaluate(
        "() => sessionStorage.getItem(`flashSurfaceView:${flashTrackingCloudViewer.identity}`) === null"
    )


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
    page.evaluate(
        "async () => {switchTab('run'); await runTabInit(); await selectWorkflow('flash'); await showFlashReview();}"
    )
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
    req = flash_tracking.TrackingStart(engine="experiment", pipeline_settings=settings)
    asyncio.run(flash_tracking.start(req))
    assert captured["args"][-2] == "lerobot.policies.flash.live_surface_worker"
    assert captured["config"]["pipeline_settings"] == settings
    assert json.loads((tmp_path / "config.json").read_text())["pipeline_settings"] == json.loads(
        json.dumps(settings)
    )
    settings["group_experiment"] = None
    with pytest.raises(ValidationError, match="requires group"):
        flash_tracking.TrackingStart(engine="experiment", pipeline_settings=settings)


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
