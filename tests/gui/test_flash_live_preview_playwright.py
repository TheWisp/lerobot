"""The raw RGB-D preview works before an object is selected and never swaps frames."""

from __future__ import annotations

import base64
import json
import struct
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.requires_playwright


def test_rejected_fit_is_opt_in_for_live_and_recording_replay(gui_page):
    page = gui_page
    page.get_by_role("button", name="FLASH", exact=True).click()
    page.evaluate("""async () => {
        const image = document.createElement('canvas'); image.width = image.height = 32;
        const context = image.getContext('2d'); context.fillStyle = '#153f6a';
        context.fillRect(0, 0, 32, 32);
        flashLive.visible = true; flashLive.view = '2d'; setFlashSource(true);
        flashLive.bitmap = await createImageBitmap(image);
        const pose = [[1,0,0,0.1],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
        flashLive.overlay = {camera:{intrinsics:[[32,0,16],[0,32,16],[0,0,1]]},
            objects:{object:{pose:null,status:'lost',surface_used:[[0,0,1]],
                axis_points:[[0,0,1],[0.03,0,1],[0,0.03,1],[0,0,1.03]],
                rejected_poses:{fit:pose,prediction:null},contours:[]}},scene_group_tracks:[]};
        await drawFlashLive();
    }""")
    pixel = "Array.from(document.getElementById('flash-live-image').getContext('2d').getImageData(19,16,1,1).data)"
    background = gui_page.evaluate(pixel)
    assert not page.locator("#flash-live-rejected").is_checked()
    page.check("#flash-live-rejected")
    page.evaluate("drawFlashLive()")
    assert page.evaluate(pixel) != background
    # The same overlay packet is displayed over a saved episode frame.
    page.evaluate("""() => {
        flashLive.episodePlayer = {}; flashLive.episodeImage = flashLive.bitmap;
        const object = flashLive.overlay.objects.object;
        object.diagnostics = {fit: {rejected_camera_from_reference: object.rejected_poses.fit}};
        delete object.rejected_poses; // A recording saved before the display field existed.
        drawFlashLive();
    }""")
    assert page.evaluate(pixel) != background
    page.uncheck("#flash-live-rejected")
    page.evaluate("drawFlashLive()")
    assert page.evaluate(pixel) == background


def test_unselected_camera_recording_displays_raw_rgb_and_depth(gui_page):
    from lerobot.policies.flash.perception import RGBDFrame
    from lerobot.policies.flash.tracking_history import capture_packet

    rgb = np.full((4, 4, 3), [21, 92, 173], np.uint8)
    depth = np.full((4, 4), 427, np.uint16)
    frame = RGBDFrame(7, rgb, depth, np.eye(3), 0.001, np.eye(4), 7.0, 7.0, "test")
    encoded = base64.b64encode(capture_packet(frame)).decode()
    gui_page.evaluate("flashLive.visible = true; flashLive.view = '2d'")
    gui_page.evaluate(
        """async data => {
            const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0));
            await displayFlashLivePacket(bytes.buffer, flashLive.generation, 0);
        }""",
        encoded,
    )
    assert gui_page.evaluate("flashLive.packet.depth[0]") == pytest.approx(0.427, abs=1e-6)
    assert gui_page.evaluate(
        "Array.from(document.getElementById('flash-live-image').getContext('2d').getImageData(0,0,1,1).data)"
    ) == [21, 92, 173, 255]
    assert gui_page.evaluate("Object.keys(flashLive.packet.objects).length") == 0


def test_unselected_model_preview_exposes_sensor_and_model_depth(gui_page):
    from dataclasses import replace

    from lerobot.gui.flash_live_scene import encode_rgbd_scene
    from lerobot.policies.flash.perception import RGBDFrame

    sensor = RGBDFrame(
        9,
        np.full((8, 8, 3), 80, np.uint8),
        np.full((8, 8), 427, np.uint16),
        np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]]),
        0.001,
        np.eye(4),
        9.0,
        9.0,
        "test",
    )
    model = replace(sensor, depth=sensor.depth + 10, depth_before_filters=sensor.depth)
    packet = encode_rgbd_scene(
        model,
        {
            "frame": 9,
            "objects": {},
            "scene_group_tracks": [],
            "depth_source": "foundation_stereo",
            "preview_model": True,
        },
        rgb_format="jpeg",
    )
    gui_page.route(
        "**/api/run/flash/tracking/state",
        lambda route: route.fulfill(
            json={
                "running": True,
                "config": {"engine": "experiment", "depth_source": "foundation_stereo"},
                "estimate": {
                    "status": "preview_model",
                    "frame_id": 9,
                    "depth_source": "foundation_stereo",
                    "timings": {"depth_ms": 37},
                },
            }
        ),
    )
    gui_page.route(
        "**/api/run/flash/tracking/scene",
        lambda route: route.fulfill(body=packet, content_type="application/octet-stream"),
    )
    gui_page.route(
        "**/api/run/flash/tracking/history*",
        lambda route: route.fulfill(json={"recording": "recent", "recordings": [], "frames": [], "bytes": 0}),
    )
    gui_page.evaluate(
        "flashLive.visible = true; flashLive.view = '2d'; "
        "document.getElementById('tab-flash').classList.add('active'); setFlashSource(true); void pollFlashLive()"
    )
    gui_page.wait_for_function("flashLive.processedMode === 'preview_model' && flashLive.packet?.frame === 9")
    gui_page.evaluate("setFlashLiveView('depth')")
    assert gui_page.evaluate("flashLive.packet.depth[0]") == pytest.approx(0.437, abs=1e-6)
    assert gui_page.evaluate("flashLive.packet.depthBeforeFilters[0]") == pytest.approx(0.427, abs=1e-6)
    assert "Fast-FoundationStereo" in gui_page.locator("#flash-live-depth-info").inner_text()
    assert (
        gui_page.locator('#flash-live-stream-mode button[data-mode="processed"]').get_attribute(
            "aria-pressed"
        )
        == "true"
    )


def test_live_processed_image_arrives_before_the_pose_result(gui_page):
    from lerobot.gui.flash_live_scene import encode_rgbd_scene
    from lerobot.policies.flash.perception import RGBDFrame

    measured = RGBDFrame(
        4,
        np.full((8, 8, 3), 60, np.uint8),
        np.full((8, 8), 427, np.uint16),
        np.eye(3),
        0.001,
        np.eye(4),
        4.0,
        4.0,
        "test",
    )
    masks = {"part": [[9, 2]]}
    pending = encode_rgbd_scene(
        measured,
        {
            "frame": 4,
            "time_s": 4.0,
            "depth_source": "sensor",
            "processed_input": True,
            "masks_rle": masks,
            "objects": {},
            "scene_group_tracks": [],
        },
        rgb_format="jpeg",
    )
    completed = encode_rgbd_scene(
        measured,
        {
            "frame": 4,
            "time_s": 4.0,
            "depth_source": "sensor",
            "masks_rle": masks,
            "result": {"frame": 4},
            "objects": {
                "part": {"name": "part", "status": "observed", "contours": [], "surface": [], "pose": None}
            },
            "scene_group_tracks": [],
        },
        rgb_format="jpeg",
    )
    stage = {"status": "processed"}
    gui_page.route(
        "**/api/run/flash/tracking/state",
        lambda route: route.fulfill(
            json={
                "running": True,
                "config": {"engine": "experiment"},
                "estimate": {"status": stage["status"], "frame_id": 4},
            }
        ),
    )
    gui_page.route(
        "**/api/run/flash/tracking/scene",
        lambda route: route.fulfill(
            body=pending if stage["status"] == "processed" else completed,
            content_type="application/octet-stream",
        ),
    )
    gui_page.route(
        "**/api/run/flash/tracking/history*",
        lambda route: route.fulfill(json={"recording": "recent", "recordings": [], "frames": [], "bytes": 0}),
    )
    gui_page.evaluate(
        "flashLive.visible=true; flashLive.view='2d'; setFlashSource(true); void pollFlashLive()"
    )
    gui_page.wait_for_function("flashLive.processedMode === 'processed' && flashLive.packet?.frame === 4")
    assert gui_page.evaluate("flashLive.packet.masks_rle.part[0]") == [9, 2]
    assert gui_page.evaluate("flashLive.overlay === null")
    stage["status"] = "tracking"
    gui_page.wait_for_function("flashLive.processedMode === 'tracking' && flashLive.overlay?.frame === 4")


def test_filter_created_depth_toggle_marks_only_new_values(gui_page):
    from lerobot.policies.flash.perception import RGBDFrame
    from lerobot.policies.flash.tracking_history import capture_packet

    rgb = np.full((8, 8, 3), 80, np.uint8)
    before = np.full((8, 8), 427, np.uint16)
    before[2, 3] = 0
    after = before.copy()
    after[2, 3] = 430
    frame = RGBDFrame(
        7,
        rgb,
        after,
        np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]]),
        0.001,
        np.eye(4),
        7.0,
        7.0,
        "test",
        {"hole_filling": True},
        1,
        before,
    )
    encoded = base64.b64encode(capture_packet(frame)).decode()
    gui_page.evaluate(
        "flashLive.visible = true; flashLive.view = '2d'; document.getElementById('tab-flash').classList.add('active'); setFlashSource(true)"
    )
    gui_page.evaluate(
        """async data => {
            const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0));
            await displayFlashLivePacket(bytes.buffer, flashLive.generation, 0);
            setFlashLiveView('depth');
        }""",
        encoded,
    )
    assert gui_page.evaluate("flashLive.packet.depthBeforeFilters[2*8+3]") == 0
    assert not gui_page.evaluate("document.getElementById('flash-live-filled-control').hidden")
    gui_page.evaluate("document.getElementById('flash-live-filled').checked = true; drawFlashLive()")
    assert gui_page.evaluate(
        "Array.from(document.getElementById('flash-live-image').getContext('2d').getImageData(3,2,1,1).data)"
    ) == [255, 89, 191, 255]
    assert gui_page.evaluate(
        "Array.from(document.getElementById('flash-live-image').getContext('2d').getImageData(4,2,1,1).data)"
    ) != [255, 89, 191, 255]
    gui_page.evaluate("document.getElementById('flash-live-filled').checked = false; drawFlashLive()")
    assert gui_page.evaluate(
        "Array.from(document.getElementById('flash-live-image').getContext('2d').getImageData(3,2,1,1).data)"
    ) != [255, 89, 191, 255]
    gui_page.evaluate("setFlashLiveView('3d')")
    gui_page.wait_for_function("!!flashLive.viewer")
    plain_scene = gui_page.evaluate("document.getElementById('flash-live-scene').toDataURL()")
    gui_page.evaluate("document.getElementById('flash-live-filled').checked = true; drawFlashLive()")
    gui_page.wait_for_timeout(100)
    marked_scene = gui_page.evaluate("document.getElementById('flash-live-scene').toDataURL()")
    assert marked_scene != plain_scene


def test_depth_and_3d_preview_keep_a_matched_frame_and_fixed_color_scale(gui_page):
    from lerobot.gui.flash_live_scene import encode_rgbd_scene
    from lerobot.policies.flash.perception import RGBDFrame
    from lerobot.policies.flash.surface_session_config import defaults

    rgb = np.full((16, 16, 3), (20, 90, 170), dtype=np.uint8)
    ok, png = cv2.imencode(".png", rgb)
    assert ok
    image = "data:image/png;base64," + base64.b64encode(png).decode()
    source = {"frame": 1, "processed_frame": 1, "depth_frame": 1, "objects": 0}
    measured = (np.arange(256, dtype="<u2") + 400).reshape(16, 16)
    measured[0, 1] = 0
    settings = defaults()

    def state(route):
        frame = source["frame"]
        route.fulfill(
            json={
                "running": True,
                "config": {"engine": "experiment", "pipeline_settings": settings, "depth_source": "sensor"},
                "estimate": {"status": "tracking", "frame_id": source["processed_frame"]}
                if source["objects"]
                else {"status": "preview"},
                "preview": {
                    "frame_id": frame,
                    "captured_s": float(frame),
                    "width": 16,
                    "height": 16,
                    "intrinsics": [[16, 0, 8], [0, 16, 8], [0, 0, 1]],
                    "depth_m_per_unit": 0.001,
                    "depth_processing": {"spatial": True, "temporal": False, "hole_filling": False},
                    "depth_filter_revision": 5,
                    "image": image,
                },
            }
        )

    def depth(route):
        shifted = measured.copy()
        shifted[shifted > 0] += 5 * (source["depth_frame"] - 1)
        packet = struct.pack("<I", source["depth_frame"]) + shifted.tobytes()
        route.fulfill(body=packet, content_type="application/octet-stream")

    def control(route):
        source["objects"] += 1
        source["frame"] += 1
        source["processed_frame"] = source["frame"]
        route.fulfill(json={"id": "ok"})

    def scene(route):
        frame_id = source["processed_frame"]
        objects = {}
        for i in range(source["objects"]):
            objects[f"object-{i}"] = {
                "name": f"object-{i}",
                "status": "observed",
                "pose": [[1, 0, 0, 0.02 * (frame_id - 2)], [0, 1, 0, 0], [0, 0, 1, 0.45], [0, 0, 0, 1]],
                "surface": [[-0.01, 0, 0], [0.01, 0, 0], [0, 0.01, 0]],
                "axis_points": [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
                "contours": [],
            }
        frame = RGBDFrame(
            frame_id,
            rgb,
            measured,
            np.array([[16, 0, 8], [0, 16, 8], [0, 0, 1]]),
            0.001,
            np.eye(4),
            float(frame_id),
            float(frame_id),
            "test",
        )
        packet = encode_rgbd_scene(
            frame, {"frame": frame_id, "objects": objects, "scene_group_tracks": []}, rgb_format="jpeg"
        )
        route.fulfill(body=packet, content_type="application/octet-stream")

    gui_page.route("**/api/run/flash/tracking/state", state)
    gui_page.route("**/api/run/flash/tracking/preview-depth", depth)
    gui_page.route("**/api/run/flash/tracking/control", control)
    gui_page.route("**/api/run/flash/tracking/scene", scene)
    gui_page.route("**/api/run/flash/tracking/cameras", lambda route: route.fulfill(json={"cameras": []}))
    gui_page.route(
        "**/api/run/flash/tracking/history*",
        lambda route: route.fulfill(json={"recording": "recent", "recordings": [], "frames": [], "bytes": 0}),
    )
    gui_page.get_by_role("button", name="FLASH", exact=True).click()
    gui_page.locator("#flash-source-live").click()
    gui_page.locator('#flash-live-view-mode button[data-mode="3d"]').click()
    gui_page.wait_for_function("flashLive.viewer !== null && flashLive.frame?.frame_id === 1")
    assert gui_page.evaluate(
        "flashLive.packet === null && !document.getElementById('flash-live-scene').hidden"
    ), "the RGB-D scene must work with no selected object"
    assert (
        "aligned to color · Depth: RealSense depth · Camera filters: spatial · revision 5"
        in gui_page.locator("#flash-live-depth-info").inner_text()
    )

    gui_page.locator('#flash-live-view-mode button[data-mode="depth"]').click()
    gui_page.wait_for_function("flashLive.depthColorRange !== null")
    color_range = gui_page.evaluate("flashLive.depthColorRange")
    source.update(frame=2, depth_frame=1)  # depth belongs to an older RGB image
    gui_page.wait_for_timeout(400)
    assert gui_page.evaluate("flashLive.frame.frame_id") == 1
    assert gui_page.locator("#flash-live-depth-info").is_visible()

    source["depth_frame"] = 2
    gui_page.wait_for_function("flashLive.frame?.frame_id === 2")
    assert gui_page.evaluate("flashLive.depthColorRange") == color_range

    # Choose objects by name while staying in 3D. The same scene viewer must
    # replace the raw preview with processed surfaces and update on later frames.
    gui_page.locator('#flash-live-view-mode button[data-mode="3d"]').click()
    gui_page.evaluate("""() => {
        const viewer = flashLive.viewer, update = viewer.update.bind(viewer);
        viewer.stableMarker = 'same viewer';
        viewer.seen = [];
        viewer.update = (...args) => {
            viewer.seen.push(args[3].filter(d => d.layer === 'estimate').map(d => d.pose[0][3]));
            return update(...args);
        };
    }""")
    gui_page.locator("#flash-live-name").fill("first object")
    gui_page.get_by_role("button", name="Track by name", exact=True).click()
    gui_page.wait_for_function("flashLive.processed === 3 && flashLive.viewer.seen.at(-1)?.length === 1")
    assert gui_page.locator("#flash-live-scene").is_visible()
    assert gui_page.evaluate("flashLive.viewer.seen.at(-1)") == [0.02]

    gui_page.locator("#flash-live-name").fill("second object")
    gui_page.get_by_role("button", name="Track by name", exact=True).click()
    gui_page.wait_for_function("flashLive.processed === 4 && flashLive.viewer.seen.at(-1)?.length === 2")
    assert gui_page.evaluate("flashLive.viewer.seen.at(-1)") == [0.04, 0.04]
    assert gui_page.evaluate("flashLive.viewer.stableMarker") == "same viewer"

    # The processed view holds its last completed frame. The raw view follows
    # new camera captures without presenting an older fit as a new result.
    source.update(frame=5, depth_frame=5)
    gui_page.wait_for_timeout(400)
    assert gui_page.evaluate("flashLive.frame?.frame_id") == 4
    gui_page.locator('#flash-live-stream-mode button[data-mode="raw"]').click()
    gui_page.wait_for_function("flashLive.frame?.frame_id === 5")
    assert gui_page.evaluate("flashLive.streamMode === 'raw'")


def test_replay_holds_last_processed_overlay_and_recordings_can_be_renamed(gui_page):
    from lerobot.gui.flash_live_scene import encode_rgbd_scene
    from lerobot.policies.flash.perception import RGBDFrame
    from lerobot.policies.flash.tracking_history import capture_packet

    rgb = np.full((24, 24, 3), 40, np.uint8)
    depth = np.full((24, 24), 450, np.uint16)
    frames = [
        RGBDFrame(
            i,
            rgb,
            depth,
            np.array([[24, 0, 12], [0, 24, 12], [0, 0, 1]]),
            0.001,
            np.eye(4),
            float(i),
            float(i),
            "test",
        )
        for i in range(3)
    ]
    tracked = encode_rgbd_scene(
        frames[1],
        {
            "frame": 1,
            "time_s": 1.0,
            "result": {"frame": 1},
            "objects": {
                "part": {
                    "name": "part",
                    "status": "observed",
                    "pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.45], [0, 0, 0, 1]],
                    "surface": [[0, 0, 0]],
                    "axis_points": [],
                    "contours": [[[6, 6], [18, 6], [18, 18], [6, 18]]],
                }
            },
            "scene_group_tracks": [],
        },
    )
    packets = {0: capture_packet(frames[0]), 1: tracked, 2: capture_packet(frames[2])}
    recorded = [
        {
            "frame": i,
            "time_s": float(i),
            "processed_ready_time_s": float(i) + 0.1,
            "objects": {"part": {"name": "part"}} if i == 1 else {},
        }
        for i in range(3)
    ]
    saved_name = {"value": None}

    def history(route):
        url = urlparse(route.request.url)
        if url.path.endswith("/history/scene"):
            index = int(parse_qs(url.query)["frame"][0])
            route.fulfill(body=packets[index], content_type="application/octet-stream")
        elif url.path.endswith("/history/name"):
            saved_name["value"] = json.loads(route.request.post_data)["name"]
            route.fulfill(json={"id": "saved-" + "a" * 32, "name": saved_name["value"]})
        else:
            route.fulfill(
                json={
                    "recording": "recent",
                    "recordings": [
                        {
                            "id": "saved-" + "a" * 32,
                            "name": saved_name["value"],
                            "objects": ["part"],
                            "saved_at": 1.0,
                            "frames": 3,
                        }
                    ],
                    "frames": recorded,
                    "recent_frames": 3,
                    "bytes": 100,
                }
            )

    gui_page.route("**/api/run/flash/experiments", lambda route: route.fulfill(json=[]))
    gui_page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json={"running": False}))
    gui_page.route("**/api/run/flash/tracking/cameras", lambda route: route.fulfill(json={"cameras": []}))
    gui_page.route("**/api/run/flash/tracking/history*", history)
    gui_page.route("**/api/run/flash/tracking/history/**", history)
    gui_page.get_by_role("button", name="FLASH", exact=True).click()
    gui_page.locator("#flash-source-live").click()
    gui_page.wait_for_function("flashLive.packet?.frame === 2 && flashLive.overlay?.frame === 1")
    preferred = gui_page.evaluate(
        "async () => (await (await fetch('/api/run/flash/tracking/shared-defaults')).json()).default_depth_source"
    )
    assert gui_page.locator("#flash-live-depth-source").get_attribute("data-mode") == preferred
    assert "part: observed" in gui_page.locator("#flash-live-objects").inner_text()
    assert gui_page.evaluate(
        "Array.from(liveElement('image').getContext('2d').getImageData(6,6,1,1).data)"
    ) != [40, 40, 40, 255]
    gui_page.evaluate("void loadFlashHistory(0)")
    gui_page.wait_for_function("flashLive.packet?.frame === 0 && flashLive.overlay === null")
    gui_page.evaluate("void loadFlashHistory(2)")
    gui_page.wait_for_function("flashLive.packet?.frame === 2 && flashLive.overlay?.frame === 1")
    # The camera is stopped. Polling it must not replace recording status or
    # prevent the retained frames from playing.
    gui_page.evaluate("void pollFlashLive()")
    assert gui_page.locator("#flash-live-status").inner_text() == "Recording replay"
    gui_page.locator("#flash-live-play").click()
    gui_page.wait_for_function("flashLive.packet?.frame === 0 && flashLive.playing")
    gui_page.wait_for_function("flashLive.packet?.frame === 2 && !flashLive.playing", timeout=5000)
    assert gui_page.locator("#flash-live-status").inner_text() == "Recording replay"

    gui_page.get_by_role("button", name="Rename recording", exact=False).click()
    gui_page.get_by_label("Recording name").fill("Controller turn")
    gui_page.locator(".flash-live-recording-row").last.get_by_role("button", name="Save").click()
    gui_page.get_by_role("button", name="Controller turn · 3 frames").wait_for()
    assert (
        gui_page.locator("#flash-live-save-history").evaluate(
            "e => e.parentElement.querySelector('.flash-live-delete').textContent"
        )
        == "Clear"
    )
    gui_page.locator("#flash-live-sidebar").evaluate("e => e.style.width = '200px'")
    rows = gui_page.locator(".flash-live-recording-row")
    save_box = rows.nth(0).locator("button").nth(1).bounding_box()
    rename_box = rows.nth(1).locator("button").nth(1).bounding_box()
    assert save_box["x"] == pytest.approx(rename_box["x"], abs=1)
    assert (
        rows.nth(0).bounding_box()["width"] <= gui_page.locator("#flash-live-sidebar").bounding_box()["width"]
    )
    assert gui_page.locator("#flash-live-start").inner_text() == "Start"
    assert gui_page.locator("#flash-live-stop").inner_text() == "Stop"


def test_raw_and_processed_timelines_replay_recent_and_saved_clip(gui_page, tmp_path, monkeypatch):
    from dataclasses import replace

    from lerobot.policies.flash.tracking_history import export_episode, save_window
    from tests.policies.flash.test_shared_image_tracking import frame
    from tests.policies.flash.test_tracking_history import new_history

    recent = new_history(tmp_path / "recent")
    source = [replace(frame(i), source_time_s=10 + i / 30, available_time_s=10 + i / 30) for i in range(3)]
    for captured in source:
        recent.submit_frame(captured)
    recent.queue.join()
    mask = np.zeros(source[2].depth.shape, bool)
    mask[10:20, 30:40] = True
    exact = replace(
        source[2], depth=source[2].depth.astype(np.float32) * 0.001 + 0.00037, depth_m_per_unit=1.0
    )
    timestamps = iter([10.05, 10.24, 10.27])
    with monkeypatch.context() as patch:
        patch.setattr("lerobot.policies.flash.tracking_history.time.perf_counter", lambda: next(timestamps))
        recent.attach_processed_frame(source[0], "sensor")
        recent.attach_processed_frame(exact, "foundation_stereo", complete=False)
        recent.attach_processed_frame(exact, "foundation_stereo", masks={"part": mask})
    recent.queue.join()
    recent.close()
    saved = tmp_path / save_window(recent.directory, tmp_path)
    saved_index = json.loads((saved / "index.json").read_text())
    saved_index["frames"][0]["context_only"] = True
    (saved / "index.json").write_text(json.dumps(saved_index))
    export_episode(saved)
    monkeypatch.setenv("LEROBOT_FLASH_HISTORY_DIR", str(tmp_path))

    gui_page.get_by_role("button", name="FLASH", exact=True).click()
    gui_page.locator("#flash-source-live").click()
    gui_page.wait_for_function("flashLive.historyActive && flashLive.packet?.frame === 2")
    assert gui_page.evaluate("flashLive.historyFrames.map(row=>row.frame)") == [0, 2]
    assert gui_page.evaluate("flashLive.packet.masks_rle.part.length > 0")
    assert gui_page.evaluate("flashLive.packet.modelDepth[0]") == pytest.approx(float(exact.depth[0, 0]))
    mask_pixel = "Array.from(liveElement('image').getContext('2d').getImageData(35,15,1,1).data)"
    processed_pixel = gui_page.evaluate(mask_pixel)

    gui_page.locator('#flash-live-stream-mode button[data-mode="raw"]').click()
    gui_page.wait_for_function("flashLive.historyFrames.length === 3")
    gui_page.evaluate("void loadFlashHistory(2)")
    gui_page.wait_for_function("flashLive.packet?.frame === 2")
    assert gui_page.evaluate(mask_pixel) != processed_pixel
    gui_page.evaluate("void loadFlashHistory(1)")
    gui_page.wait_for_function("flashLive.packet?.frame === 1")
    assert gui_page.evaluate("flashLive.packet.modelDepth === undefined")

    gui_page.locator('#flash-live-stream-mode button[data-mode="processed"]').click()
    gui_page.locator(".flash-live-recording-row").last.locator(".flash-live-recording-select").click()
    gui_page.wait_for_function(
        "flashLive.episodePlayer && flashLive.packet?.source_frame_id === 2", timeout=15000
    )
    assert gui_page.evaluate("flashLive.historyFrames.map(row=>row.frame)") == [2]
    assert gui_page.evaluate("flashLive.rawIndexByFrame.get(2)") == 2
    assert gui_page.evaluate("flashLive.packet.masks_rle.part.length > 0")
    assert gui_page.evaluate("flashLive.episodeModelDepth.depth[0]") == pytest.approx(
        float(exact.depth[0, 0])
    )
    saved_processed_pixel = gui_page.evaluate(mask_pixel)

    gui_page.locator('#flash-live-stream-mode button[data-mode="raw"]').click()
    gui_page.wait_for_function("flashLive.historyFrames.length === 2")
    gui_page.wait_for_function("flashLive.frame?.frame_id === 2")
    assert gui_page.evaluate(mask_pixel) != saved_processed_pixel
    assert gui_page.evaluate("flashLive.streamMode") == "raw"
    assert gui_page.evaluate("flashLive.playbackTimeS") == pytest.approx(10.27)


def test_switching_recording_tracks_preserves_wall_clock_time(gui_page):
    gui_page.get_by_role("button", name="FLASH", exact=True).click()
    gui_page.evaluate("""() => {
        flashLive.visible=true;flashLive.historyActive=true;flashLive.streamMode='raw';
        flashLive.rawFrames=[
            {frame:0,time_s:0,processed_ready_time_s:0.2},
            {frame:1,time_s:3.0,processed_ready_time_s:4.1},
            {frame:2,time_s:3.8,processed_ready_time_s:4.3},
        ];
        flashLive.historyFrames=flashLive.rawFrames;
        flashLive.playbackTimeS=3.8;
        flashLive.frame={frame_id:2};
        liveElement('history-position').value='2';
        flashLive.switchLoads=[];
        window.refreshFlashHistory=async () => {
            flashLive.historyFrames=flashLive.streamMode==='raw'?flashLive.rawFrames:
                flashLive.rawFrames.filter(row=>row.processed_ready_time_s!=null);
        };
        window.loadFlashHistory=async (index,asOf) => {
            flashLive.switchLoads.push({frame:flashLive.historyFrames[index].frame,asOf});
        };
    }""")
    gui_page.evaluate("setFlashLiveStream('processed')")
    assert gui_page.evaluate("flashLive.switchLoads.at(-1)") == {"frame": 0, "asOf": 3.8}
    gui_page.evaluate("setFlashLiveStream('raw')")
    assert gui_page.evaluate("flashLive.switchLoads.at(-1)") == {"frame": 2, "asOf": 3.8}


def test_recent_recording_without_completion_timestamps_still_plays(gui_page):
    from lerobot.gui.flash_live_scene import encode_rgbd_scene
    from lerobot.policies.flash.tracking_history import capture_packet
    from tests.policies.flash.test_shared_image_tracking import frame

    first, second = frame(0), frame(1)
    packets = {
        0: capture_packet(first),
        1: encode_rgbd_scene(
            second,
            {
                "frame": 1,
                "time_s": second.source_time_s,
                "objects": {"part": {"name": "part", "status": "observed"}},
                "scene_group_tracks": [],
            },
        ),
    }

    def history(route):
        url = urlparse(route.request.url)
        if url.path.endswith("/history/scene"):
            frame_id = int(parse_qs(url.query)["frame"][0])
            route.fulfill(body=packets[frame_id], content_type="application/octet-stream")
        else:
            route.fulfill(
                json={
                    "recording": "recent",
                    "recordings": [],
                    "recent_frames": 2,
                    "overlay_indices": [1],
                    "bytes": 0,
                    "frames": [
                        {"frame": 0, "time_s": 0.0, "objects": {}},
                        {"frame": 1, "time_s": 0.1, "objects": {"part": {"name": "part"}}},
                    ],
                }
            )

    gui_page.route("**/api/run/flash/experiments", lambda route: route.fulfill(json=[]))
    gui_page.route("**/api/run/flash/tracking/state", lambda route: route.fulfill(json={"running": False}))
    gui_page.route("**/api/run/flash/tracking/cameras", lambda route: route.fulfill(json={"cameras": []}))
    gui_page.route("**/api/run/flash/tracking/history*", history)
    gui_page.route("**/api/run/flash/tracking/history/**", history)
    gui_page.get_by_role("button", name="FLASH", exact=True).click()
    gui_page.locator("#flash-source-live").click()
    gui_page.wait_for_function("flashLive.packet?.frame === 1 && flashLive.historyActive")
    assert gui_page.evaluate("flashLive.historyFrames.map(row=>row.frame)") == [1]
    assert "completion times unavailable" in gui_page.locator("#flash-live-history-info").inner_text()
    gui_page.locator("#flash-live-play").click()
    gui_page.wait_for_function("!flashLive.playing && flashLive.packet?.frame === 1")
    assert gui_page.locator("#flash-live-status").inner_text() == "Recording replay"


def test_processed_playback_uses_ready_times_and_delays_tracking_result(gui_page):
    gui_page.evaluate("""() => {
        flashLive.visible=true;flashLive.historyActive=true;flashLive.streamMode='processed';
        flashLive.rawFrames=[
            {frame:0,time_s:10,processed_ready_time_s:10.03,objects:{}},
            {frame:1,time_s:10.10,objects:{}},
            {frame:2,time_s:10.20,processed_ready_time_s:10.21,
             tracking_ready_time_s:10.23,objects:{part:{name:'part'}}},
        ];
        flashLive.historyFrames=[flashLive.rawFrames[0],flashLive.rawFrames[2]];
        flashLive.playbackCalls=[];
        window.loadFlashHistory=async (index,asOf) => {
            flashLive.playbackCalls.push({index,asOf,clock:performance.now()});
        };
        liveElement('history-position').value='0';
    }""")
    started = gui_page.evaluate("performance.now()")
    gui_page.evaluate("void playFlashHistory()")
    gui_page.wait_for_function("flashLive.playbackCalls.length >= 1 && !flashLive.playing")
    calls = gui_page.evaluate("flashLive.playbackCalls")
    assert [call["index"] for call in calls] == [1]
    assert calls[0]["clock"] - started == pytest.approx(200, abs=80)
    assert calls[0]["asOf"] >= 10.23
