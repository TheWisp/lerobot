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
                "config": {"engine": "experiment", "pipeline_settings": settings},
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
        packet = encode_rgbd_scene(frame, {"frame": frame_id, "objects": objects, "scene_group_tracks": []})
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

    # The camera advances while the tracker is still processing frame 4. Keep
    # both object surfaces on the newer RGB-D preview until frame 5 is fitted.
    source.update(frame=5, depth_frame=5)
    gui_page.wait_for_function("flashLive.frame?.frame_id === 5 && flashLive.overlay?.frame === 4")
    assert gui_page.evaluate("flashLive.viewer.seen.at(-1)") == [0.04, 0.04]


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
        {"frame": i, "time_s": float(i), "objects": {"part": {"name": "part"}} if i == 1 else {}}
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
    assert "part: observed" in gui_page.locator("#flash-live-objects").inner_text()
    assert gui_page.evaluate(
        "Array.from(liveElement('image').getContext('2d').getImageData(6,6,1,1).data)"
    ) != [40, 40, 40, 255]
    gui_page.evaluate("void loadFlashHistory(0)")
    gui_page.wait_for_function("flashLive.packet?.frame === 0 && flashLive.overlay === null")
    gui_page.evaluate("void loadFlashHistory(2)")
    gui_page.wait_for_function("flashLive.packet?.frame === 2 && flashLive.overlay?.frame === 1")

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
