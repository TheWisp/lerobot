"""Display actual shared-tracker evidence on its recorded source frame."""

import pytest
from playwright.sync_api import expect

from lerobot.gui.api import flash_editor
from lerobot.policies.flash.demo_document import (
    DemoObject,
    atomic_json,
    document_path,
    load_document,
    save_document,
)
from lerobot.policies.flash.demo_tracking import track_demo
from tests.policies.flash.test_demo_tracking import Masks, rgbd_episode  # noqa: F401


def test_tracked_episode_overlay_and_keyframe_edits(gui_page, rgbd_episode, tmp_path):  # noqa: F811 - imported pytest fixture
    atomic_json(
        flash_editor.EDITOR_DIR / "settings.json",
        flash_editor.EditorSettings(depth_source="sensor").model_dump(),
    )
    document, _ = load_document(rgbd_episode, 0)
    document.objects = [DemoObject(id="object_1", name="left"), DemoObject(id="object_2", name="right")]
    document = save_document(rgbd_episode, 0, document)
    folder = document_path(rgbd_episode, 0).with_suffix("")
    output = folder / "test-result"
    track_demo(rgbd_episode, 0, document, output, mask_source=Masks())
    atomic_json(folder / "latest.json", {"directory": output.name})
    atomic_json(
        flash_editor.EDITOR_DIR / "demos.json",
        {
            "recent": {
                "id": "fixture",
                "root": str(rgbd_episode.root),
                "episode": 0,
                "created_at": 1,
                "name": "Last replay",
            },
            "demos": [],
        },
    )
    page = gui_page
    base = page.url.split("#")[0].rstrip("/")
    page.goto(f"{base}/#/flash/editor")
    expect(page.locator("#flash-editor-workspace")).to_be_visible()
    # Observe the rendered boundary, not eventual evidence arrival. A frame
    # with a known pose must never first be shown without that pose's axes.
    page.evaluate("""() => {
        window.demoPaints=[];
        const create=ChunkPlayer.create;
        ChunkPlayer.create=opts=>create({...opts,onPaint(...args){
            opts.onPaint(...args);
            const canvas=document.getElementById('flash-editor-marks');
            const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
            if(flashEditor.view==='2d')demoPaints.push({frame:args[0],evidence:flashEditor.overlay?.frame,
                ink:pixels.some((v,i)=>i%4===3&&v>0)});
        }});
    }""")
    page.locator("#flash-editor-demos button").first.click()
    expect(page.locator("#flash-editor-tracking-status")).to_contain_text("Tracking complete", timeout=30_000)
    expect(page.locator("#flash-editor-save")).to_be_disabled()
    page.wait_for_function("flashEditor.overlay?.frame === 0")
    page.locator("#flash-editor-position").fill("2")
    page.wait_for_function("flashEditor.overlay?.frame === 2")
    assert page.evaluate("flashEditor.overlay.objects.object_1.pose !== null")
    assert page.evaluate("flashEditor.overlay.scene_group_tracks.length > 0")
    assert page.evaluate(
        "Array.from(document.getElementById('flash-editor-marks').getContext('2d').getImageData(0,0,128,96).data).some((v,i)=>i%4===3 && v>0)"
    )
    page.locator("#flash-editor-play").click()
    page.wait_for_function("demoPaints.length >= 10")
    page.locator("#flash-editor-play").click()
    paints = page.evaluate("demoPaints")
    assert {0, 1, 2, 3} <= {value["frame"] for value in paints}
    assert all(value.get("evidence") == value["frame"] and value["ink"] for value in paints)
    page.locator("#flash-editor-position").fill("2")
    page.wait_for_function("flashEditor.overlay?.frame === 2")
    page.locator("#flash-editor-outline").uncheck()
    page.locator("#flash-editor-pose").uncheck()
    # The tiny four-frame tracking fixture does not establish wireframe links.
    # Supply one measured and one carried edge to verify the editor renderer.
    page.evaluate("""() => {
        flashEditor.overlay.scene_group_tracks = [{id:7, pixels:[[20,20],[30,20]],
            camera_xyz:[[0,0,1],[0.01,0,1]], links:[[0,1]],
            predicted_pixels:[[30,30]], predicted_camera_xyz:[[0.01,0.01,1]],
            predicted_links:[[1,2]]}];
        flashEditor.overlay.objects.object_1.group_tracks = {group_id:7};
        drawFlashDemoMarks();
    }""")
    assert page.evaluate(
        "document.getElementById('flash-editor-marks').getContext('2d').getImageData(0,0,128,96).data.some((v,i)=>i%4===3&&v>0)"
    )
    page.locator('#flash-editor-groups button[data-mode="none"]').click()
    assert not page.evaluate(
        "document.getElementById('flash-editor-marks').getContext('2d').getImageData(0,0,128,96).data.some((v,i)=>i%4===3&&v>0)"
    )
    page.locator('#flash-editor-groups button[data-mode="objects"]').click()
    assert page.evaluate(
        "document.getElementById('flash-editor-marks').getContext('2d').getImageData(0,0,128,96).data.some((v,i)=>i%4===3&&v>0)"
    )
    page.locator("#flash-editor-pose").check()
    assert page.evaluate(
        "document.getElementById('flash-editor-marks').getContext('2d').getImageData(0,0,128,96).data.some((v,i)=>i%4===3&&v>0)"
    )
    page.locator("#flash-editor-view-depth").click()
    page.wait_for_function(
        "flashEditor.overlay?.frame === 2 && flashEditor.overlay?.depth?.length === 128*96"
    )
    expect(page.locator("#flash-editor-depth")).to_be_visible()
    expect(page.locator("#flash-editor-depth-info")).to_contain_text("RealSense")
    # Match the browser's consumed float32 depth to the source observation,
    # including real holes, rather than comparing two copies of a JSON header.
    import numpy as np

    from tests.policies.flash.test_shared_image_tracking import frame

    depth = page.evaluate("Array.from(flashEditor.overlay.depth)")
    np.testing.assert_allclose(depth, frame(2).metric_depth().ravel(), atol=1e-6)
    page.locator("#flash-editor-view-3d").click()
    expect(page.locator("#flash-editor-scene")).to_be_visible()
    expect(page.locator("#flash-editor-image-host .rgbd-point-label").filter(has_text="left")).to_be_visible()
    page.locator("#flash-editor-position").fill("3")
    page.wait_for_function("flashEditor.overlay?.frame === 3 && flashEditor.overlay?.depth")
    page.locator("#flash-editor-view-2d").click()
    expect(page.locator("#flash-editor-scene-host")).to_be_hidden()
    expect(page.locator("#flash-editor-image-host .rgbd-point-label").filter(has_text="left")).to_be_hidden()
    page.locator("#flash-editor-position").fill("2")
    page.wait_for_function("flashEditor.overlay?.frame === 2")
    page.locator("#flash-editor-references").get_by_role("button", name="left", exact=True).click()
    page.locator("#flash-editor-mark").click()
    expect(page.locator("#flash-editor-save")).to_be_enabled()
    assert page.evaluate("flashEditor.overlay?.frame === 2")
    assert page.evaluate(
        "document.getElementById('flash-editor-marks').getContext('2d').getImageData(0,0,128,96).data.some((v,i)=>i%4===3&&v>0)"
    )
    page.locator("#flash-editor-save").click()
    expect(page.locator("#flash-editor-save-state")).to_have_text("Saved")
    expect(page.locator("#flash-editor-save")).to_be_disabled()
    # Marking a keyframe does not invalidate already measured poses or demand SAM again.
    result = page.request.get(
        f"{base}/api/run/flash/editor/tracking?dataset_id={rgbd_episode.root}&episode=0&frame=2"
    ).json()
    assert result["stale"] is False
    assert result["frame_result"]["frame"] == 2
    page.wait_for_function("flashEditor.overlay?.frame === 2")
    page.screenshot(path=str(tmp_path / "editor.png"))
    saved, _ = load_document(rgbd_episode, 0)
    saved.objects[0].name = "different object"
    save_document(rgbd_episode, 0, saved)
    result = page.request.get(
        f"{base}/api/run/flash/editor/tracking?dataset_id={rgbd_episode.root}&episode=0&frame=2"
    ).json()
    assert result["stale"] is True
    assert "frame_result" not in result
    url = f"{base}/api/run/flash/editor/display?dataset_id={rgbd_episode.root}&episode=0"
    assert page.request.get(url + "&start=0&count=4").json() == [None] * 4
    assert page.request.get(url + "&start=2&count=4").status == 400
    assert page.request.get(url + "&start=0&count=241").status == 422


@pytest.mark.asyncio
async def test_saved_pipeline_settings_reach_demo_worker(rgbd_episode, monkeypatch):  # noqa: F811
    import json
    from types import SimpleNamespace

    from lerobot.gui.api import run

    document, _ = load_document(rgbd_episode, 0)
    document.objects = [DemoObject(id="object_1", name="left")]
    document = save_document(rgbd_episode, 0, document)
    settings = flash_editor.EditorSettings()
    settings.pipeline_settings["config"]["shape"]["registration"]["backend"] = "open3d_plane"
    atomic_json(flash_editor.EDITOR_DIR / "settings.json", settings.model_dump())
    captured = []

    async def launch(args, **kwargs):
        captured.extend(args)

    monkeypatch.setattr(run, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(run, "_launch_subprocess", launch)
    monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=1, returncode=None))
    monkeypatch.setattr(flash_editor, "_dataset", lambda identity: rgbd_episode)
    monkeypatch.setattr(flash_editor, "_tracking_job", None)
    await flash_editor.track(flash_editor.TrackDemo(dataset_id="test", episode=0, revision=document.revision))
    from pathlib import Path

    config_path = Path(captured[captured.index("--settings") + 1])
    assert json.loads(config_path.read_text()) == json.loads(json.dumps(settings.pipeline_settings))
