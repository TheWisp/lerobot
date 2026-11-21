"""Editor → normal recorder / dataset / player, with no physical hardware."""

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.gui.api import flash_editor, run
from lerobot.policies.flash.demo_document import (
    DemoDocument,
    DemoKeyframe,
    DemoObject,
    atomic_json,
    load_document,
    save_document,
)
from tests.mocks.editorrobot import calibrated_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing", "robot", "camera", "matrix"])
async def test_uncalibrated_demo_fails_before_hardware_launch(tmp_path, monkeypatch, fault):
    import json

    profile = calibrated_profile(tmp_path)
    if fault == "missing":
        profile["fields"].pop("geometry_path")
    elif fault == "robot":
        profile["fields"]["id"] = "different-robot"
    elif fault == "camera":
        profile["cameras"]["scene"]["index_or_path"] = 1
    else:
        path = Path(profile["fields"]["geometry_path"])
        data = json.loads(path.read_text())
        data["cameras"]["scene"]["T_base_cam"] = np.zeros((4, 4)).tolist()
        path.write_text(json.dumps(data))
    launch = AsyncMock(return_value={"pid": 42})
    monkeypatch.setattr(run, "start_record", launch)
    status = await flash_editor.robot_geometry(profile)
    assert status["ready"] is False
    with pytest.raises(HTTPException) as error:
        await flash_editor.record(
            run.RecordRequest(
                robot=profile,
                teleop={"type": "scripted_bimanual_ee", "fields": {}},
                repo_id="test/demo",
                root=str(tmp_path / "recording"),
                single_task="test",
            )
        )
    assert error.value.status_code == 400
    assert error.value.detail == status["reason"]
    launch.assert_not_called()


@pytest.fixture(autouse=True)
def child_observation_namespace(tmp_path, monkeypatch):
    """The real recorder must publish into pytest's private observation stream.

    The suite tags the parent stream at configure; a fresh child interpreter
    does not inherit that monkeypatch. Pass its name without touching the
    developer's running GUI stream (see tests/fixtures/shm_namespace.py).
    """
    from lerobot.robots.obs_stream import SHM_PREFIX

    startup = tmp_path / "child-startup"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text(
        f"import lerobot.robots.obs_stream as stream\nstream.SHM_PREFIX = {SHM_PREFIX!r}\n"
        "import tests.mocks.editorrobot\n"
    )
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(startup), str(root / "src"), str(root)]))


@pytest.fixture
def episode(tmp_path, request):
    dataset = LeRobotDataset.create(
        repo_id="test/editor",
        root=tmp_path / "demo",
        fps=10,
        robot_type="virtual_bi_so107",
        use_videos=True,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["joint.pos", "gripper.pos"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["joint.pos", "gripper.pos"]},
            "observation.images.scene": {
                "dtype": "video",
                "shape": (64, 96, 3),
                "names": ["height", "width", "channels"],
            },
        },
    )
    for frame in range(getattr(request, "param", 10)):
        rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        rgb[:, :, 0] = (20 + frame * 20) % 256
        rgb[15:30, 20:40, 1] = 200
        dataset.add_frame(
            {
                "task": "move the cube",
                "observation.state": np.array([frame, 0], dtype=np.float32),
                "action": np.array([frame, 1], dtype=np.float32),
                "observation.images.scene": rgb,
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return LeRobotDataset("test/editor", root=dataset.root)


def test_document_survives_reopen_and_rejects_stale_edits(episode):
    document, source = load_document(episode, 0)
    document.objects = [DemoObject(id="object_1", name="cube", point=(25, 20))]
    document.keyframes = [DemoKeyframe(frame=7, reference="object_1")]
    saved = save_document(episode, 0, document)
    reopened = LeRobotDataset(episode.repo_id, root=episode.root)
    actual, actual_source = load_document(reopened, 0)
    assert actual == saved
    assert actual_source == source
    assert actual.keyframes[0].reference == actual.objects[0].id
    # The original camera and motor data remain readable through LeRobot.
    assert reopened[7]["action"].tolist() == [7.0, 1.0]
    with pytest.raises(ValueError, match="newer edit"):
        save_document(reopened, 0, document)


def test_document_bounds_and_reference_validation(episode):
    document, _ = load_document(episode, 0)
    document.objects = [DemoObject(id="object_1", name="cube", point=(96, 20))]
    with pytest.raises(ValueError, match="outside.*image"):
        save_document(episode, 0, document)
    document.objects = []
    document.keyframes = [DemoKeyframe(frame=10, reference="world")]
    with pytest.raises(ValueError, match="outside.*episode"):
        save_document(episode, 0, document)
    with pytest.raises(ValidationError, match="not selected"):
        DemoDocument(source="source", keyframes=[DemoKeyframe(frame=0, reference="object_1")])
    with pytest.raises(ValidationError):
        DemoDocument(source="source", objects=[DemoObject(id="object_1", name=str(i)) for i in range(4)])


@pytest.mark.parametrize("episode", [241], indirect=True)
def test_dense_contact_interval_saves_every_recorded_frame(episode):
    document, source = load_document(episode, 0)
    document.keyframes = [DemoKeyframe(frame=i, reference="world") for i in range(source["length"])]
    # Exercise the same JSON validation boundary as a GUI save, not an unchecked
    # assignment. Long intervals used to fail at the arbitrary 200-keyframe cap.
    saved = save_document(episode, 0, DemoDocument.model_validate_json(document.model_dump_json()))
    reopened, _ = load_document(LeRobotDataset(episode.repo_id, root=episode.root), 0)
    assert reopened == saved
    assert [key.frame for key in reopened.keyframes] == list(range(241))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "robot,teleop",
    [
        (
            {"type": "so101_follower", "fields": {"port": "/dev/test"}},
            {"type": "so101_leader", "fields": {"port": "/dev/test-leader"}},
        ),
        (
            {"type": "bi_so107_follower", "fields": {"left_arm_port": "/dev/l", "right_arm_port": "/dev/r"}},
            {"type": "bi_so107_leader", "fields": {"left_arm_port": "/dev/ll", "right_arm_port": "/dev/lr"}},
        ),
        ({"type": "editor_test_robot", "fields": {}}, {"type": "scripted_bimanual_ee", "fields": {}}),
    ],
)
async def test_editor_record_reaches_real_record_config(robot, teleop, tmp_path, monkeypatch):
    from lerobot.scripts.lerobot_record import RecordConfig
    from tests.gui.test_cli_contract import _parse

    robot = {
        **robot,
        "fields": {**robot["fields"], **calibrated_profile(tmp_path)["fields"]},
        "cameras": calibrated_profile(tmp_path)["cameras"],
    }
    argv = []

    async def launch(args, **kwargs):
        argv.extend(args)

    monkeypatch.setattr(run, "_launch_subprocess", launch)
    monkeypatch.setattr(run, "_release_preview_cameras", AsyncMock())
    monkeypatch.setattr(run, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(
        run, "_active_process", SimpleNamespace(pid=42, returncode=None, wait=AsyncMock(return_value=0))
    )
    monkeypatch.setattr(flash_editor, "_record_process", None)
    monkeypatch.setattr(flash_editor, "_recording", None)
    response = await flash_editor.record(
        run.RecordRequest(
            robot=robot,
            teleop=teleop,
            repo_id="test/demo",
            root=str(tmp_path / "recording"),
            single_task="pick cube",
            num_episodes=50,
        )
    )
    config = _parse(RecordConfig, argv)
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    # SO-100 and SO-101 are registry aliases for the same config class.
    assert type(config.robot) is RobotConfig.get_choice_class(robot["type"])
    assert type(config.teleop) is TeleoperatorConfig.get_choice_class(teleop["type"])
    assert config.dataset.num_episodes == 1
    assert str(config.dataset.root) == response["root"]
    # Reload knows this process, but cannot stop a replacement process.
    assert (await flash_editor.recording())["recording"]["pid"] == 42
    monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=43, returncode=None))
    with pytest.raises(HTTPException, match="no longer"):
        await flash_editor.stop(flash_editor.StopRecording(pid=42))


def test_playback_defaults_to_keyframe_span_without_cutting_recording(gui_page, episode):
    from playwright.sync_api import expect

    document, _ = load_document(episode, 0)
    document.objects = [DemoObject(id="object_1", name="cube", point=(25, 20))]
    document.keyframes = [DemoKeyframe(frame=i, reference="object_1") for i in (2, 7)]
    save_document(episode, 0, document)
    atomic_json(
        flash_editor.EDITOR_DIR / "demos.json",
        {
            "recent": {
                "id": "fixture",
                "root": str(episode.root),
                "episode": 0,
                "created_at": 1,
                "name": "Last replay",
            },
            "demos": [],
        },
    )
    page = gui_page
    page.get_by_role("button", name="FLASH", exact=True).click()
    page.get_by_role("button", name="Demo editor", exact=True).click()
    page.locator("#flash-editor-demos button").first.click()
    page.wait_for_function("flashEditor.player?.state().lastPainted === 2")
    expect(page.locator("#flash-editor-camera")).to_have_count(0)
    expect(page.locator("#flash-editor-resolution")).to_have_text("96 × 64")
    expect(page.locator("#flash-editor-resolution")).to_be_visible()
    expect(page.locator("#flash-editor-position")).to_have_attribute("min", "2")
    expect(page.locator("#flash-editor-position")).to_have_attribute("max", "7")
    expect(page.locator("#flash-editor-full-recording")).not_to_be_checked()
    page.locator("#flash-editor-play").click()
    page.wait_for_function(
        "flashEditor.player.metrics.wraps.length >= 2 || "
        "flashEditor.player.metrics.painted.some(p=>p.frame<2 || p.frame>7)"
    )
    page.locator("#flash-editor-play").click()
    # The actual shared player's painted RGB frames stay in the span, including
    # its unmarked middle, rather than jumping between the two keyframes.
    painted = page.evaluate("flashEditor.player.metrics.painted.map(p=>p.frame)")
    assert set(painted) == set(range(2, 8))
    page.locator("#flash-editor-full-recording").check()
    page.locator("#flash-editor-position").fill("9")
    page.wait_for_function("flashEditor.player.state().lastPainted === 9")
    page.locator("#flash-editor-full-recording").uncheck()
    page.wait_for_function("flashEditor.player.state().lastPainted === 7")
    # Object selection can still return to its earlier image outside the span.
    page.get_by_role("button", name="Picked", exact=True).click()
    page.wait_for_function("flashEditor.player.state().lastPainted === 0")
    expect(page.locator("#flash-editor-full-recording")).to_be_checked()
    assert episode.episode_rows(0)[1] == 10
    assert len(load_document(episode, 0)[0].keyframes) == 2


def test_saved_keyframe_references_follow_playback_not_new_key_choice(gui_page, episode):
    from playwright.sync_api import expect

    document, _ = load_document(episode, 0)
    document.objects = [DemoObject(id="object_1", name="cube", point=(25, 20))]
    document.keyframes = [DemoKeyframe(frame=2, reference="world")] + [
        DemoKeyframe(frame=i, reference="object_1") for i in range(5, 8)
    ]
    save_document(episode, 0, document)
    atomic_json(
        flash_editor.EDITOR_DIR / "demos.json",
        {
            "recent": None,
            "demos": [
                {
                    "id": "fixture",
                    "root": str(episode.root),
                    "episode": 0,
                    "name": "References",
                    "created_at": 1,
                }
            ],
        },
    )
    page = gui_page
    page.get_by_role("button", name="FLASH", exact=True).click()
    page.get_by_role("button", name="Demo editor", exact=True).click()
    page.locator("#flash-editor-demos button").click()
    keys = page.locator("#flash-editor-keys")
    expect(keys).to_contain_text("Keyframe 2")
    expect(keys).to_contain_text("Relative to World (robot base)")
    expect(keys).to_contain_text("Frames 5–7 · 3 keyframes")
    expect(keys).to_contain_text("Relative to cube")
    current = page.locator("#flash-editor-current-key")
    expect(current).to_have_text("Keyframe 2 · Relative to World (robot base)")
    page.locator("#flash-editor-position").fill("6")
    expect(current).to_have_text("Keyframe 6 · Relative to cube")
    expect(keys.locator('[aria-current="true"]')).to_contain_text("Frames 5–7")
    # Choosing a reference for a future edit must not relabel a saved keyframe.
    page.locator("#flash-editor-references").get_by_role("button", name="World", exact=True).click()
    expect(current).to_have_text("Keyframe 6 · Relative to cube")
    page.locator("#flash-editor-position").fill("4")
    expect(current).to_have_text("Frame 4 · Not a keyframe")
    expect(keys.locator('[aria-current="true"]')).to_have_count(0)
    assert load_document(episode, 0)[0].keyframes == document.keyframes


def test_editor_ui_edits_real_episode(gui_page, episode):
    from playwright.sync_api import expect

    page = gui_page
    base = page.url.split("#")[0].rstrip("/")
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    atomic_json(
        flash_editor.EDITOR_DIR / "demos.json",
        {
            "recent": {
                "id": "fixture",
                "root": str(episode.root),
                "episode": 0,
                "created_at": 1,
                "name": "Last replay",
            },
            "demos": [],
        },
    )
    # Profile availability follows the real registries, including newly added
    # robot and teleop types. The editor has no second allowlist.
    for schemas, profiles in [("schemas", "profiles"), ("teleop-schemas", "teleop-profiles")]:
        entries = page.request.get(f"{base}/api/robot/{schemas}").json()
        for entry in entries:
            response = page.request.post(
                f"{base}/api/robot/{profiles}",
                data={
                    "name": entry["type_name"],
                    "type": entry["type_name"],
                    "fields": {},
                },
            )
            assert response.ok
    page.get_by_role("button", name="FLASH", exact=True).click()
    page.get_by_role("button", name="Demo editor", exact=True).click()
    for control, schemas in [("robot", "schemas"), ("teleop", "teleop-schemas")]:
        entries = page.request.get(f"{base}/api/robot/{schemas}").json()
        expect(page.locator(f"#flash-editor-{control} option")).to_have_count(len(entries) + 1)
    page.locator("#flash-editor-settings").click()
    page.locator("#flash-editor-robot").select_option("virtual_bi_so107")
    expect(page.locator("#flash-editor-geometry")).to_contain_text("camera-to-base calibration")
    page.locator("#flash-editor-teleop").select_option("scripted_bimanual_ee")
    page.locator("#flash-editor-settings-save").click()
    page.locator("#flash-editor-demos button").first.click()
    expect(page.locator("#flash-editor-play")).to_be_enabled(timeout=30_000)
    page.wait_for_function("document.getElementById('flash-editor-image').width === 96", timeout=30_000)
    page.get_by_label("Demo object name").fill("cube")
    page.locator("#flash-editor-add-object").click()
    page.get_by_role("button", name="Pick", exact=True).click()
    page.locator("#flash-editor-image").click(position={"x": 25, "y": 20})
    page.locator("#flash-editor-references").get_by_role("button", name="cube").click()
    page.locator("#flash-editor-position").fill("7")
    page.wait_for_function("flashEditor.frame === 7")
    page.locator("#flash-editor-mark").click()
    page.locator("#flash-editor-save").click()
    expect(page.locator("#flash-editor-save-state")).to_have_text("Saved")
    saved, _ = load_document(episode, 0)
    assert saved.objects[0].point is not None
    assert saved.keyframes == [DemoKeyframe(frame=7, reference="object_1")]
    # Dense range crosses the browser -> API -> durable document boundary.
    page.locator("#flash-editor-position").fill("5")
    page.wait_for_function("flashEditor.frame === 5")
    page.locator("#flash-editor-range-start").click()
    page.locator("#flash-editor-position").fill("4")
    page.wait_for_function("flashEditor.frame === 4")
    page.locator("#flash-editor-range-end").click()
    expect(page.locator("#flash-editor-status")).to_contain_text("at or after")
    page.locator("#flash-editor-position").fill("9")
    page.wait_for_function("flashEditor.frame === 9")
    page.locator("#flash-editor-range-end").click()
    page.locator("#flash-editor-save").click()
    expect(page.locator("#flash-editor-save-state")).to_have_text("Saved")
    saved, _ = load_document(episode, 0)
    assert saved.keyframes == [DemoKeyframe(frame=i, reference="object_1") for i in range(5, 10)]
    expect(page.locator("#flash-editor-keys > div")).to_have_count(1)
    # An overlapping interval with a different reference must not silently
    # reassign existing contact frames or partially insert new ones.
    page.locator("#flash-editor-references").get_by_role("button", name="World", exact=True).click()
    page.locator("#flash-editor-position").fill("3")
    page.wait_for_function("flashEditor.frame === 3")
    page.locator("#flash-editor-range-start").click()
    page.locator("#flash-editor-position").fill("6")
    page.wait_for_function("flashEditor.frame === 6")
    page.locator("#flash-editor-range-end").click()
    expect(page.locator("#flash-editor-status")).to_contain_text("another reference")
    # The rejected edit changes nothing, so there is nothing new to save.
    expect(page.locator("#flash-editor-save")).to_be_disabled()
    expect(page.locator("#flash-editor-save-state")).to_have_text("Saved")
    assert load_document(episode, 0)[0].keyframes == saved.keyframes
    page.locator("#flash-editor-range-cancel").click()
    for name in ["ring", "dowel"]:
        page.get_by_label("Demo object name").fill(name)
        page.locator("#flash-editor-add-object").click()
    expect(page.locator("#flash-editor-add-object")).to_be_disabled()
    expect(page.locator("#flash-editor-object-count")).to_have_text("3 / 3")
    page.locator("#flash-editor-save").click()
    expect(page.locator("#flash-editor-save-state")).to_have_text("Saved")
    assert len(load_document(episode, 0)[0].objects) == 3
    page.locator("#flash-editor-settings").click()
    page.get_by_role("button", name="Robot / cameras…", exact=True).click()
    expect(page.locator("#tab-robot")).to_be_visible()
    page.wait_for_function("currentProfile?.name === 'virtual_bi_so107'")
    page.get_by_role("button", name="Run", exact=True).first.click()
    page.get_by_role("button", name="FLASH", exact=True).click()
    expect(page.locator("#flash-editor-workspace")).to_be_visible()
    expect(page.locator("#flash-editor-objects")).to_contain_text("cube")
    expect(page.locator("#flash-editor-robot")).to_have_value("virtual_bi_so107")
    expect(page.locator("#flash-editor-status")).not_to_have_text("Buffering recording…")
    assert not errors


@pytest.mark.asyncio
@pytest.mark.parametrize("stereo", [False, True])
async def test_editor_records_virtual_robot_through_real_writer(tmp_path, monkeypatch, stereo):
    from lerobot.gui.api._run_core import get_run_output_tail

    monkeypatch.setattr(flash_editor, "_record_process", None)
    monkeypatch.setattr(flash_editor, "_recording", None)
    launched = await flash_editor.record(
        run.RecordRequest(
            robot=calibrated_profile(tmp_path, stereo=stereo),
            teleop={"type": "scripted_bimanual_ee", "fields": {"id": "editor-test", "shape": "circle"}},
            repo_id="test/virtual-demo",
            root=str(tmp_path / "virtual-demo"),
            single_task="circle",
            fps=10,
            episode_time_s=1,
            play_sounds=False,
        )
    )
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            status = (await flash_editor.recording())["recording"]
            if not status["running"]:
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail(f"Recorder did not finish: {get_run_output_tail(40)}")
        assert status["returncode"] == 0, "\n".join(get_run_output_tail(40)["lines"])
        dataset = LeRobotDataset("test/virtual-demo", root=launched["root"])
        assert dataset.meta.total_episodes == 1
        assert dataset.meta.info.robot_geometry["robot_id"] == "editor-test"
        assert dataset.meta.info.robot_geometry["cameras"]["scene"]["T_base_cam"] == np.eye(4).tolist()
        assert dataset.num_frames >= 5
        actions = np.asarray(dataset.episode_column("action", 0))
        assert np.isfinite(actions).all()
        assert np.ptp(actions, axis=0).max() > 0.01
        if stereo:
            from lerobot.policies.flash.demo_tracking import DemoFrames

            document, source = load_document(dataset, 0)
            assert list(source["cameras"]) == ["observation.images.scene"]
            frames = DemoFrames(dataset, 0, document)
            frames.require_depth_source("foundation_stereo")
            assert frames.stereo_geometry[2] == 0.05
            for current in frames:
                row = dataset[current.frame_id]
                sequence = row["observation.cameras.scene.frame_sequence"].item()
                value = sequence % 255
                assert np.all(current.infrared_left == value)
                assert np.all(current.infrared_right == value + 1)
                np.testing.assert_allclose(current.metric_depth(), (400 + value) * 0.001, atol=1e-7)
                # The robot's independently read RGB was 90: recording must use
                # RGB from the IR snapshot, not attach the IR pair to that image.
                np.testing.assert_allclose(current.rgb, value, atol=2)
                assert current.source_time_s == pytest.approx((sequence - 1) * 0.103, abs=1e-9)
    finally:
        if (await flash_editor.recording())["recording"]["running"]:
            await flash_editor.stop(flash_editor.StopRecording(pid=launched["pid"]))


def test_record_stop_auto_opens_episode_in_editor(gui_page, tmp_path, monkeypatch):
    from playwright.sync_api import expect

    from lerobot.utils import constants

    monkeypatch.setattr(constants, "HF_LEROBOT_HOME", tmp_path / "datasets")
    monkeypatch.setattr(flash_editor, "_record_process", None)
    monkeypatch.setattr(flash_editor, "_recording", None)
    page = gui_page
    base = page.url.split("#")[0].rstrip("/")
    for endpoint, profile in [
        ("profiles", {"name": "virtual", **calibrated_profile(tmp_path, "editor-ui")}),
        (
            "teleop-profiles",
            {"name": "script", "type": "scripted_bimanual_ee", "fields": {"id": "editor-ui"}},
        ),
    ]:
        assert page.request.post(f"{base}/api/robot/{endpoint}", data=profile).ok
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.get_by_role("button", name="FLASH", exact=True).click()
    page.get_by_role("button", name="Demo editor", exact=True).click()
    expect(page.locator("#flash-editor-robot option")).to_have_count(2)
    page.locator("#flash-editor-settings").click()
    page.locator("#flash-editor-robot").select_option("virtual")
    page.locator("#flash-editor-teleop").select_option("script")
    expect(page.locator("#flash-editor-geometry")).to_have_text("Calibrated: scene")
    page.locator("#flash-editor-settings-save").click()
    page.locator("#flash-editor-record").click()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = page.request.get(f"{base}/api/run/status").json()
            if (state.get("phase") or "").startswith("recording episode"):
                break
            page.wait_for_timeout(100)
        else:
            pytest.fail(f"Recorder did not reach recording: {state}")
        expect(page.locator("#flash-editor-live-host #rerun-viewer")).to_be_visible()
        # Moving to Run restores the shared viewer; returning to FLASH keeps the
        # same process rather than reopening the robot or the camera.
        pid = page.request.get(f"{base}/api/run/status").json()["pid"]
        page.get_by_role("button", name="Run", exact=True).first.click()
        expect(page.locator("#run-main #rerun-viewer")).to_be_visible()
        page.get_by_role("button", name="FLASH", exact=True).click()
        expect(page.locator("#flash-editor-live-host #rerun-viewer")).to_be_visible()
        assert page.request.get(f"{base}/api/run/status").json()["pid"] == pid
        first_action_time = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = page.request.get(f"{base}/api/run/obs-stream/state").json()
            if state.get("action_ts"):
                first_action_time = first_action_time or state["action_ts"]
                if state["action_ts"] - first_action_time > 0.1:
                    break
            page.wait_for_timeout(100)
        else:
            from lerobot.gui.api._run_core import get_run_output_tail

            pytest.fail(
                f"No changing robot observations: {state}; {page.request.get(f'{base}/api/run/obs-stream/meta').json()}; {get_run_output_tail(20)}"
            )
        page.locator("#flash-editor-stop").click()
        try:
            expect(page.locator("#flash-editor-edit-view")).to_be_visible(timeout=30_000)
        except AssertionError:
            from lerobot.gui.api._run_core import get_run_output_tail

            pytest.fail(
                f"Editor: {page.locator('#flash-editor-status').inner_text()}\n{get_run_output_tail(30)}"
            )
        expect(page.locator("#flash-editor-status")).to_contain_text("Select up to three objects")
        library = page.request.get(f"{base}/api/run/flash/editor/library").json()
        dataset = LeRobotDataset("local/test-demo", root=library["recent"]["root"])
        assert dataset.num_frames > 0
        assert dataset.meta.total_episodes == 1
        page.get_by_label("Demo name", exact=True).fill("Circle demo")
        # Renaming already saves on blur, without a separate save click.
        page.get_by_label("Demo name", exact=True).press("Tab")
        expect(page.locator("#flash-editor-demos")).to_contain_text("Circle demo")
        expect(page.locator("#flash-editor-save")).to_be_disabled()
        page.reload()
        expect(page.locator("#flash-editor-demos")).to_contain_text("Circle demo")
        page.locator("#flash-editor-demos button").click()
        expect(page.locator("#flash-editor-edit-view")).to_be_visible()
        expect(page.locator("#flash-editor-name")).to_have_value("Circle demo")
        assert page.request.get(f"{base}/api/run/status").json()["running"] is False
        assert not errors
    finally:
        status = page.request.get(f"{base}/api/run/flash/editor/recording").json().get("recording")
        if status and status["running"]:
            page.request.post(
                f"{base}/api/run/flash/editor/stop", data={"pid": status["pid"]}, timeout=30_000
            )


@pytest.mark.asyncio
async def test_saved_setup_drives_recorder_and_camera(tmp_path, monkeypatch):
    from lerobot.gui.api import flash_tracking, robot as robot_api

    robot = calibrated_profile(tmp_path)
    camera = {
        "type": "intelrealsense",
        "serial_number_or_name": "test-serial",
        "width": 848,
        "height": 480,
        "fps": 30,
    }
    robot["cameras"] = {"scene": camera}
    import json

    geometry_path = Path(robot["fields"]["geometry_path"])
    geometry = json.loads(geometry_path.read_text())
    geometry["cameras"]["scene"]["identity"] = {
        "type": "intelrealsense",
        "serial_number_or_name": "test-serial",
    }
    atomic_json(geometry_path, geometry)
    teleop = {"type": "scripted_bimanual_ee", "fields": {"id": "test"}}
    atomic_json(robot_api.ROBOT_PROFILES_DIR / "robot.json", robot)
    atomic_json(robot_api.TELEOP_PROFILES_DIR / "leader.json", teleop)
    settings = flash_editor.EditorSettings(robot="robot", teleop="leader", camera="scene", fps=20)
    settings.pipeline_settings["config"]["shape"]["registration"]["backend"] = "open3d_plane"
    settings.pipeline_settings["shared_matching"]["backend"] = "ufm"
    await flash_editor.save_settings(settings)
    assert await flash_editor.settings() == settings

    # Follow the profile to the same camera-worker request used by Live camera.
    started = SimpleNamespace(pid=12, returncode=None)

    async def start(req):
        assert req.serial == "test-serial"
        assert (req.width, req.height, req.fps) == (848, 480, 30)
        assert req.pipeline_settings == settings.pipeline_settings
        run._active_process = started
        return {"status": "started"}

    monkeypatch.setattr(flash_editor, "_preview_process", None)
    monkeypatch.setattr(run, "_active_process", None)
    monkeypatch.setattr(flash_tracking, "start", start)
    await flash_editor.preview()
    assert flash_editor._preview_process is started
    events = []

    async def stop():
        events.append("stop-camera")
        started.returncode = 0
        return {"status": "stopped"}

    async def record(req):
        events.append("record")
        assert req.robot == robot and req.teleop == teleop
        assert req.fps == 20
        assert req.root.startswith(str(flash_editor.EDITOR_DIR / "recordings"))
        assert req.repo_id.startswith("local/flash-demo-")
        return {"root": req.root}

    monkeypatch.setattr(run, "stop_process", stop)
    monkeypatch.setattr(flash_editor, "record", record)
    first = await flash_editor.record_current()
    second = await flash_editor.record_current()
    assert first["root"] != second["root"]
    assert events == ["stop-camera", "record", "record"]

    # Never stop a process that replaced this editor's preview.
    monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=13, returncode=None))
    await flash_editor.stop_preview()
    assert events == ["stop-camera", "record", "record"]


@pytest.mark.asyncio
async def test_last_replay_and_saved_demos_survive_new_recording_and_restart(tmp_path, monkeypatch):
    def completed(identity):
        monkeypatch.setattr(
            flash_editor, "_recording", {"id": identity, "root": str(tmp_path / identity), "created_at": 1}
        )
        monkeypatch.setattr(flash_editor, "_record_process", SimpleNamespace(returncode=0))

    completed("one")
    assert (await flash_editor.library())["recent"]["id"] == "one"
    await flash_editor.keep_demo(flash_editor.KeepDemo(id="one", name="Pick cube"))
    completed("two")
    actual = await flash_editor.library()
    assert actual["recent"]["id"] == "two"
    assert [d["id"] for d in actual["demos"]] == ["two", "one"]
    assert actual["demos"][1]["name"] == "Pick cube"
    monkeypatch.setattr(flash_editor, "_recording", None)
    monkeypatch.setattr(flash_editor, "_record_process", None)
    assert await flash_editor.library() == actual
    renamed = await flash_editor.keep_demo(flash_editor.KeepDemo(id="one", name="Place cube"))
    assert len(renamed["demos"]) == 2
    assert renamed["demos"][1]["name"] == "Place cube"
