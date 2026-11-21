"""Exercise Apply through the real editor/API/episode reader, stubbing only launch."""

import json
import shutil
from threading import Event
from time import perf_counter, sleep
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from playwright.sync_api import expect

from lerobot.gui.api import flash_editor, flash_tracking, robot as robot_api, run
from lerobot.policies.flash.demo_document import atomic_json, document_path, save_document
from tests.policies.flash.test_demo_execution import (  # noqa: F401
    make_annotated_demo,
    motor_robot,
    ready_state,
    robot,
)


@pytest.fixture
def prepared(tmp_path, robot, monkeypatch):  # noqa: F811
    ds, document, tracked = make_annotated_demo(robot, tmp_path)
    document = save_document(ds, 0, document)
    folder = document_path(ds, 0).with_suffix("")
    destination = folder / "fixture"
    shutil.copytree(tracked, destination)
    inputs = json.loads((destination / "input.json").read_text())
    inputs["document"] = document.model_dump(mode="json")
    atomic_json(destination / "input.json", inputs)
    atomic_json(folder / "latest.json", {"directory": "fixture"})
    profile = {
        "type": robot.robot_type,
        "fields": {"id": robot.id, "geometry_path": str(robot.config.geometry_path)},
        "cameras": {
            "camera": {
                "type": "intelrealsense",
                "serial_number_or_name": "test",
                "width": 16,
                "height": 16,
                "fps": 30,
            }
        },
    }
    setup = flash_editor.EditorSettings(robot="test", camera="camera", depth_source="sensor")
    atomic_json(robot_api.ROBOT_PROFILES_DIR / "test.json", profile)
    atomic_json(flash_editor.EDITOR_DIR / "settings.json", setup.model_dump())
    atomic_json(
        flash_editor.EDITOR_DIR / "demos.json",
        {
            "recent": None,
            "demos": [
                {
                    "id": "fixture",
                    "root": str(ds.root),
                    "episode": 0,
                    "name": "Three objects",
                    "created_at": 1,
                }
            ],
        },
    )
    monkeypatch.setattr(robot_api, "_make_robot_from_profile", lambda profile: robot)
    monkeypatch.setattr(flash_editor, "_preview_process", None)
    monkeypatch.setattr(flash_editor, "_execution_process", None)
    return ds, document


@pytest.mark.asyncio
async def test_prepare_apply_pins_real_episode_and_uses_shared_camera_worker(prepared, monkeypatch, tmp_path):
    result = await flash_editor.prepare_demo(flash_editor.PrepareDemo(id="fixture"))
    assert len(result["keyframes"]) == 3
    plan_folder = flash_editor.EDITOR_DIR / "plans" / result["plan"]
    assert (plan_folder / "reference.npz").is_file()
    calls = []
    process = SimpleNamespace(pid=456, returncode=None)

    async def start(request, **kwargs):
        calls.append((request, kwargs))
        monkeypatch.setattr(run, "_active_process", process)
        monkeypatch.setattr(flash_tracking, "_directory", tmp_path)

    monkeypatch.setattr(flash_tracking, "start_worker", start)
    result = await flash_editor.apply_demo(flash_editor.ApplyDemo(plan=result["plan"]))
    assert result["pid"] == 456
    assert result["mode"] == "simulation"
    assert (flash_editor.EDITOR_DIR / "rollouts" / result["rollout"] / "plan.json").exists()
    request, kwargs = calls[0]
    assert request.serial == "test"
    assert request.pipeline_settings == flash_editor.EditorSettings().pipeline_settings
    assert kwargs["module"] == "lerobot.policies.flash.demo_live_worker"
    assert kwargs["extra"]["demo_execution"]["simulate"] is True
    assert kwargs["extra"]["initial_selection"]["action"] == "select"
    assert flash_editor._preview_process is process
    assert (await flash_editor.execution())["running"] is True
    # Editor Stop cannot terminate another tab's replacement process.
    stop = AsyncMock()
    monkeypatch.setattr(run, "stop_process", stop)
    monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=999, returncode=None))
    await flash_editor.stop_preview()
    stop.assert_not_called()


@pytest.mark.asyncio
async def test_apply_rejects_edited_demo_before_camera_or_motors(prepared, monkeypatch):
    ds, document = prepared
    result = await flash_editor.prepare_demo(flash_editor.PrepareDemo(id="fixture"))
    document.keyframes.pop()
    save_document(ds, 0, document)
    launch = AsyncMock()
    monkeypatch.setattr(flash_tracking, "start_worker", launch)
    with pytest.raises(HTTPException, match="Demo changed"):
        await flash_editor.apply_demo(flash_editor.ApplyDemo(plan=result["plan"]))
    launch.assert_not_called()


@pytest.mark.asyncio
async def test_apply_rejects_old_execution_rules_before_camera_or_motors(prepared, monkeypatch):
    result = await flash_editor.prepare_demo(flash_editor.PrepareDemo(id="fixture"))
    path = flash_editor.EDITOR_DIR / "plans" / result["plan"] / "plan.json"
    plan = json.loads(path.read_text())
    plan["version"] = 1
    atomic_json(path, plan)
    launch = AsyncMock()
    monkeypatch.setattr(flash_tracking, "start_worker", launch)
    with pytest.raises(HTTPException, match="Execution rules changed"):
        await flash_editor.apply_demo(flash_editor.ApplyDemo(plan=result["plan"]))
    launch.assert_not_called()


@pytest.mark.asyncio
async def test_robot_apply_requires_saved_rest_before_launch(prepared, monkeypatch):
    result = await flash_editor.prepare_demo(flash_editor.PrepareDemo(id="fixture"))
    launch = AsyncMock()
    monkeypatch.setattr(flash_tracking, "start_worker", launch)
    with pytest.raises(HTTPException, match="Save a rest position"):
        await flash_editor.apply_demo(flash_editor.ApplyDemo(plan=result["plan"], simulate=False))
    launch.assert_not_called()


@pytest.mark.asyncio
async def test_apply_profile_reaches_worker_and_resets_before_waiting_for_objects(
    prepared,
    motor_robot,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    from lerobot.policies.flash import demo_live_worker as worker

    rest = ready_state(motor_robot.cartesian_model())
    profile_path = robot_api.ROBOT_PROFILES_DIR / "test.json"
    profile = json.loads(profile_path.read_text())
    profile["rest_position"] = rest
    atomic_json(profile_path, profile)
    result = await flash_editor.prepare_demo(flash_editor.PrepareDemo(id="fixture"))
    directory = tmp_path / "worker"

    async def launch(request, **kwargs):
        # Serialize exactly what Apply gives the real subprocess.
        atomic_json(directory / "config.json", {**request.model_dump(), **kwargs["extra"]})
        monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=456, returncode=None))
        monkeypatch.setattr(flash_tracking, "_directory", directory)

    monkeypatch.setattr(flash_tracking, "start_worker", launch)
    await flash_editor.apply_demo(flash_editor.ApplyDemo(plan=result["plan"], simulate=False))
    stop = Event()
    monkeypatch.setattr(stop, "wait", lambda _: stop.is_set())
    monkeypatch.setattr(worker, "Event", lambda: stop)
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setattr(worker.sys, "argv", ["demo_live_worker", str(directory)])
    monkeypatch.setattr(robot_api, "_make_robot_from_profile", lambda _: motor_robot)

    def camera_run(_directory, _config, **_kwargs):
        _kwargs["on_camera_ready"]()
        deadline = perf_counter() + 3
        while perf_counter() < deadline:
            path = directory / "execution.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            if str(status.get("state")).startswith("waiting:"):
                assert status["commands"] == 0
                return
            assert status.get("state") != "failed", status
            sleep(0.01)
        pytest.fail("Worker never finished reset and reached the live-reference wait")

    monkeypatch.setattr(worker, "run", camera_run)
    worker.main()
    assert len(motor_robot.sent) == 150
    assert motor_robot.sent[-1] == rest
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    requested = [e["requested"] for e in events if e["kind"] == "robot.command_requested"]
    accepted = [e["accepted"] for e in events if e["kind"] == "robot.command_accepted"]
    assert requested == motor_robot.sent == accepted
    assert events[-1]["kind"] == "run.finished"
    motor_robot.disconnect.assert_called_once()
    assert (await flash_editor.execution())["state"] == "stopped"


@pytest.mark.parametrize("entry_point", ["live", "demo"])
def test_apply_starts_robot_directly_and_shows_execution(
    gui_page, prepared, monkeypatch, tmp_path, entry_point
):
    profile_path = robot_api.ROBOT_PROFILES_DIR / "test.json"
    profile = json.loads(profile_path.read_text())
    profile["rest_position"] = {"joint.pos": 0}
    atomic_json(profile_path, profile)

    async def start(request, **kwargs):
        assert kwargs["extra"]["demo_execution"]["simulate"] is False
        (tmp_path / "output.log").write_text("Connecting to robot\n")
        monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=456, returncode=None))
        monkeypatch.setattr(flash_tracking, "_directory", tmp_path)
        atomic_json(
            tmp_path / "execution.json",
            {
                "mode": "robot",
                "state": "resetting to rest position",
                "keyframe": 0,
                "total": 3,
            },
        )

    monkeypatch.setattr(flash_tracking, "start_worker", start)
    page = gui_page
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.get_by_role("button", name="FLASH", exact=True).click()
    page.get_by_role("button", name="Demo editor", exact=True).click()
    if entry_point == "demo":
        page.locator("#flash-editor-demos button").click()
        expect(page.locator("#flash-editor-edit-view")).to_be_visible()
        page.locator("#flash-editor-run-demo").click()
        expect(page.locator("#flash-editor-live-actions")).to_be_visible()
        expect(page.locator("#flash-editor-apply-demo")).to_have_value("fixture")
    else:
        page.locator("#flash-editor-apply-demo").select_option("fixture")
        page.locator("#flash-editor-apply").click()
    expect(page.locator("#flash-editor-apply-dialog")).to_have_count(0)
    expect(page.locator("#flash-editor-rollouts button")).to_have_count(1)
    expect(page.locator("#flash-editor-execution")).to_contain_text("resetting to rest position")
    expect(page.locator("#flash-editor-run-log")).to_be_visible()
    expect(page.locator("#flash-editor-terminal")).to_contain_text("Connecting to robot")
    expect(page.locator("#flash-editor-rest")).to_be_disabled()
    atomic_json(
        tmp_path / "execution.json",
        {"mode": "robot", "state": "waiting: object_1 needs a pose", "keyframe": 0, "total": 3},
    )
    expect(page.locator("#flash-editor-execution")).to_contain_text("waiting: object_1")
    atomic_json(
        tmp_path / "execution.json",
        {"mode": "robot", "state": "complete", "keyframe": 3, "total": 3},
    )
    expect(page.locator("#flash-editor-execution")).to_contain_text("path complete — check placement")
    expect(page.locator("#flash-editor-rest")).to_be_disabled()
    process = flash_editor._execution_process
    process.returncode = 0
    expect(page.locator("#flash-editor-rest")).to_be_enabled()
    expect(page.locator("#flash-editor-execution")).to_contain_text("path complete — check placement")
    # A subsequent setup failure must reopen the same output panel and show
    # the saved traceback even when no demo command was issued.
    page.locator("#flash-editor-run-log summary").click()
    (tmp_path / "output.log").write_text("ConnectionError: wrist_roll (id=6): no status packet\n")
    atomic_json(
        tmp_path / "execution.json", {"mode": "robot", "state": "failed", "reason": "no status packet"}
    )
    process.returncode = 1
    expect(page.locator("#flash-editor-terminal")).to_be_visible()
    expect(page.locator("#flash-editor-terminal")).to_contain_text("wrist_roll (id=6)")
    expect(page.locator("#flash-editor-execution")).to_contain_text("no status packet")
    page.evaluate("window._copyTextToClipboard = async text => {window.copiedRobotLog=text; return true;}")
    page.get_by_role("button", name="Copy robot log").click()
    assert "wrist_roll (id=6)" in page.evaluate("window.copiedRobotLog")
    assert not errors


def test_saved_failure_keeps_output_and_rest_controls_within_view(gui_page, prepared, tmp_path):
    from tests.policies.flash.test_tracking_history import new_history, submit

    identity = "rollout-" + "d" * 32
    directory = flash_editor.EDITOR_DIR / "rollouts" / identity
    history = new_history(directory / "camera")
    submit(history, 0, 1)
    history.close()
    atomic_json(
        directory / "run.json",
        {
            "id": identity,
            "name": "Failed setup",
            "state": "failed",
            "reason": "No motor reply",
            "created_at": 1,
        },
    )
    (directory / "output.log").write_text("Robot configuration failed\n" + "traceback detail\n" * 30)
    page = gui_page
    page.get_by_role("button", name="FLASH", exact=True).click()
    page.get_by_role("button", name="Demo editor", exact=True).click()
    page.locator("#flash-editor-rollouts button").click()
    expect(page.locator("#flash-editor-terminal")).to_be_visible()
    expect(page.locator("#flash-editor-terminal")).to_contain_text("Robot configuration failed")
    expect(page.locator("#flash-editor-status")).to_contain_text("No motor reply")
    expect(page.locator("#flash-editor-rest")).to_be_visible()
    expect(page.locator("#flash-editor-stop")).to_be_visible()
    divider = page.locator("#flash-editor-output-divider")
    divider.focus()
    divider.press("ArrowUp")
    terminal = page.locator("#flash-editor-run-log").bounding_box()
    workspace = page.locator("#flash-editor-main").bounding_box()
    assert terminal["y"] + terminal["height"] <= workspace["y"] + workspace["height"]
    assert page.locator("#flash-editor-preview-host").bounding_box()["height"] >= 120
    page.locator("#flash-editor-workspace").screenshot(path=str(tmp_path / "saved-failure-output.png"))
