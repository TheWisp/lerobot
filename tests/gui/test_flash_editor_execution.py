"""Exercise Apply through the real editor/API/episode reader, stubbing only launch."""

import json
import shutil
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from playwright.sync_api import expect

from lerobot.gui.api import flash_editor, flash_tracking, robot as robot_api, run
from lerobot.policies.flash.demo_document import atomic_json, document_path, save_document
from tests.policies.flash.test_demo_execution import make_annotated_demo, robot  # noqa: F401


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
    assert result == {"pid": 456, "mode": "simulation"}
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


@pytest.mark.parametrize("entry_point", ["live", "demo"])
def test_apply_dialog_sends_explicit_simulation_and_shows_execution(
    gui_page, prepared, monkeypatch, tmp_path, entry_point
):
    async def start(request, **kwargs):
        assert kwargs["extra"]["demo_execution"]["simulate"] is True
        monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=456, returncode=None))
        monkeypatch.setattr(flash_tracking, "_directory", tmp_path)
        atomic_json(
            tmp_path / "execution.json",
            {
                "mode": "simulation",
                "state": "waiting: object_1 needs a current accepted pose",
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
    expect(page.locator("#flash-editor-apply-keyframes li")).to_have_count(3)
    expect(page.locator("#flash-editor-run-robot")).to_be_visible()
    page.locator("#flash-editor-simulate").click()
    expect(page.locator("#flash-editor-execution")).to_contain_text("Test run · no motion")
    expect(page.locator("#flash-editor-execution")).to_contain_text("waiting: object_1")
    atomic_json(
        tmp_path / "execution.json",
        {"mode": "simulation", "state": "complete", "keyframe": 3, "total": 3},
    )
    expect(page.locator("#flash-editor-execution")).to_contain_text("path complete — check placement")
    assert not errors
