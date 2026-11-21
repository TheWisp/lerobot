"""The explicit rest control crosses the API/worker boundary with fake motor I/O."""

import asyncio
import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from playwright.sync_api import expect

from lerobot.gui.api import flash_editor, robot as robot_api, run
from lerobot.policies.flash import demo_rest_worker as worker
from lerobot.policies.flash.demo_document import atomic_json
from tests.gui.test_flash_editor_execution import prepared  # noqa: F401
from tests.policies.flash.test_demo_execution import motor_robot, ready_state, robot  # noqa: F401


@pytest.fixture
def rest_setup(prepared, motor_robot, monkeypatch):  # noqa: F811
    path = robot_api.ROBOT_PROFILES_DIR / "test.json"
    profile = json.loads(path.read_text())
    profile["rest_position"] = ready_state(motor_robot.cartesian_model())
    atomic_json(path, profile)
    monkeypatch.setattr(run, "_active_process", None)
    monkeypatch.setattr(run, "_launch_lock", asyncio.Lock())
    monkeypatch.setattr(flash_editor, "_execution_process", None)
    monkeypatch.setattr(flash_editor, "_preview_process", None)
    monkeypatch.setattr(robot_api, "release_preview_cameras", lambda: None)
    return profile


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["complete", "stop", "connect", "write", "disconnect"])
async def test_rest_launch_reaches_cancellable_motion_and_persists_result(
    rest_setup,
    motor_robot,  # noqa: F811
    tmp_path,
    monkeypatch,
    outcome,
):
    captured = {}
    process = SimpleNamespace(pid=654, returncode=None)

    async def launch(args, **kwargs):
        captured.update(args=args, **kwargs)
        monkeypatch.setattr(run, "_active_process", process)

    monkeypatch.setattr(run, "_launch_subprocess", launch)
    result = await flash_editor.return_to_rest()
    assert result["pid"] == process.pid
    assert captured["args"][3] == worker.__name__
    directory = flash_editor._execution_directory
    assert captured["output_log"] == directory / "output.log"
    assert json.loads((directory / "config.json").read_text())["robot"] == rest_setup

    stop = Event()
    monkeypatch.setattr(stop, "wait", lambda _: stop.is_set())
    monkeypatch.setattr(worker, "Event", lambda: stop)
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setattr(worker.sys, "argv", captured["args"][3:])
    monkeypatch.setattr(robot_api, "_make_robot_from_profile", lambda p: motor_robot)
    original_send = motor_robot.send_action

    def send(action):
        if outcome == "write":
            raise ConnectionError("write failed")
        accepted = original_send(action)
        if outcome == "stop" and len(motor_robot.sent) == 3:
            stop.set()
        return accepted

    motor_robot.send_action = send
    if outcome == "connect":

        def connect(**_):
            motor_robot.is_connected = True  # Port opened, configure failed.
            raise ConnectionError("no status packet")

        motor_robot.connect = Mock(side_effect=connect)
    if outcome == "disconnect":
        motor_robot.disconnect = Mock(side_effect=ConnectionError("cleanup failed"))
    if outcome in {"connect", "write", "disconnect"}:
        with pytest.raises(ConnectionError):
            worker.main()
        process.returncode = 1
    else:
        worker.main()
        process.returncode = 0
    status = await flash_editor.execution()
    assert status["state"] == {"complete": "complete", "stop": "stopped"}.get(outcome, "failed")
    assert status["operation"] == "rest"
    motor_robot.connect.assert_called_once_with(calibrate=False)
    motor_robot.disconnect.assert_called_once()
    if outcome == "complete":
        assert len(motor_robot.sent) == 150
        assert motor_robot.sent[-1] == rest_setup["rest_position"]
        assert motor_robot.sent[0] != motor_robot.sent[-1]
    elif outcome == "stop":
        assert len(motor_robot.sent) == 3
        assert motor_robot.sent[-1] != rest_setup["rest_position"]
    elif outcome in {"write", "connect"}:
        assert motor_robot.sent == []
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert events[-1]["state"] == status["state"]


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["demo", "unrelated"])
async def test_rest_cannot_take_active_motor_owner(rest_setup, monkeypatch, owner):
    process = SimpleNamespace(pid=111, returncode=None)
    monkeypatch.setattr(run, "_active_process", process)
    monkeypatch.setattr(run, "_active_command", owner)
    if owner == "demo":
        monkeypatch.setattr(flash_editor, "_execution_process", process)
        monkeypatch.setattr(flash_editor, "_preview_process", process)
    stop = AsyncMock()
    launch = AsyncMock()
    monkeypatch.setattr(run, "stop_process", stop)
    monkeypatch.setattr(run, "_launch_subprocess", launch)
    with pytest.raises(HTTPException) as error:
        await flash_editor.return_to_rest()
    assert error.value.status_code == 409
    stop.assert_not_called()
    launch.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_rest_requests_release_only_preview_and_launch_once(rest_setup, monkeypatch):
    preview = SimpleNamespace(pid=123, returncode=None)
    process = SimpleNamespace(pid=456, returncode=None)
    monkeypatch.setattr(flash_editor, "_preview_process", preview)
    monkeypatch.setattr(run, "_active_process", preview)
    calls = []

    async def stop():
        assert run._launch_lock.locked()
        assert run._active_process is preview
        calls.append("stop preview")
        preview.returncode = 0
        monkeypatch.setattr(run, "_active_process", None)

    async def launch(*_args, **_kwargs):
        assert preview.returncode == 0
        calls.append("launch rest")
        await asyncio.sleep(0.01)
        monkeypatch.setattr(run, "_active_process", process)

    monkeypatch.setattr(run, "stop_process", stop)
    monkeypatch.setattr(run, "_launch_subprocess", launch)
    results = await asyncio.gather(
        flash_editor.return_to_rest(), flash_editor.return_to_rest(), return_exceptions=True
    )
    assert results[0] == {"pid": process.pid}
    assert isinstance(results[1], HTTPException) and results[1].status_code == 409
    assert calls == ["stop preview", "launch rest"]


@pytest.mark.asyncio
async def test_rest_requires_saved_position_and_stop_checks_process_identity(rest_setup, monkeypatch):
    profile = {**rest_setup, "rest_position": {}}
    atomic_json(robot_api.ROBOT_PROFILES_DIR / "test.json", profile)
    launch = AsyncMock()
    monkeypatch.setattr(run, "_launch_subprocess", launch)
    with pytest.raises(HTTPException, match="Save a rest position"):
        await flash_editor.return_to_rest()
    launch.assert_not_called()
    own = SimpleNamespace(pid=11, returncode=None)
    monkeypatch.setattr(flash_editor, "_execution_process", own)
    monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=12, returncode=None))
    stop = AsyncMock(return_value={"status": "stopped"})
    monkeypatch.setattr(run, "stop_process", stop)
    with pytest.raises(HTTPException):
        await flash_editor.stop_execution(flash_editor.StopRecording(pid=own.pid))
    stop.assert_not_called()
    monkeypatch.setattr(run, "_active_process", own)
    await flash_editor.stop_execution(flash_editor.StopRecording(pid=own.pid))
    stop.assert_awaited_once()


def test_rest_button_runs_and_stops_owned_operation(gui_page, rest_setup, monkeypatch, tmp_path):
    process = SimpleNamespace(pid=876, returncode=None)

    async def launch(args, **kwargs):
        monkeypatch.setattr(run, "_active_process", process)
        kwargs["output_log"].write_text("Returning to saved rest\n")

    async def stop():
        process.returncode = 0
        atomic_json(
            flash_editor._execution_directory / "execution.json",
            {"operation": "rest", "mode": "robot", "state": "stopped"},
        )
        return {"status": "stopped"}

    monkeypatch.setattr(run, "_launch_subprocess", launch)
    monkeypatch.setattr(run, "stop_process", stop)
    page = gui_page
    page.get_by_role("button", name="FLASH", exact=True).click()
    page.get_by_role("button", name="Demo editor", exact=True).click()
    page.get_by_role("button", name="Return to saved rest", exact=True).click()
    expect(page.locator("#flash-editor-terminal")).to_contain_text("Returning to saved rest")
    expect(page.locator("#flash-editor-rest")).to_be_disabled()
    expect(page.locator("#flash-editor-apply")).to_be_disabled()
    page.locator("#flash-editor-stop").click()
    expect(page.locator("#flash-editor-rest")).to_be_enabled()
    expect(page.locator("#flash-editor-execution")).to_contain_text("stopped")
    expect(page.locator("#flash-editor-terminal")).to_be_visible()
    page.screenshot(path=str(tmp_path / "rest-control.png"))
