import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lerobot.gui.api import flash_editor, flash_tracking, run
from lerobot.policies.flash.demo_document import atomic_json
from lerobot.policies.flash.rollout_recording import RunJournal


@pytest.mark.asyncio
async def test_recorder_completion_registers_demo_without_library_poll(tmp_path, monkeypatch):
    from tests.mocks.editorrobot import calibrated_profile

    monkeypatch.setattr(flash_editor, "_recording", None)
    monkeypatch.setattr(flash_editor, "_record_process", None)
    done = asyncio.Event()
    process = SimpleNamespace(returncode=None)

    async def finish():
        await done.wait()
        process.returncode = 0

    process.wait = finish
    monkeypatch.setattr(run, "_active_process", process)
    monkeypatch.setattr(run, "start_record", AsyncMock(return_value={"pid": 42}))
    recording = await flash_editor.record(
        run.RecordRequest(
            robot=calibrated_profile(tmp_path),
            teleop={"type": "scripted_bimanual_ee", "fields": {}},
            repo_id="test/auto",
            root=str(tmp_path / "dataset"),
            single_task="test",
        )
    )
    tasks = list(flash_editor._record_tasks)
    done.set()
    await asyncio.gather(*tasks)
    persisted = json.loads((flash_editor.EDITOR_DIR / "demos.json").read_text())
    assert [d["id"] for d in persisted["demos"]] == [recording["id"]]
    assert persisted["demos"][0]["name"].startswith("Demo ·")


@pytest.mark.asyncio
async def test_rollout_survives_new_worker_and_restart_with_joinable_events(tmp_path, monkeypatch):
    identity = "rollout-" + "a" * 32
    destination = flash_editor.EDITOR_DIR / "rollouts" / identity
    atomic_json(destination / "run.json", {"id": identity, "state": "complete", "created_at": 1})
    journal = RunJournal(destination)
    journal.emit("control.tick", tick={"tick": 0, "observed": {"joint": 1}})
    journal.emit("plan.published", revision=1)
    journal.close()
    monkeypatch.setattr(flash_tracking, "_check_registration_backend", AsyncMock())
    monkeypatch.setattr(run, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(run, "_launch_subprocess", AsyncMock())
    monkeypatch.setattr(run, "_active_process", None)
    from lerobot.gui.api import robot

    monkeypatch.setattr(robot, "release_preview_cameras", lambda: None)
    monkeypatch.setattr(flash_tracking, "history_root", lambda: tmp_path / "history")
    monkeypatch.setattr(flash_tracking, "_directory", destination)
    await flash_tracking.start_worker(flash_tracking.TrackingStart(depth_source="sensor"))
    monkeypatch.setattr(flash_editor, "_execution_process", None)
    monkeypatch.setattr(flash_editor, "_execution_directory", None)
    assert (await flash_editor.rollouts())[0]["state"] == "complete"
    page = await flash_editor.rollout_events(identity, cursor=0, limit=1, kind="")
    assert page["events"][0]["tick"]["observed"] == {"joint": 1}
    next_page = await flash_editor.rollout_events(identity, cursor=page["cursor"], limit=1, kind="")
    assert next_page["events"][0]["revision"] == 1
    assert not next_page["more"]
    # A process killed mid-write leaves earlier events readable, with no fake
    # final event or silently advanced cursor over the incomplete line.
    with (destination / "events.jsonl").open("a") as stream:
        stream.write('{"kind":')
    partial = await flash_editor.rollout_events(identity, cursor=next_page["cursor"], limit=1, kind="")
    assert partial["events"] == []
    assert partial["cursor"] == next_page["cursor"]


@pytest.mark.asyncio
async def test_rollout_console_survives_memory_buffer_reset(tmp_path, monkeypatch):
    identity = "rollout-" + "b" * 32
    destination = flash_editor.EDITOR_DIR / "rollouts" / identity
    atomic_json(destination / "run.json", {"id": identity, "state": "complete", "created_at": 1})
    from lerobot.gui.api import overlays, robot

    monkeypatch.setattr(robot, "release_preview_cameras", lambda: None)
    monkeypatch.setattr(overlays, "stop_data_publisher", lambda: None)
    monkeypatch.setattr(run, "_close_obs_reader", lambda: None)
    for name, value in {
        "_active_process": None,
        "_active_command": None,
        "_active_config": None,
        "_active_phase": None,
        "_active_loop": None,
        "_last_failure": None,
        "_stream_tasks": [],
        "_output_lines": [],
    }.items():
        monkeypatch.setattr(run, name, value)
    await run._launch_subprocess(
        [
            sys.executable,
            "-c",
            "import sys; print('Camera ready'); print('Motor 7 unavailable', file=sys.stderr)",
        ],
        command="flash_live",
        config={},
        output_log=destination / "output.log",
    )
    async with asyncio.timeout(10):
        while run._active_process is not None:
            await asyncio.sleep(0.01)
    monkeypatch.setattr(run, "_output_lines", [])
    text = (await flash_editor.rollout_log(identity))["text"]
    assert "\nCamera ready\n" in text
    assert "\n[stderr] Motor 7 unavailable\n" in text
    assert text.endswith("--- Process exited with code 0 ---\n")


@pytest.mark.asyncio
async def test_old_rollout_log_reports_recorded_reason_without_fabricating_console(tmp_path):
    identity = "rollout-" + "c" * 32
    destination = flash_editor.EDITOR_DIR / "rollouts" / identity
    atomic_json(destination / "run.json", {"id": identity, "state": "failed", "created_at": 1})
    journal = RunJournal(destination)
    journal.emit("run.started", mode="robot")
    journal.emit("run.finished", state="failed", reason="Motor 7 unavailable")
    journal.close()
    text = (await flash_editor.rollout_log(identity))["text"]
    assert "Console output was not saved" in text
    assert "Motor 7 unavailable" in text


@pytest.mark.asyncio
async def test_output_write_failure_does_not_stop_draining_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_output_lines", [])
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "for i in range(3000): print('worker line', i)",
        stdout=asyncio.subprocess.PIPE,
    )
    # A directory is unwritable as a log file, like an unavailable output disk.
    await asyncio.wait_for(run._read_stream(process.stdout, output_log=tmp_path), timeout=10)
    assert await asyncio.wait_for(process.wait(), timeout=10) == 0
    assert "worker line 2999" in run._output_lines
