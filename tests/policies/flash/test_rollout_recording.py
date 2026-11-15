"""Run evidence crosses the real producer, filesystem and history-reader boundaries."""

import json
from dataclasses import replace

import numpy as np
import pytest

from lerobot.policies.flash.rollout_recording import RecordedRobot, RunJournal
from lerobot.policies.flash.tracking_history import RollingHistory, decode_record, read_index, read_packet
from tests.policies.flash.test_demo_execution import robot  # noqa: F401
from tests.policies.flash.test_shared_image_tracking import frame


def test_full_run_retains_camera_inputs_beyond_live_window(tmp_path):
    journal = RunJournal(tmp_path)
    history = RollingHistory(tmp_path / "camera", seconds=None, reserve_bytes=0, journal=journal)
    try:
        for timestamp in (0, 61, 181):
            measured = replace(
                frame(timestamp), source_time_s=float(timestamp), available_time_s=float(timestamp)
            )
            history.submit_frame(measured)
            history.attach_processed_frame(measured, "sensor")
    finally:
        history.close()
        journal.close()
    assert [r["frame"] for r in read_index(history.directory)["frames"]] == [0, 61, 181]
    restored, _, _, _ = decode_record(read_packet(history.directory, 0))
    np.testing.assert_array_equal(restored.rgb, frame(0).rgb)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e["frame_id"] for e in events if e["kind"] == "camera.capture"] == [0, 61, 181]


def test_robot_journal_retains_beginning_and_requested_accepted_readback(tmp_path):
    class ClippingRobot:
        def send_action(self, action):
            return {"joint": min(action["joint"], 1)}

        def get_state_observation(self):
            return {"joint": 0.8}

    journal = RunJournal(tmp_path, capacity=20000)
    recorded = RecordedRobot(ClippingRobot(), journal)
    for i in range(3605):
        recorded.send_action({"joint": i})
        recorded.get_state_observation()
    journal.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(events) == 3605 * 3
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert events[0]["requested"] == {"joint": 0}
    assert events[-3]["requested"] == {"joint": 3604}
    assert events[-2]["accepted"] == {"joint": 1}
    assert events[-1]["observed"] == {"joint": 0.8}


def test_full_recording_failure_cannot_be_cleared_by_later_capture(tmp_path):
    journal = RunJournal(tmp_path)
    history = RollingHistory(tmp_path / "camera", seconds=None, reserve_bytes=0, journal=journal)
    history.error = "disk failed"
    with pytest.raises(RuntimeError, match="disk failed"):
        history.submit_frame(frame(0))
    with pytest.raises(RuntimeError, match="disk failed"):
        history.close()
    journal.close()


def test_worker_preserves_failure_before_robot_construction(tmp_path, monkeypatch):
    from lerobot.policies.flash import demo_live_worker as worker
    from lerobot.policies.flash.demo_document import atomic_json

    atomic_json(tmp_path / "run.json", {"state": "running"})
    atomic_json(
        tmp_path / "config.json", {"demo_execution": {"simulate": False, "plan_directory": str(tmp_path)}}
    )
    monkeypatch.setattr(worker.sys, "argv", ["demo_live_worker", str(tmp_path)])
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    with pytest.raises(FileNotFoundError):
        worker.main()
    manifest = json.loads((tmp_path / "run.json").read_text())
    assert manifest["state"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "run.finished"
    assert "plan.json" in events[-1]["reason"]


def test_control_receipts_join_exact_published_revision_and_motor_io(robot, tmp_path):  # noqa: F811
    from threading import Event
    from time import perf_counter, sleep

    from lerobot.policies.flash.demo_live_worker import ExecutionLoop
    from tests.policies.flash.test_demo_execution import plan_for

    journal = RunJournal(tmp_path)
    loop = ExecutionLoop(
        plan_for(robot, reference="world"), robot, tmp_path, Event(), simulate=True, journal=journal
    )
    try:
        loop.start()
        deadline = perf_counter() + 5
        while loop.status not in {"complete", "failed"} and perf_counter() < deadline:
            sleep(0.01)
        assert loop.status == "complete", loop.error
    finally:
        loop.close()
        journal.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    publications = {e["revision"]: e for e in events if e["kind"] == "plan.published"}
    receipts = [e["tick"]["receipt"] for e in events if e["kind"] == "control.tick" and e["tick"]["receipt"]]
    commands = [e["requested"] for e in events if e["kind"] == "robot.command_requested"]
    assert len(receipts) == loop.commands_sent > 0
    assert [r["requested"]["action"] for r in receipts] == commands
    for receipt in receipts:
        actions = publications[receipt["revision"]]["actions"]
        assert receipt["requested"] in actions
    assert any(e["kind"] == "execution.range_finished" for e in events)
