"""Observation processing stays replayable across a failed pass and crosses the GUI worker boundary."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from lerobot.gui.api import run
from lerobot.gui.api.run import FlashObservationProcessRequest
from lerobot.policies.flash import observation_processing


def test_processing_keeps_last_complete_evidence_until_replacement(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    destination = root / "flash/v1/observation-evidence/episode_000000"
    destination.mkdir(parents=True)
    (destination / "previous.json").write_text("previous")

    def fail(_root, _masks, _config, *, output, **_kwargs):
        (output / "partial.json").write_text("partial")
        raise RuntimeError("tracking interrupted")

    monkeypatch.setattr(observation_processing, "_extract_to_directory", fail)
    with pytest.raises(RuntimeError, match="tracking interrupted"):
        observation_processing.extract(root, {}, tmp_path / "config.json", prompts_by_role={"box": "box"})
    assert (destination / "previous.json").read_text() == "previous"
    assert not list(destination.parent.glob(".episode_000000-pending-*"))

    def succeed(_root, _masks, _config, *, output, **_kwargs):
        (output / "complete.json").write_text("complete")
        return output

    monkeypatch.setattr(observation_processing, "_extract_to_directory", succeed)
    assert (
        observation_processing.extract(root, {}, tmp_path / "config.json", prompts_by_role={"box": "box"})
        == destination
    )
    assert (destination / "complete.json").read_text() == "complete"
    assert not (destination / "previous.json").exists()
    assert not list(destination.parent.glob(".episode_000000-previous-*"))


def test_gui_launches_shared_observation_worker_with_frozen_config(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    extension = root / "flash/v1"
    extension.mkdir(parents=True)
    (extension / "source.json").write_text(json.dumps({"status": "perception_only"}))
    launched = {}

    async def fake_launch(args, *, command, config):
        launched.update(args=args, command=command, config=config)
        monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=123))

    monkeypatch.setattr(run, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(run, "_launch_subprocess", fake_launch)
    monkeypatch.setattr(run, "_flash_observation_job", None)
    defaults = observation_processing.default_processing_config()
    request = FlashObservationProcessRequest(
        dataset_root=str(root),
        episode=0,
        prompts_by_role={"box": "sugar box", "can": "white bottle"},
        seed_points_by_role={"box": (50.0, 30.0)},
        **defaults,
    )
    result = asyncio.run(run.flash_observation_process(request))
    assert result["pid"] == 123
    assert asyncio.run(run.flash_observation_process_status(result["job_id"]))["status"] == "running"
    assert launched["command"] == "flash_observation"
    assert launched["args"][2:4] == ["-m", "lerobot.policies.flash.observation_processing"]
    assert "--prompt" in launched["args"] and "box=sugar box" in launched["args"]
    assert "--seed" in launched["args"] and "box=50.0,30.0" in launched["args"]
    config_path = Path(launched["args"][5])
    assert config_path.parent == extension / "observation-inputs"
    assert json.loads(config_path.read_text()) == defaults
    with pytest.raises(HTTPException, match="name at least one object"):
        asyncio.run(
            run.flash_observation_process(request.model_copy(update={"prompts_by_role": {"../bad": "box"}}))
        )
    config_path.write_text("{}")
    with pytest.raises(HTTPException, match="content hash"):
        asyncio.run(run.flash_observation_process(request))


@pytest.mark.parametrize("returncode,expected", [(0, "complete"), (2, "failed")])
def test_observation_job_retain_completion_after_generic_run_clears_command(
    monkeypatch, returncode, expected
):
    class FinishedProcess:
        async def wait(self):
            self.returncode = returncode

        returncode = None

    process = FinishedProcess()
    monkeypatch.setattr(run, "_active_process", process)
    monkeypatch.setattr(run, "_active_command", "flash_observation")
    monkeypatch.setattr(run, "_active_config", {})
    monkeypatch.setattr(run, "_flash_observation_job", {"job_id": "job", "status": "running"})
    monkeypatch.setattr(run, "_stream_tasks", [])
    monkeypatch.setattr(run, "_output_lines", ["ValueError: failed to track object"])
    monkeypatch.setattr(run, "_last_failure", None)
    monkeypatch.setattr(run, "_close_obs_reader", lambda: None)
    asyncio.run(run._wait_for_exit())
    assert run._active_command is None
    job = asyncio.run(run.flash_observation_process_status("job"))
    assert job["status"] == expected
    assert job["returncode"] == returncode
    if returncode:
        assert job["reason"] == "ValueError: failed to track object"
