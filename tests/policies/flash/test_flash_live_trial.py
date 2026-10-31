"""The GUI trial launcher keeps each attempt's diagnostic evidence with its dataset."""

import json
import sys

from lerobot.policies.flash import live_trial
from lerobot.policies.flash.live_trial import _new_attempt_settings


def test_new_attempt_settings_preserve_rig_config_and_allocate_new_dataset(tmp_path):
    settings = {
        "output_root": str(tmp_path / "trial"),
        "output_repo_id": "local/flash-trial",
        "robot_profile": "/some/calibrated/robot.json",
    }
    updated = _new_attempt_settings(settings, "20261020-001")
    assert updated["output_root"] == str(tmp_path / "trial-20261020-001")
    assert updated["output_repo_id"] == "local/flash-trial-20261020-001"
    assert updated["robot_profile"] == settings["robot_profile"]
    assert settings["output_root"] == str(tmp_path / "trial")


def test_gui_trial_persists_camera_and_planner_diagnostics_on_failure(tmp_path, monkeypatch):
    dataset_root = tmp_path / "failed-attempt"
    report = {
        "dataset_root": str(dataset_root),
        "outcome": "failure",
        "reason": "perception failed: no camera frame",
        "capture_metrics": {
            "timeouts": 2,
            "camera_reader": {"inflight_stage": "processing", "max_processing_s": 0.7},
        },
        "planner_compute_max_s": 0.4,
    }
    monkeypatch.setattr(live_trial, "trial", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(live_trial.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["flash-trial", "bundle", "config"])

    try:
        live_trial.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("a failed trial must exit unsuccessfully")

    saved = json.loads((dataset_root / "flash" / "v1" / "trial_report.json").read_text())
    assert saved == report
