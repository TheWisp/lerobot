"""The GUI trial launcher allocates a fresh standard dataset per attempt."""

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
