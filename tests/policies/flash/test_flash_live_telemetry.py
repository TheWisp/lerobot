"""The GUI consumes exactly the live and saved tracker views produced by a trial."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lerobot.gui.api.run import _flash_tracker_replay
from lerobot.policies.flash.live_telemetry import FlashLiveTelemetry
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def _estimate(source_s: float, estimator: str, evidence: Evidence) -> PoseEstimate:
    pose = np.eye(4)
    pose[0, 3] = 0.1
    return PoseEstimate(
        "object",
        pose,
        source_s,
        source_s + 0.01,
        evidence,
        "reference-v1",
        (True, True, True),
        (True, True, True),
        estimator=estimator,
    )


def test_live_tracker_layers_are_replayed_with_camera_frame(tmp_path: Path) -> None:
    extension = tmp_path / "flash" / "v1"
    extension.mkdir(parents=True)
    (extension / "episode_000000.jsonl").write_text(
        json.dumps({"frame_index": 7, "camera_source_time_s": 10.0})
        + "\n"
        + json.dumps({"frame_index": 8, "camera_source_time_s": 10.1})
        + "\n"
    )
    live = tmp_path / "live.json"
    telemetry = FlashLiveTelemetry(
        live,
        extension / "tracker_episode_000000.jsonl",
        {"object": np.array([0.02, 0.0, 0.0])},
        points_by_role={"object": np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])},
    )
    frame = RGBDFrame(
        9,
        np.zeros((12, 16, 3), dtype=np.uint8),
        np.ones((12, 16), dtype=np.uint16),
        np.eye(3),
        0.001,
        np.eye(4),
        10.1,
        10.11,
        "calibration-v1",
    )
    direct = _estimate(10.0, "appearance_rgbd", Evidence.OBSERVED)
    group = _estimate(10.1, "environment_group_motion", Evidence.PREDICTED)
    telemetry.frame(frame)
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:8, 4:10] = True
    telemetry.proposals((direct,), {"object": mask}, frame_id=9)
    telemetry.layer(direct)
    telemetry.layer(group)
    telemetry.accepted({"object": group})
    telemetry.close()

    current = json.loads(live.read_text())
    assert current["camera"]["frame_id"] == 9
    assert len(current["points_object_m"]["object"]) == 2
    assert current["observed_masks"]["object"]["frame_id"] == 9
    assert len(current["observed_masks"]["object"]["boundary_xy"]) >= 4
    assert set(current["layers"]["object"]) == {"appearance_rgbd", "environment_group_motion"}
    assert current["accepted"]["object"]["estimator"] == "environment_group_motion"
    replay = _flash_tracker_replay(tmp_path, 0)
    assert replay["layer_history"]
    assert replay["frames"][-1]["frame_index"] == 8
    assert replay["frames"][-1]["snapshot"]["accepted"] == current["accepted"]


def test_older_trace_does_not_claim_unrecorded_tracker_layers(tmp_path: Path) -> None:
    extension = tmp_path / "flash" / "v1"
    extension.mkdir(parents=True)
    (extension / "episode_000000.jsonl").write_text(
        json.dumps(
            {"frame_index": 0, "scheduled_s": 2.0, "poses": {"object": {"estimator": "object_point_motion"}}}
        )
        + "\n"
    )
    replay = _flash_tracker_replay(tmp_path, 0)
    assert not replay["layer_history"]
    assert replay["frames"][0]["snapshot"]["layers"] == {}
    assert replay["frames"][0]["snapshot"]["accepted"]["object"]["estimator"] == "object_point_motion"
