"""The GUI consumes exactly the live and saved tracker views produced by a trial."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lerobot.gui.api.run import _flash_recordings, _flash_tracker_replay
from lerobot.policies.flash.live_telemetry import FlashLiveTelemetry
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.trajectory import AttachmentState, Evidence, PoseEstimate


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


def test_held_prediction_is_recorded_separately_from_stale_independent_pose(tmp_path: Path) -> None:
    class Kinematics:
        def forward_kinematics(self, joints):
            result = np.eye(4)
            result[0, 3] = joints[0]
            return result

    live = tmp_path / "live.json"
    telemetry = FlashLiveTelemetry(
        live, tmp_path / "tracker.jsonl", {"object": np.zeros(3)}, kinematics=Kinematics()
    )
    telemetry.accepted({"object": _estimate(1.0, "stationary_prior", Evidence.PREDICTED)})
    gripper_from_object = np.eye(4)
    gripper_from_object[0, 3] = 0.2
    telemetry.held_attachment(
        "object",
        AttachmentState(gripper_from_object, 2.0, 2.0, "reference-v1"),
        np.array([0.3]),
        3.0,
        independently_observed=False,
    )
    telemetry.close()
    snapshot = json.loads(live.read_text())
    assert snapshot["accepted"]["object"]["estimator"] == "stationary_prior"
    held = snapshot["held_prediction"]
    assert held["estimator"] == "grasp_attachment_prediction"
    assert held["pose"][0][3] == 0.5
    assert held["last_direct_observation_s"] is None


def test_flash_recordings_lists_source_datasets_with_outcomes(tmp_path: Path) -> None:
    older = tmp_path / "demos" / "older" / "flash" / "v1"
    latest = tmp_path / "demos" / "latest" / "flash" / "v1"
    incomplete = tmp_path / "demos" / "incomplete" / "flash" / "v1"
    for extension in (older, latest, incomplete):
        extension.mkdir(parents=True)
        (extension.parent.parent / "meta").mkdir()
        (extension.parent.parent / "meta" / "info.json").write_text('{"total_episodes": 1}')
    for extension in (older, latest):
        (extension / "episode_000000.jsonl").write_text("{}\n")
    (latest / "manifest.json").write_text(
        json.dumps({"episodes": {"0": {"frames": 42, "outcome": "interrupted", "reason": "IK failed"}}})
    )
    bundle = older / "prepared" / "one-demo"
    bundle.mkdir(parents=True)
    (bundle / "bundle.json").write_text('{"skill": "skill.json"}')
    (bundle / "skill.json").write_text('{"source_episode": 0}')
    recordings = _flash_recordings((tmp_path,))
    assert {item["dataset_root"] for item in recordings} == {
        str(older.parent.parent),
        str(latest.parent.parent),
    }
    assert next(item for item in recordings if item["kind"] == "demo") == {
        "kind": "demo",
        "dataset_root": str(older.parent.parent),
        "episode": 0,
        "modified_at_s": (bundle / "bundle.json").stat().st_mtime,
        "skills": ["one-demo"],
        "bundle_path": str((bundle / "bundle.json").resolve()),
        "prepared_skill": "one-demo",
    }
    assert (
        next(
            item
            for item in recordings
            if item["kind"] == "rollout" and item["dataset_root"] == str(latest.parent.parent)
        )["reason"]
        == "IK failed"
    )
