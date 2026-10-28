"""The GUI consumes exactly the live and saved tracker views produced by a trial."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

import lerobot.gui.api.run as run_api
from lerobot.gui.api.run import _flash_recordings, _flash_tracker_replay
from lerobot.policies.flash.live_telemetry import FlashLiveTelemetry, live_frame_path
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.pose_disagreement import (
    FreeSpaceComparison,
    FreeSpaceSupport,
    FreeSpaceVerdict,
    PoseComparison,
    PoseComparisonVerdict,
    SurfaceSupport,
)
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
    support = SurfaceSupport(20, 16, 0.8, 0.003)
    telemetry.point_comparison(
        "object", PoseComparison(PoseComparisonVerdict.AMBIGUOUS, 0.012, 0.1, support, support)
    )
    telemetry.free_space_comparison(
        "object",
        FreeSpaceComparison(
            FreeSpaceVerdict.CONTRADICTED,
            FreeSpaceSupport(24, 20, 0.04, 0.9),
            0.01,
        ),
    )
    telemetry.accepted({"object": group})
    deadline = time.monotonic() + 2.0
    rgb_file = live_frame_path(live, frame.frame_id, "rgb")
    depth_file = live_frame_path(live, frame.frame_id, "depth")
    while not depth_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert np.frombuffer(depth_file.read_bytes(), dtype="<u2").reshape(12, 16)[0, 0] == 1
    assert cv2.imdecode(np.frombuffer(rgb_file.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR).shape == (
        12,
        16,
        3,
    )
    telemetry.close()
    assert not depth_file.exists()

    current = json.loads(live.read_text())
    assert current["camera"]["frame_id"] == 9
    assert len(current["points_object_m"]["object"]) == 2
    assert current["observed_masks"]["object"]["frame_id"] == 9
    assert len(current["observed_masks"]["object"]["boundary_xy"]) >= 4
    assert set(current["layers"]["object"]) == {"appearance_rgbd", "environment_group_motion"}
    assert current["accepted"]["object"]["estimator"] == "environment_group_motion"
    assert current["direct_point_comparisons"]["object"]["verdict"] == "ambiguous"
    assert current["direct_free_space_comparisons"]["object"]["verdict"] == "contradicted"
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.front": {"dtype": "image"},
                    "observation.images.front_depth": {"dtype": "image", "info": {"is_depth_map": True}},
                }
            }
        )
    )
    replay = _flash_tracker_replay(tmp_path, 0)
    assert replay["kind"] == "rollout"
    assert replay["camera_features"] == {
        "rgb": "observation.images.front",
        "depth": "observation.images.front_depth",
    }
    assert replay["layer_history"]
    assert replay["frames"][-1]["frame_index"] == 8
    assert replay["frames"][-1]["snapshot"]["accepted"] == current["accepted"]
    assert replay["frames"][-1]["snapshot"]["direct_point_comparisons"] == current["direct_point_comparisons"]
    assert (
        replay["frames"][-1]["snapshot"]["direct_free_space_comparisons"]
        == current["direct_free_space_comparisons"]
    )


@pytest.mark.asyncio
async def test_live_display_endpoint_serves_only_the_active_frame_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    live = tmp_path / "trial.json"
    depth = live_frame_path(live, 9, "depth")
    depth.parent.mkdir()
    depth.write_bytes(np.array([1000, 1001], dtype="<u2").tobytes())
    monkeypatch.setattr(run_api, "_flash_live_path", live)
    monkeypatch.setattr(run_api, "_active_command", "flash")
    monkeypatch.setattr(run_api, "is_run_active", lambda: True)
    response = await run_api.flash_live_frame(9, "depth")
    assert response.body == depth.read_bytes()
    with pytest.raises(HTTPException) as missing:
        await run_api.flash_live_frame(8, "depth")
    assert missing.value.status_code == 404
    with pytest.raises(HTTPException) as invalid:
        await run_api.flash_live_frame(9, "../../other")
    assert invalid.value.status_code == 400
    monkeypatch.setattr(run_api, "is_run_active", lambda: False)
    with pytest.raises(HTTPException) as stopped:
        await run_api.flash_live_frame(9, "depth")
    assert stopped.value.status_code == 404


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


def test_tracker_replay_requires_the_same_recorded_camera_frame_id(tmp_path: Path) -> None:
    extension = tmp_path / "flash" / "v1"
    extension.mkdir(parents=True)
    (extension / "episode_000000.jsonl").write_text(
        "\n".join(
            json.dumps({"frame_index": index, "camera_frame_id": frame_id, "camera_source_time_s": source_s})
            for index, frame_id, source_s in ((0, 65, 10.0), (1, 66, 10.1))
        )
        + "\n"
    )
    (extension / "tracker_episode_000000.jsonl").write_text(
        "\n".join(
            json.dumps({"camera": {"frame_id": frame_id, "source_time_s": source_s}, "accepted": {}})
            for frame_id, source_s in ((4, 9.99), (66, 10.01))
        )
        + "\n"
    )
    replay = _flash_tracker_replay(tmp_path, 0)
    assert [
        (frame["frame_index"], frame["alignment"], frame["snapshot"]["camera"]["frame_id"])
        for frame in replay["frames"]
    ] == [(1, "camera_frame_id", 66)]


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
        "prepared_versions": [
            {
                "name": "one-demo",
                "bundle_path": str((bundle / "bundle.json").resolve()),
                "candidate": False,
            }
        ],
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
