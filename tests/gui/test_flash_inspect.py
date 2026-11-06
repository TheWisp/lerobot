"""The Run → FLASH entry validates a skill without touching robot hardware."""

import asyncio
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from lerobot.gui.api.run import (
    FlashConfigInspectRequest,
    FlashDatasetInspectRequest,
    FlashDemoPrepareRequest,
    FlashInspectRequest,
    FlashRunRequest,
    _flash_demo_preparation,
    _flash_recordings,
    inspect_flash_config,
    inspect_flash_dataset,
    inspect_flash_skill,
    prepare_flash_demo,
    start_flash,
    stop_process,
)
from lerobot.policies.flash.demo_replay import load_raw_legacy_demo_replay
from lerobot.policies.flash.inspect import inspect_dataset, inspect_skill
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample


def test_flash_inspect_subprocess_returns_spatial_intervals(tmp_path):
    identity = np.eye(4).tolist()
    path = tmp_path / "skill.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_repo_id": "example/dataset",
                "source_episode": 2,
                "milestones": [
                    {
                        "name": "placed",
                        "depends_on_roles": ["held", "target"],
                        "pose_relation": {
                            "moving_role": "held",
                            "reference_role": "target",
                            "reference_from_moving": identity,
                            "moving_version": "held-v1",
                            "reference_version": "target-v1",
                            "translation_tolerance_m": 0.01,
                            "rotation_tolerance_rad": None,
                            "moving_point_m": [0.04, 0, 0],
                            "moving_direction": [1, 0, 0],
                            "direction_tolerance_rad": 0.1,
                            "max_pose_age_s": 0.2,
                        },
                    }
                ],
                "intervals": [
                    {
                        "name": "grasp",
                        "mode": "demonstrated",
                        "relation": "ee_to_object",
                        "reference_role": "held",
                        "samples": [
                            {"time_s": t, "demo_ee": identity, "demo_reference": identity} for t in (3.0, 4.0)
                        ],
                    }
                ],
            }
        )
    )
    inspected = asyncio.run(inspect_flash_skill(FlashInspectRequest(skill_path=str(path))))
    assert inspected["source_repo_id"] == "example/dataset"
    assert inspected["intervals"][0]["demo_duration_s"] == 1.0
    assert inspected["milestones"][0]["pose_relation"]["translation_tolerance_m"] == 0.01
    assert inspected["milestones"][0]["pose_relation"]["moving_point_m"] == [0.04, 0, 0]
    assert inspected["milestones"][0]["pose_relation"]["direction_tolerance_rad"] == 0.1


def test_flash_trial_config_inspection_shows_effective_thresholds(tmp_path):
    from pathlib import Path

    from lerobot.policies.flash.causal_service import CausalTrackingConfig

    example = (
        Path(__file__).resolve().parents[2] / "src/lerobot/policies/flash/docs/trial_config.example.json"
    )
    config = json.loads(example.read_text())
    config["attachment"]["max_observation_shift_m"] = 0.014
    config["first_pose_timeout_s"] = 24
    path = tmp_path / "trial.json"
    path.write_text(json.dumps(config))
    result = asyncio.run(inspect_flash_config(FlashConfigInspectRequest(config_path=str(path))))
    assert result["settings"]["attachment"]["max_observation_shift_m"] == 0.014
    assert result["settings"]["first_pose_timeout_s"] == 24
    assert CausalTrackingConfig.from_config(result["settings"]["causal_tracking"])


def test_demo_editor_shows_the_path_that_a_legacy_skill_actually_uses(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("{}")
    np.savez(tmp_path / "showservo_demo.npz", t=np.arange(3, dtype=float))
    bundle = tmp_path / "flash/v1/prepared/original/bundle.json"
    bundle.parent.mkdir(parents=True)
    bundle.write_text(json.dumps({"skill": "skill.json", "references": []}))
    (bundle.parent / "skill.json").write_text(
        json.dumps(
            {
                "source_episode": 0,
                "intervals": [
                    {
                        "mode": "demonstrated",
                        "samples": [
                            {"source_frame_index": 0},
                            {"source_frame_index": 1},
                        ],
                    }
                ],
            }
        )
    )
    (bundle.parent / "annotation.json").write_text(json.dumps({"episode": 0}))
    original = {"keypoints": [{"t": 1.0, "move": "line"}, {"t": 2.0, "move": "line"}]}
    keypoints_path = bundle.parent / "keypoints.json"
    keypoints_path.write_text(json.dumps(original))

    result = _flash_demo_preparation(str(tmp_path), str(bundle), 0)
    assert result["path_selection_migrated"]
    assert result["keypoints_document"]["path_selection_version"] == 2
    assert result["keypoints_document"]["keypoints"][0]["move"] == "follow"
    assert json.loads(keypoints_path.read_text()) == original


def test_demo_preparation_versions_edited_inputs_and_launches_existing_importer(tmp_path, monkeypatch):
    from lerobot.gui.api import run as run_api

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("{}")
    bundle = tmp_path / "flash/v1/prepared/original/bundle.json"
    bundle.parent.mkdir(parents=True)
    bundle.write_text(json.dumps({"skill": "skill.json", "references": []}))
    (bundle.parent / "skill.json").write_text(json.dumps({"source_episode": 0}))
    annotation = {
        "episode": 0,
        "prompts_by_role": {"held": "dowel", "target": "ring"},
        "held_role": "held",
        "target_role": "target",
    }
    (bundle.parent / "annotation.json").write_text(json.dumps(annotation))
    keypoints = {"keypoints": [{"t": 1.0, "move": "line"}]}
    (bundle.parent / "keypoints.json").write_text(json.dumps(keypoints))
    assert _flash_demo_preparation(str(tmp_path), str(bundle), 0)["annotation"] == annotation

    launched = []

    async def launch(args, *, command, config):
        launched.append((args, command))
        run_api._active_process = SimpleNamespace(pid=321)

    monkeypatch.setattr(run_api, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(run_api, "_launch_subprocess", launch)
    monkeypatch.setenv("LEROBOT_FLASH_TRACKING_PYTHON", "/test/shared-vision/python")
    edited = {
        **annotation,
        "grasp_reference_frame": 17,
        "prompts_by_role": {**annotation["prompts_by_role"], "moved_obstacle": "yellow block"},
        "reference_frames_by_role": {"moved_obstacle": 24},
    }
    request = FlashDemoPrepareRequest(
        dataset_root=str(tmp_path),
        episode=0,
        bundle_path=str(bundle),
        annotation=edited,
        keypoints_document=keypoints,
        output_name="revised",
    )
    result = asyncio.run(prepare_flash_demo(request))
    assert result["status"] == "started"
    assert (
        json.loads((tmp_path / "flash/v1/preparation-inputs/revised/annotation.json").read_text()) == edited
    )
    assert (
        json.loads((tmp_path / "flash/v1/preparation-inputs/revised/keypoints.json").read_text()) == keypoints
    )
    args, command = launched[0]
    assert args[0] == "/test/shared-vision/python"
    assert command == "flash_prepare"
    assert args[2].endswith("/benchmarks/flash_prepare_legacy.py")
    assert args[-4:] == [
        "--keypoints",
        str(tmp_path / "flash/v1/preparation-inputs/revised/keypoints.json"),
        "--output-name",
        "revised",
    ]


def test_demo_preparation_snapshots_matching_scene_tracking_settings(tmp_path, monkeypatch):
    from dataclasses import asdict

    from lerobot.gui.api import run as run_api
    from lerobot.policies.flash.causal_service import CausalTrackingConfig

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("{}")
    (tmp_path / "showservo_demo.npz").write_bytes(b"fixture")
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    config = tmp_path / "trial.json"
    tracking = asdict(CausalTrackingConfig(60, 64, 12, 0.1, 0.005))
    settings = {
        "source_dataset_root": str(tmp_path),
        "calibration_path": str(calibration),
        "robot_arm": "left",
        "causal_tracking": tracking,
    }
    config.write_text(json.dumps(settings))

    async def launch(args, *, command, config):
        run_api._active_process = SimpleNamespace(pid=321)

    monkeypatch.setattr(run_api, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(run_api, "_launch_subprocess", launch)
    annotation = {
        "episode": 0,
        "calibration_path": str(calibration),
        "robot_arm": "left",
        "prompts_by_role": {"held": "dowel", "target": "ring"},
        "held_role": "held",
        "target_role": "target",
    }

    def request(name):
        return FlashDemoPrepareRequest(
            dataset_root=str(tmp_path),
            episode=0,
            annotation=annotation,
            keypoints_document={"keypoints": [{"t": 1.0}]},
            tracking_config_path=str(config),
            output_name=name,
        )

    result = asyncio.run(prepare_flash_demo(request("with-scene")))
    assert result["status"] == "started"
    saved = json.loads((tmp_path / "flash/v1/preparation-inputs/with-scene/annotation.json").read_text())
    assert saved["causal_tracking"] == tracking
    assert saved["causal_tracking_source"] == str(config)
    assert "causal_tracking" not in annotation

    settings["source_dataset_root"] = str(tmp_path / "another-demo")
    config.write_text(json.dumps(settings))
    with pytest.raises(HTTPException, match="another demo dataset"):
        asyncio.run(prepare_flash_demo(request("wrong-scene")))
    assert not (tmp_path / "flash/v1/preparation-inputs/wrong-scene").exists()


def test_raw_demo_can_be_reviewed_without_preexisting_flash_labels(tmp_path):
    dataset = tmp_path / "unprepared_recording"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(json.dumps({"total_episodes": 1, "fps": 30}))
    (dataset / "recording/rgb").mkdir(parents=True)
    (dataset / "recording/depth").mkdir(parents=True)
    np.savez(dataset / "showservo_demo.npz", t=np.array([0.0, 0.03]), t0=np.array(0.0))
    np.savetxt(dataset / "recording/times.txt", [0.0, 0.03])
    for index in (0, 1):
        cv2.imwrite(str(dataset / "recording/rgb" / f"{index:06d}.jpg"), np.zeros((4, 6, 3), np.uint8))
        cv2.imwrite(str(dataset / "recording/depth" / f"{index:06d}.png"), np.zeros((4, 6), np.uint16))

    listing = _flash_recordings((tmp_path,))
    assert len(listing) == 1
    assert listing[0]["bundle_path"] is None
    replay = load_raw_legacy_demo_replay(dataset, 0)
    assert replay["kind"] == "raw_demo"
    assert replay["frames"][1]["snapshot"]["camera"]["frame_id"] == 1
    assert replay["frames"][1]["snapshot"]["accepted"] == {}
    assert "prompts_by_role" not in replay


def test_raw_demo_preparation_accepts_object_roles_entered_in_editor(tmp_path, monkeypatch):
    from lerobot.gui.api import run as run_api

    dataset = tmp_path / "raw"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    (dataset / "showservo_demo.npz").touch()
    launches = []

    async def launch(args, *, command, config):
        launches.append((args, command, config))
        run_api._active_process = SimpleNamespace(pid=123)

    monkeypatch.setattr(run_api, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(run_api, "_launch_subprocess", launch)
    annotation = {
        "episode": 0,
        "prompts_by_role": {
            "workpiece": "wooden cylinder",
            "receiver": "green ring",
            "moved_obstacle": "yellow block",
        },
        "held_role": "workpiece",
        "target_role": "receiver",
    }
    request = FlashDemoPrepareRequest(
        dataset_root=str(dataset),
        episode=0,
        annotation=annotation,
        keypoints_document={"keypoints": [{"t": 1.0, "frame": "receiver", "move": "follow"}]},
        output_name="from-ui",
    )
    result = asyncio.run(prepare_flash_demo(request))
    assert result["status"] == "started"
    assert (
        json.loads((dataset / "flash/v1/preparation-inputs/from-ui/annotation.json").read_text())
        == annotation
    )
    assert launches[0][1] == "flash_prepare"


def test_new_preparation_candidate_opens_for_review_by_default(tmp_path):
    dataset = tmp_path / "demo"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(json.dumps({"total_episodes": 1}))
    prepared = dataset / "flash/v1/prepared"
    for name, stamp in [("accepted", 100), ("new", 200)]:
        path = prepared / name
        path.mkdir(parents=True)
        (path / "skill.json").write_text(json.dumps({"source_episode": 0}))
        bundle = path / "bundle.json"
        bundle.write_text(json.dumps({"skill": "skill.json"}))
        os.utime(bundle, (stamp, stamp))
    (prepared / "new/preparation-report.json").write_text(json.dumps({"review_status": "candidate"}))
    (prepared / "new/annotation.json").write_text(
        json.dumps({"held_role": "workpiece", "target_role": "receiver"})
    )
    recordings = _flash_recordings((tmp_path,))
    demo = next(recording for recording in recordings if recording["kind"] == "demo")
    assert Path(demo["bundle_path"]).parent.name == "new"
    assert demo["prepared_skill"] == "new"
    assert [version["name"] for version in demo["prepared_versions"]] == ["accepted", "new"]
    assert demo["prepared_versions"][-1]["candidate"]
    assert demo["interaction"] == {"held_role": "workpiece", "target_role": "receiver"}


def test_inspection_flags_held_interval_without_an_attachment_event(tmp_path):
    identity = np.eye(4)
    skill = FlashSkill(
        "example/dataset",
        0,
        (
            Interval(
                "place",
                MotionMode.DEMONSTRATED,
                Relation.HELD_TO_TARGET,
                "target",
                (
                    SpatialSample(0, identity, identity, identity, held_evidence="predicted"),
                    SpatialSample(1, identity, identity, identity, held_evidence="predicted"),
                ),
                reference_version="target-v1",
                held_reference_version="held-v1",
                held_role="held",
            ),
        ),
    )
    path = tmp_path / "skill.json"
    skill.save(path)
    inspected = inspect_skill(str(path))
    assert "place: matching demonstrated attachment event" in inspected["missing"]
    assert inspected["reference_frames_declared"]
    assert not inspected["runtime_bindings_complete"]
    assert inspected["intervals"][0]["held_evidence"] == {"predicted": 2}
    assert inspected["conflicted_held_samples"] == 0
    assert inspected["warnings"] == ["place: no independently observed held-object demo correction"]

    document = json.loads(path.read_text())
    document["intervals"][0]["samples"][0]["held_evidence"] = "conflicted_prediction"
    path.write_text(json.dumps(document))
    conflicted = inspect_skill(str(path))
    assert conflicted["conflicted_held_samples"] == 1
    assert conflicted["warnings"] == ["place: held-object demo pose has unresolved visual conflict"]


def test_flash_inspection_uses_regular_dataset_directory(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 2}))
    status = asyncio.run(
        inspect_flash_dataset(FlashDatasetInspectRequest(dataset_root=str(tmp_path), episode=1))
    )
    assert status["skills"] == []
    assert not status["has_flash_extension"]
    assert not status["can_launch"]
    assert "No FLASH spatial skill is stored for this episode" in status["missing"]

    identity = np.eye(4)
    skill = FlashSkill(
        "example/dataset",
        1,
        (
            Interval(
                "approach",
                MotionMode.DEMONSTRATED,
                Relation.EE_TO_OBJECT,
                "target",
                (SpatialSample(0, identity, identity), SpatialSample(1, identity, identity)),
                reference_version="target-v1",
            ),
        ),
    )
    sidecar = tmp_path / "flash/v1/skills/episode_000001.json"
    sidecar.parent.mkdir(parents=True)
    skill.save(sidecar)
    status = inspect_dataset(str(tmp_path), 1)
    assert status["has_flash_extension"]
    assert [item["kind"] for item in status["skills"]] == ["annotation"]
    assert status["skills"][0]["inspection"]["source_episode"] == 1
    assert "No prepared RGB-D object references are stored for this episode" in status["missing"]
    assert inspect_dataset(str(tmp_path), 0)["skills"] == []


def test_flash_launch_uses_shared_process_owner_and_prepared_episode(tmp_path, monkeypatch):
    from lerobot.gui.api import run as run_api

    bundle = tmp_path / "flash/v1/prepared/example/bundle.json"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("{}")
    config = tmp_path / "trial.json"
    config.write_text("{}")
    launched = []

    async def inspect(*_args):
        return {
            "skills": [
                {"kind": "prepared", "path": str(bundle), "inspection": {"runtime_bindings_complete": True}}
            ]
        }

    async def release():
        return None

    async def launch(args, *, command, config):
        launched.append((args, command, config))
        run_api._active_process = SimpleNamespace(pid=123)

    monkeypatch.setattr(run_api, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(run_api, "_inspect_flash_subprocess", inspect)
    monkeypatch.setattr(run_api, "_release_preview_cameras", release)
    monkeypatch.setattr(run_api, "_launch_subprocess", launch)
    monkeypatch.setattr(run_api, "_active_process", None)
    monkeypatch.setenv("LEROBOT_FLASH_TRACKING_PYTHON", "/test/shared-vision/python")
    request = FlashRunRequest(
        dataset_root=str(tmp_path), episode=0, bundle_path=str(bundle), config_path=str(config)
    )
    assert asyncio.run(start_flash(request))["pid"] == 123
    assert launched[0][0][0] == "/test/shared-vision/python"
    assert launched[0][1] == "flash"
    assert launched[0][0][-7:-2] == [
        str(bundle),
        str(config),
        "--auto-output",
        "--source-dataset-root",
        str(tmp_path),
    ]
    assert launched[0][0][-2] == "--telemetry-path"
    assert launched[0][0][-1] == str(run_api._flash_live_path)


def test_flash_launch_requires_explicit_override_for_conflicted_demo_pose(tmp_path, monkeypatch):
    from lerobot.gui.api import run as run_api

    bundle = tmp_path / "flash/v1/prepared/example/bundle.json"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("{}")
    config = tmp_path / "trial.json"
    config.write_text("{}")
    launched = []
    released = []

    async def inspect(*_args):
        return {
            "skills": [
                {
                    "kind": "prepared",
                    "path": str(bundle),
                    "inspection": {"runtime_bindings_complete": True, "conflicted_held_samples": 3},
                }
            ]
        }

    async def release():
        released.append(True)

    async def launch(args, *, command, config):
        launched.append(args)
        run_api._active_process = SimpleNamespace(pid=123)

    monkeypatch.setattr(run_api, "_ensure_no_active_process", lambda: None)
    monkeypatch.setattr(run_api, "_inspect_flash_subprocess", inspect)
    monkeypatch.setattr(run_api, "_release_preview_cameras", release)
    monkeypatch.setattr(run_api, "_launch_subprocess", launch)
    monkeypatch.setattr(run_api, "_active_process", None)
    request = FlashRunRequest(
        dataset_root=str(tmp_path), episode=0, bundle_path=str(bundle), config_path=str(config)
    )
    with pytest.raises(HTTPException, match="conflicted held-object poses"):
        asyncio.run(start_flash(request))
    assert not released and not launched
    approved = request.model_copy(update={"allow_conflicted_demo_pose": True})
    assert asyncio.run(start_flash(approved))["pid"] == 123
    assert launched[0][-1] == "--allow-conflicted-demo-pose"


def test_flash_cli_rejects_conflicted_demo_before_hardware_setup(tmp_path, monkeypatch):
    from lerobot.policies.flash.live_trial import trial
    from lerobot.policies.flash.prepared import PreparedFlashSkill

    identity = np.eye(4)
    skill = FlashSkill(
        "example/dataset",
        0,
        (
            Interval(
                "carry",
                MotionMode.DEMONSTRATED,
                Relation.HELD_TO_TARGET,
                "target",
                (
                    SpatialSample(0, identity, identity, identity, held_evidence="conflicted_prediction"),
                    SpatialSample(1, identity, identity, identity, held_evidence="conflicted_prediction"),
                ),
                reference_version="target-v1",
                held_reference_version="held-v1",
                held_role="held",
            ),
        ),
    )
    monkeypatch.setattr(PreparedFlashSkill, "load", staticmethod(lambda _path: SimpleNamespace(skill=skill)))
    with pytest.raises(ValueError, match="conflicted held-object poses"):
        trial(tmp_path / "bundle.json", tmp_path / "trial.json", preflight=False)


def test_flash_stop_waits_for_trial_dataset_save(monkeypatch):
    from lerobot.gui.api import run as run_api

    class Process:
        pid = 123
        stdin = None
        returncode = None

        def __init__(self):
            self.signals = []

        def send_signal(self, value):
            self.signals.append(value)
            self.returncode = 0

        async def wait(self):
            return self.returncode

    process = Process()
    monkeypatch.setattr(run_api, "_active_process", process)
    monkeypatch.setattr(run_api, "_active_command", "flash")
    monkeypatch.setattr(run_api, "_active_config", {})
    monkeypatch.setattr(run_api, "_active_phase", None)
    monkeypatch.setattr(run_api, "_graceful_stop_inflight", False)
    monkeypatch.setattr(run_api, "_close_obs_reader", lambda: None)
    result = asyncio.run(stop_process())
    assert result["graceful"]
    assert process.signals == [signal.SIGINT]
