"""The Run → FLASH entry validates a skill without touching robot hardware."""

import asyncio
import importlib.util
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
import pytest

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
    annotation = {"episode": 0, "prompts_by_role": {"held": "dowel", "target": "ring"}}
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
    edited = {**annotation, "grasp_reference_frame": 17}
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
        "prompts_by_role": {"workpiece": "wooden cylinder", "receiver": "green ring"},
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


def test_new_preparation_candidate_does_not_replace_default_demo(tmp_path):
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
    recordings = _flash_recordings((tmp_path,))
    demo = next(recording for recording in recordings if recording["kind"] == "demo")
    assert Path(demo["bundle_path"]).parent.name == "accepted"
    assert [version["name"] for version in demo["prepared_versions"]] == ["accepted", "new"]
    assert demo["prepared_versions"][-1]["candidate"]


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
    assert inspected["warnings"] == ["place: no independently observed held-object demo pose"]


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


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright is not installed")
def test_run_flash_subtab_inspects_without_launching_robot(tmp_path, gui_page):
    identity = np.eye(4).tolist()
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 1}))
    path = tmp_path / "flash/v1/skills/episode_000000.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_repo_id": "example/dataset",
                "source_episode": 0,
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
                            "max_pose_age_s": 0.2,
                        },
                    }
                ],
                "intervals": [
                    {
                        "name": "place",
                        "mode": "demonstrated",
                        "relation": "ee_to_object",
                        "reference_role": "target",
                        "samples": [
                            {"time_s": t, "demo_ee": identity, "demo_reference": identity} for t in (0.0, 1.0)
                        ],
                    }
                ],
            }
        )
    )
    page = gui_page
    page.evaluate("switchTab('run')")
    page.evaluate("() => selectWorkflow('flash')")
    assert page.locator("#run-episode-control").is_hidden()
    assert page.locator("#run-launch-btn").inner_text() == "Run FLASH"
    assert page.locator("#run-launch-btn").is_disabled()
    page.get_by_role("button", name="Set up live trial").click()
    page.locator("#run-flash-root").fill(str(tmp_path))
    page.locator("#run-flash-inspect-btn").click()
    page.wait_for_function(
        "() => document.getElementById('run-flash-status').textContent.includes('No runnable prepared skill')"
    )
    assert page.locator("#run-launch-btn").is_disabled()
    assert "Prepared skill:" not in page.locator("#run-terminal").inner_text()


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright is not installed")
def test_run_flash_view_demo_uses_data_view_without_inspection_or_robot_launch(gui_page):
    page = gui_page
    requests = []

    def flash_response(route):
        if route.request.url.endswith("/recordings"):
            route.fulfill(json=[])
        else:
            requests.append(route.request.url)
            route.fulfill(status=500)

    page.route("**/api/run/flash/**", flash_response)
    page.evaluate("switchTab('run')")
    page.evaluate("() => selectWorkflow('flash')")
    page.evaluate(
        """() => {
            window.openDataset = async root => {
                datasets.demo = { id: 'demo', root, repo_id: 'demo', fps: 30 };
                episodes.demo = [{ episode_index: 2, length: 42 }];
            };
            window.selectEpisode = (id, episode, length) => {
                window.flashViewedEpisode = { id, episode, length };
            };
        }"""
    )
    page.get_by_role("button", name="Set up live trial").click()
    page.locator("#run-flash-root").fill("/tmp/demo")
    page.locator("#run-flash-episode").fill("2")
    page.locator("#run-flash-view-btn").click()
    page.wait_for_function("() => window.flashViewedEpisode !== undefined")
    assert page.evaluate("window.flashViewedEpisode") == {"id": "demo", "episode": 2, "length": 42}
    assert "active" in page.locator('.tab[data-tab="data"]').get_attribute("class").split()
    assert requests == []


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright is not installed")
def test_run_flash_lists_demos_and_shows_extracted_pose_overlay(gui_page):
    page = gui_page
    identity = np.eye(4).tolist()
    page.route(
        "**/api/run/flash/recordings",
        lambda route: route.fulfill(
            json=[
                {
                    "kind": "demo",
                    "dataset_root": "/tmp/one-demo",
                    "episode": 2,
                    "skills": ["spatial-skill", "new-skill"],
                    "bundle_path": "/tmp/one-demo/flash/v1/prepared/spatial-skill/bundle.json",
                    "prepared_skill": "spatial-skill",
                    "prepared_versions": [
                        {
                            "name": "spatial-skill",
                            "bundle_path": "/tmp/one-demo/flash/v1/prepared/spatial-skill/bundle.json",
                            "candidate": False,
                        },
                        {
                            "name": "new-skill",
                            "bundle_path": "/tmp/one-demo/flash/v1/prepared/new-skill/bundle.json",
                            "candidate": True,
                        },
                    ],
                    "modified_at_s": 12.0,
                }
            ]
        ),
    )
    page.route(
        "**/api/run/flash/demo/replay?*",
        lambda route: route.fulfill(
            json={
                "available": True,
                "kind": "demo",
                "bundle_path": parse_qs(urlsplit(route.request.url).query)["bundle_path"][0],
                "tracker_candidate": {
                    "candidate_id": "test-tracker",
                    "tracker": "trackeverything",
                    "role": "object",
                }
                if "candidate_id" in parse_qs(urlsplit(route.request.url).query)
                else None,
                "fps": 30,
                "geometry": {
                    "points_object_m": {"object": [[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]]},
                    "centers_object_m": {"object": [0.03, 0.03, 1]},
                },
                "frames": [
                    {
                        "frame_index": 4,
                        "snapshot": {
                            "recorded_at_s": 1.0,
                            "camera": {
                                "frame_id": 4,
                                "width": 640,
                                "height": 480,
                                "intrinsics": identity,
                                "base_from_camera": identity,
                            },
                            "accepted": {
                                "object": {
                                    "pose": identity,
                                    "evidence": "predicted",
                                    "source_time_s": 1.0,
                                    "estimator": "demo_reference",
                                },
                                "__gripper__": {
                                    "pose": identity,
                                    "evidence": "recorded",
                                    "source_time_s": 1.0,
                                    "estimator": "demo_ee",
                                },
                            },
                            "held_prediction": None,
                            "layers": {
                                "object": {
                                    "candidate_tracker": {
                                        "pose": identity,
                                        "evidence": "candidate",
                                        "estimator": "trackeverything",
                                        "source_time_s": 1.0,
                                    }
                                }
                            }
                            if "candidate_id" in parse_qs(urlsplit(route.request.url).query)
                            else {},
                            "demo": {
                                "phase": "demonstrated_sample",
                                "time_s": 1.0,
                                "camera_arm_delta_s": 0.01,
                                "interval_name": "contact",
                                "relation": "ee_to_object",
                                "reference_role": "object",
                                "source_arm_frame_index": 5,
                                "sample_time_s": 1.0,
                                "event_id": None,
                            },
                        },
                    }
                ],
            }
        ),
    )
    page.evaluate("switchTab('run')")
    page.evaluate("() => selectWorkflow('flash')")
    page.wait_for_function(
        "() => document.querySelectorAll('#run-flash-demos .flash-recording').length === 1"
    )
    assert "spatial-skill" in page.locator("#run-flash-demos").inner_text()
    assert "review candidate" in page.locator("#run-flash-demos .flash-recording-version").inner_text()
    page.locator("#run-flash-demos .flash-recording").click()
    page.wait_for_function("() => document.querySelector('#flash-tracker-mode').textContent === 'Demo'")
    assert "flash-reviewing" in page.locator("#tab-run").get_attribute("class").split()
    assert page.locator("#run-terminal-container").is_hidden()
    assert page.locator("#overlays-panel-run").is_hidden()
    assert "active" in page.locator('.tab[data-tab="run"]').get_attribute("class").split()
    assert "Reference pose used by skill" in page.locator("#flash-tracker-content").inner_text()
    assert "predicted" in page.locator("#flash-tracker-content").inner_text()
    assert (
        "no mask here does not mean segmentation was lost"
        in page.locator("#flash-tracker-content").inner_text()
    )
    assert "/api/run/flash/demo/frame?" in page.locator("#rerun-viewer .flash-replay-grid img").get_attribute(
        "src"
    )
    viewer = page.locator("#rerun-viewer").bounding_box()
    controls = page.locator("#flash-tracker-replay-controls")
    control_y = controls.bounding_box()["y"]
    assert control_y >= viewer["y"] + viewer["height"] - 1
    page.locator("#flash-tracker-content").evaluate("el => el.append(' extra detail '.repeat(100))")
    assert controls.bounding_box()["y"] == control_y
    page.locator("#run-flash-demos .flash-recording-version").select_option(
        "/tmp/one-demo/flash/v1/prepared/new-skill/bundle.json"
    )
    page.wait_for_function("() => flashReplay?.bundle_path?.includes('/new-skill/')")
    assert "Viewing: new-skill" in page.locator("#run-flash-demos .flash-recording-meta").inner_text()
    page.route(
        "**/api/run/flash/demo/preparation?*",
        lambda route: route.fulfill(
            json={
                "annotation": {
                    "episode": 2,
                    "reference_frame": 4,
                    "grasp_reference_frame": 4,
                    "grasp_time_s": 1.0,
                    "release_time_s": 2.0,
                    "start_anchored_until_s": 1.5,
                    "held_role": "object",
                    "prompts_by_role": {"object": "dowel"},
                    "axial_symmetry_roles": {},
                },
                "keypoints_document": {"keypoints": [{"t": 1.0, "move": "line", "frame": "object"}]},
                "reference_masks": {"object": {"frame_index": 4, "available": True}},
            }
        ),
    )
    candidate = {
        "candidate_id": "test-tracker",
        "tracker": "trackeverything",
        "role": "object",
        "valid_frames": 1,
        "total_frames": 1,
    }
    page.route(
        "**/api/run/flash/demo/trackers?*",
        lambda route: route.fulfill(
            json={
                "trackers": [{"id": "trackeverything", "name": "TrackEverything", "runnable": True}],
                "candidates": [candidate],
            }
        ),
    )
    process_requests = []

    def process_tracker(route):
        process_requests.append(route.request.post_data_json)
        route.fulfill(json=candidate)

    page.route("**/api/run/flash/demo/trackers/process", process_tracker)
    page.route(
        "**/api/run/flash/demo/track-points?*",
        lambda route: route.fulfill(
            json={
                "available": True,
                "candidate_id": "test-tracker",
                "frames": [4],
                "fit_threshold_mm": 5,
                "roles": [
                    {"role": "object", "count": 3, "positions": [[[600, 400, 0], [610, 400, 7], None]]}
                ],
            }
        ),
    )
    page.locator("#flash-demo-edit-toggle").click()
    page.wait_for_function("() => !!document.querySelector('#flash-demo-editor')")
    assert "Edit demo" in page.locator("#flash-demo-editor").inner_text()
    assert page.locator("#flash-demo-editor-frame").inner_text().startswith("Frame 4")
    assert page.locator("#flash-demo-editor .flash-editor-pane[data-section='objects']").is_visible()
    page.get_by_role("button", name="Draw replacement mask").click()
    page.evaluate("""() => {
        const image = document.querySelector('#rerun-viewer .flash-replay-grid img');
        const box = image.getBoundingClientRect();
        for (const [x, y] of [[0.4, 0.4], [0.6, 0.4], [0.5, 0.6]])
            _flashEditorPolygonClick({clientX: box.left + box.width * x, clientY: box.top + box.height * y});
    }""")
    page.get_by_role("button", name="Finish polygon").click()
    assert "Manual object mask: 3 vertices" in page.locator("#flash-demo-editor-polygon-status").inner_text()
    assert len(page.evaluate("flashDemoEditor.annotation.reference_polygons_by_role.object")) == 3
    page.locator(".flash-editor-settings > summary").click()
    assert page.get_by_label("Processing scope", exact=True).input_value() == ""
    assert page.get_by_label("Grow points from later views").is_checked()
    page.get_by_label("Maximum points per object").fill("256")
    page.get_by_label("Maximum points per object").press("Tab")
    page.get_by_label("Maximum new points per view").fill("64")
    page.get_by_label("Maximum new points per view").press("Tab")
    page.get_by_role("button", name="Process demo and preview", exact=True).click()
    page.wait_for_function("() => flashReplay.tracker_candidate?.candidate_id === 'test-tracker'")
    page.wait_for_function("() => !document.getElementById('flash-track-tools').hidden")
    assert "Raw 3D point paths" in page.locator("#flash-tracker-content").inner_text()
    arc_count = page.evaluate("""() => {
        let n = 0; const original = CanvasRenderingContext2D.prototype.arc;
        CanvasRenderingContext2D.prototype.arc = function(...args) { n++; return original.apply(this, args); };
        try { _flashDrawOverlay(); } finally { CanvasRenderingContext2D.prototype.arc = original; }
        return n;
    }""")
    page.locator("#flash-track-visible").uncheck()
    without_points = page.evaluate("""() => {
        let n = 0; const original = CanvasRenderingContext2D.prototype.arc;
        CanvasRenderingContext2D.prototype.arc = function(...args) { n++; return original.apply(this, args); };
        try { _flashDrawOverlay(); } finally { CanvasRenderingContext2D.prototype.arc = original; }
        return n;
    }""")
    assert arc_count - without_points == 2
    page.locator("#flash-track-visible").check()
    point_xy = page.evaluate("""() => {
        const box = document.querySelector('#rerun-viewer .flash-pose-canvas').getBoundingClientRect();
        const camera = flashSnapshot.camera;
        const scale = Math.min(box.width / camera.width, box.height / camera.height);
        return [box.left + (box.width - camera.width * scale) / 2 + 600 * scale,
            box.top + (box.height - camera.height * scale) / 2 + 400 * scale];
    }""")
    page.mouse.click(*point_xy)
    assert page.evaluate("flashSelectedTrack") == {"role": "object", "index": 0}
    assert process_requests == [
        {
            "dataset_root": "/tmp/one-demo",
            "bundle_path": "/tmp/one-demo/flash/v1/prepared/new-skill/bundle.json",
            "episode": 2,
            "tracker": "trackeverything",
            "role": None,
            "max_points_per_object": 256,
            "new_points_per_view": 64,
            "surface_tolerance_m": 0.005,
        }
    ]
    assert "Candidate pose on this camera frame" in page.locator("#flash-tracker-content").inner_text()
    colors = page.evaluate("""() => {
        const colors = [], original = CanvasRenderingContext2D.prototype.stroke;
        CanvasRenderingContext2D.prototype.stroke = function(...args) {
            colors.push(this.strokeStyle); return original.apply(this, args);
        };
        try { _flashDrawOverlay(); } finally { CanvasRenderingContext2D.prototype.stroke = original; }
        return colors;
    }""")
    assert "#f472b6" in colors, "candidate must remain visible above the saved pose"
    assert colors.index("#f472b6") < colors.index("#22d3ee"), "raw points must remain visible above the fit"
    page.locator(".flash-editor-keypoint").filter(has_text="trackeverything").hover()
    assert page.evaluate("flashHighlightedOverlay?.estimator") == "candidate_tracker"
    assert page.locator("#flash-demo-editor .flash-editor-footer").is_visible()
    applied = []

    def apply_tracker(route):
        applied.append(route.request.post_data_json)
        route.fulfill(
            json={
                "bundle_path": "/tmp/one-demo/flash/v1/prepared/applied/bundle.json",
                "report": {"changed_reference_samples": 1, "changed_held_samples": 0},
            }
        )

    page.route("**/api/run/flash/demo/trackers/apply", apply_tracker)
    page.get_by_role("button", name="Apply as new demo version", exact=True).click()
    page.wait_for_function("() => flashReplay.bundle_path.endsWith('/applied/bundle.json')")
    assert applied[0]["candidate_id"] == "test-tracker"
    assert page.locator("#flash-demo-editor").count() == 0


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright is not installed")
def test_run_flash_subtab_launches_selected_prepared_skill(gui_page):
    page = gui_page
    bundle = "/tmp/example/flash/v1/prepared/ready/bundle.json"
    requests = []
    inspections = []

    def inspect_response(route):
        inspections.append(route.request.post_data_json)
        route.fulfill(
            json={
                "dataset_root": "/tmp/example",
                "episode": 0,
                "skills": [
                    {
                        "kind": "prepared",
                        "path": bundle,
                        "inspection": {
                            "runtime_bindings_complete": True,
                            "intervals": [],
                            "attachment_transitions": [],
                            "prepared_references": [],
                            "milestones": [
                                {
                                    "name": "placed",
                                    "pose_relation": {
                                        "moving_role": "held",
                                        "reference_role": "target",
                                        "translation_tolerance_m": 0.01,
                                    },
                                }
                            ],
                            "missing": [],
                            "warnings": [],
                        },
                    }
                ],
                "missing": [],
                "warnings": [],
                "can_launch": True,
            }
        )

    def launch_response(route):
        requests.append(route.request.post_data_json)
        route.fulfill(json={"status": "started", "command": "flash", "pid": 123})

    page.route("**/api/run/flash/dataset-inspect", inspect_response)
    page.route("**/api/run/flash/run", launch_response)
    page.route(
        "**/api/run/flash/config-inspect",
        lambda route: route.fulfill(
            json={
                "path": "/tmp/trial.json",
                "settings": {
                    "attachment": {"max_observation_shift_m": 0.015, "max_unverified_age_s": None},
                    "first_pose_timeout_s": 20,
                    "causal_tracking": {
                        "max_pose_disagreement_translation_m": 0.01,
                        "max_pose_disagreement_rotation_rad": 0.15,
                    },
                },
            }
        ),
    )
    page.evaluate("switchTab('run')")
    page.evaluate("() => selectWorkflow('flash')")
    page.get_by_role("button", name="Set up live trial").click()
    page.locator("#run-flash-root").fill("/tmp/example")
    page.locator("#run-flash-inspect-btn").click()
    page.wait_for_function(
        "() => [...document.querySelectorAll('#run-flash-skill option')].some(option => option.value.endsWith('/ready/bundle.json'))"
    )
    page.locator("#run-flash-skill").select_option(bundle)
    assert "0 motion intervals" in page.locator("#run-flash-skill-detail").inner_text()
    assert (
        "Milestone placed: verify held relative to target within 0.01 m"
        in page.locator("#run-flash-skill-detail").inner_text()
    )
    assert not requests
    assert inspections == [{"dataset_root": "/tmp/example", "episode": 0}]
    assert page.locator("#run-launch-btn").is_disabled()
    page.locator("#run-flash-config").fill("/tmp/trial.json")
    page.get_by_role("button", name="Inspect trial parameters").click()
    detail = page.locator("#run-flash-config-detail")
    page.wait_for_function(
        "() => document.getElementById('run-flash-config-detail').textContent.includes('Held-pose correction gate')"
    )
    assert "Held-pose correction gate: 15 mm" in detail.inner_text()
    assert "Held prediction expiry: disabled" in detail.inner_text()
    assert "Initial pose wait: 20 s" in detail.inner_text()
    assert page.locator("#run-launch-btn").inner_text() == "Run FLASH"
    page.locator("#run-launch-btn").click()
    assert requests, page.locator("#run-terminal").inner_text()
    assert requests == [
        {
            "dataset_root": "/tmp/example",
            "episode": 0,
            "bundle_path": bundle,
            "config_path": "/tmp/trial.json",
        }
    ]


def test_run_flash_tracker_panel_reuses_view_for_recorded_layers(gui_page):
    page = gui_page
    page.route(
        "**/api/run/flash/telemetry/replay?*",
        lambda route: route.fulfill(
            json={
                "available": True,
                "layer_history": True,
                "frames": [
                    {
                        "frame_index": 3,
                        "snapshot": {
                            "recorded_at_s": 12.0,
                            "camera": None,
                            "centers_object_m": {},
                            "proposals": {
                                "object": {
                                    "evidence": "observed",
                                    "source_time_s": 11.9,
                                    "estimator": "appearance_rgbd",
                                    "pose": None,
                                }
                            },
                            "layers": {
                                "object": {
                                    "environment_group_motion": {
                                        "evidence": "predicted",
                                        "source_time_s": 12.0,
                                        "pose": None,
                                    }
                                }
                            },
                            "accepted": {
                                "object": {
                                    "evidence": "predicted",
                                    "source_time_s": 12.0,
                                    "estimator": "environment_group_motion",
                                    "pose": None,
                                }
                            },
                            "anchor_decisions": {
                                "object": {
                                    "frame_id": 2,
                                    "outcome": "direct_group_unresolved",
                                    "candidate": {"estimator": "appearance_rgbd"},
                                }
                            },
                        },
                    }
                ],
            }
        ),
    )
    page.route(
        "**/api/run/flash/recordings",
        lambda route: route.fulfill(
            json=[
                {
                    "dataset_root": "/tmp/tracker-run",
                    "episode": 0,
                    "modified_at_s": 12.0,
                    "frames": 1,
                    "outcome": "interrupted",
                    "reason": "IK failed",
                }
            ]
        ),
    )
    page.evaluate("switchTab('run')")
    page.evaluate("() => selectWorkflow('flash')")
    page.evaluate("datasets['/tmp/tracker-run'] = {id: '/tmp/tracker-run', root: '/tmp/tracker-run'}")
    page.wait_for_function("() => document.querySelectorAll('.flash-recording').length === 1")
    page.get_by_role("tab", name="Rollouts").click()
    assert page.locator(".flash-recording-reason").inner_text() == "IK failed"
    page.locator(".flash-recording").click()
    page.wait_for_function("() => document.querySelector('#flash-tracker-mode').textContent === 'Recording'")
    panel = page.locator("#flash-tracker-panel")
    assert panel.is_visible()
    assert "is-collapsed" in panel.get_attribute("class").split()
    assert not page.locator("#flash-tracker-content").is_visible()
    page.locator("#flash-tracker-collapse").click()
    assert "Direct proposal" in panel.inner_text()
    assert "Verified group" in panel.inner_text()
    assert "Accepted estimator pose: Verified group" in panel.inner_text()
    assert "Direct pose disagrees with tracked scene motion" in panel.inner_text()
    assert "Candidate from camera frame 2" in panel.inner_text()
    assert "No mask for this camera frame" in panel.inner_text()
    group = page.locator("#flash-tracker-content [data-flash-index]").filter(has_text="Verified group").first
    group.hover()
    assert "flash-tracker-hover" in group.get_attribute("class")
    assert page.evaluate("flashHighlightedOverlay?.kind") == "layer"
    assert page.locator("#flash-tracker-replay-controls").is_visible()
    image = page.locator("#rerun-viewer .flash-replay-grid").bounding_box()
    inspector = panel.bounding_box()
    assert inspector["x"] >= image["x"] + image["width"] or inspector["y"] >= image["y"] + image["height"]
    page.locator("#flash-tracker-collapse").click()
    assert not page.locator("#flash-tracker-content").is_visible()
    assert page.locator("#flash-tracker-replay-controls").is_visible()
    collapsed = panel.bounding_box()
    assert collapsed["width"] < 150
    heading = page.locator("#flash-tracker-heading").bounding_box()
    before = panel.bounding_box()
    page.mouse.move(heading["x"] + 20, heading["y"] + heading["height"] / 2)
    page.mouse.down()
    page.mouse.move(heading["x"] - 60, heading["y"] + heading["height"] / 2 + 30)
    page.mouse.up()
    assert panel.bounding_box()["x"] == before["x"]
    assert panel.bounding_box()["y"] == before["y"]
    page.locator("#flash-tracker-collapse").click()
    assert page.locator("#flash-tracker-content").is_visible()


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright is not installed")
def test_rollout_replay_uses_recorded_metric_depth_in_the_shared_scene(gui_page):
    page = gui_page
    height, width = 48, 64
    _, encoded = cv2.imencode(".jpg", np.full((height, width, 3), 100, dtype=np.uint8))
    requested_depth = []
    page.route(
        "**/api/run/flash/recordings",
        lambda route: route.fulfill(
            json=[
                {
                    "dataset_root": "/tmp/metric-rollout",
                    "episode": 0,
                    "modified_at_s": 12,
                    "frames": 1,
                    "outcome": "unknown",
                }
            ]
        ),
    )
    page.route(
        "**/api/run/flash/telemetry/replay?*",
        lambda route: route.fulfill(
            json={
                "available": True,
                "kind": "rollout",
                "layer_history": True,
                "camera_features": {
                    "rgb": "observation.images.front",
                    "depth": "observation.images.front_depth",
                },
                "frames": [
                    {
                        "frame_index": 3,
                        "snapshot": {
                            "recorded_at_s": 12,
                            "camera": {
                                "frame_id": 3,
                                "source_time_s": 11.9,
                                "width": width,
                                "height": height,
                                "intrinsics": [[50, 0, 32], [0, 50, 24], [0, 0, 1]],
                                "base_from_camera": np.eye(4).tolist(),
                            },
                            "centers_object_m": {"object": [0, 0, 1]},
                            "points_object_m": {"object": [[0, 0, 1], [0.01, 0, 1], [0, 0.01, 1]]},
                            "observed_masks": {
                                "object": {"frame_id": 2, "boundary_xy": [[1, 1], [2, 1], [2, 2]]}
                            },
                            "proposals": {},
                            "layers": {},
                            "accepted": {
                                "object": {
                                    "pose": np.eye(4).tolist(),
                                    "evidence": "observed",
                                    "source_time_s": 11.9,
                                    "estimator": "appearance_rgbd",
                                }
                            },
                            "held_prediction": None,
                        },
                    }
                ],
            }
        ),
    )
    page.route(
        "**/episodes/0/frame/3?*",
        lambda route: route.fulfill(body=encoded.tobytes(), content_type="image/jpeg"),
    )
    page.route(
        "**/episodes/0/depth-frame/3?*",
        lambda route: (
            requested_depth.append(route.request.url),
            route.fulfill(
                body=np.full((height, width), 1000, dtype="<u2").tobytes(),
                content_type="application/octet-stream",
            ),
        ),
    )
    page.evaluate("switchTab('run')")
    page.evaluate("() => selectWorkflow('flash')")
    page.evaluate(
        "datasets['/tmp/metric-rollout'] = {id: '/tmp/metric-rollout', root: '/tmp/metric-rollout'}"
    )
    page.get_by_role("tab", name="Rollouts").click()
    page.locator("#run-flash-recordings .flash-recording").click()
    page.wait_for_function(
        "() => document.querySelector('.flash-replay-grid img')?.dataset.loadedFrame === '3'"
    )
    page.get_by_role("button", name="3D scene").click()
    page.wait_for_function("() => flashSceneFrameCache?.frameIndex === 3")
    assert len(requested_depth) == 1
    assert "camera=observation.images.front_depth" in requested_depth[0]
    assert page.evaluate("flashSceneFrameCache.depth[0]") == 1000
    assert "measured 3D surface unavailable" in page.locator("#flash-scene-status").inner_text()
    assert page.locator(".flash-scene-photo").evaluate("element => element.style.visibility") == "hidden"
    assert page.locator("#flash-scene-edit").count() == 0
    assert page.locator("#flash-scene-surface").count() == 0
    assert page.locator("#flash-scene-group").count() == 0


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright is not installed")
def test_live_flash_uses_the_same_frame_tagged_rgbd_scene(gui_page):
    page = gui_page
    height, width = 48, 64
    _, rgb = cv2.imencode(".jpg", np.full((height, width, 3), 120, dtype=np.uint8))
    snapshot = {
        "recorded_at_s": 12.1,
        "camera": {
            "frame_id": 9,
            "source_time_s": 12.0,
            "calibration_version": "camera-v1",
            "width": width,
            "height": height,
            "intrinsics": [[50, 0, 32], [0, 50, 24], [0, 0, 1]],
            "base_from_camera": np.eye(4).tolist(),
            "rgbd_available": True,
        },
        "centers_object_m": {"object": [0, 0, 1]},
        "points_object_m": {"object": [[-0.1, -0.1, 1], [0.1, -0.1, 1], [-0.1, 0.1, 1]]},
        "observed_masks": {"object": {"frame_id": 8, "boundary_xy": [[1, 1], [2, 1], [2, 2]]}},
        "proposal_frame_id": 9,
        "proposals": {
            "object": {"pose": None, "evidence": "lost", "source_time_s": 12.0, "estimator": "surface_rgbd"}
        },
        "raw_proposals": {
            "object": {
                "frame_id": 9,
                "calibration_version": "camera-v1",
                "pose": np.eye(4).tolist(),
                "estimator": "surface_rgbd",
                "failure": "surface fit is uncertain",
            }
        },
        "layers": {},
        "accepted": {
            "object": {
                "pose": np.eye(4).tolist(),
                "evidence": "observed",
                "source_time_s": 12.0,
                "estimator": "appearance_rgbd",
            }
        },
    }
    latest = {"snapshot": snapshot}
    page.route("**/api/run/status", lambda route: route.fulfill(json={"running": True, "command": "flash"}))
    page.route(
        "**/api/run/flash/telemetry/live",
        lambda route: route.fulfill(json={"available": True, "snapshot": latest["snapshot"]}),
    )
    page.route(
        "**/api/run/flash/telemetry/frame/9/rgb",
        lambda route: route.fulfill(body=rgb.tobytes(), content_type="image/jpeg"),
    )
    page.route(
        "**/api/run/flash/telemetry/frame/9/depth",
        lambda route: route.fulfill(
            body=np.full((height, width), 1000, dtype="<u2").tobytes(),
            content_type="application/octet-stream",
        ),
    )
    page.evaluate("switchTab('run')")
    page.evaluate("() => selectWorkflow('flash')")
    page.evaluate("startFlashTelemetry()")
    page.wait_for_function(
        "() => document.querySelector('.flash-replay-grid img')?.dataset.loadedFrame === '9'"
    )
    assert "Rejected raw fit" in page.locator("#flash-tracker-content").inner_text()
    pink_pixels = """() => {
        const canvas = document.querySelector('.flash-pose-canvas');
        const rgba = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        let count = 0;
        for (let i = 0; i < rgba.length; i += 4)
            if (rgba[i] > 220 && rgba[i+1] > 80 && rgba[i+1] < 160 && rgba[i+2] > 150 && rgba[i+2] < 220 && rgba[i+3] > 0) count++;
        return count;
    }"""
    assert page.evaluate(pink_pixels) > 0
    assert page.locator("#flash-tracker-replay-controls").is_hidden()
    page.get_by_role("button", name="3D scene").click()
    page.wait_for_function("() => flashSceneFrameCache?.frameIndex === 9")
    assert page.evaluate("flashSceneFrameCache.depth[0]") == 1000
    assert "Live RGB-D · frame 9" in page.locator("#flash-scene-status").inner_text()
    assert page.locator("#flash-scene-surface").count() == 0
    assert page.locator(".flash-scene-photo").evaluate("element => element.style.visibility") == "hidden"
    page.route(
        "**/api/run/flash/telemetry/frame/10/rgb",
        lambda route: route.fulfill(body=rgb.tobytes(), content_type="image/jpeg"),
    )
    page.route(
        "**/api/run/flash/telemetry/frame/10/depth",
        lambda route: route.fulfill(
            body=np.full((height, width), 1010, dtype="<u2").tobytes(),
            content_type="application/octet-stream",
        ),
    )
    latest["snapshot"] = {**snapshot, "camera": {**snapshot["camera"], "frame_id": 10}}
    page.wait_for_function("() => flashSceneFrameCache?.frameIndex === 10")
    assert page.evaluate("flashSceneFrameCache.depth[0]") == 1010
    assert "Live RGB-D · frame 10" in page.locator("#flash-scene-status").inner_text()
    page.get_by_role("button", name="RGB camera").click()
    page.evaluate("_flashDrawOverlay()")
    assert page.evaluate(pink_pixels) == 0
    page.evaluate("stopFlashTelemetry()")


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright is not installed")
def test_raw_demo_editor_collects_object_names_and_action_from_visible_controls(gui_page):
    page = gui_page
    submitted = []
    raw_frames = [
        {
            "frame_index": index,
            "snapshot": {
                "camera": {"frame_id": index, "width": 640, "height": 480},
                "accepted": {},
                "demo": {"phase": "raw", "time_s": float(index)},
            },
        }
        for index in range(3)
    ]
    page.route(
        "**/api/run/flash/recordings",
        lambda route: route.fulfill(
            json=[
                {
                    "kind": "demo",
                    "dataset_root": "/tmp/unprepared",
                    "episode": 0,
                    "modified_at_s": 1.0,
                    "skills": [],
                    "prepared_versions": [],
                    "bundle_path": None,
                }
            ]
        ),
    )
    page.route(
        "**/api/run/flash/demo/raw-replay?*",
        lambda route: route.fulfill(
            json={
                "available": True,
                "kind": "raw_demo",
                "fps": 0.1,
                "frames": raw_frames,
                "starter_parameters": {
                    "max_demo_pose_gap_frames": 90,
                    "connection_speed_m_s": 0.04,
                    "min_similarity": 0.75,
                    "registration_threshold_m": 0.005,
                    "registration_min_inliers": 12,
                    "registration_trials": 100,
                    "sam_resolution": 448,
                    "descriptor_model_id": "facebook/dinov2-small",
                },
            }
        ),
    )

    def capture_submission(route):
        submitted.append(route.request.post_data_json)
        route.fulfill(status=400, json={"detail": "captured test request"})

    page.route("**/api/run/flash/demo/prepare", capture_submission)
    page.evaluate("switchTab('run')")
    page.evaluate("() => selectWorkflow('flash')")
    page.locator("#run-flash-demos .flash-recording").click()
    page.wait_for_function("() => flashReplay?.kind === 'raw_demo'")
    page.locator("#flash-demo-edit-toggle").click()
    assert page.locator("#flash-demo-editor-change-status").inner_text() == "Raw demo · no preparation saved"

    current_section = "setup"

    def section(name):
        nonlocal current_section
        current_section = {
            "Objects": "objects",
            "Events": "events",
            "References": "references",
            "Advanced": "advanced",
            "Key frames": "keyframes",
        }[name]
        if current_section == "advanced" and not page.locator(".flash-editor-settings").evaluate(
            "el => el.open"
        ):
            page.locator(".flash-editor-settings > summary").click()

    def set_field(label, value):
        control = (
            page.locator(f"#flash-demo-editor .flash-editor-pane[data-section='{current_section}'] label")
            .filter(has_text=label)
            .locator("input, select")
            .first
        )
        if control.evaluate("element => element.tagName") == "SELECT":
            control.select_option(str(value))
        else:
            control.fill(str(value))
            control.dispatch_event("change")

    def seek(index):
        page.locator("#flash-tracker-scrub").evaluate(
            "(element, value) => { element.value = String(value); "
            'element.dispatchEvent(new Event("input", {bubbles: true})); }',
            index,
        )

    set_field("Arm side", "left")
    set_field("Robot/camera calibration file", "/tmp/calibration.json")
    section("Objects")
    for role, prompt in (("workpiece", "wooden cylinder"), ("receiver", "green ring")):
        set_field("New object name", role)
        set_field("What it looks like", prompt)
        page.get_by_role("button", name="Add object").click()
    page.locator(".flash-editor-object-options > summary").click()
    set_field("Rotational symmetry", "long_axis")
    assert "dowel" not in page.locator("#flash-demo-editor").inner_text().lower()
    section("References")
    page.get_by_role("button", name="Use current for grasp view").click()
    seek(1)
    section("Events")
    page.get_by_role("button", name="Mark grasp here").click()
    section("References")
    page.get_by_role("button", name="Use current for start-anchored until").click()
    seek(2)
    section("Events")
    page.get_by_role("button", name="Mark release here").click()
    section("Advanced")
    page.get_by_role("button", name="Apply SO-107 starter parameters").click()
    set_field("Vision pipeline (demo and run)", "surface")
    section("Key frames")
    seek(1)
    page.get_by_role("button", name="Add key frame at current time").click()
    set_field("Move relative to", "receiver")
    page.get_by_role("button", name="Prepare demo").click()
    page.wait_for_function(
        "() => document.querySelector('#flash-demo-editor-status').textContent.includes('captured test request')"
    )
    assert submitted[0]["annotation"]["vision_pipeline"]["backend"] == "surface"
    assert submitted[0].get("bundle_path") is None
    assert submitted[0]["annotation"]["prompts_by_role"] == {
        "workpiece": "wooden cylinder",
        "receiver": "green ring",
    }
    assert submitted[0]["annotation"]["axial_symmetry_roles"]["receiver"] == {"axis_kind": "long_axis"}
    assert submitted[0]["keypoints_document"]["keypoints"][0]["frame"] == "receiver"


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright is not installed")
def test_3d_scene_projection_and_surface_pick_edit_the_prepared_keyframe(gui_page):
    page = gui_page
    width, height = 64, 48
    intrinsics = [[50, 0, 29], [0, 50, 26], [0, 0, 1]]  # Deliberately off-centre.
    identity = np.eye(4).tolist()
    image = np.full((height, width, 3), 100, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    mask_runs = {
        role: [[row * width + left, 12] for row in range(18, 30)]
        for role, left in (("tool", 8), ("fixture", 40))
    }
    surface = {
        role: {
            "positions_object_m": [x, -0.04, 1, x + 0.1, -0.04, 1, x, 0.04, 1],
            "colors_rgb": [[200, 200, 200]] * 3,
            "indices": [0, 1, 2],
        }
        for role, x in (("tool", -0.4), ("fixture", 0.2))
    }
    snapshot = {
        "camera": {
            "frame_id": 0,
            "width": width,
            "height": height,
            "intrinsics": intrinsics,
            "base_from_camera": identity,
        },
        "accepted": {
            role: {
                "pose": identity,
                "evidence": "observed",
                "source_time_s": 1.0,
                "estimator": "demo_reference",
            }
            for role in ("tool", "fixture")
        },
        "layers": {},
        "demo": {
            "phase": "demonstrated_sample",
            "time_s": 1.0,
            "camera_arm_delta_s": 0,
            "interval_name": "contact",
            "relation": "ee_to_object",
            "reference_role": "tool",
            "source_arm_frame_index": 0,
            "sample_time_s": 1.0,
            "event_id": None,
        },
    }
    page.route(
        "**/api/run/flash/recordings",
        lambda route: route.fulfill(
            json=[
                {
                    "kind": "demo",
                    "dataset_root": "/tmp/scene-edit",
                    "episode": 0,
                    "skills": ["prepared"],
                    "prepared_skill": "prepared",
                    "modified_at_s": 1,
                    "bundle_path": "/tmp/scene-edit/flash/v1/prepared/prepared/bundle.json",
                }
            ]
        ),
    )
    page.route(
        "**/api/run/flash/demo/replay?*",
        lambda route: route.fulfill(
            json={
                "available": True,
                "kind": "demo",
                "fps": 30,
                "bundle_path": "/tmp/scene-edit/flash/v1/prepared/prepared/bundle.json",
                "geometry": {
                    "points_object_m": {role: [[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]] for role in surface},
                    "centers_object_m": {role: [0, 0, 1] for role in surface},
                },
                "frames": [{"frame_index": 0, "snapshot": snapshot}],
            }
        ),
    )
    page.route(
        "**/api/run/flash/demo/frame?*",
        lambda route: route.fulfill(body=encoded.tobytes(), content_type="image/png"),
    )
    page.route(
        "**/api/run/flash/demo/depth-frame?*",
        lambda route: route.fulfill(
            body=np.full((height, width), 1000, dtype="<u2").tobytes(),
            content_type="application/octet-stream",
        ),
    )
    page.route("**/api/run/flash/demo/reference-surfaces?*", lambda route: route.fulfill(json=surface))
    page.route(
        "**/api/run/flash/demo/scene-evidence/status?*",
        lambda route: route.fulfill(json={"available": True, "frames_completed": 1}),
    )
    page.route(
        "**/api/run/flash/demo/scene-evidence/frame?*",
        lambda route: route.fulfill(
            json={"available": True, "frame_index": 0, "masks_rle": mask_runs, "groups": {}}
        ),
    )
    page.route(
        "**/api/run/flash/demo/preparation?*",
        lambda route: route.fulfill(
            json={
                "annotation": {
                    "episode": 0,
                    "reference_frame": 0,
                    "grasp_reference_frame": 0,
                    "grasp_time_s": 2.0,
                    "release_time_s": 3.0,
                    "start_anchored_until_s": 2.0,
                    "held_role": "tool",
                    "target_role": "fixture",
                    "prompts_by_role": {"tool": "tool", "fixture": "fixture"},
                    "axial_symmetry_roles": {},
                },
                "keypoints_document": {
                    "keypoints": [{"t": 1.0, "frame": "tool", "move": "line", "grip": "first"}]
                },
                "reference_masks": {},
            }
        ),
    )
    page.route(
        "**/api/run/flash/demo/trackers?*",
        lambda route: route.fulfill(json={"trackers": [], "candidates": []}),
    )
    submitted = []
    page.route(
        "**/api/run/flash/demo/prepare",
        lambda route: (
            submitted.append(route.request.post_data_json),
            route.fulfill(status=400, json={"detail": "captured"}),
        ),
    )

    page.evaluate("switchTab('run')")
    page.evaluate("() => selectWorkflow('flash')")
    page.locator("#run-flash-demos .flash-recording").click()
    page.locator("#flash-scene-toggle").click()
    page.wait_for_function("() => flashSceneMode === '3d' && flashSceneFrameCache?.frameIndex === 0")
    # The actual WebGL projection must meet the same pixel as the CSS-contained RGB frame.
    box = page.locator(".flash-scene-canvas").bounding_box()
    scale = min(box["width"] / width, box["height"] / height)
    expected = [
        box["x"] + (box["width"] - width * scale) / 2 + 45 * scale,
        box["y"] + (box["height"] - height * scale) / 2 + 24 * scale,
    ]
    projected = page.evaluate("() => flashSceneView.projectCameraXYZ([(45-29)/50, (24-26)/50, 1])")
    assert np.allclose(projected, expected, atol=0.5)

    page.get_by_role("button", name="Edit key frames").click()
    page.wait_for_function("() => !!flashDemoEditor")
    assert not page.locator("#run-sidebar").is_visible()
    assert page.locator(".flash-editor-navigation").count() == 0
    assert page.locator("#flash-demo-editor .flash-editor-pane[data-section='objects']").is_visible()
    assert page.locator("#flash-demo-editor .flash-editor-pane[data-section='keyframes']").is_visible()
    assert page.get_by_label("Reach", exact=True).is_visible()
    assert page.get_by_role("button", name="Move key frame to playhead").is_visible()
    page.get_by_label("Path", exact=True).select_option("lift")
    assert page.get_by_label("Lift (mm)").is_visible()
    page.get_by_label("Path", exact=True).select_option("line")
    assert not page.get_by_label("Lift (mm)").count()
    assert not page.get_by_text("Pose evidence", exact=True).is_visible()
    page.locator(".flash-editor-settings > summary").click()
    assert page.get_by_text("Pose evidence", exact=True).is_visible()
    assert page.locator(".flash-editor-settings .flash-editor-group").count() == 0
    page.get_by_role("button", name="Choose reference in scene").click()
    box = page.locator(".flash-scene-canvas").bounding_box()
    scale = min(box["width"] / width, box["height"] / height)
    fixture_pixel = [
        box["x"] + (box["width"] - width * scale) / 2 + 45 * scale,
        box["y"] + (box["height"] - height * scale) / 2 + 24 * scale,
    ]
    page.wait_for_function(
        "([x, y]) => { const [px, py] = flashSceneView.projectCameraXYZ([(45-29)/50, (24-26)/50, 1]); return Math.abs(px-x) < 0.5 && Math.abs(py-y) < 0.5; }",
        arg=fixture_pixel,
    )
    page.wait_for_function(
        "() => flashDemoEditor?.pickingReference && !document.querySelector('.flash-replay-grid').classList.contains('flash-frame-loading')"
    )
    assert page.evaluate("([x, y]) => flashSceneView.pickObserved(x, y)?.role", fixture_pixel) == "fixture"
    page.mouse.click(*fixture_pixel)
    assert page.evaluate("flashDemoEditor.keypointsDocument.keypoints[0].frame") == "fixture"
    assert "flash-scene-orbit" not in page.locator(".flash-replay-grid").get_attribute("class")
    assert page.evaluate("flashDemoEditor.role") == "fixture"
    tool_pixel = [
        box["x"] + (box["width"] - width * scale) / 2 + 13 * scale,
        box["y"] + (box["height"] - height * scale) / 2 + 24 * scale,
    ]
    assert page.evaluate("([x, y]) => flashSceneView.pickObserved(x, y)?.role", tool_pixel) == "tool"
    page.mouse.click(*tool_pixel)
    assert page.evaluate("flashDemoEditor.role") == "tool"
    assert page.evaluate("flashDemoEditor.keypointsDocument.keypoints[0].frame") == "fixture"
    assert "Unsaved edits" in page.locator("#flash-demo-editor-change-status").inner_text()
    page.get_by_role("button", name="Preview updated demo").click()
    page.wait_for_function(
        "() => document.querySelector('#flash-demo-editor-status').textContent.includes('captured')"
    )
    assert submitted[0]["keypoints_document"]["keypoints"][0]["frame"] == "fixture"
    assert page.locator("#flash-demo-editor .flash-editor-pane[data-section='keyframes']").is_visible()
    assert page.locator(".flash-editor-turn").is_visible()
    page.get_by_role("button", name="Allow rotation around a point").click()
    page.wait_for_function("() => flashDemoEditor?.pickingTurn")
    picked = page.evaluate("([x, y]) => flashSceneView.pickObserved(x, y)", fixture_pixel)
    assert picked["reference_xyz_m"] == pytest.approx(picked["camera_xyz_m"])
    assert np.linalg.norm(picked["reference_normal"]) == pytest.approx(1)
    page.mouse.click(*fixture_pixel)
    rule = page.evaluate("flashDemoEditor.keypointsDocument.turn_rules[0]")
    assert "flash-scene-orbit" not in page.locator(".flash-replay-grid").get_attribute("class")
    assert rule["reference_role"] == "fixture"
    assert rule["pivot_reference_m"] == pytest.approx(picked["camera_xyz_m"])
    assert page.get_by_label("Move relative to", exact=True).is_disabled()
    page.get_by_role("button", name="Preview updated demo").click()
    page.wait_for_function(
        "() => document.querySelector('#flash-demo-editor-status').textContent.includes('captured')"
    )
    assert submitted[-1]["keypoints_document"]["turn_rules"][0]["rule_id"] == rule["rule_id"]
    assert "Any angle" in page.locator(".flash-editor-turn").inner_text()
    assert not page.get_by_label("Minimum angle (degrees)").is_visible()
    page.get_by_text("Refine turn").click()
    assert page.get_by_label("Minimum angle (degrees)").is_visible()
    page.get_by_role("button", name="Remove", exact=True).click()
    page.evaluate("""() => {
        flashDemoEditor.annotation.axial_symmetry_roles.fixture = {lock_to_baseline: true};
        flashReplay.pose_constraints = {fixture: {
            orientation_locked_to_baseline: true, center_object_m: [0, 0, 1],
            axis_object: [0, 0, 1], source_frame_index: 0,
        }};
        _flashEditorRender();
    }""")
    page.get_by_role("button", name="Allow rotation around center").click()
    invariant = page.evaluate("flashDemoEditor.keypointsDocument.turn_rules[0]")
    assert invariant["source_evidence"] == "symmetry_invariant"
    assert invariant["pivot_reference_m"] == [0, 0, 1]
    assert invariant["axis_reference"] == [0, 0, 1]
    assert "Object center" in page.locator(".flash-editor-turn").inner_text()
    page.get_by_role("button", name="Remove", exact=True).click()
    page.evaluate("""() => {
        flashDemoEditor.annotation.grasp_time_s = 0.5;
        flashDemoEditor.annotation.start_anchored_until_s = 1.5;
        _flashEditorRender();
    }""")
    assert page.get_by_role("button", name="Choose reference in scene").is_disabled()
    assert page.get_by_label("Move relative to", exact=True).is_disabled()
    assert "grasp-start reference" in page.locator("#flash-demo-editor").inner_text()
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate(
        "() => { const main = document.querySelector('#run-main'); "
        "return main.scrollWidth <= main.clientWidth && "
        "document.querySelector('#flash-demo-editor').getBoundingClientRect().left >= 0; }"
    )
    page.get_by_role("button", name="Close editor").click()
    assert page.locator("#run-sidebar").is_visible()
