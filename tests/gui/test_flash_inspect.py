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
    request = FlashRunRequest(
        dataset_root=str(tmp_path), episode=0, bundle_path=str(bundle), config_path=str(config)
    )
    assert asyncio.run(start_flash(request))["pid"] == 123
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
    page.locator("#flash-demo-edit-toggle").click()
    page.wait_for_function("() => !!document.querySelector('#flash-demo-editor')")
    assert "Demo preparation" in page.locator("#flash-demo-editor").inner_text()
    assert page.locator("#flash-demo-editor-frame").inner_text().startswith("Frame 4")
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
    page.locator("#flash-demo-edit-toggle").click()
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
    page.locator(".flash-recording").click()
    page.wait_for_function("() => document.querySelector('#flash-tracker-mode').textContent === 'Recording'")
    panel = page.locator("#flash-tracker-panel")
    assert panel.is_visible()
    assert "Direct proposal" in panel.inner_text()
    assert "Verified group" in panel.inner_text()
    assert "Accepted estimator pose: Verified group" in panel.inner_text()
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

    def section(name):
        page.locator("#flash-demo-editor .flash-editor-tabs button", has_text=name).click()

    def set_field(label, value):
        control = (
            page.locator("#flash-demo-editor .flash-editor-pane:not([hidden]) label")
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
    set_field("Rotational symmetry", "long_axis")
    assert "dowel" not in page.locator("#flash-demo-editor").inner_text().lower()
    section("Events")
    page.get_by_role("button", name="Use current for grasp view").click()
    seek(1)
    page.get_by_role("button", name="Use current for grasp time").click()
    page.get_by_role("button", name="Use current for start-anchored until").click()
    seek(2)
    page.get_by_role("button", name="Use current for release time").click()
    section("Advanced")
    page.get_by_role("button", name="Apply SO-107 starter parameters").click()
    section("Key frames")
    seek(1)
    page.get_by_role("button", name="Add key frame at current time").click()
    set_field("Reference", "receiver")
    page.get_by_role("button", name="Prepare spatial demo").click()
    page.wait_for_function(
        "() => document.querySelector('#flash-demo-editor-status').textContent.includes('captured test request')"
    )
    assert submitted[0].get("bundle_path") is None
    assert submitted[0]["annotation"]["prompts_by_role"] == {
        "workpiece": "wooden cylinder",
        "receiver": "green ring",
    }
    assert submitted[0]["annotation"]["axial_symmetry_roles"]["receiver"] == {"axis_kind": "long_axis"}
    assert submitted[0]["keypoints_document"]["keypoints"][0]["frame"] == "receiver"
