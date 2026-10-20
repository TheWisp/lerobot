"""The Run → FLASH entry validates a skill without touching robot hardware."""

import asyncio
import importlib.util
import json
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.gui.api.run import (
    FlashDatasetInspectRequest,
    FlashInspectRequest,
    FlashRunRequest,
    inspect_flash_dataset,
    inspect_flash_skill,
    start_flash,
    stop_process,
)
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
    page.locator("#run-flash-root").fill("/tmp/demo")
    page.locator("#run-flash-episode").fill("2")
    page.locator("#run-flash-view-btn").click()
    page.wait_for_function("() => window.flashViewedEpisode !== undefined")
    assert page.evaluate("window.flashViewedEpisode") == {"id": "demo", "episode": 2, "length": 42}
    assert "active" in page.locator('.tab[data-tab="data"]').get_attribute("class").split()
    assert requests == []


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
    page.evaluate("switchTab('run')")
    page.evaluate("() => selectWorkflow('flash')")
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
    page.locator(".flash-recording").click()
    page.wait_for_function("() => document.querySelector('#flash-tracker-mode').textContent === 'Recording'")
    panel = page.locator("#flash-tracker-panel")
    assert panel.is_visible()
    assert "Direct proposal" in panel.inner_text()
    assert "Verified group" in panel.inner_text()
    assert "Planner pose: Verified group" in panel.inner_text()
    assert page.locator("#flash-tracker-replay-controls").is_visible()
