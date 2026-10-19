"""The Run → FLASH entry validates a skill without touching robot hardware."""

import asyncio
import importlib.util
import json

import numpy as np
import pytest

from lerobot.gui.api.run import FlashInspectRequest, inspect_flash_skill
from lerobot.policies.flash.inspect import inspect_skill
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


@pytest.mark.requires_playwright
@pytest.mark.skipif(importlib.util.find_spec("playwright") is None, reason="Playwright is not installed")
def test_run_flash_subtab_displays_spatial_milestone_from_skill(tmp_path, gui_page):
    identity = np.eye(4).tolist()
    path = tmp_path / "skill.json"
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
    page.locator("#run-flash-skill").fill(str(path))
    page.evaluate("() => launchRun()")
    page.wait_for_function(
        "() => document.getElementById('run-terminal').textContent.includes('Milestone placed: verify held relative to target within 0.01 m')"
    )
