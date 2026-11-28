"""Stale demo results cannot reach playback or execution; rebuild uses real tracking."""

import asyncio
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from lerobot.gui.api import flash_editor, flash_tracking, run
from lerobot.policies.flash.demo_document import (
    DemoObject,
    atomic_json,
    document_path,
    load_document,
    save_document,
)
from lerobot.policies.flash.demo_tracking import track_demo
from lerobot.policies.flash.pipeline_version import PIPELINE_VERSION
from tests.gui.test_flash_editor_execution import prepared  # noqa: F401
from tests.policies.flash.test_demo_execution import robot  # noqa: F401
from tests.policies.flash.test_demo_tracking import Masks, rgbd_episode  # noqa: F401


@pytest.fixture
def tracked(rgbd_episode, monkeypatch):  # noqa: F811
    document, _ = load_document(rgbd_episode, 0)
    document.objects = [DemoObject(id="object_1", name="left"), DemoObject(id="object_2", name="right")]
    document = save_document(rgbd_episode, 0, document)
    output = document_path(rgbd_episode, 0).with_suffix("") / "original"
    setup = flash_editor.EditorSettings(depth_source="sensor")
    atomic_json(flash_editor.EDITOR_DIR / "settings.json", setup.model_dump())
    track_demo(rgbd_episode, 0, document, output, settings=setup.pipeline_settings, mask_source=Masks())
    atomic_json(output.parent / "latest.json", {"directory": output.name})
    monkeypatch.setattr(flash_editor, "_dataset", lambda _: rgbd_episode)
    monkeypatch.setattr(flash_editor, "_tracking_job", None)
    monkeypatch.setattr(run, "_active_process", None)
    return rgbd_episode, document, output


def make_stale(output, change):
    path = output / "input.json"
    inputs = json.loads(path.read_text())
    if change == "legacy":
        inputs.pop("pipeline_version")
    elif change == "version":
        inputs["pipeline_version"] = PIPELINE_VERSION - 1
    elif change == "settings":
        inputs["settings"]["config"]["sam_resolution"] += 1
    elif change == "depth":
        inputs["depth_source"] = "foundation_stereo"
    atomic_json(path, inputs)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["legacy", "version", "settings", "depth"])
async def test_stale_results_rebuild_once_without_changing_recording(tracked, monkeypatch, change):
    dataset, document, output = tracked
    make_stale(output, change)
    old_bytes = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    raw_bytes = {p: p.read_bytes() for p in dataset.root.rglob("*") if p.is_file() and "flash" not in p.parts}
    before = await flash_editor.tracking("fixture", frame=1)
    assert before["stale"] and "frame_result" not in before
    display = await flash_editor.display_frames("fixture", start=0, count=4, scene=False)
    assert json.loads(display.body) == [None] * 4
    assert json.loads((output.parent / "latest.json").read_text())["directory"] == "original"
    launches = []

    async def launch(args, **kwargs):
        launches.append(args)
        destination = Path(args[args.index("--output") + 1])
        settings = json.loads(Path(args[args.index("--settings") + 1]).read_text())
        await asyncio.get_running_loop().run_in_executor(
            flash_editor._io,
            partial(track_demo, dataset, 0, document, destination, settings=settings, mask_source=Masks()),
        )
        monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=11, returncode=0))

    monkeypatch.setattr(run, "_launch_subprocess", launch)
    request = flash_editor.TrackDemo(
        dataset_id="fixture", episode=0, revision=document.revision, ensure_current=True
    )
    await flash_editor.track(request)
    current = await flash_editor.tracking("fixture", frame=1)
    assert current["state"] == "complete" and not current["stale"]
    assert current["pipeline_version"] == PIPELINE_VERSION
    assert current["frame_result"]["objects"]["object_1"]["pose"] is not None
    assert (await flash_editor.track(request))["state"] == "complete"
    assert len(launches) == 1
    assert all(path.read_bytes() == value for path, value in old_bytes.items())
    assert all(path.read_bytes() == value for path, value in raw_bytes.items())


@pytest.mark.asyncio
async def test_duplicate_requests_join_job_and_failure_needs_explicit_retry(tracked, monkeypatch):
    _, document, output = tracked
    make_stale(output, "version")
    launches = []
    process = SimpleNamespace(pid=12, returncode=None)

    async def launch(args, **kwargs):
        launches.append(args)
        monkeypatch.setattr(run, "_active_process", process)

    monkeypatch.setattr(run, "_launch_subprocess", launch)
    request = flash_editor.TrackDemo(
        dataset_id="fixture", episode=0, revision=document.revision, ensure_current=True
    )
    first, second = await asyncio.gather(flash_editor.track(request), flash_editor.track(request))
    assert first["pid"] == second["pid"] == 12
    assert len(launches) == 1
    process.returncode = 1
    failed = await flash_editor.track(request)
    assert failed["state"] == "failed" and len(launches) == 1
    await flash_editor.track(request.model_copy(update={"ensure_current": False}))
    assert len(launches) == 2


@pytest.mark.asyncio
async def test_busy_run_does_not_replace_latest_tracking(tracked, monkeypatch):
    _, document, output = tracked
    make_stale(output, "version")
    latest = (output.parent / "latest.json").read_bytes()
    monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=99, returncode=None))
    with pytest.raises(HTTPException):
        await flash_editor.track(
            flash_editor.TrackDemo(
                dataset_id="fixture",
                episode=0,
                revision=document.revision,
                ensure_current=True,
            )
        )
    assert (output.parent / "latest.json").read_bytes() == latest


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["legacy", "version"])
async def test_old_tracking_and_old_prepared_plan_never_launch_hardware(prepared, monkeypatch, change):  # noqa: F811
    dataset, _ = prepared
    plan = await flash_editor.prepare_demo(flash_editor.PrepareDemo(id="fixture"))
    plan_path = flash_editor.EDITOR_DIR / "plans" / plan["plan"] / "plan.json"
    value = json.loads(plan_path.read_text())
    if change == "legacy":
        value.pop("pipeline_version")
    else:
        value["pipeline_version"] = PIPELINE_VERSION - 1
    atomic_json(plan_path, value)
    launch = AsyncMock()
    monkeypatch.setattr(flash_tracking, "start_worker", launch)
    with pytest.raises(HTTPException, match="Tracking code changed"):
        await flash_editor.apply_demo(flash_editor.ApplyDemo(plan=plan["plan"]))
    output = document_path(dataset, 0).with_suffix("") / "fixture"
    make_stale(output, change)
    with pytest.raises(HTTPException, match="Tracking code changed"):
        await flash_editor.prepare_demo(flash_editor.PrepareDemo(id="fixture"))
    launch.assert_not_called()


def test_opening_old_demo_rebuilds_and_displays_new_results(gui_page, tracked, monkeypatch):
    from playwright.sync_api import expect

    dataset, document, output = tracked
    make_stale(output, "legacy")
    atomic_json(
        flash_editor.EDITOR_DIR / "demos.json",
        {
            "recent": None,
            "demos": [
                {
                    "id": "fixture",
                    "root": str(dataset.root),
                    "episode": 0,
                    "name": "Versioned demo",
                    "created_at": 1,
                }
            ],
        },
    )
    launches = []

    async def launch(args, **kwargs):
        destination = Path(args[args.index("--output") + 1])
        launches.append(destination)
        await asyncio.get_running_loop().run_in_executor(
            flash_editor._io,
            partial(track_demo, dataset, 0, document, destination, mask_source=Masks()),
        )
        monkeypatch.setattr(run, "_active_process", SimpleNamespace(pid=13, returncode=0))

    monkeypatch.setattr(run, "_launch_subprocess", launch)
    page = gui_page
    base = page.url.split("#")[0].rstrip("/")
    page.goto(f"{base}/#/flash/editor")
    page.locator("#flash-editor-demos button").first.click()
    expect(page.locator("#flash-editor-tracking-status")).to_contain_text(
        f"Tracking complete · pipeline v{PIPELINE_VERSION}",
        timeout=60_000,
    )
    page.wait_for_function("flashEditor.overlay?.objects?.object_1?.pose != null")
    assert len(launches) == 1
    assert page.evaluate("flashEditor.displayResult") == launches[0].name
    # Reopening a current result does not run the tracker again.
    page.locator("#flash-editor-demos button").first.click()
    expect(page.locator("#flash-editor-tracking-status")).to_contain_text("Tracking complete")
    assert len(launches) == 1


@pytest.mark.parametrize("outcome", ["complete", "failed", "stopped"])
def test_apply_waits_for_rebuild_and_never_runs_on_failure(gui_page, prepared, outcome):  # noqa: F811
    from playwright.sync_api import expect

    page = gui_page
    base = page.url.split("#")[0].rstrip("/")
    stage = {"state": "running", "done": 1, "total": 4, "pid": 42}
    runs = []
    page.route("**/api/run/flash/editor/track", lambda route: route.fulfill(json=stage))
    page.route("**/api/run/flash/editor/tracking?*", lambda route: route.fulfill(json=stage))
    page.route(
        "**/api/run/flash/editor/run-demo", lambda route: (runs.append(True), route.fulfill(json={"pid": 42}))
    )
    page.goto(f"{base}/#/flash/editor")
    page.locator("#flash-editor-apply-demo").select_option("fixture")
    page.locator("#flash-editor-apply").click()
    expect(page.locator("#flash-editor-status")).to_contain_text("Updating demo tracking")
    assert not runs
    if outcome == "stopped":
        page.route(
            "**/api/run/flash/editor/cancel-tracking", lambda route: route.fulfill(json={"status": "stopped"})
        )
        page.locator("#flash-editor-stop").click()
        expected = "Apply cancelled"
    else:
        stage.update(state=outcome, stale=False, reason="Tracking failed; Track again to retry.")
        expected = "Running robot" if outcome == "complete" else "Tracking failed"
    expect(page.locator("#flash-editor-status")).to_contain_text(expected)
    assert len(runs) == (1 if outcome == "complete" else 0)
