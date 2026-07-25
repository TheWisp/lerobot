# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Browser smoke for training health metrics and checkpoint recovery.

The full FastAPI app runs on an ephemeral loopback port, but every training
artifact lives under ``tmp_path`` and process launch is replaced with a no-op.
This exercises the real HTML/CSS/JavaScript and API without Docker, SSH,
robot hardware, or the operator's live GUI.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import (
    Error as PlaywrightError,  # noqa: E402
    sync_playwright,  # noqa: E402
)

from lerobot.common.training_log import format_training_log_record  # noqa: E402
from lerobot.gui.api import training as training_api  # noqa: E402
from lerobot.gui.training.hosts import HostRegistry, TrainingHost  # noqa: E402
from lerobot.gui.training.orchestrator import Orchestrator  # noqa: E402
from lerobot.gui.training.runs import (  # noqa: E402
    Run,
    RunPaths,
    RunRegistry,
    RunState,
    append_event,
)
from lerobot.gui.training.transport import SubprocessTransport  # noqa: E402

pytestmark = pytest.mark.requires_playwright


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _seed_interrupted_run(registry: RunRegistry) -> tuple[Run, RunPaths, str]:
    now = time.time()
    run = Run(
        run_id="browser-partial-run",
        host_id="this-server",
        recipe_name="HVLA browser smoke",
        dataset_id="robot/browser-smoke",
        args={
            "__recipe__": "hvla_flow_s1",
            "steps": 500,
            "batch_size": 8,
            "num_workers": 2,
        },
        state=RunState.COMPLETED,
        created_at=now - 300,
        started_at=now - 290,
        finished_at=now - 10,
    )
    registry.save(run)
    paths = RunPaths.for_run(run.run_id, registry.runs_dir)

    checkpoint = paths.root / "output/checkpoints/checkpoint-200"
    pretrained = checkpoint / "pretrained_model"
    training_state = checkpoint / "training_state"
    pretrained.mkdir(parents=True)
    training_state.mkdir()
    (pretrained / "model.safetensors").write_bytes(b"browser-smoke")
    (pretrained / "train_config.json").write_text("{}")
    (training_state / "training_step.json").write_text('{"step": 200}')
    (training_state / "optimizer.pt").write_bytes(b"browser-smoke")

    relative_model = "output/checkpoints/checkpoint-200/pretrained_model/model.safetensors"
    paths.checkpoints_jsonl.write_text(
        json.dumps(
            {
                "step": 200,
                "path": relative_model,
                "sha256": "a" * 64,
                "ts": now - 10,
            }
        )
        + "\n"
    )
    append_event(paths.events_jsonl, "completed_naturally", final_step=200)

    samples = [
        {
            "step": 100,
            "loss": 0.12,
            "grdn": 1.8,
            "lr": 2.5e-5,
            "updt_s": 0.18,
            "data_s": 0.02,
            "samples_per_s": 40.0,
            "mem_gb": 6.5,
        },
        {
            "step": 150,
            "loss": 0.08,
            "grdn": 1.2,
            "lr": 2.0e-5,
            "updt_s": 0.16,
            "data_s": 0.015,
            "samples_per_s": 45.7,
            "mem_gb": 7.0,
        },
        {
            "step": 200,
            "loss": 0.05,
            "grdn": 0.9,
            "lr": 1.5e-5,
            "updt_s": 0.14,
            "data_s": 0.01,
            "samples_per_s": 53.3,
            "mem_gb": 7.5,
        },
    ]
    log_lines = []
    for sample in samples:
        values = {key: value for key, value in sample.items() if key != "step"}
        record = format_training_log_record(
            step=sample["step"],
            total_steps=500,
            eta_seconds=(500 - sample["step"]) * 0.15,
            **values,
        )
        log_lines.append(f"2026-07-24 12:00:00 [INFO] step {sample['step']}/500 | {record}")
    log_text = "\n".join(log_lines) + "\n"
    paths.stderr_log.write_text(log_text)
    return run, paths, log_text


@pytest.fixture
def training_gui_server(tmp_path: Path, monkeypatch):
    from lerobot.gui import server as gui_server_mod
    from lerobot.gui.api import datasets as datasets_api

    # The app restores user-persisted datasets on load. Keep the browser test
    # hermetic so a stale path in ~/.config cannot create unrelated 404s.
    monkeypatch.setattr(datasets_api, "_read_opened", lambda: [])
    monkeypatch.setattr(datasets_api, "_read_sources", lambda: [])

    workstation = TrainingHost(
        id="this-server",
        display_name="This server",
        transport=SubprocessTransport(workdir=tmp_path / "workdir"),
        capabilities={"gpu_name": "Browser Test GPU", "vram_mb": 24 * 1024},
    )
    registry = RunRegistry(runs_dir=tmp_path / "runs")
    run, paths, log_text = _seed_interrupted_run(registry)
    hosts = HostRegistry(hosts=[workstation])
    orchestrator = Orchestrator(host_registry=hosts, run_registry=registry)
    monkeypatch.setattr(orchestrator, "_prepare_and_launch", lambda *_args: None)
    training_api.init_state(orch=orchestrator, host_registry=hosts)

    port = _free_port()
    config = uvicorn.Config(gui_server_mod.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import requests

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base_url}/api/training/hosts", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        server.should_exit = True
        pytest.fail("GUI server did not become ready within 15 seconds")

    yield base_url, run, paths, log_text, tmp_path

    server.should_exit = True
    thread.join(timeout=10)
    training_api.reset_state_for_testing()


def test_training_dashboard_metrics_repair_and_resume(training_gui_server):
    base_url, source_run, _paths, log_text, tmp_path = training_gui_server
    browser_errors: list[str] = []

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except PlaywrightError as exc:
            pytest.skip(f"chromium not available: {exc}")

        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        context.grant_permissions(["clipboard-read", "clipboard-write"], origin=base_url)
        page = context.new_page()
        page.on("pageerror", lambda error: browser_errors.append(f"pageerror: {error}"))
        page.on(
            "console",
            lambda message: browser_errors.append(
                f"console: {message.text} ({message.location.get('url', 'unknown')})"
            )
            if message.type == "error"
            else None,
        )
        page.on(
            "response",
            lambda response: browser_errors.append(f"http {response.status}: {response.url}")
            if response.status >= 400
            else None,
        )

        page.goto(base_url)
        page.wait_for_function("typeof switchTab === 'function'")
        page.evaluate("switchTab('model')")
        page.wait_for_function("typeof trainingSelectRun === 'function'")
        page.evaluate("(runId) => trainingSelectRun(runId)", source_run.run_id)
        page.wait_for_selector(".training-detail-pane")

        # The old any-checkpoint heuristic is repaired through the real API
        # before the browser renders the detail.
        assert page.text_content(".training-state-badge") == "stopped"
        assert "200 / 500" in page.text_content(".training-stats-row")
        assert "53.3 samples/s" in page.text_content(".training-stats-row")
        assert "7.5 GB" in page.text_content(".training-stats-row")
        assert page.is_visible(f"#training-resume-{source_run.run_id}")
        detail_badge = page.locator(".training-detail-actions .training-state-badge")
        detail_badge.hover()
        page.wait_for_selector(".training-state-tooltip.visible")
        assert "choice or interruption" in page.text_content(".training-state-tooltip")
        detail_badge.focus()
        assert detail_badge.get_attribute("aria-describedby") == "training-state-tooltip"

        chart_titles = page.locator(".training-chart-title").all_text_contents()
        assert chart_titles == [
            "Loss",
            "Gradient norm",
            "Learning rate",
            "Peak GPU allocation (GB)",
            "Step time (ms)",
        ]
        assert page.locator(".training-chart-empty").count() == 0
        assert page.locator("canvas.training-chart-canvas").count() == 5
        # Logged metric samples are sparse global steps, not consecutive
        # points. The shared chart must preserve 100/150/200 instead of
        # inventing 198/199/200 from "3 points ending at step 200".
        assert page.evaluate("_chartGroups.training.xValues") == [100, 150, 200]
        assert page.evaluate("_chartStepAtIndex(_chartGroups.training, 0)") == 100
        assert page.evaluate("_chartStepAtIndex(_chartGroups.training, 2)") == 200
        assert page.eval_on_selector_all(
            "canvas.training-chart-canvas",
            """(canvases) => canvases.every((canvas) => {
              if (canvas.width <= 0 || canvas.height <= 0) return false;
              const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
              return pixels.some((channel) => channel !== 0);
            })""",
        )
        assert (
            len(
                page.eval_on_selector(
                    ".training-charts", "el => getComputedStyle(el).gridTemplateColumns.split(' ')"
                )
            )
            == 2
        )

        page.click(".training-log-copy")
        page.wait_for_function("navigator.clipboard.readText().then((text) => text.includes('step 200/500'))")
        assert awaitable_clipboard_text(page).rstrip() == log_text.rstrip()

        screenshot = tmp_path / "training-dashboard-smoke.png"
        page.screenshot(path=str(screenshot), full_page=True)

        # The responsive layout collapses to one column without losing charts.
        page.set_viewport_size({"width": 700, "height": 900})
        assert (
            len(
                page.eval_on_selector(
                    ".training-charts", "el => getComputedStyle(el).gridTemplateColumns.split(' ')"
                )
            )
            == 1
        )
        assert page.locator("canvas.training-chart-canvas").count() == 5

        # Resume uses the real API but a no-op launch callback. It must create
        # a new run, preserve the source, and surface checkpoint lineage.
        page.once("dialog", lambda dialog: dialog.accept())
        page.click(f"#training-resume-{source_run.run_id}")
        page.wait_for_function(
            "() => document.querySelector('.training-detail-title')?.textContent.includes('(resume 200)')"
        )
        assert page.text_content(".training-state-badge") == "pending"
        assert f"{source_run.run_id} · step 200" in page.text_content(".training-args-table")

        context.close()
        browser.close()

    assert browser_errors == []


def awaitable_clipboard_text(page) -> str:
    """Read clipboard text through Playwright's promise-aware evaluate."""
    return page.evaluate("navigator.clipboard.readText()")
