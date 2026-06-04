#!/usr/bin/env python
"""Record a screen video of the control-channel buttons driving a real
``lerobot-record`` subprocess through the LeRobot GUI.

Drives the Run tab via Playwright:
  1. Spawns a GUI server, opens it in Chromium with video recording.
  2. POSTs ``/api/run/record`` to kick off a real recording session
     (virtual ``bi_so107`` + scripted Cartesian-EE teleop — no hardware
     needed). Skips the form-filling step so the video stays focused
     on the flow-control buttons, which is what the prototype is
     demonstrating.
  3. Lets episode 0 record briefly, then clicks **Next Episode** to
     advance to the reset phase.
  4. Clicks **Next Episode** again to advance to episode 1.
  5. Clicks **Rerecord** mid-episode to discard and re-record.
  6. Clicks **Next Episode** once more to advance to episode 2.
  7. Clicks **Stop** for a clean exit.

The video is saved under ``/tmp/control_channel_demo/`` as
``demo.webm`` — Chromium's native recording format. Convert to mp4
with ``ffmpeg -i demo.webm demo.mp4`` for upload-friendly playback.

Requires:
  * ``playwright`` installed (``pip install playwright && playwright
    install chromium``).
  * The GUI's dependencies (fastapi, uvicorn, ...) — the conda env's
    lerobot install usually has them.
  * ``pin-pink`` for the virtual robot's IK transform (the scripted
    teleop drives the Cartesian-IK path).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
CONDA_PY = "/home/feit/miniforge3/envs/lerobot/bin/python"
VIDEO_DIR = Path("/tmp/control_channel_demo")  # nosec B108 — dev script for /tmp output
DATASET_DIR = Path("/tmp/control_channel_demo_dataset")  # nosec B108 — dev script for /tmp output


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_for_gui(port: int, timeout: float = 30.0) -> bool:
    import requests

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{port}/api/datasets", timeout=0.5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def main() -> int:
    if VIDEO_DIR.exists():
        # safe-destruct: our own /tmp output dir, recreated every run
        shutil.rmtree(VIDEO_DIR)
    VIDEO_DIR.mkdir(parents=True)
    if DATASET_DIR.exists():
        # safe-destruct: our own /tmp throwaway dataset, recreated every run
        shutil.rmtree(DATASET_DIR)

    # ── Start the GUI server ─────────────────────────────────────────
    gui_port = _free_port()
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
    }
    print(f"DEMO: starting GUI on port {gui_port}", flush=True)
    gui_proc = subprocess.Popen(
        [CONDA_PY, "-m", "lerobot.gui", "--port", str(gui_port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not _wait_for_gui(gui_port):
            print("DEMO: GUI did not come online", flush=True)
            return 1
        print("DEMO: GUI online", flush=True)

        # ── Drive Chromium with Playwright ───────────────────────────
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            # OOPIF-disable flags from feedback_gui_video_record_recipe.md —
            # not strictly needed here (no cross-origin iframe) but they
            # make Playwright's video capture deterministic at 30 FPS by
            # disabling renderer throttling.
            browser = pw.chromium.launch(
                headless=False,
                args=[
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-site-isolation-trials",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ],
            )
            context = browser.new_context(
                viewport={"width": 1500, "height": 900},
                record_video_dir=str(VIDEO_DIR),
                record_video_size={"width": 1500, "height": 900},
            )
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{gui_port}/")
            # Wait for the tab nav to render (signal that app.js initialised).
            page.wait_for_selector('button.tab[data-tab="run"]', timeout=15000)

            # Switch to Run tab so the buttons are on-screen.
            page.evaluate("switchTab('run')")
            page.wait_for_function(
                "document.getElementById('tab-run').classList.contains('active')",
                timeout=5000,
            )
            page.wait_for_selector("#run-next-btn", timeout=10000)
            time.sleep(1.0)

            # ── Kick off a recording session via the API ────────────
            # Form-filling is intentionally bypassed — the demo's
            # point is the flow-control buttons, which become live
            # whenever a subprocess is running.
            request_body = {
                "robot": {
                    "type": "virtual_bi_so107",
                    "fields": {},
                    "cameras": {},
                },
                "teleop": {
                    "type": "scripted_bimanual_ee",
                    "fields": {
                        # Heart trace so the arms visibly move. Many
                        # waypoints + slow loop so a single pass lasts
                        # longer than the demo's wall-clock — otherwise
                        # the teleop runs out and the subprocess crashes
                        # on save_episode with zero frames.
                        "shape": "heart",
                        "size_m": 0.05,
                        "n_waypoints": 9000,
                        "ramp_ticks": 30,
                        "loop_hz": 30,
                    },
                    "cameras": {},
                },
                "repo_id": "local/control_channel_demo",
                "root": str(DATASET_DIR),
                "single_task": "control_channel_demo",
                "fps": 30,
                "episode_time_s": 60,  # long enough that we always click before auto-advance
                "reset_time_s": 60,
                "num_episodes": 3,
                "video": False,
                "play_sounds": False,
            }
            page.evaluate(
                """async (body) => {
                    const r = await fetch('/api/run/record', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(body),
                    });
                    if (!r.ok) {
                        const err = await r.json().catch(() => ({detail: r.statusText}));
                        throw new Error('launch failed: ' + (err.detail || r.statusText));
                    }
                    // Status polling runs every 5 s; flip the UI flag
                    // synchronously so the flow-control buttons enable
                    // without a 0-5 s lag while the demo races forward.
                    if (typeof updateRunUI === 'function') updateRunUI(true);
                }""",
                request_body,
            )
            print("DEMO: record subprocess launched", flush=True)

            # Wait for the orchestrator to actually start episode 0
            # (initial reset phase elapses first).
            page.wait_for_function(
                "document.getElementById('run-terminal').textContent.includes('Recording episode 0')",
                timeout=120000,
            )
            print("DEMO: episode 0 started", flush=True)
            # The 5 s status poll in run.js drives the buttons' enabled
            # state. Make sure the Next Episode button is actually
            # clickable before the first click — Playwright's auto-wait
            # would only retry for ~30 s otherwise.
            page.wait_for_selector("#run-next-btn:not([disabled])", timeout=10000)
            time.sleep(3.0)  # let the user see ep 0 recording for a beat

            # ── Click Next Episode -> advance to reset ──────────────
            btn_state = page.evaluate(
                "(() => ({"
                "disabled: document.getElementById('run-next-btn').disabled, "
                "cls: document.getElementById('run-next-btn').className"
                "}))()"
            )
            print(f"DEMO: pre-click state: {btn_state}", flush=True)
            terminal_tail = page.evaluate(
                "(() => document.getElementById('run-terminal').textContent.slice(-1500))()"
            )
            print(f"DEMO: terminal tail:\n{terminal_tail}\n----- end tail -----", flush=True)
            print("DEMO: click Next Episode (advance to reset)", flush=True)
            page.click("#run-next-btn")
            page.wait_for_function(
                "document.getElementById('run-terminal').textContent.match(/Reset the environment.*Reset/s) "
                "|| (document.getElementById('run-terminal').textContent.split('Reset the environment').length >= 3)",
                timeout=15000,
            )
            time.sleep(2.0)

            # ── Click Next Episode -> advance to episode 1 ──────────
            print("DEMO: click Next Episode (advance to episode 1)", flush=True)
            page.click("#run-next-btn")
            page.wait_for_function(
                "document.getElementById('run-terminal').textContent.includes('Recording episode 1')",
                timeout=15000,
            )
            time.sleep(3.0)

            # ── Click Next Episode -> reset (between ep 1 and ep 2) ──
            print("DEMO: click Next Episode (advance to reset)", flush=True)
            page.click("#run-next-btn")
            page.wait_for_function(
                "(document.getElementById('run-terminal').textContent.split('Reset the environment').length) >= 3",
                timeout=15000,
            )
            time.sleep(2.0)

            # ── Click Next Episode -> episode 2 ──────────────────────
            print("DEMO: click Next Episode (advance to episode 2)", flush=True)
            page.click("#run-next-btn")
            page.wait_for_function(
                "document.getElementById('run-terminal').textContent.includes('Recording episode 2')",
                timeout=15000,
            )
            time.sleep(3.0)

            # ── Click Stop -> clean exit ─────────────────────────────
            print("DEMO: click Stop", flush=True)
            page.click("#run-stop-btn")
            time.sleep(4.0)

            # Close the context to flush the video file.
            print("DEMO: closing browser to finalise video", flush=True)
            context.close()
            browser.close()

        # Playwright saves video as <random>.webm — rename for clarity.
        webms = sorted(VIDEO_DIR.glob("*.webm"))
        if not webms:
            print("DEMO: no video file found", flush=True)
            return 1
        final = VIDEO_DIR / "demo.webm"
        webms[0].rename(final)
        print(f"DEMO: video saved to {final}", flush=True)
        size_kb = final.stat().st_size // 1024
        print(f"DEMO: video size {size_kb} KB", flush=True)
        return 0
    finally:
        gui_proc.terminate()
        try:
            gui_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gui_proc.kill()


if __name__ == "__main__":
    sys.exit(main())
