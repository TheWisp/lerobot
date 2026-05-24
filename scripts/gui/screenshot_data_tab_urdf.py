"""Screenshot the data-tab URDF viz with the action ghost enabled.

Drives the GUI through CDP up to the point the URDF iframe is armed +
the playhead has moved off frame 0 (so the action ghost is visually
separated from the solid state pose), then closes the CDP session and
captures the rendered window with ffmpeg's x11grab demuxer.

Mirrors the recipe in [scripts/gui/screenshot_gui.py] on
``teleop/learned-ik`` — we keep CDP only for arming, not capture, because
Chrome's CDP debugger gets stuck once an iframe attaches.

Requires:
    - X display (DISPLAY=:0)
    - ffmpeg with x11grab support
    - google-chrome
    - dataset ``thewisp/intervention_cylinder_ring_assembly`` cached locally
      (already present in this repo's HF cache).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
import websocket

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "thewisp/intervention_cylinder_ring_assembly"
EPISODE_IDX = 0
TARGET_FRAME = 80  # far enough into the episode that the operator is moving
GUI_PORT = 8765
GUI_URL = f"http://127.0.0.1:{GUI_PORT}/?urdfGhost=on"
OUT = REPO_ROOT / "src/lerobot/gui/docs/images/urdf_viz_data_tab_ghost.png"
SCRATCH = Path(tempfile.gettempdir()) / "lerobot_urdf_viz_screenshot"
SCRATCH.mkdir(parents=True, exist_ok=True)
CDP_PORT = 9700 + (os.getpid() % 100)
WIN_W = 1600
WIN_H = 900


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cdp(ws, mid: int, method: str, params: dict | None = None, timeout: float = 6.0) -> dict:
    payload: dict = {"id": mid, "method": method}
    if params:
        payload["params"] = params
    ws.send(json.dumps(payload))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ws.settimeout(max(deadline - time.monotonic(), 0.01))
        try:
            msg = json.loads(ws.recv())
        except websocket.WebSocketTimeoutException:
            return {}
        if msg.get("id") == mid:
            return msg
    return {}


def js(ws, mid: int, expr: str):
    r = cdp(ws, mid, "Runtime.evaluate", {"expression": expr, "awaitPromise": True, "returnByValue": True})
    return r.get("result", {}).get("result", {}).get("value")


def wait_for_gui(timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{GUI_PORT}/api/datasets", timeout=0.5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def open_dataset() -> bool:
    """Open the target dataset via the GUI's REST API."""
    try:
        r = requests.post(
            f"http://127.0.0.1:{GUI_PORT}/api/datasets",
            json={"repo_id": DATASET_ID},
            timeout=120.0,
        )
    except Exception as e:
        log(f"open_dataset POST failed: {e}")
        return False
    if r.status_code not in (200, 201):
        log(f"open_dataset HTTP {r.status_code}: {r.text[:200]}")
        return False
    return True


def x11_screenshot(out_path: Path, x: int, y: int, w: int, h: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "x11grab",
            "-video_size",
            f"{w}x{h}",
            "-i",
            f":0.0+{x},{y}",
            "-frames:v",
            "1",
            str(out_path),
        ],
        check=True,
        timeout=15,
    )


def main() -> int:
    out_dir = OUT.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"starting GUI on port {GUI_PORT}")
    src_path = REPO_ROOT / "src"
    env = {**os.environ, "PYTHONPATH": f"{src_path}:{os.environ.get('PYTHONPATH', '')}"}
    gui = subprocess.Popen(
        [
            "/home/feit/Documents/lerobot/.venv/bin/python",
            "-m",
            "lerobot.gui",
            "--port",
            str(GUI_PORT),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for_gui():
            log("GUI never came online")
            return 1
        log("GUI online")

        # Pre-load on the server side so the front-end openDataset call below
        # finishes quickly (`/api/datasets` POST is idempotent for an already-
        # opened repo). With 1k+ episodes the episode-list fetch dominates;
        # priming the server first means CDP only has to wait on the JS-side
        # `datasets[id]` populate.
        if not open_dataset():
            return 1
        log(f"opened dataset {DATASET_ID} (server-side)")

        profile = SCRATCH / f"chrome_profile_{CDP_PORT}"
        profile.mkdir(parents=True, exist_ok=True)
        log(f"launching chrome cdp_port={CDP_PORT}")
        chrome = subprocess.Popen(
            [
                "google-chrome",
                f"--remote-debugging-port={CDP_PORT}",
                f"--user-data-dir={profile}",
                "--remote-allow-origins=*",
                "--no-first-run",
                "--no-default-browser-check",
                f"--window-size={WIN_W},{WIN_H}",
                "--window-position=0,0",
                GUI_URL,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            ws_url = None
            for _ in range(80):
                try:
                    tabs = requests.get(f"http://127.0.0.1:{CDP_PORT}/json", timeout=0.5).json()
                    pages = [
                        t
                        for t in tabs
                        if t.get("type") == "page" and f"127.0.0.1:{GUI_PORT}" in t.get("url", "")
                    ]
                    if not pages:
                        pages = [t for t in tabs if t.get("type") == "page"]
                    if pages:
                        ws_url = pages[0]["webSocketDebuggerUrl"]
                        break
                except Exception:
                    pass
                time.sleep(0.2)
            if not ws_url:
                log("CDP never came online")
                return 1

            ws = websocket.create_connection(ws_url, timeout=6.0)
            cdp(ws, 1, "Page.enable")

            for i in range(60):
                if js(ws, 100 + i, "typeof switchTab") == "function":
                    log(f"app.js loaded after {i * 0.3:.1f}s")
                    break
                time.sleep(0.3)

            js(ws, 200, "switchTab('data'); 'ok'")
            time.sleep(0.5)

            # Drive the open through the front-end function — it populates
            # the `datasets` global and the `episodes` lookup that
            # selectEpisode depends on (a raw server POST doesn't, the JS
            # state is independent of server state).
            log("calling openDataset() in the page")
            js(ws, 250, f"openDataset({DATASET_ID!r}); 'kicked'")

            # Wait for the front-end to finish populating episode metadata.
            length = 0
            for i in range(160):
                # episodes is a top-level global ({id: [{episode_index, length, ...}]});
                # use it directly rather than chasing datasets[id].episodes which
                # the open-flow doesn't set.
                v = js(
                    ws,
                    300 + i,
                    f"(window.episodes && episodes[{DATASET_ID!r}] && episodes[{DATASET_ID!r}][0] "
                    f"&& (episodes[{DATASET_ID!r}][0].video_length || episodes[{DATASET_ID!r}][0].length)) || 0",
                )
                if v and int(v) > 0:
                    length = int(v)
                    log(f"episode 0 length={length} after {i * 0.25:.1f}s")
                    break
                time.sleep(0.25)
            if length == 0:
                log("episode metadata never arrived in the data tab")
                return 1

            js(ws, 400, f"selectEpisode({DATASET_ID!r}, {EPISODE_IDX}, {length}); 'ok'")

            # Give the URDF probe + iframe mount a head start, then seek so
            # the action ghost has somewhere to diverge to.
            time.sleep(2.0)
            js(ws, 500, f"loadAllFrames({TARGET_FRAME}); 'ok'")
            log(f"sought to frame {TARGET_FRAME}")

            # Close CDP before the iframe finishes loading meshes — past that
            # point the parent CDP socket can wedge per the memory recipe.
            with contextlib.suppress(Exception):
                ws.close()

            # Let the iframe finish loading 4x STL meshes (state + ghost,
            # both arms) and paint a few frames.
            log("waiting for STL load + paint")
            time.sleep(12)

            log(f"x11grab {WIN_W}x{WIN_H} -> {OUT}")
            x11_screenshot(OUT, 0, 0, WIN_W, WIN_H)
            log("wrote screenshot")
            return 0
        finally:
            chrome.terminate()
            with contextlib.suppress(Exception):
                chrome.wait(timeout=3)
            chrome.kill()
    finally:
        gui.terminate()
        with contextlib.suppress(Exception):
            gui.wait(timeout=3)
        gui.kill()


if __name__ == "__main__":
    sys.exit(main())
