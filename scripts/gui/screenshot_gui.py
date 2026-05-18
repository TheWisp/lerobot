"""GUI screenshot via X11 grab instead of CDP capture.

Bypasses the CDP capture deadlock that fires when the MeshCat iframe
attaches: we still use CDP to drive the page (navigate, switch tab,
kick obs-stream viewer), but the actual snapshot comes from ffmpeg
reading the X server. CDP can stop responding all it wants — we don't
care, because we've already armed the page state we need.

Requires:
    - X display (DISPLAY=:0)
    - ffmpeg with x11grab support
    - google-chrome
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

GUI_URL = "http://127.0.0.1:8000/"
# A per-run scratch dir under the platform's tempdir; not a fixed
# /tmp path so concurrent invocations don't collide.
SCRATCH = Path(tempfile.gettempdir()) / "lerobot_screenshot_gui"
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT = SCRATCH / "gui.png"
CDP_PORT = 9700 + (os.getpid() % 100)
WIN_W = 1600
WIN_H = 900


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cdp(ws, mid, method, params=None, timeout=6.0):
    payload = {"id": mid, "method": method}
    if params:
        payload["params"] = params
    ws.send(json.dumps(payload))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ws.settimeout(max(deadline - time.monotonic(), 0.01))
        try:
            msg = json.loads(ws.recv())
        except websocket.WebSocketTimeoutException:
            return {}  # don't block forever; caller can ignore
        if msg.get("id") == mid:
            return msg
    return {}


def js(ws, mid, expr):
    r = cdp(ws, mid, "Runtime.evaluate", {"expression": expr})
    return r.get("result", {}).get("result", {}).get("value")


def x11_screenshot(out_path: Path, x: int, y: int, w: int, h: int) -> None:
    """Capture a region of :0 via ffmpeg's x11grab demuxer."""
    cmd = [
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
    ]
    subprocess.run(cmd, check=True, timeout=15)


def main():
    profile = SCRATCH / f"chrome_profile_{CDP_PORT}"
    profile.mkdir(parents=True, exist_ok=True)
    log(f"chrome cdp_port={CDP_PORT}")
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

    ws_url = None
    for _ in range(80):
        try:
            tabs = requests.get(f"http://127.0.0.1:{CDP_PORT}/json", timeout=0.5).json()
            page_tabs = [t for t in tabs if t.get("type") == "page" and "127.0.0.1:8000" in t.get("url", "")]
            if not page_tabs:
                page_tabs = [t for t in tabs if t.get("type") == "page"]
            if page_tabs:
                ws_url = page_tabs[0]["webSocketDebuggerUrl"]
                break
        except Exception:
            pass
        time.sleep(0.2)
    if not ws_url:
        log("CDP never came online")
        chrome.terminate()
        return 1

    ws = websocket.create_connection(ws_url, timeout=6.0)
    cdp(ws, 1, "Page.enable")

    # Wait until app.js's switchTab is defined.
    for i in range(40):
        if js(ws, 100 + i, "typeof switchTab") == "function":
            log(f"switchTab loaded after {i * 0.3:.1f}s")
            break
        time.sleep(0.3)

    js(ws, 200, "switchTab('run'); 'ok'")
    time.sleep(0.5)
    js(ws, 201, "selectWorkflow('replay'); 'ok'")
    time.sleep(0.5)
    workflow = js(ws, 202, "selectedWorkflow")
    log(f"workflow = {workflow!r}")

    # Kick the obs viewer. We don't need to interpret its result — the X11
    # snapshot doesn't care about CDP responsiveness from this point on.
    js(ws, 300, "startObsStreamViewer(); 'kicked'")
    log("kicked startObsStreamViewer; closing CDP and waiting for iframe paint")

    # Close CDP cleanly so it doesn't choke on the iframe attach.
    with contextlib.suppress(Exception):
        ws.close()

    # Let the MeshCat WebSocket connect + paint the first frame.
    time.sleep(10)

    # X11 grab the visible Chrome window region.
    log(f"x11grab {WIN_W}x{WIN_H} -> {OUT}")
    x11_screenshot(OUT, 0, 0, WIN_W, WIN_H)
    log("wrote screenshot")

    chrome.terminate()
    try:
        chrome.wait(timeout=3)
    except subprocess.TimeoutExpired:
        chrome.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
