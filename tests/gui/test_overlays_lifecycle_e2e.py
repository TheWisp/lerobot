"""End-to-end lifecycle/ordering regression for the live Overlays panel.

Covers the four orderings that unit tests can't reach — they exercise the real frontend
(overlays.js), the obs-stream shm, and a real CUDA standalone, driven over Chrome/CDP,
which is exactly where the field bugs lived:

  A  teleop-first  -> overlay         happy path
  B  overlay-first -> teleop          panel opened before cameras exist; must recover
  C  teleop restart (stop + start)    standalone must re-attach to the fresh shm segment
  D  overlay restart (clear + select) relaunch cleanly while the stream stays up

Heavy + hardware-bound: spins up the GUI, a continuous synthetic obs-stream publisher
(random frames, no dataset), and a real standalone. Skipped unless a display, CUDA, and
Chrome are all present — i.e. it runs locally / in the GPU lane, not in fast CI.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")  # the GUI extra
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "gui"))

pytestmark = [
    pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="overlay E2E needs an X display for CDP"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="overlay standalone needs CUDA"),
    pytest.mark.skipif(shutil.which("google-chrome") is None, reason="overlay E2E needs Chrome for CDP"),
]

PNL = "#overlays-panel-run"

# A continuous synthetic publisher: random frames at ~10 Hz, arbitrary keys via FEED_KEYS,
# no dataset. Faithful to a live robot (which streams continuously), so timing isn't at the
# mercy of a finite replay.
_FEEDER_SRC = """
import os, signal, time
import numpy as np
from lerobot.robots.obs_stream import ObservationStream
K = os.environ["FEED_KEYS"].split(","); H, W = 240, 320
s = ObservationStream({k: (H, W, 3) for k in K}, {}); _stop = False
def _sig(*a):
    global _stop; _stop = True
signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
try:
    while not _stop:
        s.write_obs({k: np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for k in K}); time.sleep(0.1)
finally:
    s.cleanup()
"""


# A state-only publisher (no cameras): SO-101-style `<motor>.pos` scalars so
# `resolve_robot` matches a vendored URDF — the camera-less "URDF run" shape.
_STATE_FEEDER_SRC = """
import math, os, signal, time
from lerobot.robots.obs_stream import ObservationStream
K = os.environ["FEED_KEYS"].split(",")
s = ObservationStream({k: 1 for k in K}, {}); _stop = False
def _sig(*a):
    global _stop; _stop = True
signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
try:
    while not _stop:
        s.write_obs({k: math.sin(time.time()) for k in K}); time.sleep(0.1)
finally:
    s.cleanup()
"""

SO101_JOINT_KEYS = ",".join(
    f"{m}.pos" for m in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _clean_shm():
    subprocess.run(["bash", "-c", "rm -f /dev/shm/lerobot_obs_* /dev/shm/lerobot_overlay_*"], check=False)


def _http(url, method="GET", timeout=5):
    with contextlib.suppress(Exception):
        return urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=timeout)
    return None


@pytest.fixture(scope="module")
def gui_url():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "lerobot.gui", "--port", str(port), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/"
    try:
        if not any(_http(url, timeout=1) for _ in _poll_iter(40)):
            proc.kill()
            pytest.skip("GUI did not come up")
        yield url
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()


@pytest.fixture
def feeder(tmp_path, gui_url):
    """A stoppable/launchable continuous obs-stream publisher (for the restart case)."""
    script = tmp_path / "feed.py"
    script.write_text(_FEEDER_SRC)
    procs = []

    state_script = tmp_path / "feed_state.py"
    state_script.write_text(_STATE_FEEDER_SRC)

    def _spawn(path, keys):
        p = subprocess.Popen(
            [sys.executable, str(path)],
            env={**os.environ, "FEED_KEYS": keys, "LEROBOT_OBS_STREAM": "1"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)
        return p

    def launch(keys="front,left_wrist,right_wrist,top"):
        return _spawn(script, keys)

    def launch_state(keys=SO101_JOINT_KEYS):
        return _spawn(state_script, keys)

    def stop(p):
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            with contextlib.suppress(Exception):
                p.wait(timeout=6)
            if p.poll() is None:
                p.kill()

    _clean_shm()
    yield type(
        "Feeder",
        (),
        {
            "launch": staticmethod(launch),
            "launch_state": staticmethod(launch_state),
            "stop": staticmethod(stop),
        },
    )
    _http(gui_url + "api/overlays/live/stop", "POST", 10)
    for p in procs:
        stop(p)
    _clean_shm()


# --- CDP helpers -------------------------------------------------------------
from screenshot_gui import GuiScreenshotSession  # noqa: E402


def _poll_iter(seconds, step=0.5):
    start = time.time()
    while time.time() - start < seconds:
        yield True
        time.sleep(step)


def _poll(fn, timeout):
    return any(fn() for _ in _poll_iter(timeout, step=1.0))


def _session(gui_url, tmp_path):
    return GuiScreenshotSession(
        output_path=str(tmp_path / "e2e.png"),
        url=gui_url,
        start_gui=False,
        window_size=(1600, 900),
        post_close_sleep=1.0,
    )


def _select_overlay(s):
    s.eval(
        f"var rp=document.querySelector('{PNL} .overlays-picker'); rp.value='sam3_track'; rp.dispatchEvent(new Event('change'));"
    )
    s.sleep(1.0)
    s.eval(
        f"var x=document.querySelector(\"{PNL} .overlays-obj-name[data-i='0']\"); if(x){{x.value='green ring'; x.dispatchEvent(new Event('input'));}}"
    )
    s.sleep(1.0)


def _clear_overlay(s):
    s.eval(
        f"var rp=document.querySelector('{PNL} .overlays-picker'); rp.value=''; rp.dispatchEvent(new Event('change'));"
    )


def _attach_viewer(s):
    s.eval("startObsStreamViewer()")
    s.wait_until("document.querySelectorAll('.obs-cam-grid img').length>0", timeout=25)
    s.eval("(document.querySelector('#rerun-viewer iframe')||{remove(){}}).remove()")


def _cam_buttons(s):
    return json.loads(
        s.eval(
            f"JSON.stringify(Array.from(document.querySelectorAll('{PNL} .overlays-cam-btn')).map(b=>b.dataset.cam))"
        )
    )


def _n_rendered(s):
    widths = json.loads(
        s.eval(
            "JSON.stringify(Array.from(document.querySelectorAll('.overlay-layer')).map(o=>o.naturalWidth))"
        )
    )
    return sum(1 for nw in widths if nw > 0)


def _live_fps(gui_url):
    r = _http(gui_url + "api/overlays/live/status")
    if r is None:
        return 0.0
    with contextlib.suppress(Exception):
        return json.loads(r.read()).get("fps", 0)
    return 0.0


# --- the four orderings ------------------------------------------------------
def test_a_teleop_first_then_overlay(gui_url, feeder, tmp_path):
    feeder.launch()
    with _session(gui_url, tmp_path) as s:
        s.eval("switchTab('run')")
        s.sleep(0.5)
        _attach_viewer(s)
        _select_overlay(s)
        assert _poll(lambda: _n_rendered(s) >= 1, 45), "overlay never rendered (teleop-first)"


def test_b_overlay_first_then_teleop(gui_url, feeder, tmp_path):
    # The regression: the panel opened while the obs stream doesn't exist yet must NOT
    # latch an empty camera selection — it must recover once the cameras appear.
    with _session(gui_url, tmp_path) as s:
        s.eval("switchTab('run')")
        s.sleep(0.5)
        _select_overlay(s)
        assert _cam_buttons(s) == [], "panel should show no cameras before the stream exists"
        feeder.launch()
        _attach_viewer(s)
        assert _poll(lambda: len(_cam_buttons(s)) > 0, 30), "panel never recovered its cameras (ordering bug)"
        assert _poll(lambda: _n_rendered(s) >= 1, 45), "overlay never rendered after the panel recovered"


def test_c_teleop_restart_reattaches(gui_url, feeder, tmp_path):
    # The regression: a teleop stop+start replaces the obs-stream shm; the standalone must
    # re-attach to the fresh segment instead of staying stuck on the dead one.
    f1 = feeder.launch()
    with _session(gui_url, tmp_path) as s:
        s.eval("switchTab('run')")
        s.sleep(0.5)
        _attach_viewer(s)
        _select_overlay(s)
        assert _poll(lambda: _live_fps(gui_url) > 0, 45), "overlay never started processing"
        feeder.stop(f1)
        assert _poll(lambda: _live_fps(gui_url) == 0, 15), "fps should fall to 0 once the stream stops"
        feeder.launch()  # teleop starts again
        _attach_viewer(s)
        assert _poll(lambda: _live_fps(gui_url) > 0, 30), (
            "standalone did not re-attach to the restarted stream (C)"
        )


def test_d_overlay_restart(gui_url, feeder, tmp_path):
    feeder.launch()
    with _session(gui_url, tmp_path) as s:
        s.eval("switchTab('run')")
        s.sleep(0.5)
        _attach_viewer(s)
        _select_overlay(s)
        assert _poll(lambda: _n_rendered(s) >= 1, 45), "overlay never rendered"
        _clear_overlay(s)
        assert _poll(lambda: _n_rendered(s) == 0, 15), "overlay did not clear on deselect"
        _select_overlay(s)  # re-select with the stream still up
        assert _poll(lambda: _n_rendered(s) >= 1, 45), "overlay did not come back after reselect (D)"


def test_e_urdf_tile_appears_when_stream_comes_up_late(gui_url, feeder, tmp_path):
    """Regression guard (the 750a061ac probe-ordering bug): the URDF-viz probe must run AFTER
    the stream wait. /api/run/urdf-viz is only answerable once the run's obs stream has data —
    probing it up front latches urdfVizActive=false for every GUI-launched run and the
    visualizer tile never renders. This drives the GUI-launched-run timing the other four
    tests can't reach (their feeder is up BEFORE the viewer starts): viewer first, stream late.
    """
    with _session(gui_url, tmp_path) as s:
        s.eval("switchTab('run')")
        s.eval("_isRunning = true")  # a launched run whose robot hasn't connected yet
        s.eval("startObsStreamViewer()")
        s.sleep(3.0)  # the buggy ordering has already probed urdf-viz (false) by now
        feeder.launch_state()  # the robot "connects": state-only stream, vendored-URDF joints
        try:
            assert _poll(lambda: bool(s.eval("!!document.querySelector('#rerun-viewer iframe')")), 30), (
                "URDF tile never appeared for a late stream (probe-ordering regression)"
            )
        finally:
            # Defensive: drop the iframe before further CDP traffic, then release the run flag.
            s.eval("(document.querySelector('#rerun-viewer iframe')||{remove(){}}).remove()")
            s.eval("_isRunning = false")
