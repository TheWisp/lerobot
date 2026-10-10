"""The point groups' live view from the GUI: Start runs it, Finish stops it, the stream is relayed.

The view (benchmarks/group_live.py --live --record) is its own process with a small HTTP server: an MJPEG stream
and /record/status. A fake view stands in here, as a server for the relay and as a script for Start and Finish.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import socket
import socketserver
import sys
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import pregrasp, showservo

FRAMES = [b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytes([i]) * 8 + b"\r\n" for i in range(2)]

# The view as a script: answers /record/status with a growing frame count until a SIGTERM, which it reports as
# the real view does when it closes its recording.
FAKE_VIEW = """
import http.server, json, signal, sys, threading, time
port = int(sys.argv[sys.argv.index("--port") + 1])
assert "--record" in sys.argv and "--server" in sys.argv
state = {"frames": 0, "recording": "/recordings/groups_fake"}

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        body = json.dumps({"recording": state["recording"], "frames": state["frames"], "last": None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = http.server.HTTPServer(("127.0.0.1", port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

def bye(*_):
    print(f"recorded {state['frames']} frames to {state['recording']}", flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, bye)
while True:
    time.sleep(0.02)
    state["frames"] += 1
"""


class FakeView:
    """The view's server as the relay sees it: /record/status answers JSON and /stream two frames, then closes."""

    def __init__(self):
        view = self
        self.calls: list[str] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                view.calls.append(self.path)
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    for frame in FRAMES:
                        self.wfile.write(frame)
                    return
                body = json.dumps({"recording": "/recordings/groups_1", "frames": 7, "last": None}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        self.server = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def address(self) -> tuple[str, int]:
        return self.server.server_address[:2]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(monkeypatch, address: tuple[str, int]) -> TestClient:
    monkeypatch.setattr(pregrasp, "GROUPS_VIEW", address)
    monkeypatch.setattr(pregrasp._state, "groups", pregrasp._GroupsView())
    app = FastAPI()
    app.include_router(pregrasp.router)
    return TestClient(app)


def test_the_stream_relay_strips_the_views_headers_and_connects_again_after_the_view_closes():
    view = FakeView()

    async def take() -> bytes:
        gen = pregrasp._relay_stream(view.address)
        got = b""
        try:
            while (
                got.count(FRAMES[1]) < 2
            ):  # the fake closes after two frames; two more mean a second connection
                got += await asyncio.wait_for(anext(gen), timeout=10.0)
        finally:
            await gen.aclose()
        return got

    try:
        got = asyncio.run(take())
    finally:
        view.close()
    assert got.startswith(
        FRAMES[0] + FRAMES[1] + FRAMES[0]
    )  # the view's own response headers never reach the page
    assert view.calls.count("/stream") == 2


def test_the_status_reads_a_view_started_elsewhere_and_finish_will_not_stop_it(monkeypatch):
    view = FakeView()
    try:
        with _client(monkeypatch, view.address) as client:
            st = client.get("/api/pregrasp/groups/status").json()
            assert (
                st["running"]
                and st["ready"]
                and st["recording"] == "/recordings/groups_1"
                and st["frames"] == 7
            )
            r = client.post("/api/pregrasp/groups/stop")
            assert r.status_code == 409 and "not started here" in r.json()["detail"]
    finally:
        view.close()


def test_start_runs_the_view_and_finish_stops_it_saying_where_the_recording_went(monkeypatch, tmp_path):
    script = tmp_path / "fake_view.py"
    script.write_text(FAKE_VIEW)
    monkeypatch.setattr(pregrasp, "_GROUPS_SCRIPT", script)
    monkeypatch.setattr(pregrasp, "P2P_PYTHON", sys.executable)
    monkeypatch.setattr(pregrasp, "P2P_REPO", str(tmp_path))
    monkeypatch.setattr(showservo, "live_camera", lambda: object())
    with _client(monkeypatch, ("127.0.0.1", _free_port())) as client:
        assert client.post("/api/pregrasp/groups/start").json() == {"status": "started"}
        assert client.post("/api/pregrasp/groups/start").status_code == 409
        st = {}
        for _ in range(200):  # the view answers once its server is up; the real one loads the tracker first
            st = client.get("/api/pregrasp/groups/status").json()
            if st["ready"] and st["frames"] > 0:
                break
            time.sleep(0.05)
        assert st["running"] and st["recording"] == "/recordings/groups_fake"
        done = client.post("/api/pregrasp/groups/stop").json()
        assert done["last"] == "/recordings/groups_fake" and done["frames"] >= st["frames"]
        assert done["recording"] is None
        st = client.get("/api/pregrasp/groups/status").json()
        assert (
            not st["running"] and st["last"] == "/recordings/groups_fake" and st["frames"] == done["frames"]
        )
        assert any(
            line.startswith("recorded ") for line in st["log"]
        )  # the view's own last word, SIGTERM honoured
        assert client.post("/api/pregrasp/groups/stop").status_code == 409


def test_without_the_camera_start_says_so(monkeypatch):
    monkeypatch.setattr(showservo, "live_camera", lambda: None)
    with _client(monkeypatch, ("127.0.0.1", _free_port())) as client:
        r = client.post("/api/pregrasp/groups/start")
        assert r.status_code == 409 and r.json()["detail"] == "start the camera first"
        st = client.get("/api/pregrasp/groups/status").json()
        assert st == {
            "running": False,
            "ready": False,
            "log": [],
            "with_acts": False,  # off in tests unless turned on (conftest.point_groups_off)
            "recording": None,
            "frames": 0,
            "last": None,
        }


@pytest.mark.parametrize("with_acts", [False, True])
def test_an_act_finishes_the_view_unless_it_keeps_running_with_acts(monkeypatch, tmp_path, with_acts):
    """Acts run with the point groups unless the switch is turned off (tests/gui/test_act_with_groups.py has what they
    carry): on, the view keeps running through the act; off, an act finishes it before the arm moves, its recording
    closed as Finish closes it."""
    script = tmp_path / "fake_view.py"
    script.write_text(FAKE_VIEW)
    monkeypatch.setattr(pregrasp, "_GROUPS_SCRIPT", script)
    monkeypatch.setattr(pregrasp, "P2P_PYTHON", sys.executable)
    monkeypatch.setattr(pregrasp, "P2P_REPO", str(tmp_path))
    monkeypatch.setattr(showservo, "live_camera", lambda: object())
    monkeypatch.setattr(pregrasp._state, "act", pregrasp._Act())
    monkeypatch.setattr(pregrasp._state, "demo", None)
    assert pregrasp._State().groups_with_acts is True, "on unless turned off"
    monkeypatch.setattr(pregrasp._state, "groups_with_acts", not with_acts)
    with _client(monkeypatch, ("127.0.0.1", _free_port())) as client:
        options = client.post("/api/pregrasp/options", json={"groups_with_acts": with_acts}).json()
        assert options["groups_with_acts"] is with_acts
        assert client.post("/api/pregrasp/groups/start").json() == {"status": "started"}
        for _ in range(200):
            if client.get("/api/pregrasp/groups/status").json()["ready"]:
                break
            time.sleep(0.05)
        asyncio.run(pregrasp._act_task(1.0))  # no demo loaded: the act ends before anything moves
        assert pregrasp._state.act.reason == "record or load a demo first"
        st = client.get("/api/pregrasp/groups/status").json()
        assert st["with_acts"] is with_acts
        assert st["running"] is with_acts
        if with_acts:
            client.post("/api/pregrasp/groups/stop")
        else:
            assert st["last"] == "/recordings/groups_fake" and st["frames"] > 0, "closed as Finish closes it"
