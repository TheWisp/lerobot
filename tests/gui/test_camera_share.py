"""The camera's frames shared in memory by the GUI server for the point groups' view: started, read by another
attachment, shared once however often it is asked, and gone when stopped."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import pregrasp, showservo
from lerobot.showservo.frame_ring import FrameRing

H, W = 480, 848  # the rig's camera


class FakeCamera:
    """A RealSense as the share thread uses it: intrinsics, and a read that waits for the next frame (30 fps)."""

    def __init__(self):
        self.n = 0

    def color_intrinsics(self):
        return {"fx": 604.2, "fy": 604.2, "cx": 419.2, "cy": 250.6, "width": W, "height": H}

    def read_color_and_aligned_depth(self):
        time.sleep(1 / 30)
        self.n += 1
        return np.full((H, W, 3), self.n % 251, np.uint8), np.full((H, W), 400 + self.n, np.uint16)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(showservo, "live_camera", lambda: camera)
    monkeypatch.setattr(pregrasp._state, "camera_share", None)
    camera = FakeCamera()
    app = FastAPI()
    app.include_router(pregrasp.router)
    with TestClient(app) as c:
        yield c
    share = pregrasp._state.camera_share
    if share is not None:
        share.stop.set()
        share.thread.join(5)
        share.ring.close()


def test_frames_are_shared_once_read_whole_elsewhere_and_gone_when_stopped(client):
    first = client.post("/api/pregrasp/camera/share/start").json()
    assert (first["height"], first["width"]) == (H, W)
    ring = FrameRing(first["name"])
    assert np.allclose(ring.k, [[604.2, 0, 419.2], [0, 604.2, 250.6], [0, 0, 1]])
    got, deadline = None, time.time() + 5
    while got is None and time.time() < deadline:
        got = ring.read()
        time.sleep(0.01)
    assert got is not None
    n, t, rgb, depth = got
    assert abs(time.time() - t) < 1.0 and (depth == depth.flat[0]).all() and (rgb == rgb.flat[0]).all()
    assert client.post("/api/pregrasp/camera/share/start").json()["name"] == first["name"], "one ring, shared"
    ring.close()
    stopped = client.post("/api/pregrasp/camera/share/stop").json()
    assert stopped["frames"] > 0 and stopped["error"] is None
    with pytest.raises(FileNotFoundError):
        FrameRing(first["name"])
    assert client.post("/api/pregrasp/camera/share/stop").status_code == 409


def test_without_the_camera_sharing_says_so(client, monkeypatch):
    monkeypatch.setattr(showservo, "live_camera", lambda: None)
    r = client.post("/api/pregrasp/camera/share/start")
    assert r.status_code == 409 and r.json()["detail"] == "start the camera first"


def test_the_camera_views_frame_is_read_and_encoded_off_the_event_loop(client, monkeypatch):
    """The camera view polls frame.jpg for as long as the camera is live and nothing is tracked. The frame comes back
    whole, as a JPEG of the camera's own picture, encoded on the camera's executor: an encode on the event loop would
    stall the act's ticks at every poll."""
    import threading

    import cv2

    where: list[str] = []
    jpeg = pregrasp._jpeg
    monkeypatch.setattr(
        pregrasp, "_jpeg", lambda bgr: (where.append(threading.current_thread().name), jpeg(bgr))[1]
    )
    r = client.get("/api/pregrasp/frame.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (H, W, 3)
    assert len(where) == 1 and where[0].startswith("showservo"), where
