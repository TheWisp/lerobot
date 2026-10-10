"""The camera's frames shared in memory by the GUI server for the point groups' view: started, read by another
attachment, shared once however often it is asked, and gone when stopped."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.gui.api import pregrasp, showservo
from lerobot.showservo.frame_ring import FrameRing

H, W = 480, 848  # the rig's camera


class PublishingCamera:
    """A RealSense as its readers see it: the class's own publishing and reads, a thread publishing frame n at 30 fps
    with its depth 400 + n mm."""

    _publish = RealSenseCamera._publish
    read_color_and_aligned_depth = RealSenseCamera.read_color_and_aligned_depth
    read_newer_color_and_depth = RealSenseCamera.read_newer_color_and_depth

    def __init__(self):
        self.is_connected = self.use_depth = self.use_rgb = True
        self.post_grab_processor = None
        self.frame_lock = threading.Lock()
        self.frame_cond = threading.Condition(self.frame_lock)
        self.new_frame_event = threading.Event()
        self.latest_color_frame = self.latest_depth_frame = self.latest_timestamp = None
        self.n, self.stop = 0, threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            time.sleep(1 / 30)
            self.n += 1
            self._publish(
                np.full((H, W, 3), self.n % 251, np.uint8),
                np.full((H, W), 400 + self.n, np.uint16),
                time.perf_counter(),
            )

    def color_intrinsics(self):
        return {"fx": 604.2, "fy": 604.2, "cx": 419.2, "cy": 250.6, "width": W, "height": H}


def test_the_acts_reads_and_the_share_each_get_every_frame_and_wait_for_none_of_the_others(monkeypatch):
    """2026-10-10: the act's tracker took 3-5 frames a second while it needed 72-110 ms a frame. The frame share's read,
    always waiting for the next frame on the camera's one-thread executor, held each of the act's reads back a frame,
    and both cleared the camera's one new-frame event, so each waited for a frame the other had not taken. Each reader
    now waits only for a frame newer than its own last: the act's reads come one frame after another while the share
    copies every frame too."""
    got: list[int] = []

    class Ring:  # the frame numbers the share copies
        def write(self, rgb, depth, t):
            got.append(int(depth.flat[0]) - 400)

    import asyncio

    camera = PublishingCamera()
    monkeypatch.setattr(showservo, "live_camera", lambda: camera)
    share = pregrasp._FrameShare(ring=Ring())
    share.thread = threading.Thread(target=pregrasp._share_frames, args=(camera, share), daemon=True)

    async def act_reads(n):  # the act's frames, read as the act reads them
        return [int(round((await pregrasp._frame())[1].flat[0] * 1000)) - 400 for _ in range(n)]

    try:
        share.thread.start()
        asyncio.run(act_reads(1))
        first = camera.n
        seen = asyncio.run(act_reads(20))
        last = camera.n
        assert all(b > a for a, b in zip(seen, seen[1:], strict=False)), "a new frame each read, none twice"
        assert sum(b - a > 1 for a, b in zip(seen, seen[1:], strict=False)) <= 2, (
            f"one frame after another: {seen}"
        )
        shared = [k for k in got if first < k <= last]
        assert len(shared) >= 0.9 * (last - first), "the share copied every frame meanwhile"
        camera.stop.set()
        while camera.thread.is_alive():  # its last frame published; the read thread itself is checked alive
            time.sleep(0.01)
        camera.thread = threading.current_thread()
        with pytest.raises(TimeoutError):
            camera.read_newer_color_and_depth(camera.latest_timestamp, timeout_ms=100)
        assert camera.read_newer_color_and_depth(None)[2] == camera.latest_timestamp, (
            "None: the newest, at once"
        )
    finally:
        camera.stop.set()
        share.stop.set()
        share.thread.join(2)


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
    whole, as a JPEG of the camera's own picture, encoded on the act's frame executor: an encode on the event loop would
    stall the act's ticks at every poll."""
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
    assert len(where) == 1 and where[0].startswith("pregrasp-grab"), where
