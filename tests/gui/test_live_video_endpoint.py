"""The one endpoint a viewer uses, and what starts and stops the pipeline.

The pipeline runs only while someone is watching: the first offer starts it,
the last viewer's departure stops it. A run that is not streaming is not an
error the page has to handle as a failure — it is "nothing to watch yet".
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest
import pytest_asyncio
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import lerobot.robots.obs_stream as obs_stream
from lerobot.gui.api import live_video as api
from lerobot.gui.link_class import CLASS_LINK
from lerobot.gui.live_video.transport import CYCLE_CHANNEL, offer_h264_only

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _own_shm_names(monkeypatch):
    monkeypatch.setattr(obs_stream, "SHM_PREFIX", f"lerobot_obs_e{os.getpid()}_")


@pytest.fixture
def app():
    application = FastAPI()
    application.include_router(api.router)
    yield application


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c
    await api.shutdown()


@pytest.fixture
def tap():
    from tests.gui.test_live_video_pipeline import SyntheticTap

    t = SyntheticTap()
    t.start()
    deadline = time.time() + 5.0
    while t.cycles_written < 2 and time.time() < deadline:
        time.sleep(0.01)
    yield t
    t.stop()


async def _offer(client, viewer: RTCPeerConnection, cameras: int = 4, messages: list | None = None) -> dict:
    """What the page does: describe itself, post it, apply the answer.

    The page opens the cycle channel, because an answerer cannot add one the
    offer did not describe — which is also why the messages arrive on the
    channel this side created, not through a ``datachannel`` event.
    """
    channel = viewer.createDataChannel(CYCLE_CHANNEL)
    if messages is not None:

        @channel.on("message")
        def on_message(raw):
            messages.append(json.loads(raw))

    for _ in range(cameras):
        offer_h264_only(viewer.addTransceiver("video", direction="recvonly"))
    await viewer.setLocalDescription(await viewer.createOffer())
    response = await client.post(
        "/api/run/live-video/offer",
        json={"sdp": viewer.localDescription.sdp, "type": viewer.localDescription.type},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    await viewer.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type=body["type"]))
    return body


class TestWithoutARun:
    async def test_the_status_says_there_is_nothing_to_watch(self, client):
        body = (await client.get("/api/run/live-video/status")).json()
        assert body["available"] is False
        assert body["viewers"] == 0
        assert body["cameras"] == []

    async def test_an_offer_is_refused_with_a_reason_the_page_can_show(self, client):
        response = await client.post("/api/run/live-video/offer", json={"sdp": "v=0", "type": "offer"})
        assert response.status_code == 503
        assert "run" in response.json()["detail"].lower()


class TestWithARun:
    async def test_the_status_describes_the_stream_before_anyone_watches(self, tap, client):
        body = (await client.get("/api/run/live-video/status")).json()
        assert body["available"] is True
        assert sorted(body["cameras"]) == sorted(tap.stream.image_keys)
        assert body["viewers"] == 0
        assert body["profile"]["width"] == 320
        assert body["profile"]["bitrate_kbit_s"] == CLASS_LINK.per_camera_kbit_s(len(tap.stream.image_keys))
        assert body["link"]["name"] == CLASS_LINK.name

    async def test_an_offer_is_answered_and_the_viewer_receives_every_camera(self, tap, client):
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        frames: dict[str, list] = {}
        messages: list[dict] = []
        done = asyncio.Event()

        @viewer.on("track")
        def on_track(track):
            got: list = []
            frames[track.id] = got

            async def pull():
                try:
                    while len(got) < 6:
                        got.append(await track.recv())
                except MediaStreamError:
                    pass
                if len(frames) == len(tap.stream.image_keys) and all(len(f) >= 6 for f in frames.values()):
                    done.set()

            asyncio.ensure_future(pull())

        try:
            answer = await _offer(client, viewer, messages=messages)
            assert answer["type"] == "answer"
            assert "H264" in answer["sdp"]
            assert "VP8" not in answer["sdp"]
            await asyncio.wait_for(done.wait(), timeout=30.0)
        finally:
            await viewer.close()
        assert len(frames) == len(tap.stream.image_keys)
        assert messages and messages[0]["state"]["cycle"] > 0

    async def test_the_pipeline_runs_only_while_someone_is_watching(self, tap, client):
        assert api.pipeline_running() is False
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        started = asyncio.Event()

        @viewer.on("track")
        def on_track(track):
            async def pull():
                try:
                    await track.recv()
                except MediaStreamError:
                    pass
                started.set()

            asyncio.ensure_future(pull())

        try:
            await _offer(client, viewer)
            await asyncio.wait_for(started.wait(), timeout=30.0)
            assert api.pipeline_running() is True
            assert (await client.get("/api/run/live-video/status")).json()["viewers"] == 1
        finally:
            await viewer.close()
        # The server notices the viewer left and gives the GPU back.
        deadline = time.time() + 15.0
        while api.pipeline_running() and time.time() < deadline:
            await asyncio.sleep(0.2)
        assert api.pipeline_running() is False
        assert (await client.get("/api/run/live-video/status")).json()["viewers"] == 0

    async def test_two_viewers_share_one_pipeline_and_one_encoder_per_camera(self, tap, client):
        viewers = [
            RTCPeerConnection(RTCConfiguration(iceServers=[])),
            RTCPeerConnection(RTCConfiguration(iceServers=[])),
        ]
        seen = [asyncio.Event(), asyncio.Event()]

        def wire(viewer, event):
            @viewer.on("track")
            def on_track(track):
                async def pull():
                    try:
                        for _ in range(4):
                            await track.recv()
                    except MediaStreamError:
                        pass
                    event.set()

                asyncio.ensure_future(pull())

        try:
            with_one_viewer = None
            for viewer, event in zip(viewers, seen, strict=True):
                wire(viewer, event)
                await _offer(client, viewer)
                if with_one_viewer is None:
                    await asyncio.wait_for(seen[0].wait(), timeout=40.0)
                    with_one_viewer = (await client.get("/api/run/live-video/status")).json()["encoders"]
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in seen)), timeout=40.0)
            status = (await client.get("/api/run/live-video/status")).json()
            assert status["viewers"] == 2
            # Counted, not derived from the camera list, and compared against
            # the same count with one viewer: a second encode per viewer is
            # what this forbids, and a number computed from the cameras could
            # not have shown one.
            assert status["encoders"] == with_one_viewer, (status["encoders"], with_one_viewer)
            assert status["encoders"] == len(tap.stream.image_keys)
        finally:
            for viewer in viewers:
                await viewer.close()

    async def test_shutdown_closes_everything(self, tap, client):
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            await _offer(client, viewer)
            await api.shutdown()
            assert api.pipeline_running() is False
            assert (await client.get("/api/run/live-video/status")).json()["viewers"] == 0
        finally:
            await viewer.close()

    async def test_an_offer_that_asks_for_another_codec_is_refused_with_the_reason(self, tap, client):
        """Our frames are H.264 and the answering side cannot change that, so
        an offer led by VP8 is refused rather than answered with noise."""
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            viewer.createDataChannel(CYCLE_CHANNEL)
            for _ in range(len(tap.stream.image_keys)):
                viewer.addTransceiver("video", direction="recvonly")
            await viewer.setLocalDescription(await viewer.createOffer())
            response = await client.post(
                "/api/run/live-video/offer",
                json={"sdp": viewer.localDescription.sdp, "type": viewer.localDescription.type},
            )
            assert response.status_code == 400
            assert "H.264" in response.json()["detail"]
            assert (await client.get("/api/run/live-video/status")).json()["viewers"] == 0
            # Refusing is the whole answer: the pipeline that offer started has
            # no viewer to leave and stop it, so it is stopped here or it
            # encodes every camera at 30 fps until the server shuts down.
            assert not api.pipeline_running(), "a refused offer left the pipeline running"
        finally:
            await viewer.close()

    async def test_an_offer_with_the_wrong_number_of_streams_is_refused_with_the_count(self, tap, client):
        """An answer cannot add a video stream the offer did not describe, so
        the page is told how many to ask for rather than given a short one."""
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            viewer.createDataChannel(CYCLE_CHANNEL)
            offer_h264_only(viewer.addTransceiver("video", direction="recvonly"))
            await viewer.setLocalDescription(await viewer.createOffer())
            response = await client.post(
                "/api/run/live-video/offer",
                json={"sdp": viewer.localDescription.sdp, "type": viewer.localDescription.type},
            )
            assert response.status_code == 400
            assert str(len(tap.stream.image_keys)) in response.json()["detail"]
            assert (await client.get("/api/run/live-video/status")).json()["viewers"] == 0
            # Refusing is the whole answer: the pipeline that offer started has
            # no viewer to leave and stop it, so it is stopped here or it
            # encodes every camera at 30 fps until the server shuts down.
            assert not api.pipeline_running(), "a refused offer left the pipeline running"
        finally:
            await viewer.close()


class TestWhatThePageNeeds:
    """The two things a page cannot work out for itself: which track is
    which camera, and where the robot is."""

    async def test_the_answer_names_the_cameras_in_the_order_of_its_streams(self, tap, client):
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            answer = await _offer(client, viewer, cameras=len(tap.stream.image_keys))
            assert answer["cameras"] == list(tap.stream.image_keys)
        finally:
            await viewer.close()

    async def test_each_cycle_carries_the_pose_the_robot_tile_draws(self, tap, client):
        """The URDF tile is fed from the stream instead of its own poll, so
        the pose has to ride with the readouts."""
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        messages: list[dict] = []
        try:
            await _offer(client, viewer, cameras=len(tap.stream.image_keys), messages=messages)
            deadline = time.time() + 20.0
            while len(messages) < 5 and time.time() < deadline:
                await asyncio.sleep(0.05)
        finally:
            await viewer.close()
        assert len(messages) >= 5
        for m in messages:
            pose = m["pose"]
            assert pose["available"] is True
            assert pose["arms"], m
            joints = pose["arms"][0]["frames"][0]["joints"]
            assert joints, "the arm carries no joint angles"
        # The pose moves with the run rather than being stamped once.
        first = messages[0]["pose"]["arms"][0]["frames"][0]["joints"]
        last = messages[-1]["pose"]["arms"][0]["frames"][0]["joints"]
        assert first != last

    async def test_a_robot_the_viewer_cannot_draw_says_so_once_per_cycle(self, tap, client, monkeypatch):
        """No vendored URDF is not an error: the readouts still ride the
        stream and the tile stays out of the way."""
        from lerobot.gui.api import live_video as module

        monkeypatch.setattr(module, "_resolve_pose", lambda state: {"available": False})
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        messages: list[dict] = []
        try:
            await _offer(client, viewer, cameras=len(tap.stream.image_keys), messages=messages)
            deadline = time.time() + 20.0
            while len(messages) < 3 and time.time() < deadline:
                await asyncio.sleep(0.05)
        finally:
            await viewer.close()
        assert len(messages) >= 3
        assert all(m["pose"] == {"available": False} for m in messages)
        assert all(m["state"] for m in messages)

    async def test_a_two_armed_robot_sends_a_pose_for_each_arm(self):
        """The rig's robot is bimanual, and the tap that stands in for it in
        the rest of this file is not: a pose that carried one arm would draw
        half the robot and pass every test above."""
        from lerobot.gui.api.live_video import _resolve_pose
        from lerobot.robots.openarm_description import VIZ_SPEC

        state = {
            f"{side}{motor}.pos": float(i)
            for side in ("left_", "right_")
            for i, motor in enumerate(VIZ_SPEC["motors"])
        }
        pose = _resolve_pose(state)

        assert pose["available"] is True
        assert [arm["prefix"] for arm in pose["arms"]] == ["left_", "right_"]
        for arm in pose["arms"]:
            joints = arm["frames"][0]["joints"]
            assert set(joints) == set(VIZ_SPEC["urdf_joints"]), arm["prefix"]


class TestOpeningEarly:
    """The connection is opened before the run's frames exist, so the first
    picture costs a keyframe and one way across the link rather than a
    connection's whole setup.

    The early offer carries one placeholder video stream. Every later video
    stream shares the transport chosen for the first one, so a connection
    opened with the data channel alone can only ever carry one camera: the
    rest are answered, and stranded.
    """

    async def _open_early(self, client, viewer) -> str:
        viewer.createDataChannel(CYCLE_CHANNEL)
        offer_h264_only(viewer.addTransceiver("video", direction="recvonly"))
        await viewer.setLocalDescription(await viewer.createOffer())
        response = await client.post(
            "/api/run/live-video/offer",
            json={
                "sdp": viewer.localDescription.sdp,
                "type": viewer.localDescription.type,
                "attach": False,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        await viewer.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type=body["type"]))
        return body

    async def test_a_connection_can_be_opened_with_no_run_at_all(self, client):
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            body = await self._open_early(client, viewer)
            assert body["cameras"] == []
            assert body["session"]
            assert api.pipeline_running() is False
            deadline = time.time() + 10.0
            while viewer.connectionState != "connected" and time.time() < deadline:
                await asyncio.sleep(0.05)
            assert viewer.connectionState == "connected"
        finally:
            await viewer.close()
            await api.shutdown()

    async def test_every_camera_delivers_on_a_connection_opened_early(self, tap, client):
        """The whole point: the same connection, one more offer, and every
        camera arrives on it — not only the first."""
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        frames: dict[str, list] = {}
        done = asyncio.Event()

        @viewer.on("track")
        def on_track(track):
            got: list = []
            frames[track.id] = got

            async def pull():
                try:
                    while len(got) < 4:
                        got.append(await track.recv())
                except MediaStreamError:
                    pass
                if len(frames) == len(tap.stream.image_keys) and all(len(f) >= 4 for f in frames.values()):
                    done.set()

            asyncio.ensure_future(pull())

        try:
            first = await self._open_early(client, viewer)
            session = first["session"]

            # The placeholder becomes the first camera; the rest are added.
            for _ in range(len(tap.stream.image_keys) - 1):
                offer_h264_only(viewer.addTransceiver("video", direction="recvonly"))
            await viewer.setLocalDescription(await viewer.createOffer())
            second = await client.post(
                "/api/run/live-video/offer",
                json={
                    "sdp": viewer.localDescription.sdp,
                    "type": viewer.localDescription.type,
                    "session": session,
                },
            )
            assert second.status_code == 200, second.text
            body = second.json()
            assert body["session"] == session
            assert body["cameras"] == list(tap.stream.image_keys)
            await viewer.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type=body["type"]))
            await asyncio.wait_for(done.wait(), timeout=30.0)
        finally:
            await viewer.close()
        assert len(frames) == len(tap.stream.image_keys)
        assert all(len(f) >= 4 for f in frames.values()), {k: len(v) for k, v in frames.items()}

    async def test_the_channel_opened_early_carries_the_run_when_it_starts(self, tap, client):
        """The readouts arrive on the channel from the first round, without a
        second one being needed to open it."""
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        messages: list[dict] = []
        try:
            channel = viewer.createDataChannel(CYCLE_CHANNEL)

            @channel.on("message")
            def on_message(raw):
                messages.append(json.loads(raw))

            offer_h264_only(viewer.addTransceiver("video", direction="recvonly"))
            await viewer.setLocalDescription(await viewer.createOffer())
            first = (
                await client.post(
                    "/api/run/live-video/offer",
                    json={
                        "sdp": viewer.localDescription.sdp,
                        "type": viewer.localDescription.type,
                        "attach": False,
                    },
                )
            ).json()
            await viewer.setRemoteDescription(RTCSessionDescription(sdp=first["sdp"], type=first["type"]))
            await asyncio.sleep(0.5)
            assert not messages, "nothing is watching yet, so nothing should be sent"

            for _ in range(len(tap.stream.image_keys) - 1):
                offer_h264_only(viewer.addTransceiver("video", direction="recvonly"))
            await viewer.setLocalDescription(await viewer.createOffer())
            second = (
                await client.post(
                    "/api/run/live-video/offer",
                    json={
                        "sdp": viewer.localDescription.sdp,
                        "type": viewer.localDescription.type,
                        "session": first["session"],
                    },
                )
            ).json()
            await viewer.setRemoteDescription(RTCSessionDescription(sdp=second["sdp"], type=second["type"]))
            deadline = time.time() + 20.0
            while len(messages) < 5 and time.time() < deadline:
                await asyncio.sleep(0.05)
        finally:
            await viewer.close()
        assert len(messages) >= 5
        assert [m["cycle"] for m in messages] == sorted(m["cycle"] for m in messages)

    async def test_a_session_nobody_opened_is_refused(self, tap, client):
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            viewer.createDataChannel(CYCLE_CHANNEL)
            for _ in range(len(tap.stream.image_keys)):
                offer_h264_only(viewer.addTransceiver("video", direction="recvonly"))
            await viewer.setLocalDescription(await viewer.createOffer())
            response = await client.post(
                "/api/run/live-video/offer",
                json={
                    "sdp": viewer.localDescription.sdp,
                    "type": viewer.localDescription.type,
                    "session": "not-a-session",
                },
            )
            assert response.status_code == 404
            assert "session" in response.json()["detail"].lower()
        finally:
            await viewer.close()

    async def test_a_connection_opened_early_and_left_open_holds_no_pipeline(self, tap, client):
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            await self._open_early(client, viewer)
            await asyncio.sleep(0.5)
            assert api.pipeline_running() is False
            assert (await client.get("/api/run/live-video/status")).json()["viewers"] == 0
        finally:
            await viewer.close()

    async def test_an_offer_that_is_not_attaching_carries_one_placeholder(self, client):
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            viewer.createDataChannel(CYCLE_CHANNEL)
            for _ in range(3):
                offer_h264_only(viewer.addTransceiver("video", direction="recvonly"))
            await viewer.setLocalDescription(await viewer.createOffer())
            response = await client.post(
                "/api/run/live-video/offer",
                json={
                    "sdp": viewer.localDescription.sdp,
                    "type": viewer.localDescription.type,
                    "attach": False,
                },
            )
            assert response.status_code == 400
            assert "placeholder" in response.json()["detail"]
        finally:
            await viewer.close()

    async def test_the_answer_says_what_profile_the_tiles_are_at(self, tap, client):
        """The page shows the profile beside the stream's state; it comes
        with the answer rather than costing a second request."""
        viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        try:
            body = await self._open_early(client, viewer)
            assert body["profile"]["width"] == 320
            assert body["link"]["name"] == CLASS_LINK.name
            for _ in range(len(tap.stream.image_keys) - 1):
                offer_h264_only(viewer.addTransceiver("video", direction="recvonly"))
            await viewer.setLocalDescription(await viewer.createOffer())
            second = (
                await client.post(
                    "/api/run/live-video/offer",
                    json={
                        "sdp": viewer.localDescription.sdp,
                        "type": viewer.localDescription.type,
                        "session": body["session"],
                    },
                )
            ).json()
            assert second["profile"]["bitrate_kbit_s"] == CLASS_LINK.per_camera_kbit_s(
                len(tap.stream.image_keys)
            )
        finally:
            await viewer.close()
