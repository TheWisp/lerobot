"""The transport: the pipeline's encoded frames over WebRTC, unmodified.

The loopback tests run two peer connections in one process over the host's
own interface, so what they prove is the library's path and our use of it,
not the network.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from fractions import Fraction

import av
import numpy as np
import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.codecs import get_encoder
from aiortc.mediastreams import MediaStreamError

import lerobot.robots.obs_stream as obs_stream
from lerobot.gui.live_video.pipeline import EncodedSample, LivePipeline
from lerobot.gui.live_video.transport import (
    VIDEO_CLOCK_RATE,
    WRAP,
    EncodedTrack,
    LiveVideoSession,
    capture_ts_from_rtp,
    learn_origin,
    offer_h264_only,
    rtp_from_capture_ts,
    track_origin,
    video_codecs_chosen,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _own_shm_names(monkeypatch):
    monkeypatch.setattr(obs_stream, "SHM_PREFIX", f"lerobot_obs_x{os.getpid()}_")


class _FakeSubscription:
    """A subscription fed by hand, so a track can be tested without a tap."""

    def __init__(self, cameras: list[str]) -> None:
        self.cameras = cameras
        self._queues: dict[str, list[EncodedSample]] = {c: [] for c in cameras}
        self.messages: list = []
        self.closed = False

    def feed(self, sample: EncodedSample) -> None:
        self._queues[sample.camera].append(sample)

    def take_video(self, camera: str, timeout: float | None = None) -> EncodedSample | None:
        q = self._queues[camera]
        return q.pop(0) if q else None

    def take_message(self, timeout: float | None = None):
        return self.messages.pop(0) if self.messages else None

    def dropped(self, camera: str) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


def _sample(camera: str, cycle: int, capture_ts: float, data: bytes = b"\x00\x00\x00\x01x") -> EncodedSample:
    return EncodedSample(
        camera=camera,
        data=data,
        keyframe=cycle == 1,
        cycle=cycle,
        capture_ts=capture_ts,
        capture_source=obs_stream.CaptureSource.OBSERVATION_READ,
        overlay_cycle=None,
        encoded_ts=capture_ts + 0.003,
    )


class TestStamp:
    """The capture time travels as the frame's timestamp in the transport's
    own 90 kHz clock, which wraps every thirteen hours."""

    @pytest.mark.parametrize("t", [0.0, 1.0, 1789344182.123456, 2**32 / VIDEO_CLOCK_RATE + 5.0])
    async def test_a_capture_time_survives_the_round_trip(self, t):
        rtp = rtp_from_capture_ts(t)
        assert 0 <= rtp < WRAP
        back = capture_ts_from_rtp(rtp, near=t)
        assert abs(back - t) <= 1.0 / VIDEO_CLOCK_RATE

    async def test_the_clock_wraps_and_the_nearest_reading_is_taken(self):
        t = 1789344182.0
        near_wrap = (WRAP - 100) / VIDEO_CLOCK_RATE  # a capture time just before a wrap
        base = t - (t % (WRAP / VIDEO_CLOCK_RATE)) + near_wrap
        for dt in (-0.5, 0.0, 0.5):
            moment = base + dt
            assert abs(capture_ts_from_rtp(rtp_from_capture_ts(moment), near=base) - moment) < 1e-3

    async def test_the_origin_is_learned_from_readings_and_announced_capture_times(self):
        """What the page does: it cannot tell which reading belongs to which
        cycle, so it takes the constant that explains the most of them."""
        t = 1789344182.0
        origin = 123456789
        noise = random.Random(7)
        captures = [t + i / 30 + noise.random() * 0.004 for i in range(40)]
        wire = [(rtp_from_capture_ts(c) + origin) % WRAP for c in captures[5:35]]
        assert learn_origin(wire, captures) == origin
        # A stream shifted by a constant is another stream with another
        # constant, and is read as such: what must be refused is readings no
        # constant explains.
        rng = random.Random(0)
        assert learn_origin([rng.randrange(WRAP) for _ in wire], captures) is None
        assert learn_origin([], captures) is None
        assert learn_origin(wire, []) is None

    async def test_a_perfectly_regular_cadence_has_no_answer_rather_than_a_wrong_one(self):
        """On an exact grid a constant one period out explains every reading
        just as well, so the age would be one period wrong on every frame and
        would read as a plausible number."""
        t = 1789344182.0
        captures = [t + i / 30 for i in range(60)]
        wire = [(rtp_from_capture_ts(c) + 4242) % WRAP for c in captures[5:40]]
        assert learn_origin(wire, captures) is None

    async def test_readings_from_cycles_nobody_announced_are_refused(self):
        """The capture times come over the data channel; if the frames on
        screen belong to cycles whose messages never arrived, the page says
        it does not know rather than inventing a constant."""
        t = 1789344182.0
        announced = [t + i / 30 for i in range(20)]
        noise = random.Random(5)
        elsewhere = [t + 100.0 + i / 30 + noise.random() * 0.004 for i in range(20)]
        wire = [(rtp_from_capture_ts(c) + 999) % WRAP for c in elsewhere]
        assert learn_origin(wire, announced) is None

    async def test_a_few_readings_are_not_enough_to_claim_an_origin(self):
        t = 1789344182.0
        captures = [t + i / 30 for i in range(40)]
        assert learn_origin([(rtp_from_capture_ts(captures[0]) + 42) % WRAP], captures) is None

    async def test_the_origin_is_what_turns_a_wire_reading_into_a_capture_time(self):
        """aiortc adds a random origin to every timestamp it sends, so the
        page learns the offset once per track and applies it to the rest."""
        t = 1789344182.0
        origin = 123456789
        wire = [(rtp_from_capture_ts(t + i / 30) + origin) % WRAP for i in range(5)]
        learned = track_origin(wire_rtp=wire[0], capture_ts=t)
        assert learned == origin
        for i, w in enumerate(wire):
            recovered = capture_ts_from_rtp((w - learned) % WRAP, near=t)
            assert abs(recovered - (t + i / 30)) < 1e-3


class TestTrack:
    async def test_it_hands_over_the_encoded_bytes_with_the_capture_time(self):
        sub = _FakeSubscription(["top"])
        t = time.time()
        sub.feed(_sample("top", 1, t, b"\x00\x00\x00\x01\x65abc"))
        track = EncodedTrack(sub, "top")
        try:
            packet = await track.recv()
            assert isinstance(packet, av.Packet)
            assert bytes(packet) == b"\x00\x00\x00\x01\x65abc"
            assert packet.pts == rtp_from_capture_ts(t)
            assert packet.time_base == Fraction(1, VIDEO_CLOCK_RATE)
        finally:
            track.stop()

    async def test_it_is_a_video_track_aiortc_will_take(self):
        sub = _FakeSubscription(["top"])
        track = EncodedTrack(sub, "top")
        try:
            assert track.kind == "video"
        finally:
            track.stop()

    async def test_it_ends_when_the_viewer_goes_away(self):
        sub = _FakeSubscription(["top"])
        track = EncodedTrack(sub, "top")
        track.stop()
        with pytest.raises(MediaStreamError):
            await asyncio.wait_for(track.recv(), timeout=5.0)

    async def test_the_library_packetizes_our_packet_under_our_timestamp(self):
        """The pre-encoded path: aiortc splits what we hand it and keeps our
        timestamp, rather than encoding anything of its own."""
        sub = _FakeSubscription(["top"])
        t = time.time()
        au = b"\x00\x00\x00\x01\x65" + bytes(range(256)) * 12  # more than one packet's worth
        sub.feed(_sample("top", 1, t, au))
        track = EncodedTrack(sub, "top")
        try:
            packet = await track.recv()
        finally:
            track.stop()
        encoder = get_encoder(
            type("C", (), {"name": "H264", "clockRate": VIDEO_CLOCK_RATE, "mimeType": "video/H264"})()
        )
        payloads, timestamp = encoder.pack(packet)
        assert timestamp == rtp_from_capture_ts(t)
        assert len(payloads) > 1
        assert sum(len(p) for p in payloads) >= len(au) - 8
        assert all(len(p) <= 1300 for p in payloads)


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


@pytest.fixture
def pipeline(tap):
    p = LivePipeline(fps=30)
    p.start()
    yield p
    p.stop()


def _add_h264_track(connection: RTCPeerConnection, track) -> None:
    sender = connection.addTrack(track)
    offer_h264_only(next(t for t in connection.getTransceivers() if t.sender is sender))


async def _connect(offerer: RTCPeerConnection, answerer: RTCPeerConnection) -> None:
    await offerer.setLocalDescription(await offerer.createOffer())
    await answerer.setRemoteDescription(
        RTCSessionDescription(sdp=offerer.localDescription.sdp, type=offerer.localDescription.type)
    )
    await answerer.setLocalDescription(await answerer.createAnswer())
    await offerer.setRemoteDescription(
        RTCSessionDescription(sdp=answerer.localDescription.sdp, type=answerer.localDescription.type)
    )


class TestLoopback:
    """A real peer connection between two objects in this process."""

    async def test_a_viewer_receives_every_camera_and_the_cycle_messages(self, tap, pipeline):
        server = RTCPeerConnection()
        viewer = RTCPeerConnection()
        received: dict[str, list] = {}
        messages: list[dict] = []
        done = asyncio.Event()

        @viewer.on("track")
        def on_track(track):
            frames: list = []
            received[track.id] = frames

            async def pull():
                try:
                    while len(frames) < 12:
                        frames.append(await track.recv())
                except MediaStreamError:
                    pass
                if len(received) == len(pipeline.cameras) and all(len(f) >= 12 for f in received.values()):
                    done.set()

            asyncio.ensure_future(pull())

        @viewer.on("datachannel")
        def on_channel(channel):
            @channel.on("message")
            def on_message(raw):
                messages.append(json.loads(raw))

        session = LiveVideoSession(pipeline, server)
        try:
            await _connect(server, viewer)
            await asyncio.wait_for(done.wait(), timeout=30.0)
        finally:
            await session.close()
            await viewer.close()
            await server.close()

        assert len(received) == len(pipeline.cameras)
        for frames in received.values():
            assert len(frames) >= 12
            assert all(isinstance(f, av.VideoFrame) for f in frames)
            assert frames[0].width == 320
            # Every picture decoded: the stream the page gets is playable.
            pictures = [f.to_ndarray(format="rgb24") for f in frames]
            assert all(p.std() > 1.0 for p in pictures), "a decoded picture was flat"
        assert len(messages) >= 10
        cycles = [m["cycle"] for m in messages]
        assert cycles == sorted(cycles)
        assert all("state" in m and "capture_ts" in m for m in messages)
        assert messages[-1]["state"]["cycle"] == float(messages[-1]["cycle"])

    async def test_one_offset_per_track_turns_wire_readings_into_capture_times(self, tap, pipeline):
        """What the page needs for R1's age: the offset is learned once from
        the first frame and holds for the rest."""
        server = RTCPeerConnection()
        viewer = RTCPeerConnection()
        camera = pipeline.cameras[0]
        frames: list = []
        got = asyncio.Event()

        @viewer.on("track")
        def on_track(track):
            async def pull():
                try:
                    while len(frames) < 20:
                        frames.append(await track.recv())
                except MediaStreamError:
                    pass
                got.set()

            asyncio.ensure_future(pull())

        sub = pipeline.subscribe()
        sent: list[EncodedSample] = []
        track = EncodedTrack(sub, camera, on_sent=sent.append)
        _add_h264_track(server, track)
        try:
            await _connect(server, viewer)
            await asyncio.wait_for(got.wait(), timeout=30.0)
        finally:
            track.stop()
            sub.close()
            await viewer.close()
            await server.close()

        assert len(frames) >= 20
        # The receiver's first picture is the sender's first frame, so one
        # pair gives the offset; every later frame must agree with it.
        origin = track_origin(wire_rtp=frames[0].pts, capture_ts=sent[0].capture_ts)
        for frame, sample in zip(frames, sent, strict=False):
            recovered = capture_ts_from_rtp((frame.pts - origin) % WRAP, near=sample.capture_ts)
            assert abs(recovered - sample.capture_ts) < 1e-3, (recovered, sample.capture_ts)

    async def test_the_viewer_sees_the_pictures_the_server_encoded(self, tap, pipeline):
        """What the viewer decodes equals a local decode of what was sent,
        picture for picture: the library carried the bytes and changed
        nothing on the way."""
        server = RTCPeerConnection()
        viewer = RTCPeerConnection()
        camera = pipeline.cameras[0]
        decoded: list[np.ndarray] = []
        enough = asyncio.Event()

        @viewer.on("track")
        def on_track(track):
            async def pull():
                try:
                    while len(decoded) < 15:
                        decoded.append((await track.recv()).to_ndarray(format="rgb24"))
                except MediaStreamError:
                    pass
                enough.set()

            asyncio.ensure_future(pull())

        sub = pipeline.subscribe()
        sent: list[EncodedSample] = []
        track = EncodedTrack(sub, camera, on_sent=sent.append)
        _add_h264_track(server, track)
        try:
            await _connect(server, viewer)
            await asyncio.wait_for(enough.wait(), timeout=30.0)
        finally:
            track.stop()
            sub.close()
            await viewer.close()
            await server.close()

        assert len(decoded) >= 10
        codec = av.CodecContext.create("h264", "r")
        local: list[np.ndarray] = []
        for s in sent:
            local += [f.to_ndarray(format="rgb24") for f in codec.decode(av.Packet(s.data))]
        assert len(local) >= len(decoded)
        for i, (mine, theirs) in enumerate(zip(local, decoded, strict=False)):
            assert mine.shape == theirs.shape
            assert np.abs(mine.astype(int) - theirs.astype(int)).max() == 0, i

    async def test_the_connection_settles_on_the_codec_our_frames_are_in(self, tap, pipeline):
        """Left to itself the library offers VP8 first and the viewer decodes
        our H.264 access units as VP8, which yields nothing."""
        server = RTCPeerConnection()
        viewer = RTCPeerConnection()
        session = LiveVideoSession(pipeline, server)
        try:
            await _connect(server, viewer)
            assert "H264" in server.localDescription.sdp
            assert "VP8" not in server.localDescription.sdp
            assert "H264" in viewer.localDescription.sdp
            assert "VP8" not in viewer.localDescription.sdp
        finally:
            await session.close()
            await viewer.close()
            await server.close()


class TestChosenCodec:
    """What `video_codecs_chosen` reads is what the sender will use: the
    first payload type on each video line."""

    SDP = (
        "v=0\r\n"
        "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
        "a=rtpmap:111 opus/48000/2\r\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 99 97\r\n"
        "a=rtpmap:99 H264/90000\r\n"
        "a=rtpmap:97 VP8/90000\r\n"
        "m=video 9 UDP/TLS/RTP/SAVPF 97 99\r\n"
        "a=rtpmap:97 VP8/90000\r\n"
        "a=rtpmap:99 H264/90000\r\n"
        "m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n"
    )

    async def test_it_reads_one_codec_per_video_line_in_order(self):
        assert video_codecs_chosen(self.SDP) == ["H264", "VP8"]

    async def test_a_description_with_no_video_has_none(self):
        assert video_codecs_chosen("v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n") == []

    async def test_an_offer_we_made_asks_for_h264_alone(self):
        connection = RTCPeerConnection()
        try:
            offer_h264_only(connection.addTransceiver("video", direction="recvonly"))
            await connection.setLocalDescription(await connection.createOffer())
            assert video_codecs_chosen(connection.localDescription.sdp) == ["H264"]
        finally:
            await connection.close()

    async def test_an_offer_left_to_the_library_does_not(self):
        connection = RTCPeerConnection()
        try:
            connection.addTransceiver("video", direction="recvonly")
            await connection.setLocalDescription(await connection.createOffer())
            assert video_codecs_chosen(connection.localDescription.sdp) == ["VP8"]
        finally:
            await connection.close()


class TestPacketSize:
    """A video packet has to fit the smallest path we serve, which is a
    tunnel at 1280 bytes and not an ethernet frame."""

    async def test_the_library_is_capped_to_what_a_tunnel_carries(self):
        from aiortc.codecs import h264

        from lerobot.gui.live_video.transport import RTP_PAYLOAD_MAX

        assert h264.PACKET_MAX <= RTP_PAYLOAD_MAX
        # A packet plus RTP, UDP and IP headers, and room for the MID and
        # abs-send-time extensions, must still fit 1280.
        assert RTP_PAYLOAD_MAX + 12 + 8 + 20 + 20 <= 1280

    async def test_no_packet_of_a_real_frame_exceeds_it(self):
        """A keyframe is the case that matters: it is the one frame big
        enough to be cut into full-size packets."""
        from lerobot.gui.live_video.transport import RTP_PAYLOAD_MAX

        sub = _FakeSubscription(["top"])
        t = time.time()
        au = b"\x00\x00\x00\x01\x65" + bytes(range(256)) * 200  # ~50 kB, a keyframe's size
        sub.feed(_sample("top", 1, t, au))
        track = EncodedTrack(sub, "top")
        try:
            packet = await track.recv()
        finally:
            track.stop()
        encoder = get_encoder(
            type("C", (), {"name": "H264", "clockRate": VIDEO_CLOCK_RATE, "mimeType": "video/H264"})()
        )
        payloads, _ = encoder.pack(packet)
        assert len(payloads) > 30, "this should have been cut into many packets"
        assert max(len(p) for p in payloads) <= RTP_PAYLOAD_MAX


@pytest.mark.asyncio
async def test_a_viewer_whose_decoder_is_lost_gets_a_keyframe(pipeline):
    """On a link that drops packets the viewer's decoder cannot resume until
    the next keyframe, which is a second away. The browser already asks — a
    Picture Loss Indication is standard and automatic — and aiortc turns it
    into a call on the sender. That call reached nothing here, because the
    flag it sets is read only by the encode path this track replaces.
    """
    server = RTCPeerConnection()
    session = LiveVideoSession(pipeline, server)
    try:
        camera = pipeline.cameras[0]
        broadcaster = pipeline._broadcaster
        # A new subscription already wants one; take it, so what is measured
        # below is the viewer's request and not the join's.
        broadcaster.consume_keyframe_request(camera)
        assert not broadcaster.consume_keyframe_request(camera)

        sender = next(s for s in server.getSenders() if getattr(s.track, "_camera", None) == camera)
        sender._send_keyframe()  # what aiortc calls when a PLI arrives

        assert broadcaster.consume_keyframe_request(camera), (
            "the viewer asked for a keyframe and the encoder was never told"
        )
        assert not broadcaster.consume_keyframe_request(camera), "asked for twice"
    finally:
        await session.close()
        await server.close()
