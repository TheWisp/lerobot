"""The pipeline's encoded frames over WebRTC, unmodified.

One video track per camera, each handing aiortc access units our own encoder
produced, and one data channel carrying the cycle's state and action. aiortc
splits an access unit into RTP packets and never encodes or decodes here.

The frame's capture time travels as its timestamp in the transport's 90 kHz
clock. aiortc adds a random origin to every timestamp it sends, so a wire
reading is a capture time plus a constant the page does not know: the page
learns that constant once per track, from its first picture, and applies it
to the rest. The cycle's own capture time also rides the data channel, where
no origin is added.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from fractions import Fraction

import av
from aiortc import RTCPeerConnection, RTCRtpSender
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from lerobot.gui.live_video.pipeline import CycleMessage, EncodedSample, LivePipeline, Subscription

logger = logging.getLogger(__name__)

#: RTP's clock for video, in ticks per second.
VIDEO_CLOCK_RATE = 90000

#: The timestamp is 32 bits, so the clock wraps about every thirteen hours.
WRAP = 2**32

_TIME_BASE = Fraction(1, VIDEO_CLOCK_RATE)

#: How long a track waits for its camera's next frame before checking whether
#: the viewer is still there.
_TAKE_TIMEOUT = 0.5

#: The data channel's label, which both sides agree on.
CYCLE_CHANNEL = "cycles"

#: The most a video packet may carry. The library's own default is 1300,
#: which is right for an ethernet path and wrong for a tunnelled one: a
#: WireGuard interface carries 1280 bytes, and a 1300-byte payload plus its
#: RTP, UDP and IP headers is 1340. The sending kernel does not fragment it,
#: it refuses it — "message too long" — so every full-size packet is lost
#: before it leaves, and the receiver reads the gap in sequence numbers as
#: loss on the wire. 1200 leaves room for the headers and the two RTP
#: extensions on the smallest path we serve ([O21]).
RTP_PAYLOAD_MAX = 1200


def _fit_packets_to_the_smallest_path() -> None:
    """Cap the library's packet size, which it exposes only as a global."""
    from aiortc.codecs import h264

    h264.PACKET_MAX = min(h264.PACKET_MAX, RTP_PAYLOAD_MAX)


_fit_packets_to_the_smallest_path()


def rtp_from_capture_ts(capture_ts: float) -> int:
    """A capture time as a timestamp in the transport's clock."""
    return round(capture_ts * VIDEO_CLOCK_RATE) % WRAP


def capture_ts_from_rtp(rtp: int, near: float) -> float:
    """The capture time a timestamp stands for, on the wrap nearest ``near``.

    Precondition: ``near`` is within half a wrap — about six and a half hours
    — of the true capture time, which any reading from the same session
    satisfies. The result is always the nearest of the three candidate wraps,
    so a reference outside that window silently returns the wrong one.
    """
    span = WRAP / VIDEO_CLOCK_RATE
    base = near - (near % span)
    candidates = [base + rtp / VIDEO_CLOCK_RATE + k * span for k in (-1, 0, 1)]
    return min(candidates, key=lambda t: abs(t - near))


def track_origin(wire_rtp: int, capture_ts: float) -> int:
    """The constant aiortc added, from one frame whose capture time is known."""
    return (wire_rtp - rtp_from_capture_ts(capture_ts)) % WRAP


def learn_origin(wire_rtps: list[int], capture_times: list[float]) -> int | None:
    """The constant a track's timestamps carry, from readings and capture times.

    A frame's timestamp is its cycle's capture time exactly, and the data
    channel names every cycle's capture time, so the right constant turns
    every reading into a time that was announced. Which reading belongs to
    which cycle is unknown, so each pairing is tried and the one that
    explains the most readings wins. A wrong pairing explains almost none:
    capture times are irregular at the clock's resolution.

    Returns None when no pairing explains more than a couple of readings —
    too few frames yet, or nothing in common — and also when two pairings
    explain equally many, which is what a perfectly regular cadence looks
    like: a constant one period out fits just as well, so there is no answer
    to give rather than one that would read as an age.
    """
    announced = {rtp_from_capture_ts(t) for t in capture_times}
    if not announced or not wire_rtps:
        return None
    best_origin, best_votes, tied = None, 0, False
    for wire in wire_rtps[:8]:
        for stamp in announced:
            candidate = (wire - stamp) % WRAP
            if candidate == best_origin:
                continue
            votes = sum(1 for w in wire_rtps if (w - candidate) % WRAP in announced)
            if votes > best_votes:
                best_origin, best_votes, tied = candidate, votes, False
            elif votes == best_votes:
                tied = True
    if tied:
        return None
    return best_origin if best_votes >= max(3, len(wire_rtps) // 4) else None


def offer_h264_only(transceiver) -> None:
    """Offer H.264 alone on a video transceiver.

    Our frames are already H.264, so a connection that settled on anything
    else would hand the viewer's decoder bytes it cannot read. Left to
    itself, the library offers its own preference order and VP8 wins. This
    only has an effect on the side that makes the offer: an answerer's codec
    list is settled when the offer is applied, before it can say anything.
    """
    codecs = [
        c
        for c in RTCRtpSender.getCapabilities("video").codecs
        if c.mimeType.lower() in ("video/h264", "video/rtx")
    ]
    assert any(c.mimeType.lower() == "video/h264" for c in codecs), "no H.264 in this build of aiortc"
    transceiver.setCodecPreferences(codecs)


def video_codecs_chosen(sdp: str) -> list[str]:
    """The codec each video stream in a description would carry, in order.

    The first payload type on a video ``m=`` line is what the sender uses,
    so this says what the other side will try to decode.
    """
    chosen: list[str] = []
    names: dict[str, str] = {}
    first_payload: list[str] = []
    for line in sdp.splitlines():
        if line.startswith("m=video "):
            parts = line.split()
            first_payload.append(parts[3] if len(parts) > 3 else "")
        elif line.startswith("a=rtpmap:"):
            payload, _, description = line[len("a=rtpmap:") :].partition(" ")
            names[payload] = description.split("/")[0]
    for payload in first_payload:
        chosen.append(names.get(payload, payload))
    return chosen


class EncodedTrack(MediaStreamTrack):
    """One camera's encoded frames as a video track.

    Each access unit is handed over whole, with the capture time as its
    timestamp; aiortc packetizes it. Ends when the viewer's subscription
    closes or the track is stopped.
    """

    kind = "video"

    def __init__(
        self,
        subscription: Subscription,
        camera: str,
        on_sent: Callable[[EncodedSample], None] | None = None,
    ) -> None:
        super().__init__()
        self._sub = subscription
        self._camera = camera
        #: Which camera this track carries, for whoever wires the sender up.
        self.camera = camera
        self._on_sent = on_sent
        # A named pool per track: taking the next frame blocks, and the GUI's
        # shared executor must not be where it waits.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"live-track-{camera}")

    async def recv(self) -> av.Packet:
        loop = asyncio.get_running_loop()
        while True:
            if self.readyState != "live" or self._sub.closed:
                raise MediaStreamError("the viewer went away")
            sample = await loop.run_in_executor(self._pool, self._sub.take_video, self._camera, _TAKE_TIMEOUT)
            if sample is None:
                continue
            packet = av.Packet(sample.data)
            packet.pts = rtp_from_capture_ts(sample.capture_ts)
            packet.time_base = _TIME_BASE
            if self._on_sent is not None:
                self._on_sent(sample)
            return packet

    def stop(self) -> None:
        super().stop()
        self._pool.shutdown(wait=False)


def _forward_keyframe_requests(sender, subscription: Subscription, camera: str) -> None:
    """Let the viewer's decoder ask for a keyframe and be heard.

    A browser sends a Picture Loss Indication when it cannot decode what it
    is receiving, and aiortc turns that into a call on the sender — which
    sets a flag its own encoder reads. This track hands over frames that are
    already encoded, so that flag is read by nobody and the viewer waits out
    the keyframe cadence with a frozen picture. On a link that loses packets
    that wait is most of what the operator sees.
    """
    original = sender._send_keyframe

    def _send_keyframe() -> None:
        original()
        subscription.request_keyframe(camera)

    sender._send_keyframe = _send_keyframe


class LiveVideoSession:
    """One viewer: a track per camera and the cycle messages, on one connection.

    The subscription is the viewer's own, so a slow viewer costs the others
    nothing. Closing the session closes the subscription, which ends every
    track.

    The cycle channel belongs to whoever offered: an answerer cannot add a
    channel the offer did not describe, so when the page offers, the page
    opens the channel and this session sends on the one it receives.

    ``decorate`` turns a cycle into the message the page receives, which is
    where anything the page cannot work out from the pictures is added —
    the robot's pose for the visualizer tile. It defaults to the cycle as it
    is, so this layer stays about the stream.
    """

    def __init__(
        self,
        pipeline: LivePipeline,
        connection: RTCPeerConnection,
        decorate: Callable[[CycleMessage], dict] | None = None,
        channel=None,
    ) -> None:
        self._pipeline = pipeline
        self._connection = connection
        self._decorate = decorate or asdict
        self._sub = pipeline.subscribe()
        self._tracks = [EncodedTrack(self._sub, cam) for cam in pipeline.cameras]
        for track in self._tracks:
            sender = connection.addTrack(track)
            transceiver = next(t for t in connection.getTransceivers() if t.sender is sender)
            offer_h264_only(transceiver)
            _forward_keyframe_requests(sender, self._sub, track.camera)
        # A channel the caller already holds — opened before this session
        # existed, on a connection the page opened early — is used as it is;
        # otherwise this side opens one when it is the offerer, or waits for
        # the page's.
        self._channel = channel
        if self._channel is None:
            if connection.remoteDescription is None:
                self._channel = connection.createDataChannel(CYCLE_CHANNEL, ordered=True)
            else:

                @connection.on("datachannel")
                def _on_channel(opened) -> None:
                    if opened.label == CYCLE_CHANNEL:
                        self._channel = opened

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-cycles")
        self._pump = asyncio.ensure_future(self._pump_messages())

    @property
    def cameras(self) -> list[str]:
        return list(self._pipeline.cameras)

    async def _pump_messages(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not self._sub.closed:
                message = await loop.run_in_executor(self._pool, self._sub.take_message, _TAKE_TIMEOUT)
                if message is None:
                    continue
                if self._channel is None or self._channel.readyState != "open":
                    continue
                self._channel.send(json.dumps(self._decorate(message)))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("live video: the cycle channel stopped")

    async def close(self) -> None:
        self._sub.close()
        for track in self._tracks:
            track.stop()
        self._pump.cancel()
        # CancelledError is a BaseException, so suppressing Exception alone
        # lets the cancellation escape into whoever is closing the session.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._pump
        self._pool.shutdown(wait=False)
