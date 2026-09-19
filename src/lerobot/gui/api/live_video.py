"""The Run tab's live camera video: one endpoint a viewer offers itself to.

A viewer opens its connection as soon as the tab is shown, before the run has
frames — an offer with no video streams is answered with the cycle channel
alone — and offers again on the same connection when the cameras appear. The
first picture then costs a keyframe and one way across the link, rather than
a connection's whole setup.

The pipeline runs only while someone is watching a camera: the first offer
that asks for streams starts it, the last viewer's departure stops it, so a
run nobody is watching pays nothing. Every viewer shares the one pipeline and
its one encoder per camera; each gets its own connection, its own
subscription and its own packets.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lerobot.gui.link_class import CLASS_LINK
from lerobot.gui.live_video.pipeline import PROFILE_WIDTH, CycleMessage, LivePipeline
from lerobot.gui.live_video.transport import CYCLE_CHANNEL, LiveVideoSession, video_codecs_chosen

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/run/live-video", tags=["live-video"])

#: Starting and stopping the pipeline join threads, so they wait off the
#: event loop — in their own pool, never the shared one a stalled job
#: elsewhere could fill.
_lifecycle = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-video-lifecycle")


@dataclass
class _Viewer:
    """One page's connection, from before the run until it goes away."""

    connection: RTCPeerConnection
    #: None until the viewer asks for cameras; the pipeline's subscription
    #: and the tracks on it.
    session: LiveVideoSession | None = None
    cameras: list[str] = field(default_factory=list)
    #: The cycle channel the page opened, which it does in its first offer —
    #: before any session exists to hear the event.
    channel: object | None = None


_pipeline: LivePipeline | None = None
_viewers: dict[str, _Viewer] = {}
_lock: asyncio.Lock | None = None
_lock_loop: asyncio.AbstractEventLoop | None = None


def _viewers_lock() -> asyncio.Lock:
    """The lock every path that reads or replaces the pipeline holds.

    Built on first use rather than at import: a lock belongs to one event
    loop, and one built at import belongs to whichever loop happens to touch
    it first — which is not necessarily the loop the server ends up running
    on, and never the same one twice across a suite.
    """
    global _lock, _lock_loop
    loop = asyncio.get_running_loop()
    if _lock is None or _lock_loop is not loop:
        _lock, _lock_loop = asyncio.Lock(), loop
    return _lock


class Offer(BaseModel):
    sdp: str
    type: str
    #: The connection this offer belongs to, from a previous answer. Absent
    #: for the first offer, which opens one.
    session: str | None = None
    #: Whether this offer's video streams are meant to carry cameras. The
    #: offer that opens a connection before the run has any sets it false:
    #: its one video stream is a placeholder, there because every later
    #: stream shares the transport chosen for the first one ([O20]).
    attach: bool = True


def pipeline_running() -> bool:
    return _pipeline is not None


def _watching() -> int:
    return sum(1 for v in _viewers.values() if v.session is not None)


def _reader_or_none():
    from lerobot.robots.obs_stream import ObservationStreamReader

    try:
        return ObservationStreamReader()
    except Exception:
        return None


def _profile_for(cameras: int) -> dict:
    """What the page shows beside the stream's state, and what it is against."""
    return {
        "profile": {
            "width": PROFILE_WIDTH,
            "bitrate_kbit_s": CLASS_LINK.per_camera_kbit_s(cameras) if cameras else None,
        },
        "link": {
            "name": CLASS_LINK.name,
            "down_kbit_s": CLASS_LINK.down_kbit_s,
            "rtt_ms": CLASS_LINK.rtt_ms,
            "stream_budget_kbit_s": CLASS_LINK.stream_budget_kbit_s,
        },
    }


def _resolve_pose(state: dict[str, float]) -> dict:
    """The visualizer tile's joint angles for one cycle's state.

    The same shape the tile's own endpoint returns, so the tile applies it
    unchanged — it is fed from the stream instead of polling, which is what
    keeps it as fresh as the pictures. A robot with no vendored description
    is not an error: the tile is simply not shown.
    """
    from lerobot.gui.urdf_viz import compute_joint_angles, resolve_robot

    spec = resolve_robot(state.keys())
    if spec is None:
        return {"available": False}
    angles = compute_joint_angles(spec, state)
    return {
        "available": True,
        "arms": [
            {"prefix": a.obs_prefix, "frames": [{"joints": angles.get(a.obs_prefix, {})}]} for a in spec.arms
        ],
    }


def _message_for_the_page(message: CycleMessage) -> dict:
    body = asdict(message)
    body["pose"] = _resolve_pose(message.state)
    return body


@router.get("/status")
async def status() -> dict:
    """What the controls bar shows: whether there is anything to watch, at
    what profile, and how many people are watching."""
    reader = _pipeline or _reader_or_none()
    if reader is None:
        return {"available": False, "cameras": [], "viewers": 0, "encoders": 0}
    cameras = list(reader.cameras if _pipeline is not None else reader.image_keys)
    if _pipeline is None:
        reader.close()
    return {
        "available": True,
        "cameras": cameras,
        "viewers": _watching(),
        "encoders": _pipeline.encoder_count if _pipeline is not None else 0,
        **_profile_for(len(cameras)),
        "cameras_detail": _pipeline.snapshot() if _pipeline is not None else {},
    }


def _build_and_start() -> LivePipeline:
    pipeline = LivePipeline()
    # Starting pays the device's one-time costs before a frame is asked
    # for, so the first viewer's first picture is not the one that waits.
    pipeline.start()
    return pipeline


async def _ensure_pipeline() -> LivePipeline:
    global _pipeline
    if _pipeline is not None and not _pipeline.reads_the_current_tap():
        # The run it was built for has ended. Its reader holds segments
        # nobody writes to any more, so serving from it would be a stream
        # that never advances again — indistinguishable, on the page, from
        # a run standing still.
        logger.info("live video: the run this pipeline read has ended, building one for the new tap")
        pipeline, _pipeline = _pipeline, None
        await asyncio.get_running_loop().run_in_executor(_lifecycle, pipeline.stop)
    if _pipeline is None:
        try:
            # Building asks the device which encoders it has, which is the
            # first CUDA call in this process and costs seconds of it: the
            # loop this runs on is the one serving every other request.
            pipeline = await asyncio.get_running_loop().run_in_executor(_lifecycle, _build_and_start)
        except FileNotFoundError as e:
            # No tap: there is genuinely no run, which is the ordinary state
            # between runs and not a fault. This is the answer the page treats
            # as "nothing to watch yet" rather than as a failure.
            logger.info("live video: no run to watch (%s)", e)
            raise HTTPException(503, "No run is streaming, so there is nothing to watch yet") from e
        except Exception as e:
            # A run IS streaming and the pipeline would not start -- no encoder,
            # a device that refused, a camera the reader cannot map. Answering
            # "nothing to watch" sends the operator to look at the robot while
            # the only account of what happened sits in a log line nobody is
            # reading. The design requires this state to carry a reason.
            logger.exception("live video: the pipeline would not start")
            raise HTTPException(503, f"The live stream could not start: {type(e).__name__}: {e}") from e
        _pipeline = pipeline
        logger.info("live video: pipeline started for %s", pipeline.cameras)
    return _pipeline


async def _stop_pipeline_if_nobody_is_watching() -> None:
    """Stop the pipeline when the last viewer of it has gone.

    The pipeline is started by the first offer that attaches cameras, so a
    viewer that never becomes one — a refused offer, a connection that died
    during its own negotiation — would otherwise leave it encoding for
    nobody, with no viewer left to leave and stop it.
    """
    global _pipeline
    if _watching() == 0 and _pipeline is not None:
        pipeline, _pipeline = _pipeline, None
        await asyncio.get_running_loop().run_in_executor(_lifecycle, pipeline.stop)


def _check_offer(wanted: list[str], pipeline: LivePipeline) -> None:
    """What an offer asking for cameras must look like.

    An answer cannot add a video stream the offer did not describe, and
    cannot change the codec the offer put first, so an offer wrong in either
    way would be answered with a stream that carries nothing the viewer can
    decode. Say so instead.
    """
    if len(wanted) != len(pipeline.cameras):
        raise HTTPException(
            400,
            f"This run has {len(pipeline.cameras)} cameras and the offer describes "
            f"{len(wanted)} video streams. Ask /status for the cameras, then offer one "
            "receive-only video transceiver for each.",
        )
    wrong = sorted({c for c in wanted if c.upper() != "H264"})
    if wrong:
        raise HTTPException(
            400,
            f"This stream is H.264; the offer asks for {', '.join(wrong)} first. "
            "Put H.264 first on each video transceiver before offering.",
        )


@router.post("/offer")
async def offer(body: Offer) -> dict:
    """Answer a viewer's offer, on a new connection or the one it names."""
    async with _viewers_lock():
        wanted = video_codecs_chosen(body.sdp)
        # `attach` is the page saying which of its two moves this is, so an
        # offer that claims to carry cameras and describes none is wrong in
        # the same way as one that describes too few.
        attaching = body.attach

        if body.session is None:
            # No ICE servers: the page reaches this server directly — the
            # same host, the LAN, or the tailnet — so host candidates are the
            # whole answer, and asking a public one costs seconds of setup
            # for a candidate nothing here uses.
            token = secrets.token_urlsafe(12)
            connection = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            viewer = _Viewer(connection=connection)
            _viewers[token] = viewer

            @connection.on("datachannel")
            def on_channel(channel) -> None:
                if channel.label == CYCLE_CHANNEL:
                    viewer.channel = channel

            @connection.on("connectionstatechange")
            async def on_state_change() -> None:
                if connection.connectionState in ("failed", "closed", "disconnected"):
                    await _drop(token)
        else:
            token = body.session
            viewer = _viewers.get(token)
            if viewer is None:
                raise HTTPException(404, "That session is not open here; offer without one to open it")
            connection = viewer.connection

        if attaching:
            pipeline = await _ensure_pipeline()
            try:
                _check_offer(wanted, pipeline)
            except HTTPException:
                await _stop_pipeline_if_nobody_is_watching()
                raise
        elif wanted and len(wanted) > 1:
            raise HTTPException(
                400,
                "An offer that is not attaching cameras carries one placeholder video "
                f"stream, not {len(wanted)}.",
            )

        await connection.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
        if attaching and viewer.session is None:
            # The session reads the remote description to learn which side
            # opened the cycle channel, so it is built after the offer is
            # applied — and only now, because a connection with no cameras
            # has nothing to subscribe to.
            viewer.session = LiveVideoSession(
                pipeline, connection, decorate=_message_for_the_page, channel=viewer.channel
            )
            viewer.cameras = viewer.session.cameras
        await connection.setLocalDescription(await connection.createAnswer())
        # Which stream is which camera: the page has no way to tell from the
        # description, and the order is this session's, not the status
        # endpoint's — a run that restarts with a camera unplugged would
        # otherwise relabel every tile.
        return {
            "sdp": connection.localDescription.sdp,
            "type": connection.localDescription.type,
            "cameras": viewer.cameras,
            "session": token,
            # The profile the tiles are at, so the page can say what it is
            # showing without a second request for it.
            **_profile_for(len(viewer.cameras)),
        }


async def _drop(token: str) -> None:
    """Close one viewer's connection and stop the pipeline if it was the last.

    Takes the same lock the offer path holds: both read and replace the
    pipeline, and a viewer leaving while another is negotiating would
    otherwise take the pipeline out from under it.
    """
    async with _viewers_lock():
        await _drop_held(token)


async def _drop_held(token: str) -> None:
    """Precondition: the caller holds the viewers lock."""
    viewer = _viewers.pop(token, None)
    if viewer is None:
        return
    if viewer.session is not None:
        await viewer.session.close()
    await viewer.connection.close()
    logger.info("live video: a viewer left, %d watching", _watching())
    await _stop_pipeline_if_nobody_is_watching()


async def shutdown() -> None:
    """Close every viewer and stop the pipeline. For server shutdown and tests."""
    global _pipeline
    async with _viewers_lock():
        for token in list(_viewers):
            await _drop_held(token)
        # A pipeline with nobody watching is normally stopped by the last
        # drop; stopping it here too means shutdown holds whatever happened.
        if _pipeline is not None:
            pipeline, _pipeline = _pipeline, None
            await asyncio.get_running_loop().run_in_executor(_lifecycle, pipeline.stop)
