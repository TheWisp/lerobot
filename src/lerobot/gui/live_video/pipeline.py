"""The pipeline from the tap to encoded frames, one camera at a time.

Per camera: the tap's newest frame lands in a mailbox; the camera's encode
thread takes it, uploads it once, resizes it to the profile, blends the
newest overlay there is, encodes it, and hands the access unit to every
viewer's queue. Nothing between the tap and the encoder holds more than one
frame, so a slow stage drops frames rather than ageing them. After the
encoder every access unit is delivered in order, because a decoder cannot
skip one; a viewer that stops taking has its queue emptied instead, and the
next thing it takes is a keyframe. Beside the pictures, one message per
cycle carries the state and the newest action.

Preconditions: a tap with data exists when the pipeline is built; frames of
one camera keep their size for the life of the pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.gui.link_class import CLASS_LINK, PROFILE_WIDTH, LinkClass
from lerobot.gui.live_video.encoder import Encoder, available_backends, make_encoder
from lerobot.gui.live_video.mailbox import Mailbox
from lerobot.gui.live_video.stages import blend_overlay, resize_to_width
from lerobot.robots.obs_stream import (
    BlockStamp,
    CaptureSource,
    ObservationStreamReader,
    stream_identity,
)

logger = logging.getLogger(__name__)

#: How often a pipeline with no overlay worker beside it looks for one.
OVERLAY_ATTACH_INTERVAL_S = 1.0

#: How many encoded frames a viewer may leave untaken before its queue is
#: emptied; a few frame periods, so a hiccup costs nothing and a stall costs
#: a keyframe.
VIEWER_QUEUE_DEPTH = 8

#: How many cycle messages a viewer may leave untaken; the oldest go first,
#: since each shows its newest.
MESSAGE_QUEUE_DEPTH = 64


@dataclass(frozen=True)
class EncodedSample:
    camera: str
    data: bytes
    keyframe: bool
    cycle: int
    capture_ts: float
    capture_source: CaptureSource
    #: The cycle the overlay drawn on this frame was computed for; None when
    #: the frame went out bare.
    overlay_cycle: int | None
    encoded_ts: float


@dataclass(frozen=True)
class CycleMessage:
    cycle: int
    capture_ts: float
    state: dict[str, float]
    #: The newest action the tap had when the cycle's observation appeared,
    #: normally the previous cycle's; ``action_cycle`` says which.
    action: dict[str, float] | None
    action_cycle: int | None
    overlay_cycles: dict[str, int]
    #: Cameras whose encoder is failing, and why. Empty while the stream is
    #: working. Encoding is the one stage with nothing downstream to notice
    #: it stopped — the frames simply cease — so what is wrong travels on the
    #: channel that is still running rather than only into the server's log.
    failing: dict[str, str]


@dataclass(frozen=True)
class _TapFrame:
    image: np.ndarray
    stamp: BlockStamp


class _VideoQueue:
    """One viewer's queue for one camera: in order and bounded.

    Overflow empties it and marks the viewer as needing a keyframe; until
    that keyframe arrives, other frames are dropped, since a decoder could
    not use them. The request for the keyframe goes to the encoder when the
    viewer next takes, so a viewer that has gone away costs nobody a keyframe.
    """

    def __init__(self, depth: int) -> None:
        self._items: deque[EncodedSample] = deque()
        self._depth = depth
        self._cond = threading.Condition()
        self._closed = False
        self.needs_keyframe = True
        self.keyframe_requested = True
        self.dropped = 0

    def push(self, sample: EncodedSample) -> None:
        with self._cond:
            if self._closed:
                return
            if len(self._items) >= self._depth:
                self.dropped += len(self._items)
                self._items.clear()
                self.needs_keyframe = True
            if self.needs_keyframe and not sample.keyframe:
                self.dropped += 1
                return
            self.needs_keyframe = False
            self._items.append(sample)
            self._cond.notify()

    def take(self, timeout: float | None) -> EncodedSample | None:
        with self._cond:
            if self.needs_keyframe:
                self.keyframe_requested = True
            if not self._items and not self._closed:
                self._cond.wait_for(lambda: bool(self._items) or self._closed, timeout)
            return self._items.popleft() if self._items else None

    def want_keyframe(self) -> None:
        """This viewer's decoder cannot use what it is being sent.

        Its next frame for this camera is an IDR with the parameter sets
        beside it, whatever the keyframe cadence would have given it.
        """
        self.keyframe_requested = True

    def consume_request(self) -> bool:
        with self._cond:
            wanted, self.keyframe_requested = self.keyframe_requested, False
            return wanted

    def close(self) -> None:
        # Whatever is still queued belongs to a viewer that has gone. Leaving it
        # meant a take after close served a frame from the run that ended, and
        # the buffer stayed alive as long as anything held the subscription.
        #
        # Clearing is the whole mechanism: `push` already refuses once closed,
        # under this same lock, so nothing can arrive afterwards and `take`
        # finds an empty deque. A second guard in `take` would be unreachable.
        with self._cond:
            self._closed = True
            self._items.clear()
            self._cond.notify_all()


class _MessageQueue:
    def __init__(self, depth: int) -> None:
        self._items: deque[CycleMessage] = deque(maxlen=depth)
        self._cond = threading.Condition()
        self._closed = False

    def push(self, message: CycleMessage) -> None:
        with self._cond:
            if not self._closed:
                self._items.append(message)
                self._cond.notify()

    def take(self, timeout: float | None) -> CycleMessage | None:
        with self._cond:
            if not self._items and not self._closed:
                self._cond.wait_for(lambda: bool(self._items) or self._closed, timeout)
            return self._items.popleft() if self._items else None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._items.clear()
            self._cond.notify_all()


class Subscription:
    """One viewer's side of the pipeline: a queue per camera and one for messages."""

    def __init__(self, broadcaster: _Broadcaster, cameras: list[str], depth: int) -> None:
        self._broadcaster = broadcaster
        self._videos = {cam: _VideoQueue(depth) for cam in cameras}
        self._messages = _MessageQueue(MESSAGE_QUEUE_DEPTH)
        self.closed = False

    def take_video(self, camera: str, timeout: float | None = None) -> EncodedSample | None:
        return self._videos[camera].take(timeout)

    def take_message(self, timeout: float | None = None) -> CycleMessage | None:
        return self._messages.take(timeout)

    def dropped(self, camera: str) -> int:
        return self._videos[camera].dropped

    def request_keyframe(self, camera: str) -> None:
        """This viewer's decoder cannot use what it is being sent.

        There is one encode for every viewer, so forcing it serves them all;
        a viewer that did not ask loses nothing by receiving one.
        """
        queue = self._videos.get(camera)
        if queue is not None:
            queue.keyframe_requested = True

    def close(self) -> None:
        self.closed = True
        for q in self._videos.values():
            q.close()
        self._messages.close()
        self._broadcaster.unsubscribe(self)


class _Broadcaster:
    def __init__(self, cameras: list[str]) -> None:
        self._cameras = cameras
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()

    def subscribe(self, depth: int) -> Subscription:
        sub = Subscription(self, self._cameras, depth)
        with self._lock:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def publish_video(self, sample: EncodedSample) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub._videos[sample.camera].push(sample)

    def publish_message(self, message: CycleMessage) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub._messages.push(message)

    def consume_keyframe_request(self, camera: str) -> bool:
        with self._lock:
            subs = list(self._subs)
        wanted = False
        for sub in subs:
            wanted |= sub._videos[camera].consume_request()
        return wanted

    def close_all(self) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub.close()


class LivePipeline:
    def __init__(
        self,
        *,
        fps: int = 30,
        width: int = PROFILE_WIDTH,
        link: LinkClass = CLASS_LINK,
        device: str | None = None,
        encoder_backend: str | None = None,
        encoder_factory: Callable[..., Encoder] = make_encoder,
        reader_factory: Callable[[], ObservationStreamReader] = ObservationStreamReader,
    ) -> None:
        self._reader = reader_factory()
        # Which tap this pipeline is reading. A run that ends unlinks its
        # segments and the next run creates new ones under the same names,
        # so a reader that keeps the old mapping sees a sequence that never
        # advances: this is what says so, rather than a silence that looks
        # like a paused run.
        self._tap = stream_identity()
        self.cameras: list[str] = list(self._reader.image_keys)
        self.fps = fps
        self.width = width
        # What a camera may send, and what its encoder is asked for: the
        # second is lower, because rate control lands above its target and
        # the link cares about what arrives.
        self.bitrate_kbit_s = link.per_camera_kbit_s(len(self.cameras))
        self.encoder_target_kbit_s = link.encoder_target_kbit_s(len(self.cameras))
        backends = available_backends()
        # Two questions, not one. The stages go wherever the tensors can go;
        # the encoder goes where its library is. Deriving the first from the
        # second sent every upload, resize and blend to the CPU on a host
        # with a GPU but without the hardware encoder — and the software
        # encoder copies its frame back itself, so nothing needed that.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_backend = encoder_backend or (
            "nvenc" if self.device == "cuda" and "nvenc" in backends else "libx264"
        )
        self._encoder_factory = encoder_factory
        self._mailboxes: dict[str, Mailbox[_TapFrame]] = {cam: Mailbox() for cam in self.cameras}
        # Camera -> the reason its last encode failed, cleared by a success.
        self._failing: dict[str, str] = {}
        self._overlays: dict[str, tuple[torch.Tensor, int]] = {}
        self._overlay_lock = threading.Lock()
        # The worker's channel, and what has already been taken from it.
        self._overlays_in = None
        self._overlay_seen: dict[str, int] = {}
        self._overlay_attach_at = 0.0
        self._broadcaster = _Broadcaster(self.cameras)
        self._stats: dict[str, dict] = {
            cam: {"frames_in": 0, "encoded": 0, "bytes": 0, "errors": 0, "size": None, "backend": None}
            for cam in self.cameras
        }
        self._encoders: dict[str, Encoder] = {}
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _make_encoder(self, camera: str, height: int, width: int) -> Encoder:
        old = self._encoders.pop(camera, None)
        if old is not None:
            old.close()
        encoder = self._encoder_factory(
            width, height, self.fps, self.encoder_target_kbit_s, backend=self.encoder_backend
        )
        self._encoders[camera] = encoder
        self._stats[camera]["size"] = (height, width)
        self._stats[camera]["backend"] = encoder.backend
        return encoder

    def _warm_up(self) -> None:
        """Pay the one-time costs before a frame exists — the device's first
        kernels and each camera's encoder session at the profile's size — so
        the first frame is treated like every other. The warm frame is never
        published; a viewer's first frame is a keyframe forced for it."""
        for cam, dims in self._reader.image_keys.items():
            t = time.perf_counter()
            h, w = int(dims[0]), int(dims[1])
            picture = resize_to_width(self._upload(np.zeros((h, w, 3), dtype=np.uint8)), self.width)
            picture = blend_overlay(picture, torch.zeros((h, w, 4), dtype=torch.uint8, device=self.device))
            encoder = self._make_encoder(cam, int(picture.shape[0]), int(picture.shape[1]))
            encoder.encode(picture)
            self._stats[cam]["warm_up_ms"] = round((time.perf_counter() - t) * 1000.0, 1)

    def start(self) -> None:
        assert not self._threads, "already started"
        self._warm_up()
        self._threads.append(threading.Thread(target=self._tap_loop, name="live-video-tap", daemon=True))
        for cam in self.cameras:
            self._threads.append(
                threading.Thread(target=self._encode_loop, args=(cam,), name=f"live-video-{cam}", daemon=True)
            )
        for t in self._threads:
            t.start()

    def reads_the_current_tap(self) -> bool:
        """Whether the run this pipeline was built for is still the one running.

        False once that run has ended, whether or not another has started:
        either way there is nothing more to read here, and a pipeline built
        on the tap that is there now is what a viewer needs.
        """
        return self._tap is not None and stream_identity() == self._tap

    def stop(self) -> None:
        self._stop.set()
        for mb in self._mailboxes.values():
            mb.close()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()
        self._broadcaster.close_all()
        for encoder in self._encoders.values():
            encoder.close()
        self._encoders.clear()
        self._reader.close()
        if self._overlays_in is not None:
            # Reader side: closes the mapping, unlinks nothing the worker owns.
            with contextlib.suppress(Exception):
                self._overlays_in.cleanup()
            self._overlays_in = None

    def subscribe(self, depth: int = VIEWER_QUEUE_DEPTH) -> Subscription:
        """A viewer's queues; its first frame per camera is a keyframe."""
        return self._broadcaster.subscribe(depth)

    def set_overlay(self, camera: str, rgba: np.ndarray | torch.Tensor, cycle: int) -> None:
        """The newest overlay for a camera and the cycle it was computed for;
        the next frame encoded carries it. Moved to the pipeline's device on
        the caller's thread."""
        tensor = torch.as_tensor(rgba)
        assert tensor.ndim == 3 and tensor.shape[2] == 4 and tensor.dtype == torch.uint8, tensor.shape
        tensor = tensor.to(self.device)
        with self._overlay_lock:
            self._overlays[camera] = (tensor, cycle)

    @property
    def encoder_count(self) -> int:
        """How many encoders exist, which is the claim R8 rests on: one per
        camera however many people are watching. Counted rather than derived
        from the camera list, so a second encoder per viewer would show."""
        return len(self._encoders)

    def snapshot(self) -> dict[str, dict]:
        out = {}
        for cam, s in self._stats.items():
            out[cam] = {
                **s,
                "dropped": self._mailboxes[cam].dropped,
                "device": self.device,
                "bitrate_kbit_s": self.bitrate_kbit_s,
            }
        return out

    def _overlay_buffer(self):
        """The overlay worker's channel, once there is one.

        The worker creates it only after its model is loaded, and a run
        watched with no overlay at all is the ordinary case, so this attaches
        lazily and looks again on an interval rather than paying for the
        absence every cycle.
        """
        if self._overlays_in is not None:
            return self._overlays_in
        now = time.monotonic()
        if now < self._overlay_attach_at:
            return None
        self._overlay_attach_at = now + OVERLAY_ATTACH_INTERVAL_S
        try:
            from lerobot.overlays.overlay_ipc import SharedOverlayBuffer

            self._overlays_in = SharedOverlayBuffer(create=False)
        except FileNotFoundError:
            self._overlays_in = None
        except Exception:
            logger.warning("live video: could not attach to the overlay buffer", exc_info=True)
            self._overlays_in = None
        return self._overlays_in

    def _forget_overlays(self) -> None:
        """The worker has gone: stop drawing what it last published.

        Without this the overlay an operator turned off stays burnt into
        every frame for the rest of the run, which is worse than never
        having drawn it.
        """
        self._overlays_in = None
        self._overlay_seen.clear()
        with self._overlay_lock:
            self._overlays.clear()

    def _pick_up_overlays(self, cycle: int) -> None:
        """Take whatever the worker has published since the last look.

        The buffer carries no cycle of its own, so what rides with the frame
        is the cycle the overlay arrived on. That is what the lag is for: how
        far behind the picture the overlay drawn on it was taken. The cycle it
        was computed for would need a field the worker does not publish.
        """
        buffer = self._overlay_buffer()
        if buffer is None:
            return
        for camera in self.cameras:
            if camera not in buffer.cameras:
                continue
            try:
                seq = buffer.overlay_seq(camera)
                if seq == 0 or seq == self._overlay_seen.get(camera):
                    continue
                result = buffer.read_overlay(camera)
            except Exception:
                # The worker can unload its model between the look and the
                # read, which unlinks these segments underneath us.
                self._forget_overlays()
                return
            if result is None:
                continue
            self._overlay_seen[camera] = seq
            self.set_overlay(camera, result[0], cycle)

    def _overlay_for(self, camera: str) -> tuple[torch.Tensor, int] | None:
        with self._overlay_lock:
            return self._overlays.get(camera)

    def _overlay_cycles(self) -> dict[str, int]:
        with self._overlay_lock:
            return {cam: cycle for cam, (_, cycle) in self._overlays.items()}

    def _tap_loop(self) -> None:
        reader = self._reader
        last_seq = dict.fromkeys(self.cameras, 0)
        last_cycle = 0
        while not self._stop.is_set():
            advanced = False
            for cam in self.cameras:
                seq = reader.image_seq(cam)
                if seq == last_seq[cam]:
                    continue
                result = reader.read_image_stamped(cam)
                if result is None:
                    continue
                last_seq[cam] = seq
                self._mailboxes[cam].put(_TapFrame(result[0], result[1]))
                self._stats[cam]["frames_in"] += 1
                advanced = True
            obs = reader.read_obs_stamped()
            if obs is not None and obs[1].cycle != last_cycle:
                last_cycle = obs[1].cycle
                # Before the message that reports the overlay cycles, so a
                # cycle's message describes the overlay its frames carry.
                self._pick_up_overlays(last_cycle)
                act = reader.read_action_stamped()
                self._broadcaster.publish_message(
                    CycleMessage(
                        cycle=obs[1].cycle,
                        capture_ts=obs[1].capture_ts,
                        state=obs[0],
                        action=act[0] if act is not None else None,
                        action_cycle=act[1].cycle if act is not None else None,
                        overlay_cycles=self._overlay_cycles(),
                        failing=dict(self._failing),
                    )
                )
                advanced = True
            if not advanced:
                time.sleep(0.001)

    def _upload(self, image: np.ndarray) -> torch.Tensor:
        with warnings.catch_warnings():
            # The tap's copy is read-only and no stage writes to it.
            warnings.simplefilter("ignore", UserWarning)
            return torch.from_numpy(image).to(self.device)

    def _encode_loop(self, camera: str) -> None:
        mailbox = self._mailboxes[camera]
        stats = self._stats[camera]
        while not self._stop.is_set():
            frame = mailbox.take(timeout=0.1)
            if frame is None:
                continue
            try:
                picture = resize_to_width(self._upload(frame.image), self.width)
                overlay = self._overlay_for(camera)
                overlay_cycle = None
                if overlay is not None:
                    picture = blend_overlay(picture, overlay[0])
                    overlay_cycle = overlay[1]
                h, w = int(picture.shape[0]), int(picture.shape[1])
                encoder = self._encoders.get(camera)
                if encoder is None or stats["size"] != (h, w):
                    encoder = self._make_encoder(camera, h, w)
                force = self._broadcaster.consume_keyframe_request(camera)
                out = encoder.encode(picture, force_keyframe=force)
                self._broadcaster.publish_video(
                    EncodedSample(
                        camera=camera,
                        data=out.data,
                        keyframe=out.keyframe,
                        cycle=frame.stamp.cycle,
                        capture_ts=frame.stamp.capture_ts,
                        capture_source=frame.stamp.capture_source,
                        overlay_cycle=overlay_cycle,
                        encoded_ts=time.time(),
                    )
                )
                stats["encoded"] += 1
                stats["bytes"] += len(out.data)
                self._failing.pop(camera, None)
            except Exception as e:
                stats["errors"] += 1
                self._failing[camera] = str(e) or type(e).__name__
                logger.exception("live video: %s frame failed", camera)
                time.sleep(0.05)
