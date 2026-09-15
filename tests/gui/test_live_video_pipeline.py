"""The pipeline from the tap to encoded frames, at the Workload's shape.

A synthetic tap writes four cameras at the rig's sizes at 30 fps with the
cycle number in the state; the pipeline is measured against what the tap
wrote in the same window, never against a number of seconds, so a slow
machine cannot fail these for being slow and a fast one cannot pass them for
the wrong reason.
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from collections import defaultdict

import av
import numpy as np
import pytest
import torch

import lerobot.robots.obs_stream as obs_stream
from lerobot.gui.link_class import CLASS_LINK
from lerobot.gui.live_video.encoder import available_backends, make_encoder
from lerobot.gui.live_video.pipeline import CycleMessage, EncodedSample, LivePipeline
from lerobot.robots.obs_stream import CaptureSource, ObservationStream

FPS = 30
CAMERAS = {
    "top": (720, 1280, 3),
    "front": (600, 960, 3),
    "left_wrist": (600, 960, 3),
    "right_wrist": (720, 1280, 3),
}
PATTERNS = 8


@pytest.fixture(autouse=True)
def _own_shm_names(monkeypatch):
    """Never the names a live GUI on this host uses."""
    monkeypatch.setattr(obs_stream, "SHM_PREFIX", f"lerobot_obs_t{os.getpid()}_")


class SyntheticTap:
    """A run's tap at the Workload: four cameras, 30 fps, the cycle in the state."""

    #: The rig's own motor names, so what reads this tap resolves a robot the
    #: way it will in a run, rather than a shape only tests have.
    JOINTS = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )

    def __init__(self, fps: int = FPS) -> None:
        self.fps = fps
        obs_features = {"cycle": float, **dict.fromkeys(self.JOINTS, float), **CAMERAS}
        self.stream = ObservationStream(obs_features, dict.fromkeys(self.JOINTS, float))
        rng = np.random.default_rng(0)
        self._frames: dict[str, list[np.ndarray]] = {}
        for cam, (h, w, _) in CAMERAS.items():
            y = np.linspace(0, 255, h, dtype=np.float32)[:, None, None]
            x = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
            base = np.concatenate(
                [np.broadcast_to(y, (h, w, 1)), np.broadcast_to(x, (h, w, 1)), (y + x) / 2], 2
            )
            base = base.astype(np.uint8)
            noise = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
            frames = []
            for i in range(PATTERNS):
                f = base.copy()
                x0 = (i * w // PATTERNS) % (w - 80)
                f[h // 3 : h // 3 + 80, x0 : x0 + 80] = (255, 40, 40)
                f = np.clip(f.astype(np.int16) + np.roll(noise, i * 13, axis=1) - 20, 0, 255).astype(np.uint8)
                frames.append(np.ascontiguousarray(f))
            self._frames[cam] = frames
        self.cycles_written = 0
        self.written_at: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="synthetic-tap", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self.stream.cleanup()

    def _run(self) -> None:
        period = 1.0 / self.fps
        t_next = time.perf_counter()
        while not self._stop.is_set():
            i = self.cycles_written
            self.stream.mark_observation_start()
            obs = {"cycle": float(i + 1)}
            for j, joint in enumerate(self.JOINTS):
                obs[joint] = float(np.sin(i / 10.0 + j))
            for cam in CAMERAS:
                obs[cam] = self._frames[cam][i % PATTERNS]
            self.stream.write_obs(obs)
            self.stream.write_action({j: float(np.cos(i / 10.0 + k)) for k, j in enumerate(self.JOINTS)})
            self.cycles_written = i + 1
            self.written_at.append(time.time())
            t_next += period
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

    def written_between(self, t0: float, t1: float) -> int:
        return sum(1 for t in self.written_at if t0 <= t <= t1)


@pytest.fixture
def tap():
    t = SyntheticTap()
    t.start()
    # The pipeline attaches to a tap that has data.
    deadline = time.time() + 5.0
    while t.cycles_written < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert t.cycles_written >= 2
    yield t
    t.stop()


BACKENDS = available_backends()


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


@pytest.fixture
def pipeline(tap, backend):
    p = LivePipeline(fps=FPS, encoder_backend=backend, device="cuda" if backend == "nvenc" else "cpu")
    p.start()
    yield p
    p.stop()


def _collect(sub, seconds: float) -> tuple[dict[str, list[EncodedSample]], list[CycleMessage], float, float]:
    """Everything a subscription delivers in a window, with the window's bounds."""
    videos: dict[str, list[EncodedSample]] = defaultdict(list)
    messages: list[CycleMessage] = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        for cam in CAMERAS:
            while (s := sub.take_video(cam, timeout=0)) is not None:
                videos[cam].append(s)
        while (m := sub.take_message(timeout=0)) is not None:
            messages.append(m)
        time.sleep(0.002)
    return videos, messages, t0, time.time()


def _decode(samples: list[EncodedSample]) -> list[np.ndarray]:
    codec = av.CodecContext.create("h264", "r")
    out: list[np.ndarray] = []
    for s in samples:
        out += [fr.to_ndarray(format="rgb24") for fr in codec.decode(av.Packet(s.data))]
    out += [fr.to_ndarray(format="rgb24") for fr in codec.decode(None)]
    return out


def _ages_ms(samples: list[EncodedSample]) -> list[float]:
    return [(s.encoded_ts - s.capture_ts) * 1000.0 for s in samples]


def test_every_camera_streams_at_the_taps_own_rate(tap, pipeline):
    sub = pipeline.subscribe()
    videos, _, t0, t1 = _collect(sub, 2.0)
    written = tap.written_between(t0, t1)
    assert written >= FPS, written
    for cam in CAMERAS:
        assert len(videos[cam]) >= 0.75 * written, (cam, len(videos[cam]), written)
    for samples in videos.values():
        cycles = [s.cycle for s in samples]
        assert cycles == sorted(cycles)
        assert len(set(cycles)) == len(cycles), "a frame was encoded twice"


def test_the_first_frame_is_treated_like_every_other(tap, pipeline):
    """The one-time costs are paid before a frame exists, so the first frame
    encoded is as young as the rest: the mechanism behind the first picture
    following the first frame."""
    sub = pipeline.subscribe()
    videos, _, _, _ = _collect(sub, 2.0)
    period_ms = 1000.0 / FPS
    for cam, samples in videos.items():
        ages = _ages_ms(samples)
        # The first frame taken already existed for up to a period when the
        # pipeline started; what must not appear is a start-up cost on top.
        assert ages[0] <= statistics.median(ages) + 2 * period_ms, (cam, ages[0], statistics.median(ages))
        # A wall-clock delta is always non-negative, so the number alone
        # says nothing: what matters is that warming happened at all and
        # cost less than the run it is there to protect.
        warmed = pipeline.snapshot()[cam]["warm_up_ms"]
        assert 0 < warmed < 5000, warmed


def test_the_profile_derives_from_the_shared_link_constant(pipeline):
    assert pipeline.bitrate_kbit_s == CLASS_LINK.per_camera_kbit_s(len(CAMERAS))
    assert pipeline.bitrate_kbit_s * len(CAMERAS) <= CLASS_LINK.stream_budget_kbit_s


def test_the_stream_fits_the_shared_link_budget(tap, pipeline):
    """All cameras together within a quarter over the budget the constant
    derives — on this fixture's synthetic noise, which is harder to encode
    than a room and costs the encoder more than the content a run carries.

    This is a ceiling that catches a stream running away, not R3's target.
    The target is asserted at its stated value on a recorded episode in
    `test_live_video_real_footage.py`, which is where the claim about the
    profile fitting the link is actually made.
    """
    sub = pipeline.subscribe()
    _collect(sub, 1.0)  # the encoders settle
    videos, _, t0, t1 = _collect(sub, 2.0)
    total_bits = sum(len(s.data) for samples in videos.values() for s in samples) * 8
    kbit_s = total_bits / (t1 - t0) / 1000.0
    assert kbit_s <= CLASS_LINK.stream_budget_kbit_s * 1.25, (kbit_s, dict(pipeline.snapshot()))
    for cam, samples in videos.items():
        cam_kbit_s = sum(len(s.data) for s in samples) * 8 / (t1 - t0) / 1000.0
        assert cam_kbit_s <= pipeline.bitrate_kbit_s * 1.4, (cam, cam_kbit_s)


def test_the_encoded_stream_decodes_at_the_profiles_size(tap, pipeline):
    sub = pipeline.subscribe()
    videos, _, _, _ = _collect(sub, 1.5)
    for cam, (h, w, _) in CAMERAS.items():
        samples = videos[cam]
        assert samples and samples[0].keyframe, cam
        pictures = _decode(samples)
        assert len(pictures) == len(samples), cam
        assert pictures[0].shape == (round(h * 320 / w) // 2 * 2, 320, 3), (cam, pictures[0].shape)


def test_a_new_viewer_starts_on_a_keyframe(tap, pipeline):
    first = pipeline.subscribe()
    _collect(first, 1.0)
    late = pipeline.subscribe()
    videos, _, _, _ = _collect(late, 1.0)
    for cam in CAMERAS:
        assert videos[cam], cam
        assert videos[cam][0].keyframe, cam
        assert len(_decode(videos[cam])) == len(videos[cam]), cam


def test_a_stalled_viewer_is_caught_up_with_a_keyframe(tap, pipeline):
    """A viewer that stops taking has its queue emptied rather than aged; what
    it takes next starts at a keyframe and decodes."""
    sub = pipeline.subscribe(depth=4)
    time.sleep(1.0)
    videos, _, _, _ = _collect(sub, 1.0)
    for cam in CAMERAS:
        assert sub.dropped(cam) > 0, cam
        assert videos[cam][0].keyframe, cam
        assert len(_decode(videos[cam])) == len(videos[cam]), cam


def test_a_slow_camera_does_not_delay_the_others(tap, backend):
    """One camera's encoder made ten times slower than a frame period: that
    camera drops frames at its mailbox and its age does not grow; the others
    keep the tap's rate."""
    period = 1.0 / FPS

    def slow_factory(width, height, fps, bitrate_kbit_s, backend=None):
        enc = make_encoder(width, height, fps, bitrate_kbit_s, backend=backend)
        if (height, width) != (180, 320):
            return enc  # only the 1280x720 cameras are slowed
        real = enc.encode

        def encode(frame, *, force_keyframe=False):
            time.sleep(10 * period)
            return real(frame, force_keyframe=force_keyframe)

        enc.encode = encode  # type: ignore[method-assign]
        return enc

    p = LivePipeline(
        fps=FPS,
        encoder_factory=slow_factory,
        encoder_backend=backend,
        device="cuda" if backend == "nvenc" else "cpu",
    )
    p.start()
    try:
        sub = p.subscribe()
        videos, _, t0, t1 = _collect(sub, 3.0)
    finally:
        p.stop()
    written = tap.written_between(t0, t1)
    for cam in ("front", "left_wrist"):
        assert len(videos[cam]) >= 0.75 * written, (cam, len(videos[cam]), written)
    for cam in ("top", "right_wrist"):
        slow = videos[cam]
        assert len(slow) <= written / 8, (cam, len(slow), written)
        assert p.snapshot()[cam]["dropped"] > 0, cam
        ages = _ages_ms(slow)
        first, last = ages[: len(ages) // 3], ages[-len(ages) // 3 :]
        assert statistics.median(last) <= statistics.median(first) + period * 1000.0, (first, last)


def test_an_overlay_never_delays_the_picture(tap, pipeline):
    """An overlay produced ten times slower than the frames: the cadence is
    the tap's, every frame carries the newest overlay there was, and the lag
    is reported in cycles."""
    published: list[tuple[float, int]] = []
    stop = threading.Event()

    def produce():
        h, w, _ = CAMERAS["front"]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[h // 2 :, :, 1] = 255
        rgba[h // 2 :, :, 3] = 160
        while not stop.is_set():
            time.sleep(10.0 / FPS)
            cycle = tap.cycles_written
            pipeline.set_overlay("front", rgba, cycle)
            published.append((time.time(), cycle))

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        sub = pipeline.subscribe()
        videos, _, t0, t1 = _collect(sub, 2.0)
    finally:
        stop.set()
        producer.join(timeout=5.0)
    written = tap.written_between(t0, t1)
    front = videos["front"]
    assert len(front) >= 0.75 * written, (len(front), written)
    with_overlay = [s for s in front if s.overlay_cycle is not None]
    assert len(with_overlay) >= len(front) // 2
    lags = [s.cycle - s.overlay_cycle for s in with_overlay]
    assert min(lags) >= 0
    assert max(lags) >= 5, "the injected delay should show as lag in cycles"

    def newest_at(t: float) -> int | None:
        return max((c for pt, c in published if pt <= t), default=None)

    for s in with_overlay:
        # The overlay is read before the blend and the encode; one published
        # while those ran is legitimately missed, so what was newest at any
        # instant in the moments before the frame left the encoder counts.
        allowed = {newest_at(s.encoded_ts), newest_at(s.encoded_ts - 0.1)}
        assert s.overlay_cycle in allowed, (s.overlay_cycle, allowed)
    assert all(s.overlay_cycle is None for s in videos["top"]), "no overlay was set for other cameras"


def test_the_overlay_is_in_the_pixels(tap, pipeline):
    h, w, _ = CAMERAS["front"]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = 255
    pipeline.set_overlay("front", rgba, tap.cycles_written)
    sub = pipeline.subscribe()
    videos, _, _, _ = _collect(sub, 1.0)
    pictures = _decode([s for s in videos["front"] if s.overlay_cycle is not None])
    assert pictures
    last = pictures[-1].astype(np.int32)
    assert last[..., 2].mean() > 200
    assert last[..., 0].mean() < 60


def test_state_and_action_ride_the_stream_per_cycle(tap, pipeline):
    sub = pipeline.subscribe()
    _, messages, t0, t1 = _collect(sub, 2.0)
    written = tap.written_between(t0, t1)
    assert len(messages) >= 0.75 * written, (len(messages), written)
    cycles = [m.cycle for m in messages]
    assert cycles == sorted(cycles)
    assert len(set(cycles)) == len(cycles)
    for m in messages:
        assert m.state["cycle"] == float(m.cycle)
        assert m.capture_ts > 0
        assert m.action is not None
        assert "gripper.pos" in m.action
    assert messages[-1].cycle >= tap.cycles_written - 3


def test_stop_ends_every_thread(tap):
    before = {t.name for t in threading.enumerate()}
    p = LivePipeline(fps=FPS)
    p.start()
    sub = p.subscribe()
    _collect(sub, 0.5)
    p.stop()
    after = {t.name for t in threading.enumerate()}
    assert not (after - before), after - before
    # Closing sets `closed`, so an "or closed" would be true whatever the
    # queue did: the subscription must be closed AND handing out nothing.
    assert sub.closed
    assert sub.take_video("front", timeout=0.05) is None


def test_the_pipeline_reports_its_state(tap, pipeline):
    sub = pipeline.subscribe()
    _collect(sub, 1.0)
    snap = pipeline.snapshot()
    assert set(snap) == set(CAMERAS)
    for cam, s in snap.items():
        assert s["encoded"] > 0, cam
        assert s["size"][1] == 320
        assert s["backend"] in ("nvenc", "libx264")
        assert s["device"] == pipeline.device
        assert s["backend"] == pipeline.encoder_backend
        assert s["bitrate_kbit_s"] == pipeline.bitrate_kbit_s


def test_a_pipeline_knows_when_the_run_it_reads_has_ended(tap, monkeypatch):
    """A run ending unlinks the tap's segments and the next run creates new
    ones under the same names, so a pipeline that keeps its reader sees a
    sequence that never advances again — which on the page is a robot that
    has stopped moving, not a stream that has died. One viewer staying
    connected across the two runs keeps that pipeline alive, because nothing
    else stops it, so the pipeline has to say so itself."""
    from lerobot.gui.live_video import pipeline as module

    monkeypatch.setattr(module, "stream_identity", lambda: 111)
    p = LivePipeline(fps=FPS)
    try:
        assert p.reads_the_current_tap()

        # The run ends: no tap at all.
        monkeypatch.setattr(module, "stream_identity", lambda: None)
        assert not p.reads_the_current_tap()

        # And the next run's, under the same names.
        monkeypatch.setattr(module, "stream_identity", lambda: 222)
        assert not p.reads_the_current_tap()
    finally:
        p.stop()


def test_a_pipeline_built_without_a_tap_identity_is_never_current(tap, monkeypatch):
    """Nothing to compare against is not the same as a match: a pipeline that
    could not read the tap's identity must not claim the run is still its
    own, or it would be reused forever."""
    from lerobot.gui.live_video import pipeline as module

    monkeypatch.setattr(module, "stream_identity", lambda: None)
    p = LivePipeline(fps=FPS)
    try:
        assert not p.reads_the_current_tap()
    finally:
        p.stop()


def test_an_overlay_the_worker_publishes_reaches_the_encoded_frame(tap, monkeypatch):
    """The overlay an operator turned on has to be in the picture they watch.

    The adapter runs in the overlay worker and publishes each camera's RGBA
    into a shared buffer. The GUI process already reads that buffer — it is
    where the JPEG path's overlay PNG comes from — and the pipeline runs in
    that process, so the overlay is one attach away rather than a process
    boundary away. Without this the pipeline blends an overlay nobody sets,
    every frame goes out bare, and Low Bandwidth silently loses a feature
    the JPEG path has.
    """
    from lerobot.overlays import overlay_ipc

    # A namespace of this test's own: the buffer's names are well known, and
    # creating them under the real ones would take the overlay away from a
    # worker running beside this.
    monkeypatch.setattr(overlay_ipc, "_PREFIX", f"lerobot_overlay_t{os.getpid()}_")
    h, w, _ = CAMERAS["front"]
    worker = overlay_ipc.SharedOverlayBuffer(cameras={"front": (h, w)}, model="test", create=True)
    try:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 2] = 255  # blue, opaque: nothing in the tap's pattern is
        rgba[..., 3] = 255
        worker.write_overlay("front", rgba)

        p = LivePipeline(fps=FPS)
        p.start()
        try:
            sub = p.subscribe()
            videos, _, _, _ = _collect(sub, 2.0)
        finally:
            p.stop()

        assert [s for s in videos["front"] if s.overlay_cycle is not None], (
            "no frame carried the overlay the worker published"
        )
        # Decoded from the start of the stream rather than from the first
        # frame that carries an overlay: the pick-up lands a frame or two in,
        # so that subset can begin mid-group, which a decoder cannot enter.
        pictures = _decode(videos["front"])
        assert pictures
        last = pictures[-1].astype(np.int32)
        assert last[..., 2].mean() > 200, "the overlay is not in the pixels"
        assert last[..., 0].mean() < 60
        # A camera the worker never produced is left alone.
        assert all(s.overlay_cycle is None for s in videos["top"])
    finally:
        worker.cleanup()


def test_a_camera_that_cannot_encode_says_so_on_the_channel(tap):
    """A failure the server keeps hitting has to reach the operator.

    Encoding is the one stage with nothing downstream to notice it failed:
    the frames simply stop. The tap loop and its messages keep running, so
    the channel that carries the readouts is what carries the bad news —
    otherwise the page sits at connecting for the rest of the run and a
    stream that produces nothing looks exactly like a run that has not
    started.
    """

    real_factory = make_encoder

    def fails_once_running(*args, **kwargs):
        inner = real_factory(*args, **kwargs)
        warmed = []

        class LosesItsSession:
            backend = inner.backend

            def encode(self, frame, *, force_keyframe=False):
                # The warm-up has to succeed: an encoder that never worked is
                # refused when the pipeline is built, and what this is about
                # is the session lost once frames are flowing.
                if len(warmed) < 2:
                    warmed.append(1)
                    return inner.encode(frame, force_keyframe=force_keyframe)
                raise RuntimeError("no encoder session")

            def close(self):
                inner.close()

        return LosesItsSession()

    p = LivePipeline(fps=FPS, encoder_factory=fails_once_running)
    p.start()
    try:
        sub = p.subscribe()
        _, messages, _, _ = _collect(sub, 2.0)
    finally:
        p.stop()

    assert messages, "the channel stopped too"
    failing = [m for m in messages if m.failing]
    assert failing, "no message said a camera had stopped encoding"
    last = failing[-1]
    assert set(last.failing) == set(p.cameras), last.failing
    assert "no encoder session" in last.failing[p.cameras[0]]


def test_a_camera_that_encodes_reports_nothing(tap, pipeline):
    """The complement: the field is empty while the stream is working, so a
    page that reads it does not show a failure for every healthy run."""
    sub = pipeline.subscribe()
    videos, messages, _, _ = _collect(sub, 1.5)
    assert videos["front"], "the stream was not working, so this proves nothing"
    assert messages
    assert all(not m.failing for m in messages)


def test_the_saliency_adapters_overlay_reaches_the_encoded_frame(tap, monkeypatch):
    """The other adapter, end to end, with the policy stood in for.

    Policy saliency runs no model of its own: the policy process publishes a
    per-camera grid for the action it just took, and the adapter colourises
    the newest one. So what this stands in for is the policy's write, and
    what it exercises is the real adapter, the real overlay buffer, and the
    blend into the encoded picture.

    Not covered: the policy publishing a grid at all, which needs a
    checkpoint and a run.
    """
    pytest.importorskip("cv2")
    from lerobot.overlays import aux_ipc, overlay_ipc
    from lerobot.overlays.adapters import build_adapter

    monkeypatch.setattr(aux_ipc, "_PREFIX", f"lerobot_aux_t{os.getpid()}_")
    monkeypatch.setattr(overlay_ipc, "_PREFIX", f"lerobot_overlay_t{os.getpid()}_")
    camera = "front"
    h, w, _ = CAMERAS[camera]
    gh, gw = 12, 16

    policy = aux_ipc.SharedAuxBuffer(cameras={camera: (gh, gw)}, model="test", create=True)
    worker = overlay_ipc.SharedOverlayBuffer(cameras={camera: (h, w)}, model="test", create=True)
    try:
        # What the policy publishes: attention on the right-hand side only.
        grid = np.zeros((gh, gw), dtype=np.float32)
        grid[:, gw // 2 :] = 1.0
        policy.write_saliency(camera, grid)

        adapter = build_adapter("policy_saliency", device="cpu")
        adapter.set_camera(camera)
        rgba = adapter.infer(np.zeros((h, w, 3), dtype=np.uint8))
        assert rgba.shape == (h, w, 4)
        assert rgba[..., 3].max() > 0, "the adapter drew nothing from the published grid"
        worker.write_overlay(camera, rgba)

        p = LivePipeline(fps=FPS)
        p.start()
        try:
            sub = p.subscribe()
            videos, _, _, _ = _collect(sub, 2.0)
        finally:
            p.stop()

        assert [s for s in videos[camera] if s.overlay_cycle is not None]
        with_overlay = _decode(videos[camera])[-1].astype(np.int32)
    finally:
        worker.cleanup()
        policy.cleanup()

    # The same run with nothing published, so what is compared is the overlay
    # and not the two halves of a picture that differ on their own.
    bare = LivePipeline(fps=FPS)
    bare.start()
    try:
        videos, _, _, _ = _collect(bare.subscribe(), 2.0)
    finally:
        bare.stop()
    without_overlay = _decode(videos[camera])[-1].astype(np.int32)

    half = with_overlay.shape[1] // 2
    changed = np.abs(with_overlay[:, half:] - without_overlay[:, half:]).mean()
    untouched = np.abs(with_overlay[:, :half] - without_overlay[:, :half]).mean()
    assert changed > 10, f"the heatmap is not in the pixels where the policy looked ({changed})"
    assert changed > 3 * untouched, (
        f"the half the policy ignored changed as much as the half it attended to ({untouched} vs {changed})"
    )


def test_the_stages_go_to_the_gpu_even_when_the_encoder_cannot(tap, monkeypatch):
    """Where the pictures are processed and where they are encoded are two
    questions, and tying them together answered the first one wrongly.

    A host with a GPU but without the hardware encoder's library ran upload,
    resize and blend on CPU tensors for every camera at 30 fps, because the
    device was chosen from the encoder's availability. The software encoder
    copies its frame back itself, so there was never anything stopping the
    stages from running where the tensors can go.
    """
    from lerobot.gui.live_video import pipeline as module

    monkeypatch.setattr(module, "available_backends", lambda: ["libx264"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    p = LivePipeline(fps=FPS)
    try:
        assert p.device == "cuda", "the stages were sent to the CPU by the encoder's absence"
        assert p.encoder_backend == "libx264", "and the encoder still has to be the software one"
    finally:
        p.stop()


def test_without_a_gpu_everything_is_on_the_cpu(tap, monkeypatch):
    """The complement, so the rule above cannot be 'always cuda'."""
    from lerobot.gui.live_video import pipeline as module

    monkeypatch.setattr(module, "available_backends", lambda: ["libx264"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    p = LivePipeline(fps=FPS)
    try:
        assert p.device == "cpu"
        assert p.encoder_backend == "libx264"
    finally:
        p.stop()


def _sample(data: bytes, cycle: int) -> EncodedSample:
    """A keyframe, filled in so the queue has something real to hold."""
    return EncodedSample(
        camera="front",
        data=data,
        keyframe=True,
        cycle=cycle,
        capture_ts=0.0,
        capture_source=CaptureSource.WRITE,
        overlay_cycle=None,
        encoded_ts=0.0,
    )


def _message(cycle: int) -> CycleMessage:
    return CycleMessage(
        cycle=cycle,
        capture_ts=0.0,
        state={},
        action=None,
        action_cycle=None,
        overlay_cycles={},
        failing={},
    )


class TestAClosedQueueHandsOutNothing:
    """`test_stop_ends_every_thread` states this contract but can only catch a
    breach when a frame happens to be queued at the moment stop lands. It was
    breached: both queues kept their buffer through close and `take` returned
    whatever was in it, so the assertion failed on about half of the runs here
    and on nearly every run of the shared CI host, where stopping is slower
    relative to the encode. These drive the queues directly, so the contract is
    pinned by construction rather than by timing.
    """

    def test_a_video_frame_queued_before_the_close_is_not_served_after_it(self):
        from lerobot.gui.live_video.pipeline import _VideoQueue

        q = _VideoQueue(depth=4)
        q.push(_sample(b"\x00", 1))
        assert q.take(0.05) is not None, "the queue never accepted the frame; the test proves nothing"

        q.push(_sample(b"\x01", 2))
        q.close()
        assert q.take(0.05) is None, "a closed queue served the frame it was holding"

    def test_a_message_queued_before_the_close_is_not_served_after_it(self):
        from lerobot.gui.live_video.pipeline import _MessageQueue

        q = _MessageQueue(depth=4)
        q.push(_message(1))
        assert q.take(0.05) is not None, "the queue never accepted the message; the test proves nothing"

        q.push(_message(2))
        q.close()
        assert q.take(0.05) is None, "a closed queue served the message it was holding"

    def test_closing_releases_what_was_buffered(self):
        """The same defect seen as memory: a subscription nobody reads again
        held its frames for as long as anything held the subscription."""
        from lerobot.gui.live_video.pipeline import _VideoQueue

        q = _VideoQueue(depth=4)
        q.push(_sample(b"\x00" * 1024, 1))
        assert q._items, "nothing was buffered; the test proves nothing"
        q.close()
        assert not q._items, "the closed queue is still holding the frames"
