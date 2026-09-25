"""The stream on a real recording, not on a pattern.

Bytes follow content, so what the profile costs and how the encoder behaves
can only be answered by real footage: a recorded SO-101 picking and placing,
played through the tap at its own rate. These are the numbers R3 is about,
and the cases where the encoder's rate control has a scene to work on rather
than a gradient.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict

import av
import numpy as np
import pytest

import lerobot.robots.obs_stream as obs_stream
from lerobot.gui.link_class import CLASS_LINK
from lerobot.gui.live_video.encoder import available_backends
from lerobot.gui.live_video.pipeline import PROFILE_WIDTH, LivePipeline
from tests.gui.live_video_fixtures import footage_tap


@pytest.fixture(scope="module")
def _footage_available():
    from tests.gui.live_video_fixtures import FOOTAGE_ROOT

    if not (FOOTAGE_ROOT / "meta" / "info.json").exists():
        pytest.skip("the recording is not in the local cache")


@pytest.fixture
def tap(_footage_available):
    t = footage_tap(frames=300)
    t.start()
    yield t
    t.stop()


@pytest.fixture(params=available_backends())
def pipeline(tap, request):
    p = LivePipeline(
        fps=30,
        encoder_backend=request.param,
        device="cuda" if request.param == "nvenc" else "cpu",
    )
    p.start()
    yield p
    p.stop()


def _collect(sub, cameras, seconds):
    videos = defaultdict(list)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for cam in cameras:
            while (s := sub.take_video(cam, timeout=0)) is not None:
                videos[cam].append(s)
        time.sleep(0.002)
    return videos, t0, time.time()


def test_the_recording_reaches_the_tap_as_a_run_would_write_it(tap):
    assert sorted(tap.cameras) == ["side", "up"]
    reader = obs_stream.ObservationStreamReader()
    try:
        for cam in tap.cameras:
            result = reader.read_image_stamped(cam)
            assert result is not None, cam
            frame, stamp = result
            assert frame.shape == (480, 640, 3)
            assert frame.std() > 10, "a flat frame is not footage"
            assert stamp.cycle > 0
        obs, _ = reader.read_obs_stamped()
        assert "shoulder_pan.pos" in obs
    finally:
        reader.close()


def test_the_profile_fits_the_budget_on_real_footage(tap, pipeline):
    """R3 on content that behaves like content: what the cameras cost
    together stays inside the share of the class link the pictures may use."""
    sub = pipeline.subscribe()
    _collect(sub, pipeline.cameras, 1.5)  # the rate control settles
    videos, t0, t1 = _collect(sub, pipeline.cameras, 4.0)
    total_kbit_s = sum(len(s.data) for v in videos.values() for s in v) * 8 / (t1 - t0) / 1000
    per_camera = {cam: round(sum(len(s.data) for s in v) * 8 / (t1 - t0) / 1000) for cam, v in videos.items()}
    print(f"real footage at {PROFILE_WIDTH} wide: {per_camera} kbit/s, {total_kbit_s:.0f} together")
    assert total_kbit_s <= CLASS_LINK.stream_budget_kbit_s, (per_camera, total_kbit_s)
    for cam, kbit_s in per_camera.items():
        assert kbit_s <= pipeline.bitrate_kbit_s * 1.25, (cam, kbit_s, pipeline.bitrate_kbit_s)


def test_every_frame_of_the_recording_decodes(tap, pipeline):
    sub = pipeline.subscribe()
    videos, _, _ = _collect(sub, pipeline.cameras, 2.5)
    for cam, samples in videos.items():
        assert len(samples) > 30, (cam, len(samples))
        codec = av.CodecContext.create("h264", "r")
        pictures = []
        for s in samples:
            pictures += [f.to_ndarray(format="rgb24") for f in codec.decode(av.Packet(s.data))]
        pictures += [f.to_ndarray(format="rgb24") for f in codec.decode(None)]
        assert len(pictures) == len(samples), cam
        # 640×480 at the profile's width keeps its aspect ratio.
        assert pictures[0].shape == (240, PROFILE_WIDTH, 3), (cam, pictures[0].shape)
        assert all(p.std() > 5 for p in pictures), "a decoded picture was flat"


def test_the_picture_is_the_one_that_was_recorded(tap, pipeline):
    """End to end on real content: what comes out of the encoder is the
    frame that went into the tap, at the profile's size."""
    sub = pipeline.subscribe()
    videos, _, _ = _collect(sub, pipeline.cameras, 2.0)
    camera = pipeline.cameras[0]
    samples = videos[camera]
    assert samples
    codec = av.CodecContext.create("h264", "r")
    decoded = []
    for s in samples:
        decoded += [(s, f.to_ndarray(format="rgb24")) for f in codec.decode(av.Packet(s.data))]
    assert decoded
    sample, picture = decoded[len(decoded) // 2]
    source = tap._frames[(sample.cycle - 1) % len(tap._frames)][camera]
    import cv2

    want = cv2.resize(source, (picture.shape[1], picture.shape[0]), interpolation=cv2.INTER_AREA)
    # A lossy encode at this bitrate, so agreement is a correlation rather
    # than equality; a different frame would not correlate at all.
    a = picture.astype(np.float64).ravel()
    b = want.astype(np.float64).ravel()
    correlation = float(np.corrcoef(a, b)[0, 1])
    assert correlation > 0.9, correlation


def test_the_ages_are_what_the_pipeline_costs_on_real_frames(tap, pipeline):
    sub = pipeline.subscribe()
    videos, _, _ = _collect(sub, pipeline.cameras, 3.0)
    for cam, samples in videos.items():
        ages = [(s.encoded_ts - s.capture_ts) * 1000 for s in samples]
        median = statistics.median(ages)
        print(f"{cam}: tap to encoded {median:.1f} ms median, {max(ages):.1f} worst")
        # A frame period is the bound that matters: anything slower and the
        # pipeline is adding a frame of its own to the age.
        assert median < 1000 / 30, (cam, median)
