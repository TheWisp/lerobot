"""The encoder stage: one access unit out for every frame in, nothing held.

Runs against every backend the host can run: NVENC through PyNvVideoCodec
when there is a GPU, libx264 through PyAV everywhere. The stream is handed
to a real decoder to check it, not compared to bytes we wrote ourselves.
"""

import math

import av
import numpy as np
import pytest
import torch

from lerobot.gui.live_video.encoder import EncodedFrame, available_backends, make_encoder, nal_types
from lerobot.gui.live_video.stages import resize_to_width

BACKENDS = available_backends()
W, H, FPS = 320, 200, 30
NON_IDR, IDR, SPS, PPS = 1, 5, 7, 8


def _device_for(backend: str) -> str:
    return "cuda" if backend == "nvenc" else "cpu"


def _scene(i: int, device: str, h: int = H, w: int = W) -> torch.Tensor:
    """A smooth gradient with a box that moves: compressible, with motion."""
    y = torch.linspace(0, 255, h).view(h, 1, 1)
    x = torch.linspace(0, 255, w).view(1, w, 1)
    f = torch.cat([y.expand(h, w, 1), x.expand(h, w, 1), ((y + x) / 2).expand(h, w, 1)], 2).clone()
    x0 = (i * 3) % (w - 40)
    f[60:120, x0 : x0 + 40] = torch.tensor([255.0, 40.0, 40.0])
    return f.round().to(torch.uint8).to(device)


def _noise(i: int, device: str, h: int = H, w: int = W) -> torch.Tensor:
    """Noise that scrolls: as many bytes as the rate control will allow."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return torch.from_numpy(np.roll(base, 7 * i, axis=1)).to(device)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


def _decode(aus: list[EncodedFrame]) -> list[np.ndarray]:
    """Each access unit is one packet, as the transport will hand them over."""
    codec = av.CodecContext.create("h264", "r")
    pictures: list[np.ndarray] = []
    for au in aus:
        pictures += [fr.to_ndarray(format="rgb24") for fr in codec.decode(av.Packet(au.data))]
    pictures += [fr.to_ndarray(format="rgb24") for fr in codec.decode(None)]
    return pictures


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


@pytest.fixture
def encoder(backend):
    enc = make_encoder(W, H, FPS, 300, backend=backend)
    yield enc
    enc.close()


def test_at_least_one_backend_runs_here():
    assert BACKENDS
    assert "libx264" in BACKENDS


def test_every_frame_comes_back_from_its_own_call(encoder, backend):
    dev = _device_for(backend)
    for i in range(2 * FPS):
        out = encoder.encode(_scene(i, dev))
        assert isinstance(out, EncodedFrame)
        assert len(out.data) > 0, f"frame {i} was held"


def test_the_first_frame_is_a_keyframe_with_its_parameter_sets(encoder, backend):
    dev = _device_for(backend)
    first = encoder.encode(_scene(0, dev))
    assert first.keyframe
    types = nal_types(first.data)
    assert SPS in types
    assert PPS in types
    assert IDR in types
    second = encoder.encode(_scene(1, dev))
    assert not second.keyframe
    assert IDR not in nal_types(second.data)
    assert NON_IDR in nal_types(second.data)


def test_a_keyframe_every_second_and_none_otherwise(encoder, backend):
    dev = _device_for(backend)
    keys = [encoder.encode(_scene(i, dev)).keyframe for i in range(3 * FPS)]
    assert [i for i, k in enumerate(keys) if k] == [0, FPS, 2 * FPS]


def test_a_forced_keyframe_arrives_where_asked_with_its_parameter_sets(encoder, backend):
    dev = _device_for(backend)
    for i in range(10):
        encoder.encode(_scene(i, dev))
    forced = encoder.encode(_scene(10, dev), force_keyframe=True)
    assert forced.keyframe
    types = nal_types(forced.data)
    assert IDR in types
    assert SPS in types
    assert PPS in types
    assert not encoder.encode(_scene(11, dev)).keyframe


def test_a_decoder_reconstructs_what_was_encoded(encoder, backend):
    """Hand the stream to its real consumer: one picture per access unit,
    each close to the source."""
    dev = _device_for(backend)
    frames = [_scene(i, dev) for i in range(FPS)]
    aus = [encoder.encode(f) for f in frames]
    pictures = _decode(aus)
    assert len(pictures) == len(frames)
    for src, pic in zip(frames, pictures, strict=True):
        assert pic.shape == tuple(src.shape)
        assert _psnr(src.cpu().numpy(), pic) > 30.0


def test_the_encoder_reads_the_frame_it_is_given_not_the_memory_it_had(backend):
    """The pipeline's own chain right before the call — a camera-sized frame
    uploaded and resized on the device — each frame a different solid
    colour: the decoded picture must be that frame's colour, not what the
    same memory held a frame earlier while the upload was still in flight."""
    dev = _device_for(backend)
    enc = make_encoder(320, 180, FPS, 300, backend=backend)
    colours = [(i * 37 % 200 + 30, i * 91 % 200 + 30, i * 53 % 200 + 30) for i in range(20)]
    aus = []
    try:
        for c in colours:
            source = np.full((720, 1280, 3), c, dtype=np.uint8)
            frame = resize_to_width(torch.from_numpy(source).to(dev), 320)
            aus.append(enc.encode(frame))
    finally:
        enc.close()
    pictures = _decode(aus)
    assert len(pictures) == len(colours)
    for i, (pic, c) in enumerate(zip(pictures, colours, strict=True)):
        mean = pic.reshape(-1, 3).mean(axis=0)
        assert np.abs(mean - np.array(c)).max() < 12, (i, mean, c, colours[i - 1] if i else None)


def test_a_stream_joined_at_a_forced_keyframe_decodes(encoder, backend):
    """What a late viewer receives starts at the keyframe forced for it."""
    dev = _device_for(backend)
    for i in range(7):
        encoder.encode(_scene(i, dev))
    joined = [encoder.encode(_scene(7, dev), force_keyframe=True)]
    joined += [encoder.encode(_scene(i, dev)) for i in range(8, 20)]
    assert len(_decode(joined)) == len(joined)


def test_the_bitrate_setting_bounds_the_bytes(backend):
    """Moving noise would take many times the budget uncapped; the rate
    control holds the bytes near the setting, and a higher setting spends more."""
    dev = _device_for(backend)

    def bits_per_second(kbit_s: int) -> float:
        enc = make_encoder(W, H, FPS, kbit_s, backend=backend)
        try:
            total = 0
            for i in range(3 * FPS):
                out = enc.encode(_noise(i, dev))
                if i >= FPS:
                    total += len(out.data)
        finally:
            enc.close()
        return total * 8 / 2.0

    low, high = bits_per_second(300), bits_per_second(1200)
    assert low < high
    assert low <= 300_000 * 1.5


def test_nal_types_reads_annex_b_start_codes():
    au = b"\x00\x00\x00\x01\x67abc\x00\x00\x01\x68d\x00\x00\x00\x01\x65efg"
    assert nal_types(au) == [SPS, PPS, IDR]
    assert nal_types(b"") == []


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError):
        make_encoder(W, H, FPS, 300, backend="nope")


def test_odd_sizes_are_refused():
    with pytest.raises(ValueError):
        make_encoder(W + 1, H, FPS, 300, backend="libx264")
