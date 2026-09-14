"""The encoder stage: one access unit out for every frame in, nothing held.

Two backends behind one interface, and the one place in the pipeline where a
backend is chosen: NVENC through PyNvVideoCodec on a host with an NVIDIA GPU,
libx264 through PyAV elsewhere. Both take an HWC uint8 RGB tensor on their
device and return raw H.264 in Annex B form, one access unit per call, with a
keyframe every second, a keyframe on request, and nothing held back.

Preconditions: even width and height; every frame at the size the encoder was
made for, on its device.
"""

from __future__ import annotations

import contextlib
import importlib.util
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

import numpy as np
import torch

NAL_IDR = 5
NAL_SPS = 7
NAL_PPS = 8


@dataclass(frozen=True)
class EncodedFrame:
    data: bytes
    #: An IDR picture with its parameter sets: a decoder can start here.
    keyframe: bool


class Encoder(Protocol):
    backend: str

    def encode(self, frame: torch.Tensor, *, force_keyframe: bool = False) -> EncodedFrame: ...

    def close(self) -> None: ...


def nal_types(au: bytes) -> list[int]:
    """The NAL unit types of an Annex B access unit, in order."""
    types: list[int] = []
    i = au.find(b"\x00\x00\x01")
    while i != -1 and i + 3 < len(au):
        types.append(au[i + 3] & 0x1F)
        i = au.find(b"\x00\x00\x01", i + 3)
    return types


def _is_keyframe(au: bytes) -> bool:
    return NAL_IDR in nal_types(au)


def available_backends() -> list[str]:
    """The backends this host can run, best first."""
    out: list[str] = []
    if torch.cuda.is_available() and importlib.util.find_spec("PyNvVideoCodec") is not None:
        out.append("nvenc")
    out.append("libx264")
    return out


def make_encoder(
    width: int, height: int, fps: int, bitrate_kbit_s: int, backend: str | None = None
) -> Encoder:
    if width % 2 or height % 2 or width < 2 or height < 2:
        raise ValueError(f"the encoder needs even sizes, got {width}x{height}")
    if backend is None:
        backend = available_backends()[0]
    if backend == "nvenc":
        return _NvencEncoder(width, height, fps, bitrate_kbit_s)
    if backend == "libx264":
        return _X264Encoder(width, height, fps, bitrate_kbit_s)
    raise ValueError(f"unknown encoder backend {backend!r}; this host has {available_backends()}")


class _X264Encoder:
    """libx264 through PyAV, with the Data tab stream's settings and raw
    access units out in place of its container."""

    backend = "libx264"

    def __init__(self, width: int, height: int, fps: int, bitrate_kbit_s: int) -> None:
        import av

        ctx = av.CodecContext.create("libx264", "w")
        ctx.width = width
        ctx.height = height
        ctx.pix_fmt = "yuv420p"
        ctx.framerate = Fraction(fps, 1)
        ctx.time_base = Fraction(1, fps)
        ctx.bit_rate = bitrate_kbit_s * 1000
        ctx.gop_size = fps
        ctx.max_b_frames = 0
        ctx.thread_count = 1
        ctx.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "level": "3.0",
            "maxrate": f"{bitrate_kbit_s}k",
            "bufsize": f"{max(128, bitrate_kbit_s // 5)}k",
            # A forced keyframe is an IDR, and every IDR repeats the parameter
            # sets, so a viewer can start at any keyframe.
            "forced-idr": "1",
            "x264-params": f"keyint={fps}:min-keyint={fps}:scenecut=0:bframes=0:repeat-headers=1",
        }
        ctx.open()
        self._av = av
        self._ctx = ctx
        self._size = (height, width, 3)
        self._pts = 0

    def encode(self, frame: torch.Tensor, *, force_keyframe: bool = False) -> EncodedFrame:
        assert tuple(frame.shape) == self._size, (tuple(frame.shape), self._size)
        arr = np.ascontiguousarray(frame.cpu().numpy())
        vf = self._av.VideoFrame.from_ndarray(arr, format="rgb24")
        vf.pts = self._pts
        vf.time_base = self._ctx.time_base
        self._pts += 1
        if force_keyframe:
            vf.pict_type = self._av.video.frame.PictureType.I
        data = b"".join(bytes(p) for p in self._ctx.encode(vf))
        assert data, "libx264 held a frame back"
        return EncodedFrame(data, _is_keyframe(data))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._ctx.close()


class _NvencEncoder:
    """NVENC through PyNvVideoCodec, flushed after every submission so the
    frame's bitstream comes back from its own call."""

    backend = "nvenc"

    _FORCE_IDR = 2
    _OUTPUT_SPS_PPS = 4

    def __init__(self, width: int, height: int, fps: int, bitrate_kbit_s: int) -> None:
        import PyNvVideoCodec

        # The formats are word-ordered names: ABGR is R, G, B, A in memory,
        # which is the frame's own channel order with an alpha appended.
        self._enc = PyNvVideoCodec.CreateEncoder(
            width,
            height,
            "ABGR",
            False,
            codec="h264",
            preset="P1",
            tuning_info="ultra_low_latency",
            rc="cbr",
            bitrate=f"{bitrate_kbit_s}k",
            maxbitrate=f"{bitrate_kbit_s}k",
            fps=str(fps),
            gop=str(fps),
            bf="0",
        )
        self._alpha = torch.full((height, width, 1), 255, dtype=torch.uint8, device="cuda")
        self._size = (height, width, 3)

    def encode(self, frame: torch.Tensor, *, force_keyframe: bool = False) -> EncodedFrame:
        assert frame.device.type == "cuda", frame.device
        assert tuple(frame.shape) == self._size, (tuple(frame.shape), self._size)
        rgba = torch.cat([frame, self._alpha], 2).contiguous()
        # The encoder copies the buffer on its own stream, so every kernel
        # that produced the frame must have finished first; otherwise it reads
        # whatever the allocator's reused block held before.
        torch.cuda.current_stream(frame.device).synchronize()
        flags = (self._FORCE_IDR | self._OUTPUT_SPS_PPS) if force_keyframe else 0
        packets = list(self._enc.Encode(rgba, flags))
        packets += list(self._enc.EndEncode())
        data = b"".join(bytes(p["data"]) if isinstance(p, dict) else bytes(p) for p in packets)
        assert data, "NVENC held a frame back"
        return EncodedFrame(data, _is_keyframe(data))

    def close(self) -> None:
        self._enc = None
