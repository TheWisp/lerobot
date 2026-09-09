"""Windowed playback of a stored episode, prototype (see docs/dataset_playback.md).

Two endpoints per episode:

- ``GET .../episodes/{ep}/bundle`` — everything whose size does not depend on
  what is being looked at: an envelope (min and max per column, a fixed number
  of columns) of every numeric feature, mask presence per label per frame,
  task per frame, and the cameras with their sizes. One gzipped JSON. The
  per-frame numeric values ride in the windows, so the bundle stays small for
  an hour-long episode (per-frame values made it 31 MB gzipped, 5 s to build).
- ``GET .../episodes/{ep}/window?start=F&len=S&rung=R[&codec=&rc=&q=&preset=]``
  — the next ``S`` seconds from frame ``F`` for every camera: each camera's
  frames as raw H.264 (Annex B, one keyframe first, no B-frames) or as an AV1
  OBU stream, under the requested rate control and preset, each masked camera's mask runs
  for those frames, and every numeric feature's rows for those frames, in one
  binary body. Built on demand by one ffmpeg per
  camera from the archive, cached on disk under a byte ceiling, least recently
  used out first.

Body layout: 4-byte little-endian header length, the JSON header, then the
parts back to back. The header names each part's byte offset and length, the
per-frame sizes of each H.264 part (so the page can hand the decoder one frame
at a time without parsing NAL units), and the frame range covered.

Rungs: width and bitrate; ``full`` is the archive's own samples, demuxed
from the keyframe at or before the window's first frame and re-wrapped as
the raw stream the page decodes, with the frames before the first one
marked as lead for the page to drop. Every video part carries the
presentation timestamp of each frame and which frames are keyframes, so
the page keys decoded frames by timestamp. Masks are scaled down to the
rung's width; a source narrower than a rung is not upscaled.

Pre: the dataset is open in the app state; ffmpeg with libx264 and the
``h264_metadata`` bitstream filter is on PATH. Post: nothing in the dataset
is written; the cache directory is the only side effect.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response

from lerobot.datasets.video_utils import get_video_bitrate_kbps
from lerobot.gui.api import datasets as datasets_api

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["window-playback"])

_app_state: AppState = None  # type: ignore

# Width and bitrate per rung; 0 width keeps the source. ``full`` uses constant
# quality instead of a bitrate so it is the near-lossless top of the ladder.
RUNGS: dict[str, tuple[int, str | None]] = {
    "160": (160, "150k"),
    "320": (320, "300k"),
    "640": (640, "800k"),
    "1280": (1280, "1500k"),
    "full": (0, None),
}
WINDOW_LENGTHS = (0.5, 1.0, 2.0, 4.0)
# Bump when the body layout or a part's content changes, so old cache entries are not served as new ones.
FORMAT_VERSION = 6

# Encoder options a window request may set, each with its allowed values; the
# first is the default. They are part of the cache key. ``rc`` is the rate
# control: ``cbr`` spends the rung's bitrate on every window, ``crf`` holds a
# quality level (``q``, lower is better) under the rung's bitrate as a cap.
ENCODER_OPTIONS: dict[str, tuple[str, ...]] = {
    "codec": ("h264", "av1"),
    "rc": ("cbr", "crf"),
    "preset": ("veryfast", "medium", "8", "10", "6"),  # libx264 names; SVT-AV1 numbers
}
Q_RANGE = (10, 50)
CACHE_CEILING_BYTES = int(os.environ.get("LEROBOT_WINDOW_CACHE_BYTES", 2 * 1024**3))

# Builders run one ffmpeg per camera; four in flight covers one episode's
# cameras at once. Separate from the app's single decode worker on purpose.
_build_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gui-window")


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


def cache_dir() -> Path:
    d = Path(
        os.environ.get("LEROBOT_WINDOW_CACHE_DIR")
        or Path.home() / ".cache" / "huggingface" / "lerobot" / "gui" / "windows"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dataset(dataset_id: str):
    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return dataset_id, _app_state.datasets[dataset_id]


def _video_cameras(dataset) -> dict[str, dict]:
    return {k: ft for k, ft in dataset.meta.features.items() if ft.get("dtype") == "video"}


#: What a window carries for masks: nothing, the runs at the rung's width, or
#: the saved-mask recipe rendered into the pixels (what a policy is fed).
MASK_MODES = ("none", "runs", "composited")

#: Archive codecs the page's decoder takes as they are; anything else has no ``full`` rung.
REMUX_CODECS = ("av1", "h264")

#: Archive bitrate per video file, size over duration, computed once per file.
_bitrate_cache: dict[Path, int] = {}


def _archive_kbps(path: Path) -> int:
    if path not in _bitrate_cache:
        _bitrate_cache[path] = get_video_bitrate_kbps(path)
    return _bitrate_cache[path]


#: Columns of the bundle's envelope of each numeric feature. An overview canvas
#: is a few thousand pixels wide at most; past this, min and max per column
#: draw the same band as the samples would.
ENVELOPE_COLUMNS = 1024


def dataset_generation(dataset) -> int:
    """A number that changes whenever the dataset is written.

    A window is a function of pixels and rows an edit can rewrite: a trim
    renumbers frames, a delete renumbers episodes, a mask save changes what a
    composite shows. The server drops its own cached windows on an edit, but a
    window URL is cacheable in the browser for an hour, and without this the
    same URL means something different afterwards -- the operator trims an
    episode and keeps watching the frames that were cut.

    The dataset's metadata is rewritten by every one of those paths, so its
    modification time is the generation. It survives a server restart, which an
    in-process counter would not.
    """
    try:
        return Path(dataset.root).joinpath("meta", "info.json").stat().st_mtime_ns
    except OSError:
        return 0


def _numeric_features(dataset) -> list[str]:
    """Every feature that is a number per frame: not an image, not a string, not a mask."""
    return [
        name
        for name, ft in dataset.meta.features.items()
        if ft.get("dtype") not in ("image", "video", "string") and not ft.get("mask_encoding")
    ]


def _envelope(arr: np.ndarray, columns: int = ENVELOPE_COLUMNS) -> dict[str, Any]:
    """Min and max per column of ``arr`` (frames x D) over at most ``columns``
    equal spans of frames. An episode with no more frames than columns is sent
    exactly, one frame per column, with ``hi`` omitted."""
    n = arr.shape[0]
    if n <= columns:
        return {"columns": n, "lo": arr.round(4).tolist(), "hi": None}
    edges = np.linspace(0, n, columns + 1).astype(int)
    spans = list(zip(edges[:-1], edges[1:], strict=True))
    lo = np.stack([arr[a:b].min(axis=0) for a, b in spans])
    hi = np.stack([arr[a:b].max(axis=0) for a, b in spans])
    return {"columns": columns, "lo": lo.round(4).tolist(), "hi": hi.round(4).tolist()}


# ---------------------------------------------------------------- bundle


@router.get("/{dataset_id:path}/episodes/{episode_idx}/bundle")
async def get_bundle(dataset_id: str, episode_idx: int) -> Response:
    t0 = time.perf_counter()
    dataset_id, dataset = _dataset(dataset_id)
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")
    # Per frame only what is one small value per frame: mask presence bitsets
    # and the task string, read from the arrow table's rows of this episode.
    # Reading them through the feature-series endpoint cost 770 to 800 ms per
    # call on the rig's labelled dataset, a pandas read of the mask columns of
    # the whole parquet file, on the path to the first picture. Numeric
    # features come as envelopes.
    length = dataset.episode_rows(episode_idx)[1]
    mask_feats = datasets_api._mask_features(dataset)
    series: dict[str, list[Any]] = {}
    for key in mask_feats:
        cells = dataset.episode_column(key, episode_idx)
        series[key] = [datasets_api._mask_presence_bits(v) for v in cells]
        series[f"{key}{datasets_api.MASK_DISABLED_SUFFIX}"] = [
            datasets_api._mask_disabled_bits(v) for v in cells
        ]
    if "task_index" in dataset.meta.features and dataset.meta.tasks is not None:
        names = list(dataset.meta.tasks.index)
        series["task"] = [names[int(i)] for i in dataset.episode_column("task_index", episode_idx)[:, 0]]
    envelope = {
        name: _envelope(dataset.episode_column(name, episode_idx)) for name in _numeric_features(dataset)
    }
    masks = {
        key: {"labels": ft.get("mask_labels", []), "size": ft.get("mask_size")}
        for key, ft in datasets_api._mask_features(dataset).items()
    }
    cameras = {}
    for key, ft in _video_cameras(dataset).items():
        info = ft.get("info") or {}
        rel, _f, _t = dataset.meta.get_episode_video_span(episode_idx, key)
        cameras[key] = {
            "width": info.get("video.width"),
            "height": info.get("video.height"),
            "codec": info.get("video.codec"),
            # What the full rung costs for this camera: the archive's own bitrate.
            "kbps": _archive_kbps(Path(dataset.root) / rel),
        }
    # The full rung exists only when the page can decode every camera's archive.
    full_ok = bool(cameras) and all(c["codec"] in REMUX_CODECS for c in cameras.values())
    rungs = [r for r in RUNGS if r != "full" or full_ok]
    rung_kbps = {name: (int(br[:-1]) if br else None) for name, (_, br) in RUNGS.items()}
    if full_ok:
        rung_kbps["full"] = round(sum(c["kbps"] for c in cameras.values()) / len(cameras))
    else:
        rung_kbps.pop("full", None)
    body = {
        "episode_index": episode_idx,
        "episodes": int(dataset.meta.total_episodes),  # so the page knows whether a next episode exists
        "length": length,
        "fps": dataset.meta.fps,
        "cameras": cameras,
        "masks": masks,  # presence per frame is in series[<mask key>] as a bitset, from feature-series
        "series": series,
        "envelope": envelope,
        # Changes whenever the dataset is written: the page puts it in every
        # window URL, so an edit cannot be answered from the browser's cache.
        "generation": dataset_generation(dataset),
        "rungs": rungs,
        "encoder_options": {k: list(v) for k, v in ENCODER_OPTIONS.items()},
        # Video bitrate per camera per rung, kbit/s, for the page's automatic choice: nominal caps
        # for the encoded rungs, the archive's mean over cameras for full.
        "rung_kbps": rung_kbps,
        "window_lengths": list(WINDOW_LENGTHS),
        "format_version": FORMAT_VERSION,  # the page puts it in window URLs so the browser cache cannot serve an older layout
    }
    payload = gzip.compress(json.dumps(body, separators=(",", ":")).encode(), compresslevel=6)
    logger.info(
        "window-playback bundle %s ep=%d %d B gz %.0f ms",
        dataset_id,
        episode_idx,
        len(payload),
        (time.perf_counter() - t0) * 1000,
    )
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------- window


def _split_access_units(annexb: bytes) -> list[int]:
    """Byte length of each access unit in an Annex B stream that carries an
    access-unit delimiter (NAL type 9) in front of every frame.

    The page hands its decoder one frame at a time, so these sizes are the
    frame boundaries: a wrong split is a decoder error or a frame silently
    dropped. Post: one size per delimiter, each positive, and together they
    cover the stream from the first delimiter to its end.

    Raises:
        RuntimeError: if the stream carries no access-unit delimiter, which
            means the encoder was not asked to insert them.
    """
    starts: list[int] = []
    i = 0
    n = len(annexb)
    while True:
        j = annexb.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        # A 4-byte start code has a leading zero; the AU starts at that zero.
        s = j - 1 if j > 0 and annexb[j - 1] == 0 else j
        if j + 3 < n and (annexb[j + 3] & 0x1F) == 9:
            starts.append(s)
        i = j + 3
    if not starts:
        raise RuntimeError("no access-unit delimiters in the encoded stream")
    sizes = [b - a for a, b in zip(starts, starts[1:], strict=False)] + [n - starts[-1]]
    assert len(sizes) == len(starts), (len(sizes), len(starts))
    assert all(size > 0 for size in sizes), sizes
    assert sum(sizes) == n - starts[0], (sum(sizes), n, starts[0])
    return sizes


def _split_temporal_units(obu: bytes) -> list[int]:
    """Byte length of each temporal unit in a low-overhead AV1 OBU stream: a
    frame starts with a temporal delimiter OBU (type 2).

    Post: one size per delimiter, each positive, together covering the stream
    from the first delimiter to its end.

    Raises:
        RuntimeError: if an OBU carries no size field (the stream is not the
            low-overhead format the page decodes), or none is a delimiter.
    """
    starts: list[int] = []
    i, n = 0, len(obu)
    while i < n:
        b0 = obu[i]
        obu_type = (b0 >> 3) & 0xF
        ext = (b0 >> 2) & 1
        has_size = (b0 >> 1) & 1
        j = i + 1 + ext
        if not has_size:
            raise RuntimeError("OBU without a size field")
        size, shift = 0, 0
        while True:
            c = obu[j]
            j += 1
            size |= (c & 0x7F) << shift
            shift += 7
            if not c & 0x80:
                break
        if obu_type == 2:
            starts.append(i)
        i = j + size
    if not starts:
        raise RuntimeError("no temporal delimiters in the AV1 stream")
    sizes = [b - a for a, b in zip(starts, starts[1:], strict=False)] + [n - starts[-1]]
    assert len(sizes) == len(starts), (len(sizes), len(starts))
    assert all(size > 0 for size in sizes), sizes
    assert sum(sizes) == n - starts[0], (sum(sizes), n, starts[0])
    return sizes


def _encode_camera(
    path: Path, t0: float, seconds: float, fps: float, rung: str, enc: dict[str, Any]
) -> tuple[bytes, list[int], float]:
    """One camera's window as raw frames: Annex B H.264 with one AUD per frame,
    or an AV1 OBU stream with one temporal unit per frame.

    Returns (bytes, per-frame sizes, build seconds). The first frame is an IDR,
    the GOP spans the window, no B-frames, so the page can decode from the
    first frame and every window is independent. No zero-latency tuning: it
    is for live encoding, where the next frame is unknown; on a stored
    window it cost SSIM 0.931 against 0.940 at the same bytes.
    """
    width, bitrate = RUNGS[rung]
    assert width and bitrate, "the full rung is a remux, not an encode"
    frames = round(seconds * fps)
    vf, video = _encode_args(rung, enc, frames)
    codec = enc["codec"]
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t0:.6f}", "-i", str(path), "-frames:v", str(frames), "-an",
        *vf, *video, "pipe:1",
    ]  # fmt: skip
    t = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path.name}: {out.stderr.decode(errors='replace')[-400:]}")
    sizes = _split_access_units(out.stdout) if codec == "h264" else _split_temporal_units(out.stdout)
    return out.stdout, sizes, time.perf_counter() - t


def _encode_frames(frames, fps: float, rung: str, enc: dict[str, Any]) -> tuple[bytes, list[int], float]:
    """Like :func:`_encode_camera`, from frames already in memory (HxWx3 uint8,
    an iterable) piped raw into the same encoder. The composited path: the
    recipe is rendered per frame before encoding. At ``full`` the frames are
    encoded at their own size at constant quality 18, since composited
    pixels cannot be the archive's samples."""
    import threading

    it = iter(frames)
    first = next(it)
    h, w = first.shape[:2]
    vf, video = _encode_args(rung, enc, None)
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", f"{fps:g}", "-i", "pipe:0", "-an", *vf, *video, "pipe:1",
    ]  # fmt: skip
    t = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0

    def feed():
        nonlocal n
        try:
            for frame in (first, *it):
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                n += 1
        finally:
            proc.stdin.close()

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    # Not communicate(): it closes stdin under the feeder. Drain both pipes ourselves.
    err_chunks: list[bytes] = []
    drain = threading.Thread(target=lambda: err_chunks.append(proc.stderr.read()), daemon=True)
    drain.start()
    out = proc.stdout.read()
    feeder.join()
    drain.join()
    proc.wait()
    err = b"".join(err_chunks)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on composited frames: {err.decode(errors='replace')[-400:]}")
    sizes = _split_access_units(out) if enc["codec"] == "h264" else _split_temporal_units(out)
    if len(sizes) != n:
        raise RuntimeError(f"encoded {len(sizes)} frames from {n} composited frames")
    return out, sizes, time.perf_counter() - t


def _encode_args(rung: str, enc: dict[str, Any], frames: int | None) -> tuple[list[str], list[str]]:
    """The scale filter and the video arguments for a rung and encoder options.
    ``frames`` sets the GOP to the window (one keyframe first); ``None`` leaves
    the GOP to a large value for a raw-frame input whose count is not known
    ahead. ``full`` (no width, no bitrate) means constant quality 18 at the
    source size, used only for composited frames."""
    assert rung in RUNGS, f"{rung} is not a rung of {list(RUNGS)}"
    assert enc["codec"] in ENCODER_OPTIONS["codec"], enc
    width, bitrate = RUNGS[rung]
    # Never upscale: a 960-wide wrist camera at the 1280 rung stays 960 wide.
    vf = [] if width == 0 else ["-vf", f"scale=w='min(iw,{width})':h=-2"]
    gop = str(frames if frames else 100000)
    codec, rc, q, preset = enc["codec"], enc["rc"], int(enc["q"]), enc["preset"]
    if codec == "h264":
        if bitrate is None:
            rate = ["-crf", "18"]
        elif rc == "cbr":
            rate = ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate]
        else:
            rate = ["-crf", str(q), "-maxrate", bitrate, "-bufsize", str(2 * int(bitrate[:-1])) + "k"]
        video = ["-c:v", "libx264", "-preset", preset, "-profile:v", "main", "-pix_fmt", "yuv420p", *rate,
                 "-g", gop, "-keyint_min", gop, "-sc_threshold", "0", "-bf", "0", "-forced-idr", "1",
                 "-bsf:v", "h264_metadata=aud=insert", "-f", "h264"]  # fmt: skip
    else:
        # SVT-AV1 takes its cap through its own parameter string; no maxrate/bufsize.
        if bitrate is None:
            rate = ["-crf", "18"]
        elif rc == "cbr":
            rate = ["-b:v", bitrate, "-svtav1-params", "rc=1"]
        else:
            rate = ["-crf", str(q), "-svtav1-params", f"mbr={bitrate}"]
        video = [
            "-c:v",
            "libsvtav1",
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            *rate,
            "-g",
            gop,
            "-f",
            "obu",
        ]
    return vf, video


def _composited_frames(
    dataset, episode_idx: int, key: str, spec: dict, mask_key: str, start: int, count: int, fps: float
):
    """The archive's frames of one camera for the window, each with the saved
    recipe rendered in by the library's compositor: the same pixels the
    training reader composites and the still path served."""
    import av

    from lerobot.datasets.mask_compositing import composite_from_store

    rel, from_ts, _to = dataset.meta.get_episode_video_span(episode_idx, key)
    t_from = from_ts + start / fps
    rows = dataset.episode_column(mask_key, episode_idx, start, count)
    n = 0
    with av.open(str(Path(dataset.root) / rel)) as container:
        st = container.streams.video[0]
        container.seek(int(t_from / st.time_base), stream=st, backward=True, any_frame=False)
        for frame in container.decode(st):
            if frame.time is None or frame.time < t_from - 0.5 / fps:
                continue
            rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            cell = rows[n]
            row = cell[0] if isinstance(cell, (list, tuple)) and cell else cell
            yield composite_from_store(rgb, str(row), spec, episode=episode_idx) if row else rgb
            n += 1
            if n == count:
                break
    if n != count:
        raise RuntimeError(f"{key}: decoded {n} frames for compositing, wanted {count}")


def _av1_temporal_unit(sample: bytes) -> bytes:
    """An ISOBMFF AV1 sample is a temporal unit without its delimiter; the raw
    stream the page decodes wants one per unit.

    Pre: ``sample`` is one stored sample. Post: the result starts with a
    temporal delimiter OBU and carries the sample unchanged after it.
    """
    assert sample, "an empty AV1 sample"
    if (sample[0] >> 3) & 0xF == 2:  # OBU_TEMPORAL_DELIMITER already leads
        return sample
    out = b"\x12\x00" + sample
    assert (out[0] >> 3) & 0xF == 2 and out.endswith(sample)
    return out


def _h264_codec_string(annexb: bytes) -> str | None:
    """``avc1.PPCCLL`` from the SPS in an Annex B stream, so the page's
    decoder is configured for the archive's profile, not the encoder's.

    Post: ``None`` when the stream carries no sequence parameter set, else a
    string of the exact form WebCodecs expects: ``avc1.`` and six hex digits
    naming the profile, its constraint flags and the level.
    """
    i = annexb.find(b"\x00\x00\x01")
    while i >= 0:
        nal = i + 3
        if nal + 4 <= len(annexb) and annexb[nal] & 0x1F == 7:
            out = f"avc1.{annexb[nal + 1]:02x}{annexb[nal + 2]:02x}{annexb[nal + 3]:02x}"
            assert len(out) == len("avc1.") + 6, out
            return out
        i = annexb.find(b"\x00\x00\x01", nal)
    return None


def _remux_camera(
    path: Path, t_from: float, count: int, fps: float
) -> tuple[bytes, list[int], list[int], list[int], str, str | None, float]:
    """The archive's own samples for ``count`` frames from ``t_from``, from the
    keyframe at or before it, as the raw stream the page decodes: an AV1 OBU
    stream with a temporal delimiter per unit, or Annex B H.264 with the
    parameter sets in band.

    Returns (bytes, per-frame sizes, per-frame presentation time in µs
    relative to ``t_from``, indices of the keyframes, codec, codec string,
    build seconds). Frames before ``t_from`` have a negative time: the page
    decodes and drops them.
    """
    import av
    from av.bitstream import BitStreamFilterContext

    t = time.perf_counter()
    end = t_from + (count - 0.5) / fps
    chunks: list[bytes] = []
    ts_us: list[int] = []
    keys: list[int] = []
    with av.open(str(path)) as container:
        st = container.streams.video[0]
        codec = st.codec.canonical_name
        if codec not in REMUX_CODECS:
            raise RuntimeError(f"{path.name}: the page does not decode {codec}")
        bsf = BitStreamFilterContext("h264_mp4toannexb", st) if codec == "h264" else None
        container.seek(int(t_from / st.time_base), stream=st, backward=True, any_frame=False)
        for pkt in container.demux(st):
            if pkt.pts is None:
                continue
            t_pkt = float(pkt.pts * st.time_base)
            if t_pkt > end + 1e-6:
                break
            is_key = pkt.is_keyframe  # the filter takes the packet over; read it first
            data = b"".join(bytes(x) for x in bsf.filter(pkt)) if bsf else _av1_temporal_unit(bytes(pkt))
            if is_key:
                keys.append(len(chunks))
            chunks.append(data)
            ts_us.append(round((t_pkt - t_from) * 1e6))
    if not chunks or 0 not in keys:
        raise RuntimeError(f"{path.name}: no keyframe at or before {t_from:.3f} s")
    codec_string = _h264_codec_string(chunks[0]) if codec == "h264" else None
    return (
        b"".join(chunks),
        [len(c) for c in chunks],
        ts_us,
        keys,
        codec,
        codec_string,
        time.perf_counter() - t,
    )


def _mask_part(
    dataset, key: str, episode_idx: int, start: int, count: int, size: tuple[int, int], target_w: int
) -> tuple[bytes, list[int]]:
    """The mask runs for ``count`` frames from frame ``start`` of the episode,
    scaled down to the rung's width when the rung is narrower than the mask.

    Returns the gzipped JSON and the [height, width] the runs are in. A run
    length grows with the outline, so a mask at a quarter of the width is
    about a quarter of the bytes; the page scales the canvas anyway.
    """
    import cv2

    from lerobot.datasets.mask_codec import decode_mask, encode_mask

    h, w = int(size[0]), int(size[1])
    scale = bool(target_w) and target_w < w
    nh, nw = (max(1, round(h * target_w / w)), target_w) if scale else (h, w)
    column = dataset.episode_column(key, episode_idx, start, count)
    frames = []
    for cell in column:
        raw = cell[0] if isinstance(cell, (list, tuple)) else cell
        entries = json.loads(raw) if raw else []
        if scale:
            out = []
            for entry in entries:
                label, counts, *rest = entry
                small = cv2.resize(
                    decode_mask(counts, (h, w)).astype("uint8"), (nw, nh), interpolation=cv2.INTER_NEAREST
                )
                out.append([label, encode_mask(small.astype(bool)), *rest])
            entries = out
        frames.append(entries)
    # Run-length text compresses about three to one; at the low rungs the raw
    # masks outweighed the video (108 kB against 72 kB for half a second).
    body = gzip.compress(json.dumps(frames, separators=(",", ":")).encode(), compresslevel=6)
    return body, [nh, nw]


def _build_window(
    dataset,
    dataset_id: str,
    episode_idx: int,
    start: int,
    seconds: float,
    rung: str,
    enc: dict[str, Any],
    masks: str = "runs",
    specs: dict[str, tuple[dict, str]] | None = None,
) -> tuple[bytes, dict]:
    specs = specs or {}  # camera -> (recipe, its mask column)
    fps = float(dataset.meta.fps)
    length = dataset.episode_rows(episode_idx)[1]
    count = min(round(seconds * fps), length - start)
    cams = _video_cameras(dataset)
    mask_feats = datasets_api._mask_features(dataset)

    width = RUNGS[rung][0]

    def cam_job(key):
        rel, from_ts, _to_ts = dataset.meta.get_episode_video_span(episode_idx, key)
        path = Path(dataset.root) / rel
        if key in specs:
            spec, mask_key = specs[key]
            frames = _composited_frames(dataset, episode_idx, key, spec, mask_key, start, count, fps)
            data, sizes, secs = _encode_frames(frames, fps, rung, enc)
            ts_us = [round(i * 1e6 / fps) for i in range(len(sizes))]
            return key, data, sizes, ts_us, [0], enc["codec"], None, secs
        if rung == "full":
            data, sizes, ts_us, keys, codec, codec_string, secs = _remux_camera(
                path, from_ts + start / fps, count, fps
            )
            return key, data, sizes, ts_us, keys, codec, codec_string, secs
        data, sizes, secs = _encode_camera(path, from_ts + start / fps, count / fps, fps, rung, enc)
        ts_us = [round(i * 1e6 / fps) for i in range(len(sizes))]
        return key, data, sizes, ts_us, [0], enc["codec"], None, secs

    def features_job():
        t_f = time.perf_counter()
        rows = {
            name: dataset.episode_column(name, episode_idx, start, count).round(4).tolist()
            for name in _numeric_features(dataset)
        }
        data = gzip.compress(json.dumps(rows, separators=(",", ":")).encode(), compresslevel=6)
        return data, time.perf_counter() - t_f

    def mask_job(key):
        t_m = time.perf_counter()
        data, out_size = _mask_part(
            dataset, key, episode_idx, start, count, mask_feats[key].get("mask_size") or [0, 0], width
        )
        return key, data, out_size, time.perf_counter() - t_m

    # Cameras and masks build side by side; the masks used to run after the
    # cameras and added 100 to 200 ms to a 2 s window.
    cam_futures = [_build_executor.submit(cam_job, k) for k in cams]
    mask_futures = [_build_executor.submit(mask_job, k) for k in (mask_feats if masks == "runs" else ())]
    features_future = _build_executor.submit(features_job)
    results = [f.result() for f in cam_futures]
    mask_results = [f.result() for f in mask_futures]
    features_data, features_secs = features_future.result()
    parts: list[dict] = []
    blobs: list[bytes] = []
    offset = 0
    builds = {}
    for key, data, sizes, ts_us, keys, codec, codec_string, secs in results:
        # Frames at or after the window's first frame, by timestamp, must be exactly the window's.
        wanted = [round(t * fps / 1e6) for t in ts_us if t >= 0]
        if wanted != list(range(count)):
            raise RuntimeError(
                f"{key}: frames {wanted[:3]}..{wanted[-1:] if wanted else None}, wanted 0..{count - 1}"
            )
        part = {
            "camera": key,
            "kind": "video",
            "codec": codec,
            "offset": offset,
            "length": len(data),
            "frame_sizes": sizes,
            "frame_ts_us": ts_us,
            "key_frames": keys,
            "lead": sum(1 for t in ts_us if t < 0),
        }
        if codec_string:
            part["codec_string"] = codec_string
        parts.append(part)
        blobs.append(data)
        offset += len(data)
        builds[key] = round(secs * 1000)
    for key, data, out_size, secs in mask_results:
        builds[f"masks:{key.split('.')[-1]}"] = round(secs * 1000)
        parts.append(
            {
                "camera": key,
                "kind": "masks",
                "offset": offset,
                "length": len(data),
                "encoding": "gzip",
                "size": out_size,
            }
        )
        blobs.append(data)
        offset += len(data)
    builds["features"] = round(features_secs * 1000)
    parts.append(
        {
            "camera": "",
            "kind": "features",
            "offset": offset,
            "length": len(features_data),
            "encoding": "gzip",
        }
    )
    blobs.append(features_data)
    offset += len(features_data)
    header = {
        "episode_index": episode_idx,
        "first_frame": start,
        "frames": count,
        "fps": fps,
        "rung": rung,
        "seconds": seconds,
        "enc": enc,
        "masks": masks,
        "parts": parts,
        "build_ms": builds,
    }
    hb = json.dumps(header, separators=(",", ":")).encode()
    body = struct.pack("<I", len(hb)) + hb + b"".join(blobs)
    return body, header


def _cache_key(
    dataset_id: str,
    episode_idx: int,
    start: int,
    seconds: float,
    rung: str,
    enc: dict[str, Any],
    masks: str = "runs",
    fingerprint: str = "",
    generation: int = 0,
) -> Path:
    """The cache key of a window. Stored pixels never change, so a raw window
    needs no invalidation. A composited window is keyed by the recipe
    fingerprint of every camera it renders, so an edited recipe is a new key
    on the server and, through the page's mask version in the URL, in the
    browser. The dataset's generation is in the key as well, so a window built
    before a trim, a delete or a save cannot answer for one asked afterwards,
    on this server's disk or on another instance's."""
    h = hashlib.sha1(dataset_id.encode(), usedforsecurity=False).hexdigest()[:12]
    e = (
        "archive"
        if rung == "full" and masks != "composited"
        else f"{enc['codec']}_{enc['rc']}{enc['q']}_{enc['preset']}"
    )
    m = (
        masks
        if masks != "composited"
        else "m" + hashlib.sha1(fingerprint.encode(), usedforsecurity=False).hexdigest()[:10]
    )
    return (
        cache_dir()
        / f"{h}__v{FORMAT_VERSION}__g{generation}__ep{episode_idx}__f{start}__s{seconds:g}__{rung}__{e}__{m}.bin"
    )


def invalidate_dataset(dataset_id: str) -> int:
    """Drop every cached window of one dataset. Returns the bytes freed.

    A window is a function of the archive's pixels and the dataset's rows, so
    anything that rewrites them -- a trim, a delete, a feature or mask edit --
    makes the cached windows wrong. The key carries the dataset's generation,
    so those windows can no longer be served; this reclaims their bytes at
    once rather than leaving them for the pruner, and is what the edit paths
    call where they used to clear the frame cache.
    """
    prefix = hashlib.sha1(dataset_id.encode(), usedforsecurity=False).hexdigest()[:12]
    freed = 0
    for path in cache_dir().glob(f"{prefix}__*.bin"):
        try:
            size = path.stat().st_size
            path.unlink(missing_ok=True)  # safe-destruct: our own window cache, rebuilt on demand
            freed += size
        except OSError as e:
            logger.warning("window-playback cache: could not drop %s: %s", path.name, e)
    return freed


def prune_cache(ceiling: int | None = None) -> int:
    """Drop least recently used window files until the directory fits the ceiling. Returns bytes removed.

    The ceiling is read at call time: the server sets the module's from
    ``--cache-size`` at startup, which a default bound at import would miss.
    """
    ceiling = CACHE_CEILING_BYTES if ceiling is None else ceiling
    files = sorted(cache_dir().glob("*.bin"), key=lambda p: p.stat().st_atime)
    total = sum(p.stat().st_size for p in files)
    removed = 0
    for p in files:
        if total <= ceiling:
            break
        size = p.stat().st_size
        p.unlink(missing_ok=True)  # safe-destruct: our own window cache, rebuilt on demand
        total -= size
        removed += size
    return removed


@router.get("/{dataset_id:path}/episodes/{episode_idx}/window")
async def get_window(
    dataset_id: str,
    episode_idx: int,
    start: int = Query(0, ge=0),
    len_s: float = Query(1.0, alias="len"),
    rung: str = Query("640"),
    codec: str = Query("h264"),
    rc: str = Query("cbr"),
    q: int = Query(26, ge=Q_RANGE[0], le=Q_RANGE[1]),
    preset: str = Query(""),
    v: int | None = Query(
        None, description="the format version the page expects; a cache-busting key, not checked"
    ),
    g: int | None = Query(
        None, description="the dataset generation the page holds; a cache-busting key, not checked"
    ),
    masks: str = Query(
        "runs", description="none, runs (the mask runs as a part), or composited (the recipe in the pixels)"
    ),
    mv: str = Query("", description="the page's mask version; a cache-busting key, not checked"),
) -> Response:
    t0 = time.perf_counter()
    dataset_id, dataset = _dataset(dataset_id)
    if codec not in ENCODER_OPTIONS["codec"] or rc not in ENCODER_OPTIONS["rc"]:
        raise HTTPException(
            status_code=400, detail=f"codec in {ENCODER_OPTIONS['codec']}, rc in {ENCODER_OPTIONS['rc']}"
        )
    if not preset:
        preset = "veryfast" if codec == "h264" else "8"
    if preset not in ENCODER_OPTIONS["preset"] or (codec == "h264") != preset.isalpha():
        raise HTTPException(status_code=400, detail=f"preset {preset!r} is not one for {codec}")
    enc = {"codec": codec, "rc": rc, "q": q, "preset": preset}
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")
    if rung not in RUNGS:
        raise HTTPException(status_code=400, detail=f"rung must be one of {list(RUNGS)}")
    if len_s not in WINDOW_LENGTHS:
        raise HTTPException(status_code=400, detail=f"len must be one of {list(WINDOW_LENGTHS)}")
    if masks not in MASK_MODES:
        raise HTTPException(status_code=400, detail=f"masks must be one of {MASK_MODES}")
    specs: dict[str, tuple[dict, str]] = {}
    fingerprint = ""
    if masks == "composited":
        from lerobot.datasets.mask_compositing import recipe_fingerprint

        for cam in _video_cameras(dataset):
            mask_key = datasets_api.mask_column_of(dataset, cam)
            spec = (
                datasets_api._effective_recipe(dataset_id, dataset.root, cam, mask_key) if mask_key else None
            )
            if spec is not None:
                specs[cam] = (spec, mask_key)
        # A camera whose masks the timeline draws but whose recipe did not
        # resolve would be served as stored pixels under a composited URL:
        # say so rather than let the operator read raw frames as composited.
        unresolved = sorted(set(datasets_api._mask_features(dataset)) - {k for _, k in specs.values()})
        if unresolved:
            logger.warning(
                "window-playback %s ep=%d: composited asked for, but no recipe resolved for %s; "
                "those cameras are served as stored pixels",
                dataset_id, episode_idx, unresolved,
            )  # fmt: skip
        fingerprint = ",".join(
            f"{cam}:{recipe_fingerprint(spec)}" for cam, (spec, _) in sorted(specs.items())
        )
    length = dataset.episode_rows(episode_idx)[1]
    if start >= length:
        raise HTTPException(status_code=404, detail=f"start {start} past the episode's {length} frames")

    key = _cache_key(
        dataset_id, episode_idx, start, len_s, rung, enc, masks, fingerprint, dataset_generation(dataset)
    )
    hit = key.exists()
    if hit:
        body = await asyncio.get_event_loop().run_in_executor(_build_executor, key.read_bytes)
        os.utime(key, None)
        header = json.loads(body[4 : 4 + struct.unpack("<I", body[:4])[0]])
    else:
        body, header = await asyncio.get_event_loop().run_in_executor(
            _build_executor,
            _build_window,
            dataset,
            dataset_id,
            episode_idx,
            start,
            len_s,
            rung,
            enc,
            masks,
            specs,
        )
        tmp = key.with_suffix(".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, key)
        removed = prune_cache()
        if removed:
            logger.info("window-playback cache pruned %d B", removed)
    ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "window-playback window %s ep=%d start=%d len=%g rung=%s enc=%s/%s%s/%s masks=%s %s %d B %.0f ms build=%s",
        dataset_id, episode_idx, start, len_s, rung, codec, rc, q if rc == "crf" else "", preset, masks,
        "hit" if hit else "miss", len(body), ms, header.get("build_ms"),
    )  # fmt: skip
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            # The body is a function of the URL (with the format version in it) for as long as the
            # dataset is not edited; see the cache-key note on invalidation.
            "Cache-Control": "private, max-age=3600",
            "X-Window-Cache": "hit" if hit else "miss",
            "X-Window-Ms": f"{ms:.0f}",
        },
    )
