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

Rungs: width and bitrate; ``full`` is the source width at a near-lossless
constant quality. Masks are scaled down to the rung's width. The archive's own samples are not remuxed in this prototype
because cutting an AV1 stream at an exact frame needs its keyframe index;
``full`` here is a re-encode.

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
import pyarrow as pa
from fastapi import APIRouter, HTTPException, Query, Response

from lerobot.datasets.utils import DEFAULT_VIDEO_PATH
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
FORMAT_VERSION = 4

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


#: Columns of the bundle's envelope of each numeric feature. An overview canvas
#: is a few thousand pixels wide at most; past this, min and max per column
#: draw the same band as the samples would.
ENVELOPE_COLUMNS = 1024


def _numeric_features(dataset) -> list[str]:
    """Every feature that is a number per frame: not an image, not a string, not a mask."""
    return [
        name
        for name, ft in dataset.meta.features.items()
        if ft.get("dtype") not in ("image", "video", "string") and not ft.get("mask_encoding")
    ]


def _feature_rows(dataset, name: str, first: int, count: int) -> np.ndarray:
    """Rows ``[first, first + count)`` of a numeric feature as a ``(count, D)``
    float array, straight from the arrow table: 2 ms for an hour of a 16-dim
    feature, against 4.8 s through per-row JSON coercion."""
    col = dataset.hf_dataset.data.column(name).slice(first, count).combine_chunks()
    if pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
        arr = col.flatten().to_numpy(zero_copy_only=False).reshape(count, -1)
    else:
        arr = col.to_numpy(zero_copy_only=False).reshape(count, 1)
    return np.asarray(arr, dtype=np.float64)


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


def _episode_video(dataset, ep, key: str) -> tuple[Path, float]:
    tpl = dataset.meta.info.get("video_path") or DEFAULT_VIDEO_PATH
    rel = tpl.format(
        video_key=key,
        chunk_index=int(ep[f"videos/{key}/chunk_index"]),
        file_index=int(ep[f"videos/{key}/file_index"]),
    )
    return Path(dataset.root) / rel, float(ep[f"videos/{key}/from_timestamp"])


# ---------------------------------------------------------------- bundle


@router.get("/{dataset_id:path}/episodes/{episode_idx}/bundle")
async def get_bundle(dataset_id: str, episode_idx: int) -> Response:
    t0 = time.perf_counter()
    dataset_id, dataset = _dataset(dataset_id)
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")
    ep = dataset.meta.episodes[episode_idx]
    # Per frame only what is one small value per frame: mask presence bitsets
    # and the task string, read from the arrow table's rows of this episode.
    # Reading them through the feature-series endpoint cost 770 to 800 ms per
    # call on the rig's labelled dataset, a pandas read of the mask columns of
    # the whole parquet file, on the path to the first picture. Numeric
    # features come as envelopes.
    ep_first, length = int(ep["dataset_from_index"]), int(ep["length"])
    mask_feats = datasets_api._mask_features(dataset)
    series: dict[str, list[Any]] = {}
    for key in mask_feats:
        cells = dataset.hf_dataset.data.column(key).slice(ep_first, length).to_pylist()
        series[key] = [datasets_api._mask_presence_bits(v) for v in cells]
        series[f"{key}{datasets_api.MASK_DISABLED_SUFFIX}"] = [
            datasets_api._mask_disabled_bits(v) for v in cells
        ]
    if "task_index" in dataset.meta.features and dataset.meta.tasks is not None:
        names = list(dataset.meta.tasks.index)
        series["task"] = [names[int(i)] for i in _feature_rows(dataset, "task_index", ep_first, length)[:, 0]]
    envelope = {
        name: _envelope(_feature_rows(dataset, name, ep_first, length)) for name in _numeric_features(dataset)
    }
    masks = {
        key: {"labels": ft.get("mask_labels", []), "size": ft.get("mask_size")}
        for key, ft in datasets_api._mask_features(dataset).items()
    }
    cameras = {}
    for key, ft in _video_cameras(dataset).items():
        info = ft.get("info") or {}
        cameras[key] = {
            "width": info.get("video.width"),
            "height": info.get("video.height"),
            "codec": info.get("video.codec"),
        }
    body = {
        "episode_index": episode_idx,
        "episodes": int(dataset.meta.total_episodes),  # so the page knows whether a next episode exists
        "length": int(ep["length"]),
        "fps": dataset.meta.fps,
        "cameras": cameras,
        "masks": masks,  # presence per frame is in series[<mask key>] as a bitset, from feature-series
        "series": series,
        "envelope": envelope,
        "rungs": list(RUNGS),
        "encoder_options": {k: list(v) for k, v in ENCODER_OPTIONS.items()},
        # Nominal video bitrate per camera per rung, kbit/s, for the page's automatic choice; full has none.
        "rung_kbps": {name: (int(br[:-1]) if br else None) for name, (_, br) in RUNGS.items()},
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
    access-unit delimiter (NAL type 9) in front of every frame."""
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
    assert sum(sizes) == n - starts[0]
    return sizes


def _split_temporal_units(obu: bytes) -> list[int]:
    """Byte length of each temporal unit in a low-overhead AV1 OBU stream: a
    frame starts with a temporal delimiter OBU (type 2)."""
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
    return [b - a for a, b in zip(starts, starts[1:], strict=False)] + [n - starts[-1]]


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
    frames = round(seconds * fps)
    vf = [] if width == 0 else ["-vf", f"scale={width}:-2"]
    codec, rc, q, preset = enc["codec"], enc["rc"], int(enc["q"]), enc["preset"]
    if codec == "h264":
        if rc == "cbr":
            rate = (
                ["-crf", "18"]
                if bitrate is None
                else ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate]
            )
        else:
            cap = (
                [] if bitrate is None else ["-maxrate", bitrate, "-bufsize", str(2 * int(bitrate[:-1])) + "k"]
            )
            rate = ["-crf", str(q), *cap]
        video = ["-c:v", "libx264", "-preset", preset, "-profile:v", "main", "-pix_fmt", "yuv420p", *rate,
                 "-g", str(frames), "-keyint_min", str(frames), "-sc_threshold", "0", "-bf", "0", "-forced-idr", "1",
                 "-bsf:v", "h264_metadata=aud=insert", "-f", "h264"]  # fmt: skip
    else:
        # SVT-AV1 takes its cap through its own parameter string; no maxrate/bufsize.
        if rc == "cbr":
            rate = ["-crf", "18"] if bitrate is None else ["-b:v", bitrate, "-svtav1-params", "rc=1"]
        else:
            rate = ["-crf", str(q), *([] if bitrate is None else ["-svtav1-params", f"mbr={bitrate}"])]
        video = [
            "-c:v",
            "libsvtav1",
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            *rate,
            "-g",
            str(frames),
            "-f",
            "obu",
        ]
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


def _mask_part(
    dataset, key: str, first: int, count: int, size: tuple[int, int], target_w: int
) -> tuple[bytes, list[int]]:
    """The mask runs for ``count`` frames from dataset row ``first``, scaled down
    to the rung's width when the rung is narrower than the mask.

    Returns the gzipped JSON and the [height, width] the runs are in. A run
    length grows with the outline, so a mask at a quarter of the width is
    about a quarter of the bytes; the page scales the canvas anyway.
    """
    import cv2

    from lerobot.datasets.mask_codec import decode_mask, encode_mask

    h, w = int(size[0]), int(size[1])
    scale = bool(target_w) and target_w < w
    nh, nw = (max(1, round(h * target_w / w)), target_w) if scale else (h, w)
    column = dataset.hf_dataset[key][first : first + count]
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
    dataset, dataset_id: str, episode_idx: int, start: int, seconds: float, rung: str, enc: dict[str, Any]
) -> tuple[bytes, dict]:
    ep = dataset.meta.episodes[episode_idx]
    fps = float(dataset.meta.fps)
    length = int(ep["length"])
    count = min(round(seconds * fps), length - start)
    ds_first = int(ep["dataset_from_index"]) + start
    cams = _video_cameras(dataset)
    mask_feats = datasets_api._mask_features(dataset)

    width = RUNGS[rung][0]

    def cam_job(key):
        path, from_ts = _episode_video(dataset, ep, key)
        data, sizes, secs = _encode_camera(path, from_ts + start / fps, count / fps, fps, rung, enc)
        return key, data, sizes, secs

    def features_job():
        t_f = time.perf_counter()
        rows = {
            name: _feature_rows(dataset, name, ds_first, count).round(4).tolist()
            for name in _numeric_features(dataset)
        }
        data = gzip.compress(json.dumps(rows, separators=(",", ":")).encode(), compresslevel=6)
        return data, time.perf_counter() - t_f

    def mask_job(key):
        t_m = time.perf_counter()
        data, out_size = _mask_part(
            dataset, key, ds_first, count, mask_feats[key].get("mask_size") or [0, 0], width
        )
        return key, data, out_size, time.perf_counter() - t_m

    # Cameras and masks build side by side; the masks used to run after the
    # cameras and added 100 to 200 ms to a 2 s window.
    cam_futures = [_build_executor.submit(cam_job, k) for k in cams]
    mask_futures = [_build_executor.submit(mask_job, k) for k in mask_feats]
    features_future = _build_executor.submit(features_job)
    results = [f.result() for f in cam_futures]
    mask_results = [f.result() for f in mask_futures]
    features_data, features_secs = features_future.result()
    parts: list[dict] = []
    blobs: list[bytes] = []
    offset = 0
    builds = {}
    for key, data, sizes, secs in results:
        if len(sizes) != count:
            raise RuntimeError(f"{key}: encoded {len(sizes)} frames, wanted {count}")
        parts.append(
            {
                "camera": key,
                "kind": "video",
                "codec": enc["codec"],
                "offset": offset,
                "length": len(data),
                "frame_sizes": sizes,
            }
        )
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
        "parts": parts,
        "build_ms": builds,
    }
    hb = json.dumps(header, separators=(",", ":")).encode()
    body = struct.pack("<I", len(hb)) + hb + b"".join(blobs)
    return body, header


def _cache_key(
    dataset_id: str, episode_idx: int, start: int, seconds: float, rung: str, enc: dict[str, Any]
) -> Path:
    """The cache key of a window. **Not invalidated by an edit:** it carries no
    edit generation of the dataset, so a mask written after a window was built
    is not in that window, on the server's disk cache or in the browser's
    cache, until the version changes. The Data tab integration has to add the
    dataset's edit generation here and to the URL."""
    h = hashlib.sha1(dataset_id.encode(), usedforsecurity=False).hexdigest()[:12]
    e = f"{enc['codec']}_{enc['rc']}{enc['q']}_{enc['preset']}"
    return cache_dir() / f"{h}__v{FORMAT_VERSION}__ep{episode_idx}__f{start}__s{seconds:g}__{rung}__{e}.bin"


def prune_cache(ceiling: int = CACHE_CEILING_BYTES) -> int:
    """Drop least recently used window files until the directory fits the ceiling. Returns bytes removed."""
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
    length = int(dataset.meta.episodes["length"][episode_idx])
    if start >= length:
        raise HTTPException(status_code=404, detail=f"start {start} past the episode's {length} frames")

    key = _cache_key(dataset_id, episode_idx, start, len_s, rung, enc)
    hit = key.exists()
    if hit:
        body = await asyncio.get_event_loop().run_in_executor(_build_executor, key.read_bytes)
        os.utime(key, None)
        header = json.loads(body[4 : 4 + struct.unpack("<I", body[:4])[0]])
    else:
        body, header = await asyncio.get_event_loop().run_in_executor(
            _build_executor, _build_window, dataset, dataset_id, episode_idx, start, len_s, rung, enc
        )
        tmp = key.with_suffix(".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, key)
        removed = prune_cache()
        if removed:
            logger.info("window-playback cache pruned %d B", removed)
    ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "window-playback window %s ep=%d start=%d len=%g rung=%s enc=%s/%s%s/%s %s %d B %.0f ms build=%s",
        dataset_id, episode_idx, start, len_s, rung, codec, rc, q if rc == "crf" else "", preset,
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
