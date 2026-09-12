"""Video playback of a stored episode, a few seconds at a time (gui/docs/dataset_playback.md).

``GET .../episodes/{ep}/chunk?start=F&profile=low`` returns a chunk: from frame
``F``, every camera's next frames as raw H.264 (Annex B, an access-unit
delimiter in front of every frame, one keyframe first, no B-frames) at that
camera's encoded resolution, and -- for a camera with saved masks -- the mask
RLE rows for the same frames resized to that resolution, gzipped. One request
carries every camera and its masks; nothing is requested per frame. A chunk is
built by one ffmpeg per camera in parallel from the stored file, reading the
dataset only through its accessors, cached on disk under a byte ceiling, and
dropped when the dataset is edited. Responses are never browser-cacheable, like
the JPEG path, so an edit is what plays next.

Body layout: 4-byte little-endian header length, the JSON header, then the parts
back to back. The header gives each part's offset and length and, for a video
part, the size of every frame, so the page hands its decoder one frame at a time
without parsing NAL units.

Pre: the dataset is open in the app state; ffmpeg with libx264 and the
``h264_metadata`` bitstream filter is on PATH. Post: nothing in the dataset is
written; the cache directory is the only side effect.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import json
import logging
import os
import struct
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["chunk-playback"])

_app_state: AppState = None  # type: ignore

#: Seconds of media per chunk. Above the round-trip floor and on the cheap side
#: of the keyframe cost; the design's starting point, to be measured against R2
#: and R3. The page has to agree, and checks the header's ``chunk_frames``.
CHUNK_SECONDS = 2.0

#: The one video profile. A rule applied per camera: a target width each camera
#: is scaled down to (never up), and a constant quality so bytes follow content.
#: ``high`` is the JPEG path, not an encode, and is refused here.
PROFILES: dict[str, dict[str, Any]] = {"low": {"width": 320, "crf": 26, "preset": "veryfast"}}

#: What the page configures its decoder with: H.264 Main profile, level 3.1.
CODEC_STRING = "avc1.4d401f"

#: Bump when the body layout or a part's content changes, so old cache entries are not served as new ones.
FORMAT_VERSION = 2

CACHE_CEILING_BYTES = int(os.environ.get("LEROBOT_CHUNK_CACHE_BYTES", 2 * 1024**3))

#: The explicit bound on encoders: a viewer never pushes another below the
#: frame rate by starting more ffmpeg processes than the machine has room for.
MAX_ENCODERS = 8
_encoders = threading.BoundedSemaphore(MAX_ENCODERS)
_build_executor = ThreadPoolExecutor(max_workers=MAX_ENCODERS, thread_name_prefix="gui-chunk")
_reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-chunk-io")


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


def cache_dir() -> Path:
    d = Path(
        os.environ.get("LEROBOT_CHUNK_CACHE_DIR")
        or Path.home() / ".cache" / "huggingface" / "lerobot" / "gui" / "chunks"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def chunk_frames(fps: float) -> int:
    """Frames per chunk at this frame rate; the grid every chunk starts on."""
    n = max(1, round(CHUNK_SECONDS * fps))
    assert n > 0, (CHUNK_SECONDS, fps)
    return n


def encoded_size(height: int, width: int, target_width: int) -> tuple[int, int]:
    """(height, width) one camera's video is encoded at under a profile: its own
    resolution scaled to the target width, aspect kept, never upscaled, height
    even as the encoder needs. Cameras differ, so this differs per camera, and
    the mask rows are resized to the same numbers so the two line up."""
    assert height > 0 and width > 0 and target_width > 0, (height, width, target_width)
    w = min(width, target_width)
    h = max(2, 2 * round(height * w / width / 2))
    return h, w


def _video_cameras(dataset) -> dict[str, dict]:
    return {k: ft for k, ft in dataset.meta.features.items() if ft.get("dtype") == "video"}


def _mask_columns(dataset) -> dict[str, dict]:
    """camera key -> the mask feature that describes it, for cameras that have one."""
    from lerobot.datasets.mask_compositing import mask_feature_of
    from lerobot.gui.api.datasets import _mask_features

    have = _mask_features(dataset)
    out = {}
    for cam in _video_cameras(dataset):
        key = mask_feature_of(cam)
        if key != cam and key in have:
            out[cam] = {"key": key, **have[key]}
    return out


def _split_access_units(annexb: bytes) -> list[int]:
    """Byte length of each access unit in an Annex B stream that carries an
    access-unit delimiter (NAL type 9) in front of every frame.

    The page hands its decoder one frame at a time, so these sizes are the
    frame boundaries: a wrong split is a decoder error or a frame silently
    dropped. Post: one size per delimiter, each positive, together covering
    the stream from the first delimiter to its end.

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
        s = j - 1 if j > 0 and annexb[j - 1] == 0 else j  # a 4-byte start code has a leading zero
        if j + 3 < n and (annexb[j + 3] & 0x1F) == 9:
            starts.append(s)
        i = j + 3
    if not starts:
        raise RuntimeError("no access-unit delimiters in the encoded stream")
    sizes = [b - a for a, b in zip(starts, starts[1:], strict=False)] + [n - starts[-1]]
    assert all(size > 0 for size in sizes), sizes
    assert sum(sizes) == n - starts[0], (sum(sizes), n, starts[0])
    return sizes


def _encode_camera(
    path: Path, t0: float, frames: int, fps: float, size: tuple[int, int], profile: dict[str, Any]
) -> tuple[bytes, list[int], float]:
    """One camera's chunk as raw Annex B H.264 with one AUD per frame.

    Returns (bytes, per-frame sizes, encode seconds). The first frame is an IDR,
    the GOP spans the chunk, no B-frames, so the page decodes from the first
    frame and every chunk stands on its own. The scale is given outright rather
    than as a ``-2`` rule, so the mask rows can be resized to exactly the same
    numbers. No zero-latency tuning: it is for live encoding and cost SSIM on
    stored frames.
    """
    h, w = size
    gop = str(frames)
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t0:.6f}", "-i", str(path), "-frames:v", str(frames), "-an",
        "-vf", f"scale={w}:{h}",
        "-c:v", "libx264", "-preset", profile["preset"], "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-crf", str(profile["crf"]),
        "-g", gop, "-keyint_min", gop, "-sc_threshold", "0", "-bf", "0", "-forced-idr", "1",
        "-bsf:v", "h264_metadata=aud=insert", "-f", "h264", "pipe:1",
    ]  # fmt: skip
    t = time.perf_counter()
    with _encoders:
        out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path.name}: {out.stderr.decode(errors='replace')[-400:]}")
    sizes = _split_access_units(out.stdout)
    return out.stdout, sizes, time.perf_counter() - t


def _mask_part(
    dataset,
    key: str,
    episode_idx: int,
    start: int,
    count: int,
    stored: tuple[int, int],
    size: tuple[int, int],
) -> tuple[bytes, float]:
    """The mask RLE rows for ``count`` frames from ``start``, resized from the
    stored resolution to the camera's encoded one, gzipped.

    A mask was computed at the stored resolution and is only drawn smaller: the
    resize is nearest-neighbour on the decoded mask, so the region still lines
    up with the object and what is lost is boundary pixels.
    """
    import cv2

    from lerobot.datasets.mask_codec import decode_mask, encode_mask

    sh, sw = int(stored[0]), int(stored[1])
    h, w = size
    scale = (sh, sw) != (h, w)
    t = time.perf_counter()
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
                    decode_mask(counts, (sh, sw)).astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST
                )
                out.append([label, encode_mask(small.astype(bool)), *rest])
            entries = out
        frames.append(entries)
    body = gzip.compress(json.dumps(frames, separators=(",", ":")).encode(), compresslevel=6)
    return body, time.perf_counter() - t


def build_chunk(dataset, episode_idx: int, start: int, profile: str) -> tuple[bytes, dict]:
    """The chunk body and its header. Reads the episode's rows and each camera's
    video span through the dataset's accessors only.

    Pre: ``start`` is on the chunk grid and inside the episode. Post: every video
    part holds exactly the chunk's frames, in order, and every mask part the rows
    for the same frames at the same encoded resolution.
    """
    prof = PROFILES[profile]
    fps = float(dataset.meta.fps)
    n = chunk_frames(fps)
    length = dataset.episode_rows(episode_idx)[1]
    assert start % n == 0, f"start {start} is not on the {n}-frame grid"
    assert 0 <= start < length, (start, length)
    count = min(n, length - start)
    cams = _video_cameras(dataset)
    masks = _mask_columns(dataset)

    def cam_job(cam: str):
        ft = cams[cam]
        sh, sw = int(ft["shape"][0]), int(ft["shape"][1])
        size = encoded_size(sh, sw, prof["width"])
        rel, from_ts, _to_ts = dataset.meta.get_episode_video_span(episode_idx, cam)
        data, sizes, secs = _encode_camera(
            Path(dataset.root) / rel, from_ts + start / fps, count, fps, size, prof
        )
        if len(sizes) != count:
            raise RuntimeError(f"{cam}: encoded {len(sizes)} frames, wanted {count}")
        return cam, data, sizes, size, (sh, sw), secs

    def mask_job(cam: str):
        m = masks[cam]
        ft = cams[cam]
        size = encoded_size(int(ft["shape"][0]), int(ft["shape"][1]), prof["width"])
        stored = m.get("mask_size") or (int(ft["shape"][0]), int(ft["shape"][1]))
        data, secs = _mask_part(dataset, m["key"], episode_idx, start, count, stored, size)
        return cam, data, size, m.get("mask_labels") or [], secs

    cam_futures = [_build_executor.submit(cam_job, c) for c in cams]
    mask_futures = [_build_executor.submit(mask_job, c) for c in masks]
    results = [f.result() for f in cam_futures]
    mask_results = [f.result() for f in mask_futures]

    parts: list[dict] = []
    blobs: list[bytes] = []
    offset = 0
    build_ms: dict[str, int] = {}
    for cam, data, sizes, (h, w), (sh, sw), secs in results:
        parts.append(
            {
                "camera": cam, "kind": "video", "codec": "h264", "codec_string": CODEC_STRING,
                "width": w, "height": h, "stored_width": sw, "stored_height": sh,
                "offset": offset, "length": len(data), "frame_sizes": sizes,
            }
        )  # fmt: skip
        blobs.append(data)
        offset += len(data)
        build_ms[cam.split(".")[-1]] = round(secs * 1000)
    for cam, data, (h, w), labels, secs in mask_results:
        parts.append(
            {
                "camera": cam, "kind": "masks", "encoding": "gzip", "size": [h, w], "labels": labels,
                "offset": offset, "length": len(data),
            }
        )  # fmt: skip
        blobs.append(data)
        offset += len(data)
        build_ms["masks:" + cam.split(".")[-1]] = round(secs * 1000)
    header = {
        "episode_index": episode_idx,
        "first_frame": start,
        "frames": count,
        "fps": fps,
        "chunk_frames": n,
        "profile": profile,
        "format_version": FORMAT_VERSION,
        "parts": parts,
        "build_ms": build_ms,
    }
    hb = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<I", len(hb)) + hb + b"".join(blobs), header


def _prefix(dataset_id: str) -> str:
    return hashlib.sha1(dataset_id.encode(), usedforsecurity=False).hexdigest()[:12]


def _cache_key(dataset_id: str, episode_idx: int, start: int, profile: str) -> Path:
    """Stored pixels do not change and masks are not in them, so the key is the
    request; an edit drops the dataset's entries rather than changing the key."""
    return cache_dir() / f"{_prefix(dataset_id)}__v{FORMAT_VERSION}__ep{episode_idx}__f{start}__{profile}.bin"


def invalidate_dataset(dataset_id: str) -> int:
    """Drop every cached chunk of one dataset. Returns the bytes freed.

    A chunk is a function of the stored video and the dataset's rows, so
    anything that rewrites them -- a trim, a delete, a frame removal, a mask
    save -- makes the cached chunks wrong. Called from the GUI's shared
    invalidation hook, which every edit path already calls.
    """
    freed = 0
    for path in cache_dir().glob(f"{_prefix(dataset_id)}__*.bin"):
        try:
            size = path.stat().st_size
            path.unlink(missing_ok=True)  # safe-destruct: our own chunk cache, rebuilt on demand
            freed += size
        except OSError as e:
            logger.warning("chunk-playback cache: could not drop %s: %s", path.name, e)
    return freed


def prune_cache(ceiling: int | None = None) -> int:
    """Drop least recently used chunk files until the directory fits the
    ceiling. Returns bytes removed. The ceiling is read at call time."""
    ceiling = CACHE_CEILING_BYTES if ceiling is None else ceiling
    # An edit drops its dataset's chunks while the pruner may be walking the
    # same directory, so a path the glob returned can be gone by the stat. Age
    # and size are taken once, and whatever vanished is already not taking
    # space -- raising here would fail the request that triggered the prune,
    # which is the one that had just built a chunk to store.
    entries = []
    for path in cache_dir().glob("*.bin"):
        try:
            st = path.stat()
        except OSError:
            continue
        entries.append((st.st_atime, st.st_size, path))
    entries.sort(key=lambda e: e[0])
    total = sum(size for _atime, size, _p in entries)
    removed = 0
    for _atime, size, path in entries:
        if total <= ceiling:
            break
        path.unlink(missing_ok=True)  # safe-destruct: our own chunk cache, rebuilt on demand
        total -= size
        removed += size
    return removed


def _read_and_touch(key: Path) -> bytes:
    """The cached body, with its mtime bumped so the pruner reads it as recent.

    Pre: none -- the entry may be gone. Raises FileNotFoundError if it is, which
    the caller treats as a miss.
    """
    body = key.read_bytes()
    with contextlib.suppress(FileNotFoundError):
        os.utime(key, None)  # recency is best effort; the body is already in hand
    return body


def _store_and_prune(key: Path, body: bytes) -> int:
    """Write one cache entry atomically, then bring the directory under its
    ceiling; returns the bytes pruned.

    Off the event loop, because this is file work and not request work: a
    hundred kilobytes written and a directory walked while the loop waits
    stalls every other request behind it -- the next chunk, a frame, the
    robot tile's fetch. The build already runs off the loop; so must what it
    produced.
    """
    tmp = key.with_name(key.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(body)
    os.replace(tmp, key)
    return prune_cache()


@router.get("/{dataset_id:path}/episodes/{episode_idx}/chunk")
async def get_chunk(
    dataset_id: str,
    episode_idx: int,
    start: int = Query(0, ge=0, description="first frame; must lie on the chunk grid"),
    profile: str = Query("low"),
    x_player: str | None = Header(default=None, description="what the page holds, for the log"),
) -> Response:
    t0 = time.perf_counter()
    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    dataset = _app_state.datasets[dataset_id]
    if profile not in PROFILES:
        raise HTTPException(status_code=400, detail=f"profile must be one of {list(PROFILES)}")
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")
    n = chunk_frames(float(dataset.meta.fps))
    length = dataset.episode_rows(episode_idx)[1]
    if start >= length:
        raise HTTPException(status_code=404, detail=f"start {start} past the episode's {length} frames")
    if start % n:
        raise HTTPException(status_code=400, detail=f"start must be a multiple of {n} frames")

    key = _cache_key(dataset_id, episode_idx, start, profile)
    loop = asyncio.get_event_loop()
    # An entry can go between the look and the read: an edit drops this
    # dataset's chunks and the pruner drops the coldest, either of them while a
    # request is in flight. A vanished entry is a miss to rebuild, not a 500 --
    # which is what the page was served, mid-playback, during an edit storm.
    body = None
    if key.exists():
        try:
            body = await loop.run_in_executor(_reader, _read_and_touch, key)
        except FileNotFoundError:
            body = None
    hit = body is not None
    if hit:
        header = json.loads(body[4 : 4 + struct.unpack("<I", body[:4])[0]])
    else:
        body, header = await loop.run_in_executor(
            _build_executor, build_chunk, dataset, episode_idx, start, profile
        )
        removed = await loop.run_in_executor(_reader, _store_and_prune, key, body)
        if removed:
            logger.info("chunk-playback cache pruned %d B", removed)
    ms = (time.perf_counter() - t0) * 1000
    parts = header["parts"]
    # One line per chunk, with the cameras it carried: the record that a request
    # carried every camera, and that nothing was asked for per frame. The page's
    # own state rides on the request, so the line also says what the page held
    # and had in flight when it asked -- the fact a stalled tile leaves behind.
    logger.info(
        "chunk-playback %s ep=%d start=%d frames=%d profile=%s %s %d B %.0f ms cameras=%d masks=%d build=%s%s",
        dataset_id, episode_idx, start, header["frames"], profile, "hit" if hit else "miss", len(body), ms,
        sum(p["kind"] == "video" for p in parts), sum(p["kind"] == "masks" for p in parts), header.get("build_ms"),
        f" player {x_player[:300]}" if x_player else "",
    )  # fmt: skip
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            # Like the JPEG path: the browser asks every time, and the server's
            # cache -- dropped on edit -- is the only cache between the file and the page.
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Chunk-Cache": "hit" if hit else "miss",
            "X-Chunk-Ms": f"{ms:.0f}",
        },
    )


class PlayerEvent(BaseModel):
    """What the page's player wants in the server log: `kind` names the event
    (never-ready, error), `detail` says what happened and what the page held."""

    kind: str = Field(min_length=1, max_length=64)
    detail: str = Field(min_length=1, max_length=2000)


@router.post("/{dataset_id:path}/episodes/{episode_idx}/player-event", status_code=204)
async def player_event(dataset_id: str, episode_idx: int, event: PlayerEvent) -> Response:
    """Log one event from the page's chunk player as a warning.

    A page that has stopped asking for chunks tells the chunk log nothing, and
    a rig report then reads as silence; the player posts what it saw -- a
    chunk given up on, a decoder error, a rule violation -- so the log holds
    the page's side too. Pre: the dataset is open. Post: one WARNING line.
    """
    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    logger.warning("chunk-playback client %s ep=%d %s: %s", dataset_id, episode_idx, event.kind, event.detail)
    return Response(status_code=204)
