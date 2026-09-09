"""Static playback path, prototype: the stored video file as it is, plus one
manifest per episode.

The browser plays the dataset's own MP4 through a ``<video>`` element with
range requests, seeks to the episode's time range inside the packed file,
and paints the stored masks itself. The server does no decoding and no
transcoding on this path; it serves bytes and one JSON per episode.

Endpoints
- ``GET /api/datasets/{id}/episodes/{ep}/playback`` — the manifest: per
  camera the file URL, the episode's time range in that file, the codec,
  pixel format, frame rate and size from the dataset metadata, and whether
  the file's index (``moov``) sits at the front. The masks and the
  numeric features come from the existing ``/masks`` and ``/features``
  endpoints, one request each.
- ``GET /api/datasets/{id}/video-file/{camera}/{chunk}/{file}`` — the
  file, with ``Range`` honoured by Starlette's ``FileResponse``.

Every request on this path is logged with its timing and, for the file,
the byte range asked for, so a playback session can be read back from the
server log alone: how many requests, how large, how long each took.

Pre: the dataset is open in the app state. Post: nothing is written; the
files are read only.
"""

from __future__ import annotations

import logging
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from lerobot.datasets.utils import DEFAULT_VIDEO_PATH

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["static-playback"])

_app_state: AppState = None  # type: ignore


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


def moov_before_mdat(path: Path) -> bool | None:
    """Whether the MP4's index precedes its media data.

    Walks the top-level boxes: each is a 4-byte big-endian size, a 4-byte
    type, and for size 1 an 8-byte size. Returns None when neither box is
    found in the first 64 boxes (not an MP4, or truncated).
    """
    with path.open("rb") as f:
        for _ in range(64):
            head = f.read(8)
            if len(head) < 8:
                return None
            size, kind = struct.unpack(">I4s", head)
            if size == 1:
                (size,) = struct.unpack(">Q", f.read(8))
            elif size == 0:
                size = path.stat().st_size - f.tell() + 8
            if kind == b"moov":
                return True
            if kind == b"mdat":
                return False
            f.seek(size - (16 if size > 0xFFFFFFFF else 8), 1)
    return None


def _dataset(dataset_id: str):
    dataset_id = unquote(dataset_id)
    if dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    return dataset_id, _app_state.datasets[dataset_id]


@router.get("/{dataset_id:path}/episodes/{episode_idx}/playback")
async def get_playback_manifest(dataset_id: str, episode_idx: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    dataset_id, dataset = _dataset(dataset_id)
    if episode_idx < 0 or episode_idx >= dataset.meta.total_episodes:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_idx}")
    ep = dataset.meta.episodes[episode_idx]
    video_path_tpl = dataset.meta.info.get("video_path") or DEFAULT_VIDEO_PATH
    cameras: dict[str, Any] = {}
    for key, ft in dataset.meta.features.items():
        if ft.get("dtype") != "video":
            continue
        chunk = int(ep[f"videos/{key}/chunk_index"])
        file = int(ep[f"videos/{key}/file_index"])
        rel = video_path_tpl.format(video_key=key, chunk_index=chunk, file_index=file)
        path = Path(dataset.root) / rel
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Video file missing: {rel}")
        info = ft.get("info") or {}
        cameras[key] = {
            "url": f"/api/datasets/{quote(dataset_id, safe='')}/video-file/{quote(key, safe='')}/{chunk}/{file}",
            "from_timestamp": float(ep[f"videos/{key}/from_timestamp"]),
            "to_timestamp": float(ep[f"videos/{key}/to_timestamp"]),
            "codec": info.get("video.codec"),
            "pix_fmt": info.get("video.pix_fmt"),
            "fps": info.get("video.fps"),
            "width": info.get("video.width"),
            "height": info.get("video.height"),
            "file_bytes": path.stat().st_size,
            "moov_first": moov_before_mdat(path),
        }
    body = {
        "episode_index": episode_idx,
        "length": int(ep["length"]),
        "fps": dataset.meta.fps,
        "cameras": cameras,
    }
    logger.info(
        "static-playback manifest %s ep=%d cameras=%d %.1f ms",
        dataset_id,
        episode_idx,
        len(cameras),
        (time.perf_counter() - t0) * 1000,
    )
    return body


@router.get("/{dataset_id:path}/video-file/{camera}/{chunk}/{file}")
async def get_video_file(
    dataset_id: str, camera: str, chunk: int, file: int, request: Request
) -> FileResponse:
    dataset_id, dataset = _dataset(dataset_id)
    key = unquote(camera)
    ft = dataset.meta.features.get(key)
    if not ft or ft.get("dtype") != "video":
        raise HTTPException(status_code=404, detail=f"Not a video feature: {key}")
    rel = (dataset.meta.info.get("video_path") or DEFAULT_VIDEO_PATH).format(
        video_key=key, chunk_index=chunk, file_index=file
    )
    path = Path(dataset.root) / rel
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Video file missing: {rel}")
    logger.info(
        "static-playback file %s %s range=%s",
        dataset_id,
        rel,
        request.headers.get("range", "-"),
    )
    # Cache by file identity: the file is immutable once written; a rewrite
    # produces a different path or size, and the manifest is fetched fresh.
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, max-age=3600"})
