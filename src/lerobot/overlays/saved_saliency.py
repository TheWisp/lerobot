"""Read-only, episode-indexed policy heatmaps. No model or GPU work at replay."""

from __future__ import annotations

import bisect
import hashlib
import json
import threading
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class SavedSaliency:
    metadata: dict
    frames: tuple[int, ...]
    grids: dict[str, np.ndarray]
    revision: str

    def index_at(self, frame: int) -> int | None:
        index = bisect.bisect_right(self.frames, frame) - 1
        return index if index >= 0 and frame < self.metadata["episode_length"] else None


class SavedSaliencyStore:
    """Bounded cache, invalidated when either sidecar file is replaced."""

    def __init__(self, max_bytes: int = MAX_BYTES):
        self.max_bytes = max_bytes
        self._cache: OrderedDict[tuple, SavedSaliency] = OrderedDict()
        self._lock = threading.RLock()

    def load(self, root: Path, episode: int, episode_length: int) -> SavedSaliency | None:
        if episode < 0:
            raise ValueError("Invalid episode")
        root = Path(root).resolve()
        directory = root / "diagnostics" / "policy_saliency" / f"episode_{episode:06d}"
        manifest, archive = directory / "manifest.json", directory / "grids.npz"
        if not manifest.exists():
            return None
        if not all(p.resolve().is_relative_to(root) for p in (manifest, archive)):
            raise ValueError("Heatmap files must belong to this dataset")
        try:
            signatures = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (manifest, archive))
        except FileNotFoundError as exc:
            raise ValueError("Heatmap export is incomplete") from exc
        if signatures[0][1] > 256 * 1024 or signatures[1][1] > MAX_BYTES:
            raise ValueError("Heatmap files exceed the replay size limit")
        key = (str(directory), episode_length, signatures)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            try:
                meta = json.loads(manifest.read_text(encoding="utf-8"))
                if meta["schema_version"] != 1 or meta["episode_index"] != episode:
                    raise ValueError("Unsupported heatmap format or episode")
                if meta["episode_length"] != episode_length:
                    raise ValueError("Heatmap and episode lengths differ")
                frames = tuple(meta["frames"])
                if not frames or any(type(f) is not int or not 0 <= f < episode_length for f in frames):
                    raise ValueError("Invalid heatmap frame indices")
                if any(a >= b for a, b in zip(frames, frames[1:], strict=False)):
                    raise ValueError("Heatmap frames must increase strictly")
                cameras = meta["cameras"]
                if not isinstance(cameras, dict) or not 1 <= len(cameras) <= 16:
                    raise ValueError("Invalid heatmap cameras")
                for camera, shape in cameras.items():
                    if not isinstance(camera, str) or not camera.startswith("observation.images."):
                        raise ValueError("Invalid heatmap camera key")
                    if any(
                        type(shape[k]) is not int or not 0 < shape[k] <= 16384 for k in ("height", "width")
                    ):
                        raise ValueError("Invalid camera dimensions")
                with zipfile.ZipFile(archive) as z:
                    if sum(entry.file_size for entry in z.infolist()) > MAX_BYTES:
                        raise ValueError("Uncompressed heatmaps exceed the replay size limit")
                with np.load(archive, allow_pickle=False) as arrays:
                    grids = {camera: np.asarray(arrays[camera], dtype=np.float32) for camera in cameras}
                for grid in grids.values():
                    if (
                        grid.ndim != 3
                        or grid.shape[0] != len(frames)
                        or not all(0 < d <= 256 for d in grid.shape[1:])
                    ):
                        raise ValueError("Invalid heatmap grid shape")
                    if not np.isfinite(grid).all() or (grid < 0).any():
                        raise ValueError("Heatmaps must contain finite nonnegative values")
                    grid.setflags(write=False)
                if sum(g.nbytes for g in grids.values()) > self.max_bytes:
                    raise ValueError("Heatmaps exceed the replay cache budget")
            except (KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
                raise ValueError("Invalid saved heatmap files") from exc
            item = SavedSaliency(meta, frames, grids, hashlib.sha256(repr(key).encode()).hexdigest()[:20])
            for old in list(self._cache):
                if old[0] == key[0]:
                    del self._cache[old]
            self._cache[key] = item
            while (
                len(self._cache) > 8
                or sum(g.nbytes for v in self._cache.values() for g in v.grids.values()) > self.max_bytes
            ):
                self._cache.popitem(last=False)
            return item
