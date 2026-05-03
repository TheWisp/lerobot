"""In-memory frame buffers for online flash-DAgger.

InterventionFrameBuffer
    Per-episode buffer of (obs, action) tuples, partitioned into segments —
    one segment per intervention. A sliding-window chunk dataset built from
    multiple interventions in the same episode must NOT cross segment
    boundaries: a chunk straddling two interventions would pair an obs from
    one demo with actions from a different time/scene, corrupting the
    training signal. Tracking segments here is the cleanest way to enforce
    that downstream.

    Cleared at episode end (after consumption) or on episode abort.

FlashedEpisodePool
    Session-scoped pool of intervention buffers, one per previously-flashed
    episode. Used as the "flashed" slot in the three-way batch mix to
    rehearse past corrections (Phase D recipe).

Both are thin wrappers around lists; no transactional semantics — flash-
DAgger discards on abort, RLT's commit/discard pattern is overkill here.

TODO(future): the buffer captures only while intervention is active. Frames
near the END of an intervention contribute fewer valid sliding-window chunks
(at offset L-chunk_size+1 a chunk would need actions extending past the
intervention end). Capturing for ~chunk_size additional ticks AFTER
intervention ends would let every in-intervention starting frame produce a
valid chunk. The trade-off is what those post-intervention actions
represent: policy actions or held-leader actions, not active demos. May or
may not be an acceptable training target; needs measurement.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class InterventionFrameBuffer:
    """Thread-safe append-only frame log, partitioned into per-intervention segments.

    A segment is opened by ``begin_segment()`` (called at intervention
    start) and closed by ``end_segment()`` (intervention end). Frames
    appended between are grouped into the current segment. ``snapshot()``
    returns the list of segments preserving order.

    Frames are dicts (matches the offline driver's per-tick shape).
    """

    def __init__(self) -> None:
        self._segments: list[list[dict[str, Any]]] = []
        self._segment_open: bool = False
        self._lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────────────────

    def begin_segment(self) -> None:
        """Start a new segment. Idempotent if already open (logs a warning)."""
        with self._lock:
            if self._segment_open:
                logger.warning(
                    "InterventionFrameBuffer.begin_segment called while a segment "
                    "is already open (had %d frames); closing it first.",
                    len(self._segments[-1]) if self._segments else 0,
                )
            self._segments.append([])
            self._segment_open = True

    def end_segment(self) -> None:
        """Close the current segment. Drops the segment if it has no frames."""
        with self._lock:
            self._segment_open = False
            if self._segments and len(self._segments[-1]) == 0:
                self._segments.pop()

    # ── data ───────────────────────────────────────────────────────────

    def append(self, frame: dict[str, Any]) -> None:
        """Append into the current segment. Opens an implicit segment if none."""
        with self._lock:
            if not self._segment_open:
                # Defensive: caller forgot begin_segment. Open an implicit one.
                self._segments.append([])
                self._segment_open = True
            self._segments[-1].append(frame)

    def snapshot(self) -> list[list[dict[str, Any]]]:
        """Return per-segment frame lists (shallow-copied list-of-lists)."""
        with self._lock:
            return [list(seg) for seg in self._segments]

    def clear(self) -> None:
        with self._lock:
            self._segments.clear()
            self._segment_open = False

    # ── inspection ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Total frames across all segments."""
        with self._lock:
            return sum(len(seg) for seg in self._segments)

    def is_empty(self) -> bool:
        return len(self) == 0

    def num_segments(self) -> int:
        with self._lock:
            return sum(1 for seg in self._segments if seg)


class FlashedEpisodePool:
    """Session-scoped collection of past intervention buffers.

    Each entry holds the train/val split of one previously-flashed episode's
    captured frames. Train and val are stored as flat lists (the per-segment
    structure is collapsed at registration time — by the time a correction
    enters the flashed pool, segment boundaries have already been used to
    build proper chunks during fit).

    Keyed by an opaque correction_id (monotonically increasing int) so callers
    can refer back to a specific correction for diagnostics.
    """

    def __init__(self) -> None:
        # correction_id -> {"train": list[frame], "val": list[frame]}
        self._entries: dict[int, dict[str, list[dict]]] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()

    def add(self, train_frames: list[dict], val_frames: list[dict]) -> int:
        """Register a new flashed correction. Returns its correction_id."""
        with self._lock:
            cid = self._next_id
            self._next_id += 1
            self._entries[cid] = {"train": list(train_frames), "val": list(val_frames)}
            return cid

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def correction_ids(self) -> list[int]:
        with self._lock:
            return sorted(self._entries.keys())

    def train_pool(self, cid: int) -> list[dict]:
        with self._lock:
            return list(self._entries[cid]["train"])

    def val_pool(self, cid: int) -> list[dict]:
        with self._lock:
            return list(self._entries[cid]["val"])

    def all_train_pools(self) -> list[list[dict]]:
        """For ThreeWayMix: list of train-frame lists, one per correction."""
        with self._lock:
            return [list(e["train"]) for e in self._entries.values()]
