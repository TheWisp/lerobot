"""Shared-memory auxiliary channel for policy-internal overlays (e.g. attention saliency).

The running policy process (writer, ``create=True``) publishes a small per-camera float
grid — e.g. the 16x16 cross-attention saliency map the policy computed for the action it
just took — and the debug-vision worker attaches read-only (``create=False``) and
colorizes it onto the camera tiles via a ``DebugVisionAdapter``. This is the cross-process
seam every policy-internal overlay rides on: the worker never re-runs the policy, it only
draws what the policy already computed.

Ownership mirrors ``SharedOverlayBuffer`` / ``SharedImageBuffer`` — the writer creates,
everyone else attaches; torn-read safety is inherited from ``SharedBlock``'s
sequence-counter header. Latest-wins (no frame pairing): the overlay is already
lag-tolerant, so the worker just draws the freshest grid per camera.
"""

from __future__ import annotations

import logging

import numpy as np

from lerobot.overlays.overlay_ipc import _read_json, _safe, _write_json
from lerobot.policies.hvla.ipc import SharedBlock

logger = logging.getLogger(__name__)

_PREFIX = "lerobot_aux_"
_META_BYTES = 8192


class SharedAuxBuffer:
    """Per-camera float32 saliency grids + a meta block (cameras -> grid dims).

    Writer (policy process): ``SharedAuxBuffer(cameras={cam: (gh, gw)}, model=..., create=True)``.
    Reader (debug-vision worker): ``SharedAuxBuffer(create=False)`` — reads the meta block to
    discover cameras + grid dims, then attaches each grid block.

    Precondition (reader): raises ``FileNotFoundError`` on ``create=False`` until the writer
    has created the meta block; callers attach lazily and retry. Grids are written/read
    latest-wins; ``saliency_seq`` (the write count) lets a reader skip an unchanged grid.
    """

    def __init__(
        self, cameras: dict[str, tuple[int, int]] | None = None, model: str = "", create: bool = True
    ):
        self._meta = SharedBlock(name=_PREFIX + "meta", shape=(_META_BYTES,), dtype=np.uint8, create=create)
        if create:
            assert cameras, "writer must pass cameras={cam_key: (gh, gw)}"
            self.cameras = {k: (int(v[0]), int(v[1])) for k, v in cameras.items()}
            self.model = model
            _write_json(self._meta, {"cameras": self.cameras, "model": model})
        else:
            meta = _read_json(self._meta)
            self.cameras = {k: (int(v[0]), int(v[1])) for k, v in meta.get("cameras", {}).items()}
            self.model = meta.get("model", "")

        self._blocks: dict[str, SharedBlock] = {}
        self._warned: set[str] = set()  # cam keys already warned about (dropped writes) — once each
        for cam, (gh, gw) in self.cameras.items():
            self._blocks[cam] = SharedBlock(
                name=f"{_PREFIX}grid_{_safe(cam)}", shape=(gh, gw), dtype=np.float32, create=create
            )
        # The policy-side cost of the overlay: last pass wall ms. The GUI badge reads this —
        # the worker's own fps/vram can't see work done in the policy process. Reader-side the
        # attach is LENIENT: a writer that predates the stats block (stale policy process) still
        # serves grids; read_pass_ms just returns None.
        if create:
            self._stats = SharedBlock(name=_PREFIX + "stats", shape=(1,), dtype=np.float32, create=True)
        else:
            try:
                self._stats = SharedBlock(name=_PREFIX + "stats", shape=(1,), dtype=np.float32, create=False)
            except FileNotFoundError:
                self._stats = None

    # ---- writer side (policy process) ----
    def write_saliency(self, cam_key: str, grid: np.ndarray) -> None:
        """Publish the latest saliency grid for ``cam_key``. No-op for an unknown camera or
        a grid whose shape doesn't match the meta dims (caller stays decoupled from sizing)."""
        block = self._blocks.get(cam_key)
        if block is None:
            if cam_key not in self._warned:
                self._warned.add(cam_key)
                logger.warning("aux write dropped: unknown camera %r (have %s)", cam_key, list(self.cameras))
            return
        gh, gw = self.cameras[cam_key]
        if grid.shape != (gh, gw):
            if cam_key not in self._warned:
                self._warned.add(cam_key)
                logger.warning(
                    "aux write dropped for %s: grid %s != expected %s", cam_key, grid.shape, (gh, gw)
                )
            return
        block.write(np.ascontiguousarray(grid, dtype=np.float32))

    def write_pass_ms(self, ms: float) -> None:
        """Publish the last saliency pass's wall time (the net ms added to the inference thread).
        Precondition: this buffer was constructed as the writer (``create=True``)."""
        self._stats.write(np.array([ms], dtype=np.float32))

    def read_pass_ms(self) -> tuple[float, float] | None:
        """Latest (pass_ms, write_timestamp), or None if the writer never published one.
        The timestamp lets the caller drop a stale value after the policy stops publishing."""
        if self._stats is None or self._stats.count == 0:
            return None
        data, ts = self._stats.read()
        return float(data[0]), ts

    # ---- reader side (debug-vision worker) ----
    def saliency_seq(self, cam_key: str) -> int:
        block = self._blocks.get(cam_key)
        return block.count if block is not None else 0

    def read_saliency(self, cam_key: str) -> tuple[np.ndarray, float] | None:
        """Latest (grid, timestamp) for ``cam_key``, or None if nothing has been written yet."""
        block = self._blocks.get(cam_key)
        if block is None or block.count == 0:
            return None
        return block.read()

    def cleanup(self) -> None:
        stats = (self._stats,) if self._stats is not None else ()
        for block in (*self._blocks.values(), *stats, self._meta):
            block.close()
            if block._owner:
                # safe-destruct: aux shm we created (writer) — freed when the policy stops
                block.unlink()


def read_stats_pass_ms() -> tuple[float, float] | None:
    """Attach ONLY the stats block and return (pass_ms, write_timestamp), or None when the block
    is absent or never written. The cheap read for pollers (the GUI badge) — meta and the
    per-camera grid blocks are not touched."""
    try:
        block = SharedBlock(name=_PREFIX + "stats", shape=(1,), dtype=np.float32, create=False)
    except FileNotFoundError:
        return None
    try:
        if block.count == 0:
            return None
        data, ts = block.read()
        return float(data[0]), ts
    finally:
        block.close()
