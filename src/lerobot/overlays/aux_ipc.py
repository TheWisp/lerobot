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

import json
import logging

import numpy as np

from lerobot.policies.hvla.ipc import SharedBlock

logger = logging.getLogger(__name__)

_PREFIX = "lerobot_aux_"
_META_BYTES = 8192


def _safe(cam_key: str) -> str:
    return cam_key.replace("/", "_")


def _write_json(block: SharedBlock, obj: dict) -> None:
    raw = json.dumps(obj).encode("utf-8")[: block.shape[0]]
    buf = np.zeros(block.shape[0], dtype=np.uint8)
    buf[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    block.write(buf)


def _read_json(block: SharedBlock) -> dict:
    if block.count == 0:
        return {}
    data, _ = block.read()
    raw = bytes(data).split(b"\x00", 1)[0]
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


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
        for block in (*self._blocks.values(), self._meta):
            block.close()
            if block._owner:
                # safe-destruct: aux shm we created (writer) — freed when the policy stops
                block.unlink()
