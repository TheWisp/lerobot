"""Shared-memory overlay channel for debug-vision models.

The debug-vision subprocess (owner, create=True) writes one RGBA overlay per
camera plus a small JSON meta block; the GUI backend attaches read-only
(create=False) and serves overlays as PNG. A reverse JSON 'control' block lets
the browser push a text prompt / params back to the subprocess.

Ownership mirrors the HVLA SharedImageBuffer convention: the subprocess
creates, everyone else attaches. Torn-read safety is inherited from
SharedBlock (sequence-counter header).
"""

from __future__ import annotations

import json

import numpy as np

from lerobot.policies.hvla.ipc import SharedBlock

_PREFIX = "lerobot_overlay_"
_META_BYTES = 8192
_CONTROL_BYTES = 4096


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


class SharedOverlayBuffer:
    """RGBA overlay channel keyed by camera + meta + control JSON blocks.

    Writer (subprocess): SharedOverlayBuffer(cameras={cam: (h, w)}, model=...,
    create=True). Reader (backend): SharedOverlayBuffer(create=False) — reads
    the meta block to discover cameras + dims, then attaches each RGBA block.

    Raises FileNotFoundError on create=False if the writer hasn't started yet;
    callers attach lazily and retry.
    """

    def __init__(
        self, cameras: dict[str, tuple[int, int]] | None = None, model: str = "", create: bool = True
    ):
        self._meta = SharedBlock(name=_PREFIX + "meta", shape=(_META_BYTES,), dtype=np.uint8, create=create)
        self._control = SharedBlock(
            name=_PREFIX + "control", shape=(_CONTROL_BYTES,), dtype=np.uint8, create=create
        )

        if create:
            assert cameras, "writer must pass cameras={cam_key: (h, w)}"
            self.cameras = {k: (int(v[0]), int(v[1])) for k, v in cameras.items()}
            self.model = model
            self._meta_dict = {"cameras": self.cameras, "model": model, "fps": 0.0}
            _write_json(self._meta, self._meta_dict)
        else:
            meta = _read_json(self._meta)
            self.cameras = {k: (int(v[0]), int(v[1])) for k, v in meta.get("cameras", {}).items()}
            self.model = meta.get("model", "")

        self._blocks: dict[str, SharedBlock] = {}
        for cam, (h, w) in self.cameras.items():
            self._blocks[cam] = SharedBlock(
                name=f"{_PREFIX}img_{_safe(cam)}", shape=(h, w, 4), dtype=np.uint8, create=create
            )

    # ---- writer side (subprocess) ----
    def write_overlay(self, cam_key: str, rgba: np.ndarray) -> None:
        block = self._blocks.get(cam_key)
        if block is None:
            return
        h, w = self.cameras[cam_key]
        if rgba.shape != (h, w, 4):
            return
        block.write(np.ascontiguousarray(rgba, dtype=np.uint8))

    def read_control(self) -> dict:
        return _read_json(self._control)

    # ---- reader side (backend) ----
    def overlay_seq(self, cam_key: str) -> int:
        block = self._blocks.get(cam_key)
        return block.count if block is not None else 0

    def read_overlay(self, cam_key: str) -> tuple[np.ndarray, float] | None:
        block = self._blocks.get(cam_key)
        if block is None or block.count == 0:
            return None
        return block.read()

    def write_control(self, control: dict) -> None:
        _write_json(self._control, control)

    def write_fps(self, fps: float) -> None:
        """Writer: publish the current overlay frame rate via the meta block."""
        self._meta_dict["fps"] = round(float(fps), 1)
        _write_json(self._meta, self._meta_dict)

    def read_fps(self) -> float:
        """Reader: latest overlay frame rate the writer published (0.0 if none)."""
        return float(_read_json(self._meta).get("fps", 0.0))

    def cleanup(self) -> None:
        for block in (*self._blocks.values(), self._meta, self._control):
            block.close()
            if block._owner:
                # safe-destruct: overlay shm we created (owner), freed on model unload
                block.unlink()
