# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Camera frames in shared memory: one writer (the GUI server, at the camera's rate), any number of readers (the
point groups' view), each taking the newest complete frame with no disk, encoder or decoder in between.

The segment is a header (magic, version, height, width, slots, the newest frame's number, the intrinsics K) and
``slots`` slots, each a sequence number, the time the frame was read from the camera, an RGB image (H x W x 3 uint8)
and a depth image (H x W uint16, mm). The writer marks a slot odd while it writes and even when done, a seqlock: a
reader keeps its copy only if the slot's number was even and the same before and after the copy. Pure numpy, so the
tracker's own environment can load it by path."""

from __future__ import annotations

import contextlib
import time
from multiprocessing import resource_tracker, shared_memory

import numpy as np

MAGIC = 0x5246524C  # "LRFR"
VERSION = 1
HEADER = 256  # magic, version, height, width, slots (u32 each), newest (i64 at 24), K (9 f64 at 32)


def _slot_bytes(h: int, w: int) -> int:
    return (16 + h * w * 3 + h * w * 2 + 63) // 64 * 64


@contextlib.contextmanager
def _untracked():
    """Attach without Python's resource tracker, which would unlink the writer's segment when a reader exits."""
    register, unregister = resource_tracker.register, resource_tracker.unregister
    resource_tracker.register = lambda *a, **kw: None
    resource_tracker.unregister = lambda *a, **kw: None
    try:
        yield
    finally:
        resource_tracker.register, resource_tracker.unregister = register, unregister


class FrameRing:
    """Pre: the writer creates (``create=True`` with the frame size); readers attach by name."""

    def __init__(
        self, name: str, height: int = 0, width: int = 0, slots: int = 4, k=None, create: bool = False
    ):
        if create:
            assert height > 0 and width > 0 and slots >= 2
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=HEADER + slots * _slot_bytes(height, width)
            )
            head = np.ndarray((5,), np.uint32, self.shm.buf, 0)
            head[:] = (MAGIC, VERSION, height, width, slots)
            np.ndarray((1,), np.int64, self.shm.buf, 24)[0] = -1
            np.ndarray((9,), np.float64, self.shm.buf, 32)[:] = np.asarray(
                np.eye(3) if k is None else k, float
            ).ravel()
        else:
            with _untracked():
                self.shm = shared_memory.SharedMemory(name=name)
            head = np.ndarray((5,), np.uint32, self.shm.buf, 0)
            assert head[0] == MAGIC and head[1] == VERSION, f"{name} is not a frame ring"
        self.name, self.owner = name, create
        self.h, self.w, self.slots = (int(x) for x in np.ndarray((5,), np.uint32, self.shm.buf, 0)[2:5])
        self.k = np.ndarray((9,), np.float64, self.shm.buf, 32).reshape(3, 3).copy()
        self._newest = np.ndarray((1,), np.int64, self.shm.buf, 24)
        size = _slot_bytes(self.h, self.w)
        self._seq, self._t, self._rgb, self._depth = [], [], [], []
        for i in range(self.slots):
            base = HEADER + i * size
            self._seq.append(np.ndarray((1,), np.int64, self.shm.buf, base))
            self._t.append(np.ndarray((1,), np.float64, self.shm.buf, base + 8))
            self._rgb.append(np.ndarray((self.h, self.w, 3), np.uint8, self.shm.buf, base + 16))
            self._depth.append(
                np.ndarray((self.h, self.w), np.uint16, self.shm.buf, base + 16 + self.h * self.w * 3)
            )

    @property
    def newest(self) -> int:
        """The newest complete frame's number, -1 before the first."""
        return int(self._newest[0])

    def write(self, rgb: np.ndarray, depth_mm: np.ndarray, t: float | None = None) -> int:
        """One frame into the next slot; its number."""
        n = self.newest + 1
        i = n % self.slots
        self._seq[i][0] = 2 * n + 1  # writing
        self._rgb[i][:] = rgb
        self._depth[i][:] = depth_mm
        self._t[i][0] = time.time() if t is None else t
        self._seq[i][0] = 2 * n + 2  # done
        self._newest[0] = n
        return n

    def read(self, after: int = -1) -> tuple[int, float, np.ndarray, np.ndarray] | None:
        """The newest complete frame if it is newer than ``after``: (number, time read from the camera, RGB,
        depth mm), copies. None when there is nothing newer, or the slot was being rewritten during the copy."""
        n = self.newest
        if n <= after or n < 0:
            return None
        i = n % self.slots
        before = int(self._seq[i][0])
        if before != 2 * n + 2:
            return None
        t, rgb, depth = float(self._t[i][0]), self._rgb[i].copy(), self._depth[i].copy()
        if int(self._seq[i][0]) != before:
            return None
        return n, t, rgb, depth

    def close(self) -> None:
        self.shm.close()
        if self.owner:
            with contextlib.suppress(FileNotFoundError):
                self.shm.unlink()  # safe-destruct: shm cleanup we created
