"""Camera frames in shared memory: what the writer puts in, a reader in another process gets out, newest first,
whole or not at all."""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np

from lerobot.showservo.frame_ring import FrameRing

H, W = 480, 848  # the rig's camera


def _name() -> str:
    return f"lerobot_test_ring_{os.getpid()}_{time.monotonic_ns()}"


def _frame(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full((H, W, 3), n % 251, np.uint8), np.full((H, W), 500 + n, np.uint16)


def test_a_reader_gets_the_newest_complete_frame_and_nothing_twice():
    k = np.array([[604.2, 0.0, 419.2], [0.0, 604.2, 250.6], [0.0, 0.0, 1.0]])
    ring = FrameRing(_name(), H, W, slots=4, k=k, create=True)
    try:
        reader = FrameRing(ring.name)
        assert reader.read() is None and (reader.h, reader.w) == (H, W) and np.allclose(reader.k, k)
        for n in range(6):  # wraps the 4 slots
            ring.write(*_frame(n), t=100.0 + n)
        got = reader.read()
        assert got is not None
        n, t, rgb, depth = got
        assert n == 5 and t == 105.0 and (rgb == 5).all() and (depth == 505).all()
        assert reader.read(after=n) is None, "no frame twice"
        reader.close()
    finally:
        ring.close()


def test_a_slot_rewritten_during_the_copy_is_not_handed_out():
    ring = FrameRing(_name(), H, W, slots=2, create=True)
    try:
        ring.write(*_frame(0))
        reader = FrameRing(ring.name)
        i = 0 % ring.slots
        ring._seq[i][0] = 2 * 0 + 1  # the writer is in the middle of the slot
        assert reader.read() is None
        ring._seq[i][0] = 2 * 0 + 2
        assert reader.read() is not None
        reader.close()
    finally:
        ring.close()


def _reader_process(name: str, out) -> None:
    ring = FrameRing(name)
    seen, last = [], -1
    deadline = time.time() + 5.0
    while time.time() < deadline and len(seen) < 3:
        got = ring.read(after=last)
        if got is None:
            time.sleep(0.001)
            continue
        n, _t, rgb, depth = got
        assert (rgb == n % 251).all() and (depth == 500 + n).all(), "a torn frame"
        seen.append(n)
        last = n
    ring.close()
    out.put(seen)


def test_another_process_reads_whole_frames_and_its_exit_leaves_the_segment():
    ring = FrameRing(_name(), H, W, slots=4, create=True)
    try:
        q = mp.get_context("spawn").Queue()
        p = mp.get_context("spawn").Process(target=_reader_process, args=(ring.name, q))
        p.start()
        for n in range(60):
            ring.write(*_frame(n))
            time.sleep(0.01)
        seen = q.get(timeout=10)
        p.join(timeout=10)
        assert len(seen) == 3 and seen == sorted(seen)
        ring.write(*_frame(60))  # the reader is gone; the writer's segment is still there
        assert FrameRing(ring.name).read() is not None
    finally:
        ring.close()
