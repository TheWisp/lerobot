"""Tests for InterventionFrameBuffer + FlashedEpisodePool."""

from __future__ import annotations

import threading

import torch

from lerobot.policies.hvla.flash_dagger.buffer import (
    FlashedEpisodePool,
    InterventionFrameBuffer,
)


def _frame(action: float = 1.0) -> dict:
    return {"obs": {"observation.state": torch.full((6,), action)}, "action": torch.tensor([action])}


def test_buffer_single_segment_len_clear():
    buf = InterventionFrameBuffer()
    assert buf.is_empty()
    buf.begin_segment()
    buf.append(_frame(0.0))
    buf.append(_frame(1.0))
    buf.end_segment()
    assert len(buf) == 2
    assert buf.num_segments() == 1
    assert not buf.is_empty()
    buf.clear()
    assert len(buf) == 0
    assert buf.num_segments() == 0
    assert buf.is_empty()


def test_buffer_multiple_segments_partition():
    """Two interventions in one episode should produce two distinct segments."""
    buf = InterventionFrameBuffer()
    buf.begin_segment()
    buf.append(_frame(0.0))
    buf.append(_frame(0.1))
    buf.end_segment()
    buf.begin_segment()
    buf.append(_frame(1.0))
    buf.append(_frame(1.1))
    buf.append(_frame(1.2))
    buf.end_segment()
    snap = buf.snapshot()
    assert len(snap) == 2
    assert len(snap[0]) == 2
    assert len(snap[1]) == 3
    assert len(buf) == 5
    assert buf.num_segments() == 2


def test_buffer_empty_segment_dropped_on_end():
    """begin_segment with no appends, then end_segment, should drop the empty segment."""
    buf = InterventionFrameBuffer()
    buf.begin_segment()
    buf.end_segment()
    assert len(buf) == 0
    assert buf.num_segments() == 0
    # And a subsequent populated segment still works
    buf.begin_segment()
    buf.append(_frame())
    buf.end_segment()
    assert buf.num_segments() == 1


def test_buffer_snapshot_independent_of_subsequent_clear():
    buf = InterventionFrameBuffer()
    buf.begin_segment()
    buf.append(_frame(0.0))
    buf.append(_frame(1.0))
    buf.end_segment()
    snap = buf.snapshot()
    buf.clear()
    assert len(snap) == 1
    assert len(snap[0]) == 2
    assert len(buf) == 0


def test_buffer_implicit_segment_on_orphan_append():
    """Appending without an open segment should defensively open one."""
    buf = InterventionFrameBuffer()
    buf.append(_frame())  # no begin_segment first
    assert len(buf) == 1
    assert buf.num_segments() == 1


def test_buffer_thread_safe_concurrent_appends():
    """Concurrent appends from multiple threads should not lose data."""
    buf = InterventionFrameBuffer()
    buf.begin_segment()
    n_threads = 8
    n_per_thread = 50

    def worker():
        for _ in range(n_per_thread):
            buf.append(_frame())

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(buf) == n_threads * n_per_thread


def test_pool_add_returns_monotonic_ids():
    pool = FlashedEpisodePool()
    cid0 = pool.add([_frame()], [_frame()])
    cid1 = pool.add([_frame()], [_frame()])
    cid2 = pool.add([_frame()], [_frame()])
    assert cid0 == 0
    assert cid1 == 1
    assert cid2 == 2
    assert pool.correction_ids() == [0, 1, 2]


def test_pool_train_val_separation():
    pool = FlashedEpisodePool()
    train = [_frame(0.1), _frame(0.2)]
    val = [_frame(0.9)]
    cid = pool.add(train, val)
    assert len(pool.train_pool(cid)) == 2
    assert len(pool.val_pool(cid)) == 1


def test_pool_all_train_pools_returns_one_per_correction():
    pool = FlashedEpisodePool()
    pool.add([_frame(), _frame()], [_frame()])
    pool.add([_frame(), _frame(), _frame()], [_frame()])
    all_pools = pool.all_train_pools()
    assert len(all_pools) == 2
    assert len(all_pools[0]) == 2
    assert len(all_pools[1]) == 3
