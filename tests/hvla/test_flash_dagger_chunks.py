"""Tests for InterventionChunkDataset — segment-aware chunking."""

from __future__ import annotations

import torch

from lerobot.policies.hvla.flash_dagger.fitter import InterventionChunkDataset


def _frame(state_val: float, action_val: float) -> dict:
    return {
        "obs": {"observation.state": torch.full((6,), state_val)},
        "action": torch.tensor([action_val]),
    }


def test_chunks_single_segment_produces_sliding_windows():
    seg = [_frame(float(i), float(i)) for i in range(10)]
    ds = InterventionChunkDataset([seg], chunk_size=4)
    # Valid windows: 0..6 inclusive → 7 chunks
    assert len(ds) == 7
    # First chunk: obs at i=0, actions at i=0..3
    s0 = ds[0]
    assert torch.equal(s0["action"], torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
    assert torch.equal(s0["observation.state"], torch.full((6,), 0.0))
    # Last chunk: obs at i=6, actions at i=6..9
    s_last = ds[6]
    assert torch.equal(s_last["action"], torch.tensor([[6.0], [7.0], [8.0], [9.0]]))


def test_chunks_two_segments_never_cross_boundary():
    """Critical: actions at index N from segment 0 must never appear in
    a chunk that starts in segment 1, and vice versa."""
    seg_a = [_frame(0.0, float(i)) for i in range(5)]  # actions 0..4
    seg_b = [_frame(1.0, float(10 + i)) for i in range(5)]  # actions 10..14
    ds = InterventionChunkDataset([seg_a, seg_b], chunk_size=3)
    # seg_a contributes 5-3+1 = 3 chunks; seg_b contributes 3 chunks; total 6.
    assert len(ds) == 6
    # Chunks 0..2 are from seg_a (actions 0..4 only)
    for i in range(3):
        s = ds[i]
        assert s["action"].max().item() < 5.0, f"chunk {i} crossed into seg_b"
    # Chunks 3..5 are from seg_b (actions 10..14 only)
    for i in range(3, 6):
        s = ds[i]
        assert s["action"].min().item() >= 10.0, f"chunk {i} crossed into seg_a"


def test_chunks_short_segment_contributes_zero_chunks():
    seg_short = [_frame(0.0, 0.0)]  # length 1
    seg_long = [_frame(1.0, float(i)) for i in range(5)]
    ds = InterventionChunkDataset([seg_short, seg_long], chunk_size=4)
    # Short contributes max(0, 1-4+1)=0; long contributes 5-4+1=2
    assert len(ds) == 2


def test_chunks_backward_compat_flat_list():
    """A flat list of frame dicts is treated as a single segment."""
    flat = [_frame(0.0, float(i)) for i in range(6)]
    ds = InterventionChunkDataset(flat, chunk_size=3)
    # 6-3+1 = 4 chunks
    assert len(ds) == 4


def test_chunks_empty_segments_total_zero():
    ds = InterventionChunkDataset([[], []], chunk_size=4)
    assert len(ds) == 0


def test_chunks_action_is_pad_all_false():
    """All windows are valid (no padding); action_is_pad should be all False."""
    seg = [_frame(0.0, float(i)) for i in range(5)]
    ds = InterventionChunkDataset([seg], chunk_size=3)
    s = ds[0]
    assert s["action_is_pad"].dtype == torch.bool
    assert not s["action_is_pad"].any()
