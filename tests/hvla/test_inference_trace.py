# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The inference trace must be inert and truthful.

Four wrong diagnoses of one stuck rollout came from reconstructing behaviour
out of summary statistics. This trace exists so that stops — which only helps
if the trace itself cannot lie. The properties below are ordered by how much
damage their failure would do:

  1. it never perturbs the rollout,
  2. it never raises into the control loop,
  3. what it reports is what happened.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.policies.hvla.inference_trace import InferenceTrace


def _chunk(seed: int = 0, t: int = 50, d: int = 16) -> np.ndarray:
    return np.arange(t * d, dtype=np.float32).reshape(t, d) + seed


def _record(tr: InferenceTrace, infer_id: int = 0, chunk: np.ndarray | None = None, **kw):
    tr.record_inference(
        infer_id=infer_id,
        t_obs=100.0 + infer_id,
        raw_state=kw.get("raw_state", np.arange(48, dtype=np.float32)),
        normalized_state=kw.get("normalized_state", np.zeros(48, dtype=np.float32)),
        prefix=kw.get("prefix", np.ones((2, 16), dtype=np.float32)),
        prefix_len=kw.get("prefix_len", 2),
        expected_d=kw.get("expected_d", 2),
        actual_d=kw.get("actual_d", 2),
        exec_idx=kw.get("exec_idx", 2),
        chunk=_chunk(infer_id) if chunk is None else chunk,
    )


# ── 1. inert ─────────────────────────────────────────────────────────────────


def test_recording_does_not_mutate_the_caller_arrays(tmp_path):
    """The control loop reuses its buffers; a trace that wrote into them would
    change the robot's behaviour, which is the one thing it must never do."""
    tr = InferenceTrace(tmp_path)
    chunk = _chunk()
    state = np.arange(48, dtype=np.float32)
    prefix = np.ones((2, 16), dtype=np.float32)
    before = (chunk.copy(), state.copy(), prefix.copy())

    _record(tr, chunk=chunk, raw_state=state, prefix=prefix)
    sent = np.full(16, 3.0, dtype=np.float32)
    tr.record_step(
        step=0, episode_index=0, frame_index=0, chunk_t_obs=100.0, chunk_index=2,
        chunk_action=chunk[2], sent_action=sent, jump_clamped=False,
    )

    assert np.array_equal(chunk, before[0])
    assert np.array_equal(state, before[1])
    assert np.array_equal(prefix, before[2])


def test_the_record_survives_the_caller_reusing_its_buffer(tmp_path):
    """The loop mutates chunk buffers in place. Holding a reference rather than
    a copy would silently rewrite history after the fact — a trace that changes
    retroactively is worse than no trace."""
    tr = InferenceTrace(tmp_path)
    chunk = _chunk()
    state = np.arange(48, dtype=np.float32)
    prefix = np.ones((2, 16), dtype=np.float32)
    sent = np.full(16, 7.0, dtype=np.float32)
    _record(tr, chunk=chunk, raw_state=state, normalized_state=state, prefix=prefix)
    tr.record_step(
        step=0, episode_index=0, frame_index=0, chunk_t_obs=100.0, chunk_index=2,
        chunk_action=chunk[2], sent_action=sent, jump_clamped=False,
    )

    # The loop reuses every one of these buffers on the next tick.
    chunk[:] = -999.0
    state[:] = -999.0
    prefix[:] = -999.0
    sent[:] = -999.0
    tr.close()

    inf = np.load(tmp_path / "inferences.npz")
    st = np.load(tmp_path / "steps.npz")
    for name, arr in (
        ("chunk", inf["chunk"][0]),
        ("raw_state", inf["raw_state"][0]),
        ("normalized_state", inf["normalized_state"][0]),
        ("prefix", inf["prefix"][0]),
        ("chunk_action", st["chunk_action"][0]),
        ("sent_action", st["sent_action"][0]),
    ):
        assert not np.any(arr == -999.0), f"{name} was stored by reference"
    assert np.array_equal(inf["chunk"][0], _chunk())
    assert np.array_equal(inf["raw_state"][0], np.arange(48, dtype=np.float32))


def test_recording_touches_no_rng(tmp_path):
    """Flow matching samples noise. A trace that advanced the RNG would change
    every subsequent chunk, so the traced rollout would not be the real one."""
    import random

    import torch

    tr = InferenceTrace(tmp_path)
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    t0, r0, n0 = torch.rand(4), random.random(), np.random.rand(4)

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    _record(tr)
    tr.record_step(
        step=0, episode_index=0, frame_index=0, chunk_t_obs=100.0, chunk_index=0,
        chunk_action=np.zeros(16, np.float32), sent_action=np.zeros(16, np.float32),
        jump_clamped=False,
    )
    t1, r1, n1 = torch.rand(4), random.random(), np.random.rand(4)

    assert torch.equal(t0, t1)
    assert r0 == r1
    assert np.array_equal(n0, n1)


def test_nothing_is_written_before_close(tmp_path):
    """No filesystem I/O on the control path."""
    tr = InferenceTrace(tmp_path)
    for i in range(20):
        _record(tr, infer_id=i)

    assert list(tmp_path.iterdir()) == []


# ── 2. never raises ──────────────────────────────────────────────────────────


def test_a_bad_record_disables_the_trace_instead_of_raising(tmp_path):
    """Called from the control loop. Killing a rollout over a diagnostic would
    be a far worse bug than the one being diagnosed."""
    tr = InferenceTrace(tmp_path)

    class _Explodes:
        def __array__(self, *a, **k):
            raise RuntimeError("boom")

    tr.record_inference(
        infer_id=0, t_obs=1.0, raw_state=None, normalized_state=None, prefix=None,
        prefix_len=0, expected_d=0, actual_d=0, exec_idx=None, chunk=_Explodes(),
    )
    _record(tr, infer_id=1)  # must not raise either

    assert tr.close() is None, "a disabled trace must not write a partial file"


def test_an_unwritable_directory_does_not_raise(tmp_path):
    tr = InferenceTrace(tmp_path / "sub")
    _record(tr)
    blocker = tmp_path / "sub"
    blocker.write_text("not a directory")

    assert tr.close() is None


def test_closing_an_empty_trace_writes_nothing(tmp_path):
    assert InferenceTrace(tmp_path).close() is None
    assert list(tmp_path.iterdir()) == []


# ── 3. truthful ──────────────────────────────────────────────────────────────


def test_the_chunk_read_back_is_the_chunk_recorded(tmp_path):
    tr = InferenceTrace(tmp_path)
    for i in range(3):
        _record(tr, infer_id=i)
    tr.close()

    z = np.load(tmp_path / "inferences.npz")
    assert z["chunk"].shape == (3, 50, 16)
    for i in range(3):
        assert np.array_equal(z["chunk"][i], _chunk(i))
        assert z["infer_id"][i] == i


def test_steps_join_to_inferences_by_infer_id(tmp_path):
    """The join is the whole point: it answers 'which plan, which index' without
    inferring it from action deltas."""
    tr = InferenceTrace(tmp_path)
    _record(tr, infer_id=7)  # _record sets t_obs = 100.0 + infer_id
    tr.record_step(
        step=99, episode_index=0, frame_index=99, chunk_t_obs=107.0, chunk_index=3,
        chunk_action=_chunk(7)[3], sent_action=_chunk(7)[3], jump_clamped=False,
    )
    tr.close()

    inf = np.load(tmp_path / "inferences.npz")
    st = np.load(tmp_path / "steps.npz")
    row = int(np.where(inf["t_obs"] == st["chunk_t_obs"][0])[0][0])
    assert inf["infer_id"][row] == 7

    assert np.array_equal(inf["chunk"][row][st["chunk_index"][0]], st["chunk_action"][0])


def test_the_jump_clamp_stays_visible(tmp_path):
    """The loop can clamp a chunk value to ±30° before sending. Recording only
    the sent action would attribute the clamp to the model."""
    tr = InferenceTrace(tmp_path)
    _record(tr, infer_id=0)
    planned = np.full(16, 90.0, dtype=np.float32)
    actually_sent = np.full(16, 30.0, dtype=np.float32)
    tr.record_step(
        step=1, episode_index=0, frame_index=1, chunk_t_obs=100.0, chunk_index=2,
        chunk_action=planned, sent_action=actually_sent, jump_clamped=True,
    )
    tr.close()

    st = np.load(tmp_path / "steps.npz")
    assert np.array_equal(st["chunk_action"][0], planned)
    assert np.array_equal(st["sent_action"][0], actually_sent)
    assert bool(st["jump_clamped"][0]) is True


def test_a_missing_prefix_is_distinguishable_from_a_zero_prefix(tmp_path):
    """The first chunk of an episode has no prefix. Storing that as zeros would
    read as 'the previous chunk commanded 0°', which is a different claim."""
    tr = InferenceTrace(tmp_path)
    _record(tr, infer_id=0, prefix=None, prefix_len=0)
    _record(tr, infer_id=1, prefix=np.zeros((2, 16), dtype=np.float32), prefix_len=2)
    tr.close()

    z = np.load(tmp_path / "inferences.npz")
    assert z["prefix_len"][0] == 0
    assert z["prefix_len"][1] == 2
    assert not np.array_equal(z["prefix"][0], z["prefix"][1])


def test_the_record_cap_drops_rather_than_grows_without_bound(tmp_path):
    """A long rollout must not exhaust memory; and the drop must be reported,
    because a silently truncated trace reads as a complete one."""
    tr = InferenceTrace(tmp_path, max_records=5)
    for i in range(12):
        _record(tr, infer_id=i)
    tr.close()

    import json

    meta = json.loads((tmp_path / "trace_meta.json").read_text())
    assert meta["n_inferences"] == 5
    assert meta["dropped"] == 7


@pytest.mark.parametrize("exec_idx", [None, 0, 5])
def test_exec_idx_round_trips_including_none(tmp_path, exec_idx):
    tr = InferenceTrace(tmp_path)
    _record(tr, infer_id=0, exec_idx=exec_idx)
    tr.close()

    got = int(np.load(tmp_path / "inferences.npz")["exec_idx"][0])
    assert got == (-1 if exec_idx is None else exec_idx)
