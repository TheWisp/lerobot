"""Tests for InferenceThread — state management, pause/resume, chunk production.

All tests use a MockS1Policy on CPU — no GPU required.
"""
import pytest
import threading
import time

import numpy as np
import torch

from lerobot.policies.hvla.s1_inference import InferenceThread


class MockS1Policy:
    """Minimal policy that returns a fixed chunk. No GPU needed."""

    supports_rtc = False
    needs_temporal_ensemble = False
    rtc_prefix_length = 0

    def __init__(self, chunk_size=50, action_dim=14):
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self._reset_count = 0

    def predict_action_chunk(self, batch, num_steps=None):
        return torch.randn(1, self._chunk_size, self._action_dim)

    def reset(self):
        self._reset_count += 1

    def to(self, device):
        return self

    def eval(self):
        return self


class MockSharedCache:
    """Minimal SharedLatentCache mock."""

    def read_with_age(self):
        return torch.zeros(2048), 0.0


# Use the real JOINT_NAMES so obs_to_s1_batch works without modification
from lerobot.policies.hvla.s1_process import JOINT_NAMES

def _make_obs():
    """Create a minimal observation dict matching JOINT_NAMES."""
    obs = {}
    for name in JOINT_NAMES:
        obs[name] = 0.0
    obs["front"] = np.zeros((224, 224, 3), dtype=np.uint8)
    return obs


def _make_thread(**kwargs) -> InferenceThread:
    """Create an InferenceThread with test defaults."""
    defaults = dict(
        policy=MockS1Policy(),
        preprocessor=lambda batch: batch,
        postprocessor=lambda actions: actions,
        shared_cache=MockSharedCache(),
        s2_latent_key="observation.s2_latent",
        s1_image_keys=["observation.images.front"],
        joint_names=list(JOINT_NAMES),
        device=torch.device("cpu"),
        resize_to=None,
        fps=30,
    )
    defaults.update(kwargs)
    return InferenceThread(**defaults)


class TestLifecycle:
    """Start/stop behavior."""

    def test_start_stop(self):
        """Thread starts and stops cleanly."""
        thread = _make_thread()
        thread.start()
        assert thread._thread is not None
        assert thread._thread.is_alive()
        thread.stop(timeout=2.0)
        assert not thread._thread.is_alive()

    def test_stop_without_start(self):
        """Stopping before starting doesn't crash."""
        thread = _make_thread()
        thread.stop()  # should be a no-op

    def test_double_stop(self):
        """Stopping twice doesn't crash."""
        thread = _make_thread()
        thread.start()
        thread.stop()
        thread.stop()


class TestChunkProduction:
    """Verify the thread produces chunks from observations."""

    def test_produces_chunk(self):
        """Publishing obs should produce a chunk."""
        thread = _make_thread()
        thread.start()
        try:
            obs = _make_obs()
            thread.publish_obs(obs, time.perf_counter())
            assert thread.wait_for_first_chunk(timeout=5.0)
            chunk, t_origin, t_obs = thread.get_chunk()
            assert chunk is not None
            assert chunk.shape == (50, 14)  # default MockS1Policy
            assert t_origin > 0
        finally:
            thread.stop()

    def test_multiple_chunks(self):
        """Multiple obs should produce multiple chunks (infer_times grows)."""
        thread = _make_thread()
        thread.start()
        try:
            for _ in range(3):
                obs = _make_obs()
                thread.publish_obs(obs, time.perf_counter())
                time.sleep(0.3)  # wait for inference
            assert len(thread.infer_times) >= 2
        finally:
            thread.stop()

    def test_first_chunk_timeout(self):
        """No obs published → wait_for_first_chunk times out."""
        thread = _make_thread()
        thread.start()
        try:
            assert not thread.wait_for_first_chunk(timeout=0.3)
        finally:
            thread.stop()

    def test_chunk_shape_matches_policy(self):
        """Chunk shape should match the policy's output."""
        policy = MockS1Policy(chunk_size=20, action_dim=7)
        thread = _make_thread(policy=policy)
        thread.start()
        try:
            thread.publish_obs(_make_obs(), time.perf_counter())
            thread.wait_for_first_chunk(timeout=5.0)
            chunk, _, _ = thread.get_chunk()
            assert chunk.shape == (20, 7)
        finally:
            thread.stop()


class TestPauseResume:
    """Verify pause/resume behavior."""

    def test_pause_blocks_inference(self):
        """After pause(), no new chunks should be produced."""
        thread = _make_thread()
        thread.start()
        try:
            # Produce first chunk
            thread.publish_obs(_make_obs(), time.perf_counter())
            thread.wait_for_first_chunk(timeout=5.0)

            # Pause and wait for thread to actually block
            thread.pause()
            assert thread.is_paused
            time.sleep(0.5)  # let any in-flight inference finish

            count_before = len(thread.infer_times)

            # Publish more obs while paused
            for _ in range(3):
                thread.publish_obs(_make_obs(), time.perf_counter())
                time.sleep(0.1)

            # No new chunks should have been produced
            count_after = len(thread.infer_times)
            assert count_after == count_before, (
                f"Inference ran while paused: {count_before} → {count_after}"
            )
        finally:
            thread.stop()

    def test_resume_produces_chunks(self):
        """After resume(), inference should produce chunks again."""
        thread = _make_thread()
        thread.start()
        try:
            # Produce first chunk
            thread.publish_obs(_make_obs(), time.perf_counter())
            thread.wait_for_first_chunk(timeout=5.0)

            # Pause then resume
            thread.pause()
            time.sleep(0.2)
            thread.resume()
            assert not thread.is_paused

            count_before = len(thread.infer_times)
            thread.publish_obs(_make_obs(), time.perf_counter())
            time.sleep(0.5)
            assert len(thread.infer_times) > count_before
        finally:
            thread.stop()

    def test_resume_resets_policy(self):
        """resume() should call policy.reset() to clear stale state."""
        policy = MockS1Policy()
        thread = _make_thread(policy=policy)
        thread.start()
        try:
            thread.publish_obs(_make_obs(), time.perf_counter())
            thread.wait_for_first_chunk(timeout=5.0)

            reset_count_before = policy._reset_count
            thread.pause()
            thread.resume()
            assert policy._reset_count == reset_count_before + 1
        finally:
            thread.stop()

    def test_pause_resume_cycle(self):
        """Multiple pause/resume cycles should work."""
        thread = _make_thread()
        thread.start()
        try:
            thread.publish_obs(_make_obs(), time.perf_counter())
            thread.wait_for_first_chunk(timeout=5.0)

            for _ in range(3):
                thread.pause()
                assert thread.is_paused
                time.sleep(0.1)
                thread.resume()
                assert not thread.is_paused
                thread.publish_obs(_make_obs(), time.perf_counter())
                time.sleep(0.3)

            assert len(thread.infer_times) >= 4  # initial + 3 cycles
        finally:
            thread.stop()


class TestExecIndex:
    """Verify exec index tracking for RTC."""

    def test_update_exec_index(self):
        """update_exec_index should be readable from the thread's state."""
        thread = _make_thread()
        thread.update_exec_index(42)
        with thread._main_loop_chunk_idx_lock:
            assert thread._main_loop_chunk_idx == 42


class TestGetChunkThreadSafety:
    """Verify get_chunk doesn't crash under concurrent access."""

    def test_concurrent_read_write(self):
        """Main loop reading chunks while inference thread writes shouldn't crash."""
        thread = _make_thread()
        thread.start()

        errors = []

        def reader():
            for _ in range(50):
                try:
                    thread.get_chunk()
                except Exception as e:
                    errors.append(e)
                time.sleep(0.01)

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()

        try:
            for _ in range(10):
                thread.publish_obs(_make_obs(), time.perf_counter())
                time.sleep(0.05)

            reader_thread.join(timeout=3.0)
            assert not errors, f"Concurrent read errors: {errors}"
        finally:
            thread.stop()
