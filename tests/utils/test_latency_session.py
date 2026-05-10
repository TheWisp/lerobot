#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``LatencySession`` — the per-loop monitoring lifecycle."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.utils.latency import LatencySession

# ---------------------------------------------------------------------------
# Disabled session — must be a true no-op
# ---------------------------------------------------------------------------


class TestDisabledSession:
    def test_disabled_factory_returns_disabled(self):
        session = LatencySession.from_config(enabled=False, loop_kind="teleop", target_fps=30)
        assert session.enabled is False
        assert session.tracer is None
        assert session.aggregator is None
        assert session.writer is None

    def test_disabled_iteration_yields_session(self):
        session = LatencySession.disabled()
        with session.iteration() as ls:
            # Must return *the session* so call sites can `with session.iteration() as ls: ls.span(...)`
            assert ls is session

    def test_disabled_span_is_nullcontext(self):
        session = LatencySession.disabled()
        # span() must be usable as a context manager and not raise without a prior start().
        with session.span("anything"):
            pass

    def test_disabled_methods_are_silent_noops(self, tmp_path: Path):
        session = LatencySession.disabled()
        # All these would raise on a real tracer (no start() called); on a
        # disabled session they must silently no-op.
        session.add_span("foo", time.perf_counter())
        session.cam_consume("top", time.perf_counter())
        session.cam_consume_all({"top": SimpleNamespace(latest_timestamp=time.perf_counter())})
        session.cam_consume_all(None)
        session.set_field("ep", 1)

    def test_disabled_no_snapshot_written(self, tmp_path: Path):
        # Even if from_config is asked for an output_dir, enabled=False
        # short-circuits and never creates a writer.
        session = LatencySession.from_config(
            enabled=False,
            loop_kind="teleop",
            target_fps=30,
            output_dir=tmp_path,
        )
        with session.iteration():
            pass
        assert not (tmp_path / "latency_snapshot.json").exists()


# ---------------------------------------------------------------------------
# Enabled session — iteration commits, snapshot publishes, summary fires
# ---------------------------------------------------------------------------


class TestEnabledSession:
    def test_iteration_commits_record(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with session.iteration():
            with session.span("work"):
                time.sleep(0.001)
        assert len(session.aggregator) == 1

    def test_iteration_records_multiple(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        for _ in range(5):
            with session.iteration():
                pass
        assert len(session.aggregator) == 5

    def test_iteration_commits_even_on_exception(self):
        """If the loop body raises, the iteration must still commit so the
        next start() doesn't fail with 'tracer not reset'."""
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with pytest.raises(ValueError):
            with session.iteration():
                raise ValueError("boom")
        assert len(session.aggregator) == 1
        # And the next iteration must work cleanly.
        with session.iteration():
            pass
        assert len(session.aggregator) == 2

    def test_span_records_duration_inside_iteration(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with session.iteration():
            with session.span("motor_read"):
                time.sleep(0.005)
        records = list(session.aggregator._records)
        assert "motor_read_ms" in records[0]
        assert records[0]["motor_read_ms"] >= 4.0  # tolerate scheduling jitter

    def test_cam_consume_all_skips_cameras_without_timestamp(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        cams = {
            "front": SimpleNamespace(latest_timestamp=time.perf_counter() - 0.01),
            "wrist": SimpleNamespace(latest_timestamp=None),  # None → skip
            "broken": SimpleNamespace(),  # no attr → skip
        }
        with session.iteration():
            session.cam_consume_all(cams)
        record = list(session.aggregator._records)[0]
        assert "cam_front_stale_ms" in record
        assert "cam_wrist_stale_ms" not in record
        assert "cam_broken_stale_ms" not in record

    def test_cam_consume_all_handles_none_cameras_dict(self):
        """Robots without cameras pass None / empty dict; must not raise."""
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with session.iteration():
            session.cam_consume_all(None)
            session.cam_consume_all({})

    def test_explicit_start_end_iter_for_loops_with_continue(self):
        """Loops with branchy `continue` paths use start_iter / end_iter
        explicitly — `continue` skips end_iter but the next start_iter()
        resets cleanly. Verify no leftover state."""
        session = LatencySession.from_config(enabled=True, loop_kind="record", target_fps=30)

        # Iteration 1: starts, then "continues" without end_iter.
        session.start_iter()
        with session.span("partial_work"):
            pass
        # No end_iter() — simulating continue.

        # Iteration 2: clean start + commit.
        session.start_iter()
        with session.span("real_work"):
            pass
        session.end_iter()

        # Only iteration 2 should be in the aggregator.
        assert len(session.aggregator) == 1
        record = list(session.aggregator._records)[0]
        assert "real_work_ms" in record
        assert "partial_work_ms" not in record

    def test_disabled_start_end_iter_are_noops(self):
        session = LatencySession.disabled()
        # Must not raise even though there's no real tracer underneath.
        session.start_iter()
        with session.span("foo"):
            pass
        session.end_iter()

    def test_add_span_records_outside_with_block(self):
        """Branchy code (record_loop's action selection) uses add_span
        instead of `with span()`. Verify it's accessible via the session."""
        session = LatencySession.from_config(enabled=True, loop_kind="record", target_fps=30)
        with session.iteration():
            t0 = time.perf_counter()
            time.sleep(0.002)
            session.add_span("process_action", t0)
        record = list(session.aggregator._records)[0]
        assert "process_action_ms" in record
        assert record["process_action_ms"] >= 1.5


# ---------------------------------------------------------------------------
# Snapshot publishing — throttled by interval
# ---------------------------------------------------------------------------


class TestSnapshotPublish:
    def test_snapshot_published_when_writer_configured(self, tmp_path: Path):
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="teleop",
            target_fps=30,
            output_dir=tmp_path,
        )
        # Force the writer's interval to 0 so every iteration publishes.
        session.writer._interval_s = 0.0
        with session.iteration():
            with session.span("work"):
                pass
        snap_path = tmp_path / "latency_snapshot.json"
        assert snap_path.exists()
        data = json.loads(snap_path.read_text())
        assert data["loop_kind"] == "teleop"

    def test_no_writer_when_output_dir_missing(self):
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        assert session.writer is None
        # Iteration still works; aggregator just isn't published to disk.
        with session.iteration():
            pass
        assert len(session.aggregator) == 1


# ---------------------------------------------------------------------------
# Stderr summary — throttled by summary_interval_s
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_fires_after_interval(self, caplog):
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="teleop",
            target_fps=30,
            summary_interval_s=0.0,  # fire on every iteration
        )
        with caplog.at_level("INFO", logger="lerobot.utils.latency.session"):
            with session.iteration():
                with session.span("work"):
                    pass
        # First iteration produces a summary because interval=0.
        assert any("[latency:teleop]" in r.message for r in caplog.records)

    def test_summary_skipped_when_no_records(self, caplog):
        """Even with interval=0, the summary should skip when the
        aggregator is empty (e.g., the very first call before any commit)."""
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="teleop",
            target_fps=30,
            summary_interval_s=0.0,
        )
        with caplog.at_level("INFO", logger="lerobot.utils.latency.session"):
            # Force-run _after_commit on a fresh aggregator.
            session._after_commit()
        assert not any("[latency:teleop]" in r.message for r in caplog.records)

    def test_summary_throttled_by_default(self, caplog):
        """With default 1s interval, two back-to-back iterations produce
        only one summary line."""
        session = LatencySession.from_config(enabled=True, loop_kind="teleop", target_fps=30)
        with caplog.at_level("INFO", logger="lerobot.utils.latency.session"):
            for _ in range(3):
                with session.iteration():
                    pass
        # 0 or 1 summary lines — never 3.
        n_summary = sum(1 for r in caplog.records if "[latency:teleop]" in r.message)
        assert n_summary <= 1


# ---------------------------------------------------------------------------
# Process / track schema — multi-thread loops (HVLA) need a way to identify
# which thread published a snapshot so the dashboard can stack them.
# ---------------------------------------------------------------------------


class TestProcessTrackSchema:
    def test_snapshot_carries_explicit_process_and_track(self, tmp_path: Path):
        """When passed explicitly, ``process`` and ``track`` land in the
        published snapshot envelope. HVLA main + inference both set
        ``process="hvla"`` so the dashboard can group them visually."""
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="hvla_main",
            target_fps=30,
            output_dir=tmp_path,
            process="hvla",
            track="main",
        )
        session.writer._interval_s = 0.0
        with session.iteration():
            pass
        data = json.loads((tmp_path / "latency_snapshot.json").read_text())
        assert data["process"] == "hvla"
        assert data["track"] == "main"
        assert data["loop_kind"] == "hvla_main"  # original field preserved

    def test_snapshot_defaults_process_and_track_to_loop_kind(self, tmp_path: Path):
        """Single-track loops (teleop, record) don't need to set process /
        track explicitly — they default to loop_kind so the snapshot
        envelope is always populated."""
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="teleop",
            target_fps=30,
            output_dir=tmp_path,
        )
        session.writer._interval_s = 0.0
        with session.iteration():
            pass
        data = json.loads((tmp_path / "latency_snapshot.json").read_text())
        assert data["process"] == "teleop"
        assert data["track"] == "teleop"


# ---------------------------------------------------------------------------
# Overhead microbenchmark — guards against silent regressions in the hot
# path. The thresholds below were measured on the dev box (3+ GHz x86,
# Python 3.12); they have a comfortable margin so they shouldn't flake on
# slower CI hardware. If a future change pushes us over, we either fix the
# regression or raise the bound deliberately (with a comment explaining
# why).
# ---------------------------------------------------------------------------


class TestOverhead:
    """Profiling itself must be cheap. These tests pin the per-iteration
    cost of start_iter / span / end_iter so a future refactor can't
    silently double the hot-path overhead."""

    def _measure_per_iter_us(self, n: int, body_fn) -> float:
        """Run ``body_fn`` ``n`` times and return microseconds per call."""
        # Warm up the JIT-y bits (dict allocation patterns, attribute
        # lookups), so the timed loop sees steady-state cost.
        for _ in range(min(n, 1000)):
            body_fn()
        t0 = time.perf_counter()
        for _ in range(n):
            body_fn()
        elapsed = time.perf_counter() - t0
        return (elapsed / n) * 1e6

    def test_disabled_session_is_effectively_free(self):
        """Disabled session: a full iteration with several spans should
        cost under ~2 µs per iteration. The bound is the sum of a few
        Python method calls + nullcontext enters/exits — anything more
        and we've broken the no-op contract somewhere."""
        session = LatencySession.disabled()

        def body():
            with session.iteration():
                with session.span("a"):
                    pass
                with session.span("b"):
                    pass
                session.cam_consume_all(None)

        per_iter_us = self._measure_per_iter_us(10000, body)
        assert per_iter_us < 5.0, (
            f"Disabled session iteration cost {per_iter_us:.2f} µs > 5 µs budget. "
            "The whole point of LatencySession.disabled() is that it costs "
            "essentially nothing — investigate before raising this bound."
        )

    def test_enabled_no_op_iteration_under_budget(self):
        """Enabled session with no spans / no writer: just start_iter +
        end_iter + commit (deque append, percentile-snapshot check, log
        check). Should run under ~10 µs per iteration."""
        session = LatencySession.from_config(
            enabled=True,
            loop_kind="bench",
            target_fps=60,
            summary_interval_s=1e9,  # disable the summary log path entirely
        )

        def body():
            with session.iteration():
                pass

        per_iter_us = self._measure_per_iter_us(10000, body)
        assert per_iter_us < 25.0, (
            f"Enabled empty iteration cost {per_iter_us:.2f} µs > 25 µs budget. "
            "At 60 fps the loop budget is 16667 µs — even a 25 µs cost is "
            "0.15% of the budget, but anything above that is creeping toward "
            "noticeable."
        )

    def test_enabled_realistic_iteration_under_budget(self):
        """Enabled session with a realistic shape — a few spans, a few
        cam_consume calls — matching a teleop / HVLA-main iteration.
        Should run under ~30 µs per iteration."""
        from types import SimpleNamespace

        session = LatencySession.from_config(
            enabled=True,
            loop_kind="bench",
            target_fps=60,
            summary_interval_s=1e9,
        )
        cams = {f"cam{i}": SimpleNamespace(latest_timestamp=time.perf_counter()) for i in range(3)}

        def body():
            with session.iteration():
                with session.span("get_observation"):
                    pass
                session.cam_consume_all(cams)
                with session.span("publish_obs"):
                    pass
                with session.span("get_chunk"):
                    pass
                with session.span("action_send"):
                    pass
                session.set_field("intervention", False)

        per_iter_us = self._measure_per_iter_us(5000, body)
        assert per_iter_us < 60.0, (
            f"Realistic iteration cost {per_iter_us:.2f} µs > 60 µs budget. "
            "At 30 fps (33333 µs budget) this is still <0.2%, but we want "
            "headroom; investigate any regression."
        )
