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
        session.mark_input()
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
