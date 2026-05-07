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

"""Unit tests for LatencyAggregator, LatencyTracer, LatencySnapshotWriter."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from lerobot.utils.latency import (
    LatencyAggregator,
    LatencySnapshotWriter,
    LatencyTracer,
)

# ---------------------------------------------------------------------------
# LatencyAggregator
# ---------------------------------------------------------------------------


class TestAggregator:
    def test_empty_aggregator(self):
        agg = LatencyAggregator()
        assert len(agg) == 0
        assert agg.stage_keys() == []
        assert math.isnan(agg.percentile("missing_ms", 50))
        assert agg.overrun_ratio() == 0.0
        assert agg.time_series("missing_ms") == []
        snap = agg.snapshot()
        assert snap["n_records"] == 0
        assert snap["dropped_records"] == 0
        assert snap["overrun_ratio"] == 0.0
        assert snap["stages"] == {}
        assert agg.representative_iterations() == {}

    def test_ingest_and_percentile(self):
        agg = LatencyAggregator()
        for i in range(100):
            agg.ingest({"t": float(i), "loop_dt_ms": float(i)})
        assert len(agg) == 100
        p50, p95 = agg.percentile("loop_dt_ms", [50, 95])
        assert 49.0 <= p50 <= 50.0
        assert 94.0 <= p95 <= 95.0

    def test_maxlen_eviction(self):
        agg = LatencyAggregator(maxlen=10)
        for i in range(100):
            agg.ingest({"t": float(i), "v_ms": float(i)})
        assert len(agg) == 10
        assert agg.values("v_ms") == [float(i) for i in range(90, 100)]

    def test_invalid_maxlen_rejected(self):
        with pytest.raises(AssertionError):
            LatencyAggregator(maxlen=0)
        with pytest.raises(AssertionError):
            LatencyAggregator(maxlen=-1)

    def test_overrun_ratio(self):
        agg = LatencyAggregator()
        for i in range(10):
            agg.ingest({"t": float(i), "overrun": (i % 2 == 0)})
        assert agg.overrun_ratio() == pytest.approx(0.5)
        agg.ingest({"t": 10.0})
        assert agg.overrun_ratio() == pytest.approx(0.5)

    def test_time_series_filters_by_window(self):
        agg = LatencyAggregator()
        for i in range(10):
            agg.ingest({"t": float(i), "v_ms": float(i)})
        ts = agg.time_series("v_ms", last_n_seconds=3.0)
        assert len(ts) == 4
        assert ts[0] == (6.0, 6.0)
        assert ts[-1] == (9.0, 9.0)

    def test_snapshot_shape(self):
        agg = LatencyAggregator()
        for i in range(20):
            agg.ingest({"t": float(i), "loop_dt_ms": float(i), "motor_read_ms": float(i) / 2})
        snap = agg.snapshot(percentiles=(50, 95))
        assert snap["n_records"] == 20
        assert set(snap["stages"].keys()) == {"loop_dt_ms", "motor_read_ms"}
        loop = snap["stages"]["loop_dt_ms"]
        assert {"p50", "p95", "count"} <= set(loop.keys())
        assert loop["count"] == 20

    def test_clear_resets_state(self):
        agg = LatencyAggregator()
        for i in range(5):
            agg.ingest({"t": float(i), "v_ms": float(i)})
        agg.dropped_records = 3
        agg.clear()
        assert len(agg) == 0
        assert agg.dropped_records == 0

    def test_representative_iterations_picks_percentiles(self):
        agg = LatencyAggregator()
        # Ingest records with linearly increasing loop_dt and a span dict
        # that varies by step so we can check the right one was picked.
        for i in range(100):
            agg.ingest(
                {
                    "t": float(i),
                    "step": i,
                    "loop_dt_ms": float(i),
                    "spans": {"work": [0.0, float(i)]},
                    "cam_events": {},
                }
            )
        reps = agg.representative_iterations()
        assert set(reps.keys()) == {"median", "p95", "p99", "max", "latest"}
        # Each entry should be a small subset (no other keys leak).
        assert set(reps["median"].keys()) == {"step", "t", "loop_dt_ms", "spans", "cam_events"}
        # Sanity: the chosen records should have loop_dt_ms close to the
        # corresponding percentile of the input distribution (0..99).
        assert 45.0 <= reps["median"]["loop_dt_ms"] <= 55.0
        assert 90.0 <= reps["p95"]["loop_dt_ms"] <= 99.0
        assert reps["max"]["loop_dt_ms"] == 99.0
        assert reps["latest"]["loop_dt_ms"] == 99.0


# ---------------------------------------------------------------------------
# LatencyTracer
# ---------------------------------------------------------------------------


class TestTracer:
    def test_span_records_endpoint_pair_and_duration(self):
        tracer = LatencyTracer()
        tracer.start()
        with tracer.span("work"):
            time.sleep(0.005)
        record = tracer.commit()
        # Duration scalar still emitted for sparkline / percentile queries.
        assert "work_ms" in record
        assert 3.0 <= record["work_ms"] <= 30.0
        # Endpoint pair available for Gantt rendering.
        assert "spans" in record
        assert "work" in record["spans"]
        start, end = record["spans"]["work"]
        # Endpoints are offsets from iter_start; both >= 0 and end >= start.
        assert start >= 0.0
        assert end >= start
        # Duration matches end - start.
        assert end - start == pytest.approx(record["work_ms"])

    def test_commit_without_aggregator(self):
        tracer = LatencyTracer()
        tracer.start()
        record = tracer.commit()
        assert record["loop_kind"] == "teleop"
        assert "loop_dt_ms" in record
        assert "step" in record
        # Even with no spans, the spans/cam_events keys exist (empty dicts).
        assert record["spans"] == {}
        assert record["cam_events"] == {}

    def test_commit_with_aggregator_ingests(self):
        agg = LatencyAggregator()
        tracer = LatencyTracer(agg)
        for _ in range(3):
            tracer.start()
            tracer.commit()
        assert len(agg) == 3

    def test_step_counter_increments(self):
        tracer = LatencyTracer()
        steps = []
        for _ in range(5):
            tracer.start()
            steps.append(tracer.commit()["step"])
        assert steps == [0, 1, 2, 3, 4]

    def test_cam_consume_records_capture_and_consume_offsets(self):
        tracer = LatencyTracer()
        tracer.start()
        # Frame captured 20 ms ago — captured_at_ms should be negative.
        t0 = time.perf_counter() - 0.020
        tracer.cam_consume("top", t0)
        record_first = tracer.commit()
        assert "cam_top_stale_ms" in record_first
        # No period yet (only one consume).
        assert "cam_top_period_ms" not in record_first
        ev = record_first["cam_events"]["top"]
        assert ev["captured_at_ms"] < 0  # captured before iter_start
        assert ev["consumed_at_ms"] >= 0  # consumed inside the iter
        # Stale matches consumed - captured.
        assert record_first["cam_top_stale_ms"] == pytest.approx(ev["consumed_at_ms"] - ev["captured_at_ms"])

        # Second consume with frame 33 ms after the previous → period appears.
        tracer.start()
        t1 = t0 + 0.033
        tracer.cam_consume("top", t1)
        record_second = tracer.commit()
        assert record_second["cam_top_period_ms"] == pytest.approx(33.0, abs=0.5)

    def test_overrun_flag_when_target_fps_set(self):
        tracer = LatencyTracer(target_fps=120.0)
        tracer.start()
        time.sleep(0.020)
        record = tracer.commit()
        assert record["overrun"] is True

        tracer.start()
        record_fast = tracer.commit()
        assert record_fast["overrun"] is False

    def test_no_overrun_field_when_no_target_fps(self):
        tracer = LatencyTracer()
        tracer.start()
        record = tracer.commit()
        assert "overrun" not in record

    def test_assertion_when_no_start(self):
        tracer = LatencyTracer()
        with pytest.raises(AssertionError):
            tracer.commit()
        with pytest.raises(AssertionError):
            with tracer.span("foo"):
                pass
        with pytest.raises(AssertionError):
            tracer.cam_consume("top", time.perf_counter())

    def test_set_field_extends_record(self):
        tracer = LatencyTracer()
        tracer.start()
        tracer.set_field("ep", 7)
        tracer.set_field("note", "smoke")
        record = tracer.commit()
        assert record["ep"] == 7
        assert record["note"] == "smoke"

    def test_gantt_can_be_reconstructed_from_spans(self):
        """An iteration's spans should describe a coherent timeline that the
        GUI can render: every span's [start, end] is contained within
        [0, loop_dt_ms]; cameras can sit before zero (background grab)."""
        tracer = LatencyTracer()
        tracer.start()
        # Camera captured 30 ms ago.
        cam_ts = time.perf_counter() - 0.030
        tracer.cam_consume("top", cam_ts)
        with tracer.span("motor_read"):
            time.sleep(0.002)
        with tracer.span("process_action"):
            time.sleep(0.001)
        record = tracer.commit()

        loop_dt = record["loop_dt_ms"]
        for name, (start, end) in record["spans"].items():
            assert 0.0 <= start <= end <= loop_dt + 0.5, f"span {name} out of bounds"
        ev = record["cam_events"]["top"]
        assert ev["captured_at_ms"] < 0  # before iter
        assert 0.0 <= ev["consumed_at_ms"] <= loop_dt


# ---------------------------------------------------------------------------
# Hot-path overhead — synthetic loop benchmark
# ---------------------------------------------------------------------------


class TestOverhead:
    """Capture overhead must stay well below 1 ms / iter at 60 Hz."""

    @pytest.mark.parametrize("with_aggregator", [True, False])
    def test_per_iteration_overhead(self, with_aggregator):
        agg = LatencyAggregator() if with_aggregator else None
        tracer = LatencyTracer(agg, target_fps=60.0)

        n_iters = 5000
        for _ in range(100):  # warmup
            tracer.start()
            with tracer.span("a"):
                pass
            with tracer.span("b"):
                pass
            tracer.cam_consume("top", time.perf_counter())
            tracer.commit()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            tracer.start()
            with tracer.span("motor_read"):
                pass
            tracer.cam_consume("top", time.perf_counter())
            with tracer.span("process_obs"):
                pass
            with tracer.span("process_action"):
                pass
            with tracer.span("action_send"):
                pass
            tracer.commit()
        elapsed = time.perf_counter() - t0
        per_iter_us = (elapsed / n_iters) * 1e6

        # Strict goal is ~10 µs; CI is noisy. 200 µs catches a 20× regression.
        assert per_iter_us < 200.0, f"per-iter overhead {per_iter_us:.1f} µs > 200 µs"


# ---------------------------------------------------------------------------
# Snapshot series + LatencySnapshotWriter
# ---------------------------------------------------------------------------


class TestSnapshotSeries:
    def test_snapshot_with_series_window(self):
        agg = LatencyAggregator()
        for i in range(60):
            agg.ingest({"t": float(i), "loop_dt_ms": float(i)})
        snap = agg.snapshot(series_window_s=10.0, series_max_points=10)
        assert "series" in snap
        series = snap["series"]["loop_dt_ms"]
        assert len(series) <= 10
        for t, _v in series:
            assert t >= 49.0

    def test_snapshot_without_series(self):
        agg = LatencyAggregator()
        agg.ingest({"t": 0.0, "v_ms": 1.0})
        snap = agg.snapshot()
        assert "series" not in snap


class TestSnapshotWriter:
    def test_first_write_creates_file(self, tmp_path: Path):
        agg = LatencyAggregator()
        agg.ingest(
            {
                "t": 0.0,
                "step": 0,
                "loop_dt_ms": 12.5,
                "spans": {"work": [0.0, 12.0]},
                "cam_events": {},
            }
        )
        writer = LatencySnapshotWriter(tmp_path, interval_s=0.0)
        wrote = writer.maybe_write(agg)
        assert wrote
        assert writer.path.exists()
        data = json.loads(writer.path.read_text())
        assert data["loop_kind"] == "teleop"
        assert "stages" in data
        assert "loop_dt_ms" in data["stages"]
        assert "iterations" in data
        # Every label points at a record subset with timeline keys.
        for label, rec in data["iterations"].items():
            assert "spans" in rec
            assert "cam_events" in rec
            assert "loop_dt_ms" in rec, f"missing loop_dt_ms in {label}"

    def test_throttle_skips_within_interval(self, tmp_path: Path):
        agg = LatencyAggregator()
        agg.ingest({"t": 0.0, "loop_dt_ms": 1.0})
        writer = LatencySnapshotWriter(tmp_path, interval_s=10.0)
        assert writer.maybe_write(agg, now=100.0) is True
        assert writer.maybe_write(agg, now=101.0) is False
        assert writer.maybe_write(agg, now=111.0) is True

    def test_invalid_interval_rejected(self, tmp_path: Path):
        with pytest.raises(AssertionError):
            LatencySnapshotWriter(tmp_path, interval_s=-1)

    def test_atomic_replace_no_partial_file(self, tmp_path: Path):
        agg = LatencyAggregator()
        agg.ingest({"t": 0.0, "v_ms": 1.0})
        writer = LatencySnapshotWriter(tmp_path, interval_s=0.0)
        writer.maybe_write(agg)
        assert writer.path.exists()
        tmp = tmp_path / "latency_snapshot.json.tmp"
        assert not tmp.exists()
