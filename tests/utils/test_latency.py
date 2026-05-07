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

"""Unit tests for LatencyAggregator and LatencyTracer (V1 phase 1)."""

from __future__ import annotations

import math
import time

import pytest

from lerobot.utils.latency import LatencyAggregator, LatencyTracer

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

    def test_ingest_and_percentile(self):
        agg = LatencyAggregator()
        for i in range(100):
            agg.ingest({"t": float(i), "loop_dt_ms": float(i)})
        assert len(agg) == 100
        # 0..99 → p50 ≈ 49.5, p95 ≈ 94.05
        p50, p95 = agg.percentile("loop_dt_ms", [50, 95])
        assert 49.0 <= p50 <= 50.0
        assert 94.0 <= p95 <= 95.0

    def test_maxlen_eviction(self):
        agg = LatencyAggregator(maxlen=10)
        for i in range(100):
            agg.ingest({"t": float(i), "v_ms": float(i)})
        assert len(agg) == 10
        # Only the last 10 should remain (90..99)
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
        # Records without 'overrun' key are excluded
        agg.ingest({"t": 10.0})
        assert agg.overrun_ratio() == pytest.approx(0.5)

    def test_time_series_filters_by_window(self):
        agg = LatencyAggregator()
        for i in range(10):
            agg.ingest({"t": float(i), "v_ms": float(i)})
        # Latest t = 9.0; last 3 s = t >= 6.0 → samples 6..9
        ts = agg.time_series("v_ms", last_n_seconds=3.0)
        assert len(ts) == 4
        assert ts[0] == (6.0, 6.0)
        assert ts[-1] == (9.0, 9.0)

    def test_time_series_skips_missing(self):
        agg = LatencyAggregator()
        agg.ingest({"t": 1.0, "v_ms": 1.0})
        agg.ingest({"t": 2.0})  # missing v_ms
        agg.ingest({"t": 3.0, "v_ms": 3.0})
        ts = agg.time_series("v_ms")
        assert ts == [(1.0, 1.0), (3.0, 3.0)]

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


# ---------------------------------------------------------------------------
# LatencyTracer
# ---------------------------------------------------------------------------


class TestTracer:
    def test_span_records_duration(self):
        tracer = LatencyTracer()
        tracer.start()
        with tracer.span("work"):
            time.sleep(0.005)
        record = tracer.commit()
        assert "work_ms" in record
        # 5 ms sleep: tolerate scheduling jitter generously
        assert 3.0 <= record["work_ms"] <= 30.0

    def test_commit_without_aggregator(self):
        tracer = LatencyTracer()
        tracer.start()
        record = tracer.commit()
        assert record["loop_kind"] == "teleop"
        assert "loop_dt_ms" in record
        assert "step" in record

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

    def test_cam_consume_records_stale_and_period(self):
        tracer = LatencyTracer()
        tracer.start()
        # First consume: stale only, no period
        t0 = time.perf_counter() - 0.020  # frame is 20 ms old
        tracer.cam_consume("top", t0)
        record_first = tracer.commit()
        assert "cam_top_stale_ms" in record_first
        assert "cam_top_period_ms" not in record_first
        assert 15.0 <= record_first["cam_top_stale_ms"] <= 50.0

        # Second consume: period appears
        tracer.start()
        t1 = t0 + 0.033  # 33 ms after the previous frame
        tracer.cam_consume("top", t1)
        record_second = tracer.commit()
        assert "cam_top_period_ms" in record_second
        assert record_second["cam_top_period_ms"] == pytest.approx(33.0, abs=0.5)

    def test_e2e_and_residual_computed_when_input_marked(self):
        tracer = LatencyTracer()
        tracer.start()
        # Pretend a camera frame from 50 ms ago is consumed
        cam_ts = time.perf_counter() - 0.050
        tracer.cam_consume("top", cam_ts)
        with tracer.span("process_obs"):
            time.sleep(0.002)
        with tracer.span("action_send"):
            time.sleep(0.001)
        record = tracer.commit()
        assert "e2e_obs_to_action_ms" in record
        # E2E ≈ 50 ms (initial age) + 2 ms + 1 ms + overhead ≈ 53+ ms
        assert record["e2e_obs_to_action_ms"] >= 50.0
        # residual = e2e - max_stale - (process_obs + action_send)
        # max_stale already covers most of the camera age, so residual ≈ 0
        assert "residual_ms" in record
        assert abs(record["residual_ms"]) < 5.0

    def test_residual_excludes_motor_read_from_sum(self):
        """Motor-read happens before cam-consume, so its time is already in
        cam_stale. Including it in the residual sum would double-count."""
        tracer = LatencyTracer()
        tracer.start()
        with tracer.span("motor_read"):
            time.sleep(0.005)  # 5 ms
        cam_ts = time.perf_counter() - 0.030  # cam captured 30 ms ago
        tracer.cam_consume("top", cam_ts)
        record = tracer.commit()
        # residual should be near zero, not -5 ms; motor_read is excluded.
        assert abs(record["residual_ms"]) < 3.0

    def test_overrun_flag_when_target_fps_set(self):
        tracer = LatencyTracer(target_fps=120.0)  # period = 8.33 ms
        tracer.start()
        time.sleep(0.020)  # 20 ms, definitely an overrun
        record = tracer.commit()
        assert record["overrun"] is True

        tracer.start()
        # No work; should not overrun
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

    def test_mark_input_affects_e2e(self):
        tracer = LatencyTracer()
        tracer.start()
        # Mark a synthetic input from 25 ms ago
        tracer.mark_input(time.perf_counter() - 0.025)
        record = tracer.commit()
        assert "e2e_obs_to_action_ms" in record
        assert 24.0 <= record["e2e_obs_to_action_ms"] <= 50.0


# ---------------------------------------------------------------------------
# Hot-path overhead — synthetic loop benchmark
# ---------------------------------------------------------------------------


class TestOverhead:
    """Verify capture overhead is well below 1 ms per iteration at 60 Hz.

    The doc commits to < 1 ms/iter; in practice we expect ~5–10 µs.
    Under a noisy CI runner we relax the bound to 200 µs to stay non-flaky
    while still catching catastrophic regressions.
    """

    @pytest.mark.parametrize("with_aggregator", [True, False])
    def test_per_iteration_overhead(self, with_aggregator):
        agg = LatencyAggregator() if with_aggregator else None
        tracer = LatencyTracer(agg, target_fps=60.0)

        n_iters = 5000
        # Warmup
        for _ in range(100):
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
