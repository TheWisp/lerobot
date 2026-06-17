# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the lerobot-train stdout parser (progress + auto metrics)."""

from __future__ import annotations

import math

import pytest

from lerobot.gui.training.log_parse import (
    ProgressSample,
    parse_metric_sample,
    parse_progress,
)

# ── Progress (tqdm bar) ──────────────────────────────────────────────────────


def test_progress_full_bar():
    line = "Training:   1%|▏         | 125/10000 [02:02<2:36:10,  1.05step/s]"
    s = parse_progress(line)
    assert s == ProgressSample(step=125, total_steps=10000, eta_seconds=2 * 3600 + 36 * 60 + 10)


def test_progress_mm_ss_eta():
    s = parse_progress("Training:  50%|##### | 5/10 [00:03<00:03,  1.5step/s]")
    assert s.step == 5 and s.total_steps == 10
    assert s.eta_seconds == 3


def test_progress_eta_unknown_is_none():
    # tqdm prints '?' before it can estimate.
    s = parse_progress("Training:   0%| | 0/10000 [00:00<?, ?step/s]")
    assert s is not None
    assert s.step == 0
    assert s.total_steps == 10000
    assert s.eta_seconds is None


@pytest.mark.parametrize(
    "line",
    [
        "INFO some unrelated log line",
        "step:100 loss:0.5",  # metric line, not a tqdm bar
        "",
        "Training: starting…",  # no counts
    ],
)
def test_progress_non_match_returns_none(line):
    assert parse_progress(line) is None


# ── Metrics (auto-captured bag) ──────────────────────────────────────────────


def test_metric_sample_core_fields():
    line = "INFO 2026-06-17 step:1000 smpl:32K loss:0.0343 grdn:1.234 lr:1.0e-05 updt_s:0.123"
    bag = parse_metric_sample(line)
    assert bag["step"] == 1000
    assert bag["loss"] == pytest.approx(0.0343)
    assert bag["grdn"] == pytest.approx(1.234)
    assert bag["lr"] == pytest.approx(1e-5)
    assert bag["updt_s"] == pytest.approx(0.123)


def test_metric_sample_is_auto_not_fixed():
    # A brand-new metric lerobot might add must be captured with no code change.
    line = "step:500 loss:0.2 some_new_metric:42.0 another_one:3"
    bag = parse_metric_sample(line)
    assert bag["some_new_metric"] == 42.0
    assert bag["another_one"] == 3.0


def test_metric_sample_magnitude_suffix():
    bag = parse_metric_sample("step:2000 smpl:1.5M loss:0.1")
    assert bag["smpl"] == pytest.approx(1.5e6)


def test_metric_sample_real_lerobot_line():
    # The exact shape lerobot's MetricsTracker.__str__ emits: step/smpl/ep via
    # format_big_number (K/M/B/T/Q suffixes), epch float, then the AverageMeters
    # (loss/grdn ':.3f', lr ':0.1e'). Locks the parser to the real format.
    line = "INFO 2026-06-17 12:00:00 step:10K smpl:320K ep:1K epch:0.50 loss:0.034 grdn:1.234 lr:1.0e-05 updt_s:0.123 data_s:0.001"
    bag = parse_metric_sample(line)
    assert bag["step"] == pytest.approx(10_000)
    assert bag["smpl"] == pytest.approx(320_000)
    assert bag["ep"] == pytest.approx(1_000)
    assert bag["epch"] == pytest.approx(0.50)
    assert bag["loss"] == pytest.approx(0.034)
    assert bag["grdn"] == pytest.approx(1.234)
    assert bag["lr"] == pytest.approx(1e-5)


def test_metric_sample_billion_suffix_not_dropped():
    # format_big_number uses 'B' for billion (not 'G') — a step/smpl in the
    # billions must scale, not silently drop.
    bag = parse_metric_sample("step:2B loss:0.1")
    assert bag["step"] == pytest.approx(2e9)


def test_metric_sample_skips_non_numeric_values():
    bag = parse_metric_sample("level:INFO tag:run_a step:10 loss:0.9")
    assert "level" not in bag and "tag" not in bag
    assert bag["step"] == 10 and bag["loss"] == 0.9


def test_metric_sample_requires_loss():
    # The tqdm bar and other lines carry step but no loss → not a metric sample.
    assert parse_metric_sample("step:100 lr:1e-5 grdn:0.5") is None
    assert parse_metric_sample("INFO checkpoint saved at step:100") is None


def test_metric_sample_all_values_finite_floats():
    bag = parse_metric_sample("step:1 loss:0.5 lr:1e-3")
    assert bag is not None
    assert all(isinstance(v, float) and math.isfinite(v) for v in bag.values())


def test_metric_and_progress_are_disjoint_on_their_lines():
    # The tqdm line is progress-only; the metric line is metric-only.
    tqdm = "Training:  10%| | 1000/10000 [00:30<04:30, 33.0step/s]"
    metric = "step:1000 loss:0.04 lr:1e-5"
    assert parse_progress(tqdm) is not None and parse_metric_sample(tqdm) is None
    assert parse_metric_sample(metric) is not None and parse_progress(metric) is None
