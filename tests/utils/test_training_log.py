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

from __future__ import annotations

import json

import pytest

from lerobot.common.training_log import (
    TRAINING_LOG_JSON_MARKER,
    TrainingHealthTracker,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_health_tracker_calculates_window_metrics_and_record() -> None:
    clock = FakeClock()
    reset_calls = 0

    def reset_peak_memory() -> None:
        nonlocal reset_calls
        reset_calls += 1

    tracker = TrainingHealthTracker(
        batch_size=4,
        total_steps=10,
        clock=clock,
        peak_memory_gb=lambda: 3.5,
        reset_peak_memory=reset_peak_memory,
    )
    with tracker.measure_data_loading():
        clock.advance(0.2)
    clock.advance(0.8)
    tracker.step()

    sample = tracker.sample(step=1, values={"loss": 0.25, "grdn": 1.5, "lr": 1e-5})

    assert sample.values == pytest.approx(
        {
            "loss": 0.25,
            "grdn": 1.5,
            "lr": 1e-5,
            "eta_seconds": 9.0,
            "updt_s": 0.8,
            "data_s": 0.2,
            "samples_per_s": 4.0,
            "mem_gb": 3.5,
            "step_time_ms": 1000.0,
        }
    )
    payload = json.loads(sample.record.split(TRAINING_LOG_JSON_MARKER, 1)[1])
    assert payload["step"] == 1
    assert payload["total_steps"] == 10
    assert payload["samples_per_s"] == pytest.approx(4.0)
    assert reset_calls == 1


def test_health_tracker_excludes_checkpoint_time_and_smooths_eta() -> None:
    clock = FakeClock()
    tracker = TrainingHealthTracker(batch_size=2, total_steps=10, clock=clock)

    clock.advance(1.0)
    tracker.step()
    first = tracker.sample(step=1, values={"loss": 1.0})
    assert first.values["eta_seconds"] == pytest.approx(9.0)
    tracker.reset()

    clock.advance(0.5)
    tracker.step()
    with tracker.exclude_time():
        clock.advance(20.0)
    second = tracker.sample(step=2, values={"loss": 0.5})

    assert second.values["step_time_ms"] == pytest.approx(500.0)
    assert second.values["eta_seconds"] == pytest.approx(7.2)


def test_health_tracker_omits_nonfinite_telemetry_without_raising() -> None:
    clock = FakeClock()
    tracker = TrainingHealthTracker(batch_size=1, total_steps=500, clock=clock)
    clock.advance(1.0)
    tracker.step()

    sample = tracker.sample(
        step=100,
        values={
            "loss": 0.25,
            "grdn": float("inf"),
            "lr": 1e-5,
        },
    )

    payload = json.loads(sample.record.split(TRAINING_LOG_JSON_MARKER, 1)[1])
    assert payload["loss"] == 0.25
    assert payload["lr"] == 1e-5
    assert "grdn" not in payload
    assert sample.omitted_fields == ("grdn",)
