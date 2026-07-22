#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging

import numpy as np
import pytest

from lerobot.robots.openarm_follower.telemetry import FollowerTelemetry


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_telemetry_aggregates_arm_and_excludes_j8() -> None:
    clock = FakeClock()
    telemetry = FollowerTelemetry("right", period_secs=10.0, clock=clock)

    telemetry.update(
        q_cmd=[1, 2, 3, 4, 5, 6, 7, 999],
        q_pos=[0, 4, 2, 4, 4, 8, 7, -999],
        q_torque=[2, -2, 4, -4, 6, -6, 8, 999],
        t_mos=[20, 21, 22, 23, 24, 25, 26, 999],
        tff=[1, -1, 1, -1, 1, -1, 1, 999],
    )
    telemetry.update(
        q_cmd=[0] * 8,
        q_pos=[0] * 8,
        q_torque=[0] * 8,
        t_mos=[30] * 8,
        tff=[0] * 8,
    )

    snapshot = telemetry.snapshot()
    assert snapshot is not None
    assert snapshot.count == 2
    np.testing.assert_allclose(snapshot.position_error_max, [1, 2, 1, 0, 1, 2, 0])
    np.testing.assert_allclose(snapshot.external_torque_mean, [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5])
    np.testing.assert_allclose(snapshot.external_torque_abs_mean, [0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5])
    np.testing.assert_allclose(snapshot.external_torque_abs_max, [1, 1, 3, 3, 5, 5, 7])
    assert snapshot.mos_temperature_max == 30


def test_report_waits_for_period_logs_once_and_resets(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    telemetry = FollowerTelemetry("left", period_secs=10.0, clock=clock)
    telemetry.update([0] * 7, [1] * 7, [2] * 7, [42] * 7)

    assert telemetry.maybe_report(now=109.999) is False
    with caplog.at_level(logging.INFO):
        clock.now = 110.0
        assert telemetry.maybe_report() is True

    assert telemetry.count == 0
    assert telemetry.snapshot() is None
    assert telemetry.maybe_report(now=120.0) is False
    assert "[left] telem n=1" in caplog.text
    assert "tmax=42" in caplog.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("q_cmd", [0, 0, 0, 0, 0, 0]),
        ("q_pos", [0, 0, 0, 0, 0, 0, float("nan")]),
        ("q_torque", [0, 0, 0, 0, 0, 0, float("inf")]),
        ("t_mos", [0, 0, 0, 0, 0, 0, float("-inf")]),
        ("tff", [0, 0, 0, 0, 0, 0]),
    ],
)
def test_invalid_sample_is_rejected_without_poisoning_window(field: str, value: list[float]) -> None:
    telemetry = FollowerTelemetry("arm")
    sample = {
        "q_cmd": [0] * 7,
        "q_pos": [0] * 7,
        "q_torque": [0] * 7,
        "t_mos": [20] * 7,
        "tff": [0] * 7,
    }
    sample[field] = value

    with pytest.raises(ValueError):
        telemetry.update(**sample)

    assert telemetry.count == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": ""}, "name"),
        ({"name": "arm", "n_joints": 0}, "n_joints"),
        ({"name": "arm", "period_secs": 0.0}, "period_secs"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FollowerTelemetry(**kwargs)
