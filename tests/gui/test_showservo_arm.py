# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The M1 arm's safety core, tested without a bus.

`check_arm_move` is the server's whole authority over the servo worker: whatever the
worker asks for, only what passes here reaches the motors. Every rejection must be a
refusal of the WHOLE command — a partially applied move would leave the excursion
accounting and the arm disagreeing about where it is.
"""

from __future__ import annotations

import pytest

from lerobot.gui.api.showservo import (
    ARM_EXCURSION_LIMIT,
    ARM_STEP_LIMIT,
    M1_JOINTS,
    check_arm_move,
)

START = dict.fromkeys(M1_JOINTS, 10.0)


def test_a_clean_move_returns_absolute_targets():
    targets = check_arm_move({"shoulder_pan": 1.5, "elbow_flex": -2.0}, dict(START), dict(START))
    assert targets == {"shoulder_pan": 11.5, "elbow_flex": 8.0}


def test_only_the_positioning_joints_are_servoable():
    for forbidden in ("gripper", "wrist_flex", "wrist_roll", "forearm_roll", "base"):
        with pytest.raises(ValueError, match="not servoable"):
            check_arm_move({forbidden: 0.1}, dict(START), dict(START))


def test_an_oversized_step_is_rejected_not_clamped():
    with pytest.raises(ValueError, match="exceeds"):
        check_arm_move({"shoulder_pan": ARM_STEP_LIMIT + 0.01}, dict(START), dict(START))


def test_a_non_finite_delta_is_rejected():
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="non-finite"):
            check_arm_move({"shoulder_pan": bad}, dict(START), dict(START))


def test_the_excursion_budget_is_measured_from_the_connect_pose():
    # Crept to the edge of the budget in legal steps; one more legal-sized step
    # must still be refused — the budget binds TOTAL travel, not step size.
    last = dict(START)
    last["shoulder_pan"] = START["shoulder_pan"] + ARM_EXCURSION_LIMIT - 1.0
    check_arm_move({"shoulder_pan": 1.0}, last, dict(START))  # exactly at the edge: fine
    with pytest.raises(ValueError, match="excursion"):
        check_arm_move({"shoulder_pan": 1.5}, last, dict(START))


def test_a_violation_rejects_the_whole_command():
    last = dict(START)
    with pytest.raises(ValueError):
        check_arm_move({"shoulder_pan": 1.0, "elbow_flex": ARM_STEP_LIMIT + 1.0}, last, dict(START))
    assert last == START, "a refused command must not have moved the accounting"


def test_an_empty_move_is_an_error_not_a_noop():
    with pytest.raises(ValueError, match="empty"):
        check_arm_move({}, dict(START), dict(START))
