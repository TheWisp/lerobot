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

"""Unit tests for the shared OpenArm MJCF/Mink Cartesian transform."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.openarm_description import mink_ik
from lerobot.robots.openarm_description.cartesian_ik import MOTOR_NAMES


class _FakeMinkKinematics:
    """Record the adapter boundary while returning deterministic solutions."""

    def __init__(self, *solutions: np.ndarray | None) -> None:
        self.solutions = list(solutions)
        self.fk_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.sync_calls: list[np.ndarray] = []
        self.target_calls: list[tuple[str, np.ndarray]] = []
        self.gripper_calls: list[tuple[str, float]] = []
        self.solve_calls = 0

    def fk_bimanual(self, right: np.ndarray, left: np.ndarray):
        self.fk_calls.append((np.asarray(right).copy(), np.asarray(left).copy()))
        return (
            np.array([0.12, -0.10, -0.30, 1.0, 0.0, 0.0, 0.0]),
            np.array([0.12, +0.10, -0.30, 1.0, 0.0, 0.0, 0.0]),
        )

    def sync(self, state: np.ndarray) -> None:
        self.sync_calls.append(np.asarray(state).copy())

    def set_target(self, side: str, pose: np.ndarray) -> None:
        self.target_calls.append((side, np.asarray(pose).copy()))

    def set_gripper(self, side: str, value: float) -> None:
        self.gripper_calls.append((side, value))

    def solve(self):
        self.solve_calls += 1
        if not self.solutions:
            raise AssertionError("fake solver ran out of configured solutions")
        solution = self.solutions.pop(0)
        return None if solution is None else np.asarray(solution, dtype=float).copy()


def _action(
    *,
    enabled: float = 1.0,
    reset: float = 0.0,
    left_gripper: float = 11.0,
    right_gripper: float = -12.0,
    **updates: float,
) -> dict[str, float]:
    values = {f"{side}_{key}": 0.0 for side in ("left", "right") for key in mink_ik._ACTION_KEYS}
    for side in ("left", "right"):
        values[f"{side}_enabled"] = enabled
        values[f"{side}_reset"] = reset
    values["left_gripper_pos"] = left_gripper
    values["right_gripper_pos"] = right_gripper
    values.update(updates)
    return values


def _transform(
    monkeypatch: pytest.MonkeyPatch,
    solver: _FakeMinkKinematics,
    *,
    left_seed: np.ndarray | None = None,
    right_seed: np.ndarray | None = None,
    bound_deg: float = 180.0,
) -> mink_ik.BimanualOpenArmMinkIKTransform:
    def _bounds(_self):
        lower = np.full(7, -bound_deg)
        upper = np.full(7, bound_deg)
        return {side: (lower.copy(), upper.copy()) for side in ("left", "right")}

    monkeypatch.setattr(
        mink_ik.BimanualOpenArmMinkIKTransform,
        "_resolve_joint_bounds_deg",
        _bounds,
    )
    return mink_ik.BimanualOpenArmMinkIKTransform(
        solver,
        SimpleNamespace(),
        left_seed_deg=np.zeros(8) if left_seed is None else left_seed,
        right_seed_deg=np.zeros(8) if right_seed is None else right_seed,
        workspace_min=(-10.0, -10.0, -10.0),
        workspace_max=(10.0, 10.0, 10.0),
    )


def _solution(right_deg: np.ndarray, left_deg: np.ndarray) -> np.ndarray:
    return np.deg2rad(np.concatenate([right_deg, left_deg]))


def test_fake_solver_boundary_uses_radians_right_then_left_and_wxyz(monkeypatch):
    right_seed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -8.0])
    left_seed = np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, 9.0])
    right_solved = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 99.0])
    left_solved = np.array([-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, 99.0])
    solver = _FakeMinkKinematics(_solution(right_solved, left_solved))
    transform = _transform(
        monkeypatch,
        solver,
        left_seed=left_seed,
        right_seed=right_seed,
    )

    action = _action(
        right_target_x=0.02,
        right_target_wz=np.pi / 2,
        left_gripper=13.0,
        right_gripper=-14.0,
    )
    output = transform(action)

    fk_right, fk_left = solver.fk_calls[0]
    expected_right = np.r_[right_seed[:7], action["right_gripper_pos"]]
    expected_left = np.r_[left_seed[:7], action["left_gripper_pos"]]
    np.testing.assert_allclose(fk_right, np.deg2rad(expected_right))
    np.testing.assert_allclose(fk_left, np.deg2rad(expected_left))

    state_deg = np.concatenate([expected_right, expected_left])
    np.testing.assert_allclose(solver.sync_calls[0], np.deg2rad(state_deg))
    assert [side for side, _pose in solver.target_calls] == ["right", "left"]

    right_target = solver.target_calls[0][1]
    np.testing.assert_allclose(right_target[:3], [0.14, -0.10, -0.30])
    # openarm_control pose7 is xyz + quaternion wxyz.
    np.testing.assert_allclose(
        right_target[3:],
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1e-7,
    )
    assert [side for side, _value in solver.gripper_calls] == ["right", "left"]
    assert solver.gripper_calls[0][1] == pytest.approx(np.deg2rad(-14.0))
    assert solver.gripper_calls[1][1] == pytest.approx(np.deg2rad(13.0))

    for side, solved, gripper in (
        ("right", right_solved, -14.0),
        ("left", left_solved, 13.0),
    ):
        for index, motor in enumerate(MOTOR_NAMES[:7]):
            assert output[f"{side}_{motor}.pos"] == pytest.approx(solved[index])
        assert output[f"{side}_gripper.pos"] == pytest.approx(gripper)


def test_output_has_exactly_sixteen_motor_keys(monkeypatch):
    solver = _FakeMinkKinematics(np.zeros(16))
    transform = _transform(monkeypatch, solver)

    output = transform(_action(enabled=0.0))

    expected = {f"{side}_{motor}.pos" for side in ("left", "right") for motor in MOTOR_NAMES}
    assert len(output) == 16
    assert set(output) == expected


def test_disabled_side_is_a_strict_joint_hold(monkeypatch):
    right_seed = np.arange(8, dtype=float)
    left_seed = -np.arange(8, dtype=float)
    solver_result = _solution(np.full(8, 6.0), np.full(8, -6.0))
    solver = _FakeMinkKinematics(solver_result)
    transform = _transform(
        monkeypatch,
        solver,
        left_seed=left_seed,
        right_seed=right_seed,
    )

    output = transform(_action(left_enabled=1.0, right_enabled=0.0))

    for index, motor in enumerate(MOTOR_NAMES[:7]):
        assert output[f"right_{motor}.pos"] == pytest.approx(right_seed[index])
        assert output[f"left_{motor}.pos"] == pytest.approx(-6.0)


def test_solver_failure_holds_previous_joints_and_updates_grippers(monkeypatch):
    solved_deg = np.r_[np.full(7, 6.0), 0.0]
    solver = _FakeMinkKinematics(_solution(solved_deg, solved_deg), None)
    transform = _transform(monkeypatch, solver)

    successful = transform(_action(left_gripper=10.0, right_gripper=-10.0))
    held = transform(_action(left_gripper=15.0, right_gripper=-16.0))

    for side in ("left", "right"):
        for motor in MOTOR_NAMES[:7]:
            assert held[f"{side}_{motor}.pos"] == pytest.approx(successful[f"{side}_{motor}.pos"])
    assert held["left_gripper.pos"] == pytest.approx(15.0)
    assert held["right_gripper.pos"] == pytest.approx(-16.0)
    assert transform.hold_per_arm == (True, True)


def test_out_of_bounds_solution_holds_both_arms(monkeypatch):
    right_deg = np.zeros(8)
    right_deg[0] = 11.0
    solver = _FakeMinkKinematics(_solution(right_deg, np.zeros(8)))
    transform = _transform(monkeypatch, solver, bound_deg=10.0)

    output = transform(_action(left_gripper=7.0, right_gripper=-8.0))

    for side in ("left", "right"):
        for motor in MOTOR_NAMES[:7]:
            assert output[f"{side}_{motor}.pos"] == pytest.approx(0.0)
    assert output["left_gripper.pos"] == pytest.approx(7.0)
    assert output["right_gripper.pos"] == pytest.approx(-8.0)
    assert transform.hold_per_arm == (True, True)


def test_reset_ramps_in_degrees_without_calling_solver(monkeypatch):
    solved_deg = np.r_[np.full(7, 6.0), 0.0]
    solver = _FakeMinkKinematics(_solution(solved_deg, solved_deg))
    transform = _transform(monkeypatch, solver)
    moved = transform(_action())
    assert moved["right_joint_1.pos"] == pytest.approx(6.0)
    assert moved["left_joint_1.pos"] == pytest.approx(6.0)

    timestamps = iter((100.0, 100.1))
    monkeypatch.setattr(mink_ik.time, "perf_counter", lambda: next(timestamps))
    first_reset = transform(_action(reset=1.0, left_gripper=20.0, right_gripper=-21.0))
    second_reset = transform(_action(reset=1.0, left_gripper=22.0, right_gripper=-23.0))

    # The first reset tick establishes timing; the next ramps 30 deg/s for 0.1 s.
    assert first_reset["right_joint_1.pos"] == pytest.approx(6.0)
    assert first_reset["left_joint_1.pos"] == pytest.approx(6.0)
    assert second_reset["right_joint_1.pos"] == pytest.approx(3.0)
    assert second_reset["left_joint_1.pos"] == pytest.approx(3.0)
    assert second_reset["right_gripper.pos"] == pytest.approx(-23.0)
    assert second_reset["left_gripper.pos"] == pytest.approx(22.0)
    assert solver.solve_calls == 1
    assert transform.hold_per_arm == (False, False)
