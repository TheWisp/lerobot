"""A permitted surface turn is one reference-relative choice across its lifetime."""

import time
from math import pi

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.planner import InfeasibleTrajectoryError, JointPlanner, PlannerLimits
from lerobot.policies.flash.planner_worker import PlannerWorker, PlanRequest
from lerobot.policies.flash.skill import (
    FlashSkill,
    Interval,
    MotionMode,
    Relation,
    SpatialSample,
    TurnAtPoint,
)
from lerobot.policies.flash.trajectory import ConnectionTrajectory, Evidence, PoseEstimate, spatial_goal
from lerobot.policies.flash.turn_choice import select_turn_angle


def pose(x=0.0, y=0.0):
    result = np.eye(4)
    result[:2, 3] = x, y
    return result


class OrbitKinematics:
    """One joint sweeps a gripper around a pivot; used to verify the chosen branch."""

    def __init__(self):
        self.ik_calls = 0

    def inverse_kinematics(self, seed, target):
        self.ik_calls += 1
        return np.array([np.rad2deg(np.arctan2(target[1, 3], target[0, 3]))])

    def forward_kinematics(self, joints):
        angle = np.deg2rad(joints[0])
        result = pose(np.cos(angle), np.sin(angle))
        result[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
        return result


def make_interval(name="place", rule=None):
    sample = SpatialSample(0, pose(1), pose())
    last = SpatialSample(1, pose(1), pose())
    return Interval(
        name,
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "target",
        (sample, last),
        speed_m_s=1.0,
        reference_version="target-v1",
        turn_at_point=rule,
    )


def test_turn_rule_roundtrip_and_reference_relative_goal(tmp_path):
    rule = TurnAtPoint("placement-turn", (0, 0, 0), (0, 0, 1), -pi, pi)
    interval = make_interval(rule=rule)
    skill = FlashSkill("demo/source", 0, (interval, make_interval("settle", rule)))
    path = tmp_path / "skill.json"
    skill.save(path)
    assert FlashSkill.load(path).intervals[0].turn_at_point == rule
    assert FlashSkill.load(path).intervals[1].turn_at_point == rule
    goal = spatial_goal(interval, interval.samples[-1], pose(0.4, 0.2), None, pi / 2)
    np.testing.assert_allclose(goal[:3, 3], [0.4, 1.2, 0], atol=1e-8)
    np.testing.assert_allclose(goal[:3, :3], Rotation.from_euler("z", pi / 2).as_matrix())
    with pytest.raises(ValueError, match="permitted range"):
        spatial_goal(interval, interval.samples[-1], pose(), None, pi * 1.1)


def test_rule_cannot_change_or_restart_between_keyframes():
    rule = TurnAtPoint("same", (0, 0, 0), (0, 0, 1), -pi, pi)
    changed = TurnAtPoint("same", (0.01, 0, 0), (0, 0, 1), -pi, pi)
    with pytest.raises(ValueError, match="identical"):
        FlashSkill("demo/source", 0, (make_interval("first", rule), make_interval("second", changed)))
    with pytest.raises(ValueError, match="adjacent"):
        FlashSkill(
            "demo/source",
            0,
            (make_interval("first", rule), make_interval("gap"), make_interval("third", rule)),
        )


def test_async_worker_selects_reachable_turn_and_rebuilds_connection():
    rule = TurnAtPoint("turn", (0, 0, 0), (0, 0, 1), -pi, pi, 17, "observed")
    interval = make_interval(rule=rule)
    kinematics = OrbitKinematics()
    planner = JointPlanner(kinematics, ("wrist",), "gripper", PlannerLimits((1000,), 0.01, 0.01, 10))
    seed = np.array([90.0])
    reference = PoseEstimate("target", pose(), 0, 0, Evidence.OBSERVED, "target-v1", (True,) * 3, (True,) * 3)
    chosen = select_turn_angle(planner, interval, seed, reference.pose, None)
    assert chosen == pytest.approx(pi / 2, abs=0.02)
    assert kinematics.ik_calls <= planner.limits.max_turn_ik_evaluations
    predicted = make_interval(rule=TurnAtPoint("uncertain", (0, 0, 0), (0, 0, 1), -pi, pi, 17, "predicted"))
    with pytest.raises(InfeasibleTrajectoryError, match="directly observed pose"):
        select_turn_angle(planner, predicted, seed, reference.pose, None)
    original_connection = ConnectionTrajectory.begin(
        interval, kinematics.forward_kinematics(seed), reference.pose, None, 0
    )
    assert original_connection.turn_angle_rad == 0
    published = []
    worker = PlannerWorker(planner, lambda revision, actions: published.append((revision, actions)))
    worker.start()
    try:
        worker.submit(
            PlanRequest(1, interval, 0, 2, 0, 1 / 30, 0, seed, reference, None, original_connection)
        )
        deadline = time.monotonic() + 5
        while worker.last_result is None and time.monotonic() < deadline:
            time.sleep(0.01)
        result = worker.last_result
        assert result is not None and result.error is None
        assert result.turn_rule_id == "turn"
        assert result.turn_angle_rad == pytest.approx(pi / 2, abs=0.02)
        assert result.connection.turn_angle_rad == result.turn_angle_rad
        assert published[0][0] == 1
    finally:
        worker.close()
