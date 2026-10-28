"""A permitted surface turn is one reference-relative choice across its lifetime."""

import time
from math import pi

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.appearance import AppearanceModel
from lerobot.policies.flash.demo_replay import build_demo_replay
from lerobot.policies.flash.matched_rgbd import ObjectPointModel
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet
from lerobot.policies.flash.planner import InfeasibleTrajectoryError, JointPlanner, PlannerLimits
from lerobot.policies.flash.planner_worker import PlannerWorker, PlanRequest
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.skill import (
    FlashSkill,
    Interval,
    MotionMode,
    Relation,
    SpatialSample,
    TurnAtPoint,
)
from lerobot.policies.flash.symmetry import AxialSymmetry
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


def test_prepared_skill_accepts_only_a_symmetry_invariant_center_turn(tmp_path):
    symmetry = AxialSymmetry("target", (0, 0, 1), (0, 0, 0), pose(), True, 0.3)
    reference = ObjectReference(
        "target",
        "target-v1",
        (ReferenceSnippet("demo/source", 0, 0, "target", pose(), "flash/v1/masks/target.png", "cal-v1"),),
    )
    model = AppearanceModel(
        ObjectPointModel("target", "target-v1", np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]])),
        np.ones((3, 2)),
    )

    def prepare(rule):
        skill = FlashSkill("demo/source", 0, (make_interval(rule=rule),))
        return PreparedFlashSkill(
            skill, (reference,), (model,), {"target": "target"}, "descriptor", (symmetry,)
        )

    invariant = TurnAtPoint("center", (0, 0, 0), (0, 0, 1), -pi, pi, 0, "symmetry_invariant")
    prepared = prepare(invariant)
    bundle = prepared.save(tmp_path / "prepared")
    assert PreparedFlashSkill.load(bundle).skill.intervals[0].turn_at_point == invariant
    replay = build_demo_replay(
        PreparedFlashSkill.load(bundle),
        arm_times_s=np.array([0.0, 1.0]),
        arm_tips=np.stack((pose(), pose())),
        camera_times_s=np.array([0.1, 0.9]),
        intrinsics=np.eye(3),
        base_from_camera=np.eye(4),
        image_size=(640, 480),
        fps=30,
    )
    assert replay["pose_constraints"]["target"] == {
        "orientation_locked_to_baseline": True,
        "ends_interchangeable": False,
        "center_object_m": [0.0, 0.0, 0.0],
        "axis_object": [0.0, 0.0, 1.0],
        "source_frame_index": 0,
    }
    with pytest.raises(ValueError, match="prepared reference center"):
        prepare(TurnAtPoint("offset", (0.01, 0, 0), (0, 0, 1), -pi, pi, 0, "symmetry_invariant"))
    with pytest.raises(ValueError, match="prepared reference center"):
        prepare(TurnAtPoint("wrong-direction", (0, 0, 0), (1, 0, 0), -pi, pi, 0, "symmetry_invariant"))


def test_async_worker_selects_reachable_turn_and_rebuilds_connection():
    rule = TurnAtPoint("turn", (0, 0, 0), (0, 0, 1), -pi, pi, 17, "observed")
    interval = make_interval(rule=rule)
    kinematics = OrbitKinematics()
    planner = JointPlanner(kinematics, ("wrist",), "gripper", PlannerLimits((1000,), 0.01, 0.01, 10))
    seed = np.array([90.0])
    reference = PoseEstimate("target", pose(), 0, 0, Evidence.OBSERVED, "target-v1", (True,) * 3, (True,) * 3)
    chosen = select_turn_angle(planner, interval, seed, reference.pose, None, 1 / 30)
    assert chosen == pytest.approx(pi / 2, abs=0.02)
    invariant = make_interval(
        rule=TurnAtPoint("center", (0, 0, 0), (0, 0, 1), -pi, pi, 17, "symmetry_invariant")
    )
    assert select_turn_angle(planner, invariant, seed, reference.pose, None, 1 / 30) == pytest.approx(
        pi / 2, abs=0.02
    )
    predicted = make_interval(rule=TurnAtPoint("uncertain", (0, 0, 0), (0, 0, 1), -pi, pi, 17, "predicted"))
    with pytest.raises(InfeasibleTrajectoryError, match="directly observed pose"):
        select_turn_angle(planner, predicted, seed, reference.pose, None, 1 / 30)
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


def test_turn_choice_follows_a_continuous_ik_path_when_endpoint_ik_misses():
    class LocalKinematics:
        def forward_kinematics(self, joints):
            return pose(float(joints[0]))

        def inverse_kinematics(self, seed, target):
            if np.linalg.norm(target[:2, 3] - [seed[0], 0]) > 0.2:
                return np.asarray(seed)
            return np.array([target[0, 3]])

    interval = make_interval(rule=TurnAtPoint("turn", (0, 0, 0), (0, 0, 1), -pi, pi, 0, "observed"))
    planner = JointPlanner(LocalKinematics(), ("arm",), "gripper", PlannerLimits((10,), 0.01, 0.01, 10))
    seed = np.array([0.0])
    assert planner.kinematics.inverse_kinematics(seed, pose(1))[0] == 0
    chosen = select_turn_angle(planner, interval, seed, pose(), None, 1 / 30)
    assert chosen == pytest.approx(0.0, abs=0.02)
