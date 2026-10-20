"""Declared physical symmetry must remove unobservable tracker spin."""

from pathlib import Path

import numpy as np

from lerobot.policies.flash.grasp_symmetry import select_grasp_symmetry
from lerobot.policies.flash.planner import JointPlanner, PlannerLimits
from lerobot.policies.flash.skill import (
    AttachmentAction,
    AttachmentTransition,
    EndpointRequirement,
    FlashSkill,
    Interval,
    MotionMode,
    Relation,
    SpatialSample,
)
from lerobot.policies.flash.symmetry import AxialSymmetry
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def _x_pose(x: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


class _BoundedLinearKinematics:
    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        return _x_pose(float(joints[0]) / 100)

    def inverse_kinematics(self, seed: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.array([min(float(target[0, 3]) * 100, 100)])


def test_grasp_symmetry_preview_honors_preferred_hover_and_required_carry():
    """The selector must use the same soft hover and IK rules as execution."""
    identity = np.eye(4)
    hover = Interval(
        "hover",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "held",
        (SpatialSample(0, _x_pose(0), identity), SpatialSample(1, _x_pose(1.2), identity)),
        speed_m_s=1,
        reference_version="held-v1",
        endpoint_requirement=EndpointRequirement.PREFERRED,
    )
    grasp = Interval(
        "grasp",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "held",
        (
            SpatialSample(0, _x_pose(0.8), identity),
            SpatialSample(1, _x_pose(0.8), identity, event_id="attach"),
        ),
        speed_m_s=1,
        reference_version="held-v1",
    )
    carry = Interval(
        "carry",
        MotionMode.CONNECTION,
        Relation.HELD_TO_TARGET,
        "target",
        (
            SpatialSample(0, _x_pose(0.8), identity, _x_pose(0)),
            SpatialSample(1, _x_pose(0.9), identity, _x_pose(0.1)),
        ),
        speed_m_s=1,
        reference_version="target-v1",
        held_reference_version="held-v1",
        held_role="held",
    )
    skill = FlashSkill(
        "demo",
        0,
        (hover, grasp, carry),
        attachment_transitions=(AttachmentTransition("attach", AttachmentAction.ATTACH, "held", "held-v1"),),
    )
    symmetry = AxialSymmetry("held", np.array([0, 0, 1]), np.zeros(3), identity)
    planner = JointPlanner(
        _BoundedLinearKinematics(),
        ("arm",),
        "gripper",
        PlannerLimits((1000,), 0.005, 0.05, 1),
    )
    poses = {
        role: PoseEstimate(role, identity, 0, 0, Evidence.OBSERVED, version, (True,) * 3, (True,) * 3)
        for role, version in (("held", "held-v1"), ("target", "target-v1"))
    }
    choice = select_grasp_symmetry(skill, symmetry, poses, planner, np.array([0.0]), 1 / 30)
    assert choice.angle_rad == 0
    assert not choice.exchange_ends


def _turn_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def test_annotated_dowel_axis_removes_real_static_fit_jitter():
    fixture = Path(__file__).parent / "fixtures" / "dowel_axial_jitter_20261020.npz"
    with np.load(fixture) as recorded:
        points = recorded["points"]
        direct = recorded["direct_poses"]
        baseline = recorded["demo_reference"]
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(points.T))
    symmetry = AxialSymmetry("dowel", vectors[:, -1], center, baseline)
    canonical = [symmetry.canonicalize(pose) for pose in direct]

    raw_turns = [_turn_deg(a, b) for a, b in zip(direct, direct[1:], strict=False)]
    corrected_turns = [_turn_deg(a, b) for a, b in zip(canonical, canonical[1:], strict=False)]
    assert max(raw_turns) > 15  # the live 15-degree replanning envelope was exceeded
    assert max(corrected_turns) < 7
    for raw, fixed in zip(direct, canonical, strict=True):
        np.testing.assert_allclose(raw[:3, :3] @ center + raw[:3, 3], fixed[:3, :3] @ center + fixed[:3, 3])


def test_interchangeable_ends_preserve_center_and_canonicalize_reversed_axis():
    fixture = Path(__file__).parent / "fixtures" / "dowel_axial_jitter_20261020.npz"
    with np.load(fixture) as recorded:
        points = recorded["points"]
        baseline = recorded["demo_reference"]
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(points.T))
    symmetry = AxialSymmetry("dowel", vectors[:, -1], center, baseline, ends_interchangeable=True)
    exchanged = baseline @ symmetry.equivalent_transform(exchange_ends=True)
    np.testing.assert_allclose(
        exchanged[:3, :3] @ center + exchanged[:3, 3],
        baseline[:3, :3] @ center + baseline[:3, 3],
    )
    corrected = symmetry.canonicalize(exchanged)
    np.testing.assert_allclose(
        corrected[:3, :3] @ symmetry.axis_object,
        baseline[:3, :3] @ symmetry.axis_object,
        atol=1e-8,
    )
