"""Declared physical symmetry must remove unobservable tracker spin."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.causal_service import CausalPoseService
from lerobot.policies.flash.grasp_symmetry import select_grasp_symmetry
from lerobot.policies.flash.perception import LatestPoseStore, ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.perception_step import PerceptionPipeline
from lerobot.policies.flash.planner import JointPlanner, PlannerLimits
from lerobot.policies.flash.pose_evidence import VisibleSurfaceSupport
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
from lerobot.policies.flash.symmetry import AxialSymmetry, SymmetryCanonicalizingBackend
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def test_symmetry_wrapper_keeps_rejected_surface_evidence_for_the_shared_pipeline():
    frame = RGBDFrame(
        3,
        np.zeros((4, 4, 3), np.uint8),
        np.full((4, 4), 1000, np.uint16),
        np.eye(3),
        0.001,
        np.eye(4),
        1.0,
        1.0,
        "calibration",
    )
    raw = _x_pose(0.1)
    motion_prior = _x_pose(0.02)
    support = VisibleSurfaceSupport(
        4,
        principal_surface_axis_base=(1.0, 0.0, 0.0),
        principal_surface_spreads_m=(0.001, 0.002, 0.01),
        principal_surface_points=12,
    )

    class RejectedBackend:
        last_masks = {"object": np.ones((4, 4), bool)}
        last_candidate_poses = {"object": raw}
        last_motion_prior_poses = {"object": motion_prior}
        last_visible_support = {"object": support}
        last_failure = {"object": "full pose unobservable"}

        def initialize(self, _references):
            pass

        def update(self, _frame):
            return (
                PoseEstimate(
                    "object",
                    None,
                    1.0,
                    1.0,
                    Evidence.LOST,
                    "reference",
                    (False,) * 3,
                    (False,) * 3,
                    estimator="surface_rgbd",
                ),
            )

    symmetry = AxialSymmetry("object", np.array([1, 0, 0]), np.zeros(3), np.eye(4))
    snippet = ReferenceSnippet("local/demo", 0, 0, "object", np.eye(4), "mask.png", "calibration")
    reference = ObjectReference("object", "reference", (snippet,))
    pipeline = PerceptionPipeline(
        SymmetryCanonicalizingBackend(RejectedBackend(), (symmetry,)), (reference,), require_masks=True
    )

    result = pipeline.process(frame)
    proposal = result.proposals[0]
    assert result.estimates[0].evidence == Evidence.LOST
    np.testing.assert_allclose(proposal.raw_pose, raw)
    np.testing.assert_allclose(proposal.motion_prior_pose, motion_prior)
    assert proposal.visible_support == support
    assert proposal.failure == "full pose unobservable"


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


def test_interchangeable_axis_follows_smooth_rotation_through_perpendicular():
    center = np.array([0.2, 0.1, 0.0])
    symmetry = AxialSymmetry("object", np.array([1, 0, 0]), center, np.eye(4), ends_interchangeable=True)
    observations = []
    for angle in np.deg2rad(np.arange(0, 121, 5)):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(angle * np.array([0, 1, 0])).as_matrix()
        pose[:3, 3] = center - pose[:3, :3] @ center
        observations.append(pose)

    fixed_reference = [symmetry.canonicalize(pose) for pose in observations]
    assert max(_turn_deg(a, b) for a, b in zip(fixed_reference, fixed_reference[1:], strict=False)) > 170

    continuous = []
    for pose in observations:
        continuous.append(symmetry.canonicalize(pose, reference_pose=continuous[-1] if continuous else None))
    assert max(_turn_deg(a, b) for a, b in zip(continuous, continuous[1:], strict=False)) < 6
    for observed, canonical in zip(observations, continuous, strict=True):
        np.testing.assert_allclose(canonical[:3, :3] @ center + canonical[:3, 3], center, atol=1e-9)
        np.testing.assert_allclose(
            canonical[:3, :3] @ np.array([0.3, 0.0, 0.0]) + canonical[:3, 3],
            observed[:3, :3] @ np.array([0.3, 0.0, 0.0]) + observed[:3, 3],
            atol=1e-9,
        )

    class SequenceBackend:
        def __init__(self, poses):
            self.poses = iter(poses)

        def initialize(self, _references):
            pass

        def update(self, _frame):
            return (
                PoseEstimate(
                    "object", next(self.poses), 0, 0, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3
                ),
            )

    backend = SymmetryCanonicalizingBackend(SequenceBackend(observations), (symmetry,))
    backend.initialize((type("Reference", (), {"role": "object"})(),))
    outputs = [backend.update(None)[0].pose for _ in observations]
    for actual, expected in zip(outputs, continuous, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_locked_support_plane_keeps_measured_center_with_predicted_orientation():
    class SequenceBackend:
        def __init__(self, estimates):
            self.estimates = iter(estimates)

        def initialize(self, _references):
            pass

        def update(self, _frame):
            return (next(self.estimates),)

    observed_pose = np.eye(4)
    observed_pose[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    observed_pose[0, 3] = 0.15
    estimates = (
        PoseEstimate(
            "ring",
            observed_pose,
            1.0,
            1.0,
            Evidence.OBSERVED,
            "v1",
            (True,) * 3,
            (True,) * 3,
            last_direct_observation_s=1.0,
        ),
        PoseEstimate(
            "ring",
            observed_pose,
            2.0,
            2.0,
            Evidence.PREDICTED,
            "v1",
            (True,) * 3,
            (False,) * 3,
            last_direct_observation_s=1.0,
        ),
    )
    symmetry = AxialSymmetry(
        "ring",
        np.array([0, 0, 1]),
        np.zeros(3),
        np.eye(4),
        lock_to_baseline=True,
        max_observed_tilt_rad=np.deg2rad(15),
    )
    backend = SymmetryCanonicalizingBackend(SequenceBackend(estimates), (symmetry,))
    backend.initialize((type("Reference", (), {"role": "ring"})(),))

    current = backend.update(None)[0]
    assert current.evidence == Evidence.PREDICTED
    assert current.has_direct_translation
    assert current.rotation_supported == (True,) * 3
    assert current.usable(1.0, 1.0, allow_prediction=True)
    np.testing.assert_allclose(current.pose[:3, 3], observed_pose[:3, 3])
    np.testing.assert_allclose(current.pose[:3, :3], np.eye(3))
    assert current.last_direct_observation_s == 1.0

    frame = RGBDFrame(
        0,
        np.zeros((16, 16, 3), np.uint8),
        np.full((16, 16), 1000, np.uint16),
        np.array([[100, 0, 8], [0, 100, 8], [0, 0, 1]], dtype=float),
        0.001,
        np.eye(4),
        1.0,
        1.0,
        "cal",
    )
    scene = CausalPoseService(LatestPoseStore())
    scene.process_offline(frame, (current,), {"ring": np.ones(frame.depth.shape, bool)})
    assert scene.observed["ring"][0].has_direct_translation

    carried = backend.update(None)[0]
    assert carried.evidence == Evidence.PREDICTED
    assert not carried.has_direct_translation


def test_rejected_support_orientation_clears_direct_position():
    class TiltedBackend:
        def initialize(self, _references):
            pass

        def update(self, _frame):
            tilted = np.eye(4)
            tilted[:3, :3] = Rotation.from_euler("x", 30, degrees=True).as_matrix()
            return (
                PoseEstimate(
                    "ring",
                    tilted,
                    1.0,
                    1.0,
                    Evidence.OBSERVED,
                    "v1",
                    (True,) * 3,
                    (True,) * 3,
                    last_direct_observation_s=1.0,
                ),
            )

    symmetry = AxialSymmetry(
        "ring",
        np.array([0, 0, 1]),
        np.zeros(3),
        np.eye(4),
        lock_to_baseline=True,
        max_observed_tilt_rad=np.deg2rad(15),
    )
    backend = SymmetryCanonicalizingBackend(TiltedBackend(), (symmetry,))
    backend.initialize((type("Reference", (), {"role": "ring"})(),))
    rejected = backend.update(None)[0]
    assert rejected.evidence == Evidence.LOST
    assert rejected.pose is None
    assert not rejected.has_direct_translation
    assert rejected.last_direct_observation_s is None
