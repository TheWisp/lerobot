"""Known geometry and event boundaries for the offline bridge experiment."""

import numpy as np
from bridge_shape import MotionSupportedSurfaceFit, attachment_at
from compare import move
from scipy.spatial.transform import Rotation
from shape_grounding import ShapeConfig, WholeSurfaceFit
from test_shape_grounding import fixture, observation


def test_current_geometry_can_correct_an_inaccurate_motion_prediction():
    source, k = fixture()
    prior = np.eye(4)
    prior[:3, :3] = Rotation.from_euler("z", 8, degrees=True).as_matrix()
    prior[:3, 3] = [0.006, 0, 0]
    fit = MotionSupportedSurfaceFit(source, observation(source, k), k, ShapeConfig(), prior, 0.005)
    corrected, _ = fit.correct(prior)
    error_before = np.mean(np.linalg.norm(move(source, prior) - source, axis=1))
    error_after = np.mean(np.linalg.norm(move(source, corrected) - source, axis=1))
    assert error_after < error_before
    # A rigid-gripper-only output cannot pass this correction requirement.
    assert not np.allclose(corrected, prior)


def test_motion_prior_disambiguates_geometry_without_a_symmetry_rule():
    _, k = fixture()
    theta = np.arange(128) * 2 * np.pi / 128
    source = np.column_stack((0.03 * np.cos(theta), 0.03 * np.sin(theta), np.full(128, 0.55)))
    prior = np.eye(4)
    opposite = np.eye(4)
    opposite[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    data = observation(source, k)
    visual = WholeSurfaceFit(source, data, k, ShapeConfig())
    constrained = MotionSupportedSurfaceFit(source, data, k, ShapeConfig(), prior, 0.005)
    assert np.isclose(visual.score(prior), visual.score(opposite), atol=1e-10)
    assert constrained.score(opposite) > constrained.score(prior) + 1


def test_motion_constraint_is_independent_of_arbitrary_reference_origin():
    source, k = fixture()
    prior = np.eye(4)
    shifted = np.eye(4)
    shifted[:3, 3] = [0.003, 0.002, 0.001]
    data = observation(source, k)
    original = MotionSupportedSurfaceFit(source, data, k, ShapeConfig(), prior, 0.005)
    origin_change = np.eye(4)
    origin_change[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.3, -0.1]).as_matrix()
    origin_change[:3, 3] = [1.5, -2.0, 3.0]
    changed = MotionSupportedSurfaceFit(
        move(source, origin_change), data, k, ShapeConfig(), prior @ np.linalg.inv(origin_change), 0.005
    )
    assert np.isclose(original.score(shifted), changed.score(shifted @ np.linalg.inv(origin_change)))


def test_release_removes_gripper_relationship_at_the_recorded_event():
    skill = {
        "attachment_transitions": [
            {"event_id": "closed", "action": "attach", "role": "tool"},
            {"event_id": "opened", "action": "detach", "role": "tool"},
        ],
        "intervals": [
            {
                "samples": [
                    {"time_s": 1.5, "event_id": "closed"},
                    {"time_s": 3.5, "event_id": "opened"},
                ]
            }
        ],
    }
    assert not attachment_at(skill, "tool", 1.49)
    assert attachment_at(skill, "tool", 1.5)
    assert attachment_at(skill, "tool", 3.49)
    assert not attachment_at(skill, "tool", 3.5)
    assert not attachment_at(skill, "another_object", 2.0)
