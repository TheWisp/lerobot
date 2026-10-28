"""Milestones distinguish observed success, observed failure and missing evidence."""

import numpy as np
import pytest

from lerobot.policies.flash.milestones import PoseMilestonePublisher, verify_pose_relation
from lerobot.policies.flash.progress import Progress, Verdict
from lerobot.policies.flash.skill import (
    FlashSkill,
    Interval,
    MilestoneSpec,
    MotionMode,
    PoseRelationCriterion,
    Relation,
    SpatialSample,
)
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def pose(x=0.0):
    matrix = np.eye(4)
    matrix[0, 3] = x
    return matrix


def observed(role, x, *, version, time_s=1.0, evidence=Evidence.OBSERVED, rotation=(True,) * 3):
    return PoseEstimate(role, pose(x), time_s, time_s, evidence, version, (True,) * 3, rotation)


def milestone(*, rotation_tolerance_rad=None):
    return MilestoneSpec(
        "placed",
        ("held", "target"),
        PoseRelationCriterion(
            moving_role="held",
            reference_role="target",
            reference_from_moving=pose(0.1),
            moving_version="held-v1",
            reference_version="target-v1",
            translation_tolerance_m=0.005,
            rotation_tolerance_rad=rotation_tolerance_rad,
            max_pose_age_s=0.2,
        ),
    )


def test_pose_milestone_checks_actual_object_relation_and_persists_in_skill(tmp_path):
    spec = milestone()
    sample = SpatialSample(0, pose(), pose())
    later = SpatialSample(1, pose(), pose())
    interval = Interval(
        "place",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "target",
        (sample, later),
        completion_milestone="placed",
    )
    skill = FlashSkill("demo/source", 0, (interval,), milestones=(spec,))
    path = tmp_path / "skill.json"
    skill.save(path)
    restored = FlashSkill.load(path)
    criterion = restored.milestones[0].pose_relation
    np.testing.assert_allclose(criterion.reference_from_moving, pose(0.1))

    poses = {
        "target": observed("target", 0.4, version="target-v1"),
        "held": observed("held", 0.501, version="held-v1"),
    }
    good = verify_pose_relation(restored.milestones[0], poses, pose_version=7, now_s=1.1)
    assert good.verdict == Verdict.TRUE
    published = []
    PoseMilestonePublisher(restored.milestones, published.append, clock=lambda: 1.1)(7, poses)
    assert published == [good]
    progress = Progress(restored)
    assert progress.update_milestone(good)
    assert not progress.update_milestone(good)
    same_frames_new_store_version = verify_pose_relation(
        restored.milestones[0], poses, pose_version=8, now_s=1.1
    )
    assert same_frames_new_store_version.source_id == good.source_id
    assert not progress.update_milestone(same_frames_new_store_version)

    poses["held"] = observed("held", 0.52, version="held-v1", time_s=1.2)
    bad = verify_pose_relation(restored.milestones[0], poses, pose_version=9, now_s=1.2)
    assert bad.verdict == Verdict.FALSE
    assert "0.0200 m" in bad.reason


def test_pose_milestone_never_promotes_prediction_or_unobserved_axes_to_success():
    spec = milestone(rotation_tolerance_rad=0.1)
    poses = {
        "target": observed("target", 0.4, version="target-v1"),
        "held": observed("held", 0.5, version="held-v1", evidence=Evidence.PREDICTED),
    }
    assert verify_pose_relation(spec, poses, pose_version=1, now_s=1.0).verdict == Verdict.UNKNOWN
    poses["held"] = observed("held", 0.5, version="held-v1", rotation=(False,) * 3)
    assert verify_pose_relation(spec, poses, pose_version=2, now_s=1.0).verdict == Verdict.UNKNOWN
    assert verify_pose_relation(milestone(), poses, pose_version=3, now_s=1.0).verdict == Verdict.TRUE
    assert verify_pose_relation(milestone(), poses, pose_version=4, now_s=1.3).verdict == Verdict.UNKNOWN
    poses["held"] = observed("held", 0.5, version="different")
    assert verify_pose_relation(milestone(), poses, pose_version=5, now_s=1.0).verdict == Verdict.UNKNOWN


def test_true_pose_milestone_is_revoked_once_when_observation_expires():
    spec = milestone()
    poses = {
        "target": observed("target", 0.4, version="target-v1"),
        "held": observed("held", 0.5, version="held-v1"),
    }
    now = [1.0]
    published = []
    publisher = PoseMilestonePublisher((spec,), published.append, clock=lambda: now[0])
    publisher(3, poses)
    publisher(3, poses)
    assert [event.verdict for event in published] == [Verdict.TRUE]
    now[0] = 1.3
    publisher(3, poses)
    publisher(3, poses)
    assert [event.verdict for event in published] == [Verdict.TRUE, Verdict.UNKNOWN]
    assert published[0].source_id != published[1].source_id


def test_local_contact_point_and_direction_ignore_irrelevant_roll_but_reject_bad_alignment(tmp_path):
    """A contact requires the measured point and axis, not an arbitrary frame roll."""
    spec = MilestoneSpec(
        "contact",
        ("moving", "reference"),
        PoseRelationCriterion(
            "moving",
            "reference",
            np.eye(4),
            "moving-v1",
            "reference-v1",
            0.003,
            None,
            0.2,
            moving_point_m=(0.05, 0, 0),
            moving_direction=(1, 0, 0),
            direction_tolerance_rad=0.05,
        ),
    )
    sample = SpatialSample(0, pose(), pose())
    skill = FlashSkill(
        "demo/source",
        0,
        (
            Interval(
                "move",
                MotionMode.DEMONSTRATED,
                Relation.EE_TO_OBJECT,
                "reference",
                (sample, SpatialSample(1, pose(), pose())),
            ),
        ),
        milestones=(spec,),
    )
    path = tmp_path / "skill.json"
    skill.save(path)
    spec = FlashSkill.load(path).milestones[0]
    reference = observed("reference", 0, version="reference-v1")

    roll = np.eye(4)
    angle = np.pi * 0.75
    roll[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    moving = PoseEstimate("moving", roll, 1, 1, Evidence.OBSERVED, "moving-v1", (True,) * 3, (True,) * 3)
    poses = {"reference": reference, "moving": moving}
    assert verify_pose_relation(spec, poses, pose_version=1, now_s=1).verdict == Verdict.TRUE

    poses["moving"] = PoseEstimate(
        "moving", roll, 1, 1, Evidence.OBSERVED, "moving-v1", (True,) * 3, (False, True, True)
    )
    poses["reference"] = PoseEstimate(
        "reference",
        pose(),
        1,
        1,
        Evidence.OBSERVED,
        "reference-v1",
        (True,) * 3,
        (False, True, True),
    )
    assert verify_pose_relation(spec, poses, pose_version=2, now_s=1).verdict == Verdict.TRUE

    poses["moving"] = PoseEstimate(
        "moving", roll, 1, 1, Evidence.OBSERVED, "moving-v1", (True,) * 3, (False, False, True)
    )
    assert verify_pose_relation(spec, poses, pose_version=3, now_s=1).verdict == Verdict.UNKNOWN
    poses["moving"] = moving
    poses["reference"] = reference

    tilt = np.eye(4)
    tilt[:3, :3] = [[np.cos(0.2), 0, np.sin(0.2)], [0, 1, 0], [-np.sin(0.2), 0, np.cos(0.2)]]
    local_point = np.array([0.05, 0, 0])
    tilt[:3, 3] = local_point - tilt[:3, :3] @ local_point
    poses["moving"] = PoseEstimate(
        "moving", tilt, 1, 1, Evidence.OBSERVED, "moving-v1", (True,) * 3, (True,) * 3
    )
    assert verify_pose_relation(spec, poses, pose_version=4, now_s=1).verdict == Verdict.FALSE

    displaced = roll.copy()
    displaced[0, 3] = 0.01
    poses["moving"] = PoseEstimate(
        "moving", displaced, 1, 1, Evidence.OBSERVED, "moving-v1", (True,) * 3, (True,) * 3
    )
    assert verify_pose_relation(spec, poses, pose_version=5, now_s=1).verdict == Verdict.FALSE

    poses["moving"] = PoseEstimate(
        "moving", roll, 1, 1, Evidence.OBSERVED, "moving-v1", (True,) * 3, (False,) * 3
    )
    assert verify_pose_relation(spec, poses, pose_version=6, now_s=1).verdict == Verdict.UNKNOWN


def test_feature_relation_rejects_incomplete_or_conflicting_annotations():
    base = {
        "moving_role": "moving",
        "reference_role": "reference",
        "reference_from_moving": np.eye(4),
        "moving_version": "m-v1",
        "reference_version": "r-v1",
        "translation_tolerance_m": 0.01,
        "rotation_tolerance_rad": None,
        "max_pose_age_s": 0.2,
    }
    with pytest.raises(ValueError, match="together"):
        PoseRelationCriterion(**base, moving_direction=(1, 0, 0))
    with pytest.raises(ValueError, match="unit"):
        PoseRelationCriterion(**base, moving_direction=(2, 0, 0), direction_tolerance_rad=0.1)
    with pytest.raises(ValueError, match="either"):
        PoseRelationCriterion(
            **{**base, "rotation_tolerance_rad": 0.1},
            moving_direction=(1, 0, 0),
            direction_tolerance_rad=0.1,
        )
