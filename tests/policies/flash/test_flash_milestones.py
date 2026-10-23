"""Milestones distinguish observed success, observed failure and missing evidence."""

import numpy as np

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
