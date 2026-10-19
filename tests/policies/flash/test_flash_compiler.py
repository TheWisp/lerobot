"""Skill annotations compile from normal LeRobot frame columns."""

import numpy as np
import pytest

from lerobot.policies.flash.compiler import (
    ActionEventAnnotation,
    IntervalAnnotation,
    PoseRelationAnnotation,
    compile_skill,
)
from lerobot.policies.flash.skill import (
    AttachmentAction,
    AttachmentTransition,
    FlashSkill,
    MotionMode,
    Relation,
)
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value


class FakeDataset:
    features = {
        "observation.state": {"names": ["arm.pos", "gripper.pos"]},
        "action": {"names": ["arm.pos", "gripper.pos"]},
    }

    def episode_rows(self, episode):
        assert episode == 1
        return 0, 3

    def __getitem__(self, index):
        return {
            "timestamp": index * 0.5,
            "observation.state": np.array([index * 5.0, 2.0]),
            "action": np.array([index * 5.0, 3.0]),
        }


class MultiEpisodeDataset(FakeDataset):
    features = {
        "observation.state": {"names": ["arm", "gripper"]},
        "action": {"names": ["arm", "gripper"]},
    }

    def episode_rows(self, episode):
        assert episode == 1
        return 4, 3

    def __getitem__(self, index):
        assert 4 <= index < 7
        return super().__getitem__(index - 4)


class FakeKinematics:
    def forward_kinematics(self, joints):
        return pose(joints[0] / 100)


def tracks():
    return {
        i: {
            "object": PoseEstimate(
                "object",
                pose(0),
                i * 0.5,
                i * 0.5,
                Evidence.OBSERVED,
                "object-v1",
                (True, True, True),
                (True, True, True),
            )
        }
        for i in range(3)
    }


def test_compiles_timing_speed_and_reference_version_from_dataset(tmp_path):
    skill = compile_skill(
        dataset=FakeDataset(),
        source_repo_id="demo/data",
        source_episode=1,
        kinematics=FakeKinematics(),
        arm_joint_names=("arm",),
        gripper_joint_name="gripper",
        poses_by_frame=tracks(),
        annotations=(
            IntervalAnnotation("carry", 0, 2, MotionMode.CONNECTION, Relation.EE_TO_OBJECT, "object"),
        ),
    )
    interval = skill.intervals[0]
    assert interval.speed_m_s == pytest.approx(0.1)
    assert interval.reference_version == "object-v1"
    assert [sample.source_frame_index for sample in interval.samples] == [0, 1, 2]
    assert interval.samples[1].gripper == 3.0
    assert interval.samples[1].demo_ee[0, 3] == pytest.approx(0.05)
    path = tmp_path / "flash_skill.json"
    skill.save(path)
    restored = FlashSkill.load(path)
    assert restored.intervals[0].samples[1].source_frame_index == 1
    assert restored.intervals[0].reference_version == "object-v1"


def test_compiles_bare_joint_names_from_selected_episode():
    skill = compile_skill(
        dataset=MultiEpisodeDataset(),
        source_repo_id="demo/data",
        source_episode=1,
        kinematics=FakeKinematics(),
        arm_joint_names=("arm",),
        gripper_joint_name="gripper",
        poses_by_frame=tracks(),
        annotations=(
            IntervalAnnotation("carry", 0, 2, MotionMode.CONNECTION, Relation.EE_TO_OBJECT, "object"),
        ),
    )
    assert skill.intervals[0].samples[2].demo_ee[0, 3] == pytest.approx(0.1)


def test_predicted_demo_pose_requires_explicit_annotation():
    poses = tracks()
    old = poses[1]["object"]
    poses[1]["object"] = PoseEstimate(
        "object",
        old.pose,
        old.source_time_s,
        old.available_time_s,
        Evidence.PREDICTED,
        "object-v1",
        old.translation_supported,
        old.rotation_supported,
    )
    args = {
        "dataset": FakeDataset(),
        "source_repo_id": "demo/data",
        "source_episode": 1,
        "kinematics": FakeKinematics(),
        "arm_joint_names": ("arm",),
        "gripper_joint_name": "gripper",
        "poses_by_frame": poses,
    }
    annotation = IntervalAnnotation("grasp", 0, 2, MotionMode.DEMONSTRATED, Relation.EE_TO_OBJECT, "object")
    with pytest.raises(ValueError, match="not observed"):
        compile_skill(**args, annotations=(annotation,))
    approved = IntervalAnnotation(
        "grasp",
        0,
        2,
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "object",
        allow_predicted_demo_pose=True,
    )
    skill = compile_skill(**args, annotations=(approved,))
    assert skill.intervals[0].samples[1].reference_evidence == "predicted"


def test_compiles_a_pose_milestone_from_an_observed_demo_frame(tmp_path):
    poses = tracks()
    poses[2]["target"] = PoseEstimate(
        "target", pose(0.4), 1.0, 1.0, Evidence.OBSERVED, "target-v1", (True,) * 3, (True,) * 3
    )
    poses[2]["held"] = PoseEstimate(
        "held", pose(0.5), 1.0, 1.0, Evidence.OBSERVED, "held-v1", (True,) * 3, (False,) * 3
    )
    annotation = PoseRelationAnnotation("placed", 2, "held", "target", 0.005, None, 0.2)
    skill = compile_skill(
        dataset=FakeDataset(),
        source_repo_id="demo/data",
        source_episode=1,
        kinematics=FakeKinematics(),
        arm_joint_names=("arm",),
        gripper_joint_name="gripper",
        poses_by_frame=poses,
        annotations=(
            IntervalAnnotation(
                "place",
                0,
                2,
                MotionMode.DEMONSTRATED,
                Relation.EE_TO_OBJECT,
                "object",
                completion_milestone="placed",
                events=(ActionEventAnnotation(2, "release"),),
            ),
        ),
        pose_relation_annotations=(annotation,),
        attachment_transitions=(AttachmentTransition("release", AttachmentAction.DETACH, "held", "held-v1"),),
    )
    criterion = skill.milestones[0].pose_relation
    assert criterion.reference_from_moving[0, 3] == pytest.approx(0.1)
    assert criterion.moving_version == "held-v1"
    assert criterion.rotation_tolerance_rad is None
    assert skill.intervals[0].samples[-1].event_id == "release"
    assert skill.intervals[0].samples[-1].requires_milestone is None
    path = tmp_path / "skill.json"
    skill.save(path)
    assert FlashSkill.load(path).intervals[0].samples[-1].event_id == "release"
    assert FlashSkill.load(path).attachment_transitions == skill.attachment_transitions

    poses[2]["held"] = PoseEstimate(
        "held", pose(0.5), 1.0, 1.0, Evidence.PREDICTED, "held-v1", (True,) * 3, (False,) * 3
    )
    with pytest.raises(ValueError, match="not observed"):
        compile_skill(
            dataset=FakeDataset(),
            source_repo_id="demo/data",
            source_episode=1,
            kinematics=FakeKinematics(),
            arm_joint_names=("arm",),
            gripper_joint_name="gripper",
            poses_by_frame=poses,
            annotations=(
                IntervalAnnotation("place", 0, 2, MotionMode.DEMONSTRATED, Relation.EE_TO_OBJECT, "object"),
            ),
            pose_relation_annotations=(annotation,),
        )
