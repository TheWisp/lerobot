"""Skill annotations compile from normal LeRobot frame columns."""

import numpy as np
import pytest

from lerobot.policies.flash.compiler import (
    ActionEventAnnotation,
    IntervalAnnotation,
    PoseRelationAnnotation,
    compile_skill,
)
from lerobot.policies.flash.milestones import verify_pose_relation
from lerobot.policies.flash.progress import Verdict
from lerobot.policies.flash.skill import (
    AttachmentAction,
    AttachmentTransition,
    FlashSkill,
    MotionMode,
    Relation,
)
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate, spatial_goal


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
    assert [sample.source_frame_index for sample in interval.samples] == [0, 2]
    assert interval.samples[1].gripper == 3.0
    assert interval.samples[1].demo_ee[0, 3] == pytest.approx(0.1)
    path = tmp_path / "flash_skill.json"
    skill.save(path)
    restored = FlashSkill.load(path)
    assert restored.intervals[0].samples[1].source_frame_index == 2
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
    assert skill.intervals[0].samples[1].demo_ee[0, 3] == pytest.approx(0.1)


def test_push_timeline_compiles_without_object_pose_during_start_motion_or_attachment():
    poses = tracks()
    del poses[0]
    skill = compile_skill(
        dataset=FakeDataset(),
        source_repo_id="demo/data",
        source_episode=1,
        kinematics=FakeKinematics(),
        arm_joint_names=("arm",),
        gripper_joint_name="gripper",
        poses_by_frame=poses,
        annotations=(
            IntervalAnnotation("approach", 0, 1, MotionMode.CONNECTION, Relation.EE_TO_START, "start"),
            IntervalAnnotation("push", 1, 2, MotionMode.DEMONSTRATED, Relation.EE_TO_OBJECT, "object"),
        ),
    )
    assert skill.attachment_transitions == ()
    assert [interval.name for interval in skill.intervals] == ["approach", "push"]
    approach, push = skill.intervals
    assert [sample.source_frame_index for sample in approach.samples] == [0, 1]
    assert [sample.source_frame_index for sample in push.samples] == [1, 2]
    assert all(sample.reference_evidence == "recorded_start" for sample in approach.samples)
    np.testing.assert_allclose(approach.samples[1].demo_reference, pose(0))
    np.testing.assert_allclose(spatial_goal(approach, approach.samples[-1], pose(0.2), None), pose(0.25))


def test_connection_needs_pose_at_endpoints_but_not_during_free_motion():
    poses = tracks()
    del poses[1]
    skill = compile_skill(
        dataset=FakeDataset(),
        source_repo_id="demo/data",
        source_episode=1,
        kinematics=FakeKinematics(),
        arm_joint_names=("arm",),
        gripper_joint_name="gripper",
        poses_by_frame=poses,
        annotations=(
            IntervalAnnotation("transport", 0, 2, MotionMode.CONNECTION, Relation.EE_TO_OBJECT, "object"),
        ),
    )
    assert [sample.source_frame_index for sample in skill.intervals[0].samples] == [0, 2]
    assert skill.intervals[0].speed_m_s == pytest.approx(0.1)


def test_connection_speed_uses_recorded_travel_not_endpoint_displacement():
    class BacktrackingDataset(FakeDataset):
        def episode_rows(self, episode):
            assert episode == 1
            return 0, 4

        def __getitem__(self, index):
            joint = (0, 10, 0, 10)[index]
            return {
                "timestamp": float(index),
                "observation.state": np.array([joint, 2.0]),
                "action": np.array([joint, 3.0]),
            }

    skill = compile_skill(
        dataset=BacktrackingDataset(),
        source_repo_id="demo/data",
        source_episode=1,
        kinematics=FakeKinematics(),
        arm_joint_names=("arm",),
        gripper_joint_name="gripper",
        poses_by_frame={},
        annotations=(IntervalAnnotation("free", 0, 3, MotionMode.CONNECTION, Relation.EE_TO_START, "start"),),
    )
    assert len(skill.intervals[0].samples) == 2
    assert skill.intervals[0].speed_m_s == pytest.approx(0.1)


def test_compiler_reads_only_state_action_and_timestamp_columns_when_available():
    class StateOnlyDataset(FakeDataset):
        def __getitem__(self, index):
            raise AssertionError("decoding video for skill compilation is unnecessary")

        def select_columns(self, names):
            assert names == ["timestamp", "observation.state", "action"]
            return FakeDataset()

    skill = compile_skill(
        dataset=StateOnlyDataset(),
        source_repo_id="demo/data",
        source_episode=1,
        kinematics=FakeKinematics(),
        arm_joint_names=("arm",),
        gripper_joint_name="gripper",
        poses_by_frame={},
        annotations=(IntervalAnnotation("free", 0, 2, MotionMode.CONNECTION, Relation.EE_TO_START, "start"),),
    )
    assert [sample.source_frame_index for sample in skill.intervals[0].samples] == [0, 2]


def test_connection_rejects_an_event_in_an_unreplayed_middle_frame():
    with pytest.raises(ValueError, match="events belong at the endpoint"):
        compile_skill(
            dataset=FakeDataset(),
            source_repo_id="demo/data",
            source_episode=1,
            kinematics=FakeKinematics(),
            arm_joint_names=("arm",),
            gripper_joint_name="gripper",
            poses_by_frame=tracks(),
            annotations=(
                IntervalAnnotation(
                    "travel",
                    0,
                    2,
                    MotionMode.CONNECTION,
                    Relation.EE_TO_OBJECT,
                    "object",
                    events=(ActionEventAnnotation(1, "close"),),
                ),
            ),
        )


def test_timeline_rejects_an_unmarked_gap_between_intervals():
    with pytest.raises(ValueError, match="continuous, ordered timeline"):
        compile_skill(
            dataset=FakeDataset(),
            source_repo_id="demo/data",
            source_episode=1,
            kinematics=FakeKinematics(),
            arm_joint_names=("arm",),
            gripper_joint_name="gripper",
            poses_by_frame=tracks(),
            annotations=(
                IntervalAnnotation("first", 0, 1, MotionMode.DEMONSTRATED, Relation.EE_TO_OBJECT, "object"),
                IntervalAnnotation("last", 2, 3, MotionMode.DEMONSTRATED, Relation.EE_TO_OBJECT, "object"),
            ),
        )


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


def test_compiles_and_replays_a_local_point_and_direction_milestone(tmp_path):
    poses = tracks()
    poses[2]["part"] = PoseEstimate(
        "part",
        pose(0.1),
        1.0,
        1.0,
        Evidence.OBSERVED,
        "part-v1",
        (True,) * 3,
        (False, True, True),
    )
    skill = compile_skill(
        dataset=FakeDataset(),
        source_repo_id="demo/data",
        source_episode=1,
        kinematics=FakeKinematics(),
        arm_joint_names=("arm",),
        gripper_joint_name="gripper",
        poses_by_frame=poses,
        annotations=(
            IntervalAnnotation("move", 0, 2, MotionMode.DEMONSTRATED, Relation.EE_TO_OBJECT, "object"),
        ),
        pose_relation_annotations=(
            PoseRelationAnnotation(
                "aligned",
                2,
                "part",
                "object",
                0.005,
                None,
                0.2,
                moving_point_m=(0.05, 0, 0),
                moving_direction=(1, 0, 0),
                direction_tolerance_rad=0.1,
            ),
        ),
    )
    path = tmp_path / "skill.json"
    skill.save(path)
    compiled = FlashSkill.load(path).milestones[0]
    assert compiled.pose_relation.moving_point_m == (0.05, 0, 0)
    assert compiled.pose_relation.moving_direction == (1, 0, 0)
    assert verify_pose_relation(compiled, poses[2], pose_version=1, now_s=1).verdict == Verdict.TRUE
