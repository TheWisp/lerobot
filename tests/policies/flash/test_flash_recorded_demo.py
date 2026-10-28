"""A recorded trial can become a demo without changing object coordinate frames."""

import json

import numpy as np
import pytest

from lerobot.policies.flash.appearance import AppearanceModel
from lerobot.policies.flash.compiler import IntervalAnnotation
from lerobot.policies.flash.matched_rgbd import ObjectPointModel
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.recorded_demo import prepare_recorded_episode
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample


class Episode:
    features = {
        "observation.state": {"names": ["joint.pos", "finger.pos"]},
        "action": {"names": ["joint.pos", "finger.pos"]},
    }

    def episode_rows(self, episode):
        assert episode == 0
        return 0, 3

    def __getitem__(self, index):
        return {
            "timestamp": index / 30,
            "observation.state": np.array([index, 0.0]),
            "action": np.array([index, 1.0]),
        }


class Kinematics:
    def forward_kinematics(self, joints):
        result = np.eye(4)
        result[0, 3] = joints[0] / 100
        return result


def _recorded_inputs(root, *, version="object-v1"):
    identity = np.eye(4)
    mask = root / "flash/v1/masks/reference_snapshots/episode_000000/view.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"mask")
    reference = ObjectReference(
        "object",
        "object-v1",
        (
            ReferenceSnippet(
                "demo/source", 0, 4, "object", identity, mask.relative_to(root).as_posix(), "cal"
            ),
        ),
    )
    model = AppearanceModel(
        ObjectPointModel("object", "object-v1", np.array([[0.0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])),
        np.eye(3),
    )
    source_skill = FlashSkill(
        "demo/source",
        0,
        (
            Interval(
                "move",
                MotionMode.DEMONSTRATED,
                Relation.EE_TO_OBJECT,
                "object",
                (SpatialSample(0, identity, identity), SpatialSample(1, identity, identity)),
                reference_version="object-v1",
            ),
        ),
    )
    bundle_path = PreparedFlashSkill(
        source_skill, (reference,), (model,), {"object": "patterned object"}, "encoder"
    ).save(root / "flash/v1/reference_bundles/episode_000000")
    trace = root / "flash/v1/episode_000000.jsonl"
    with trace.open("w") as stream:
        for frame in range(3):
            stream.write(
                json.dumps(
                    {
                        "episode_index": 0,
                        "frame_index": frame,
                        "poses": {
                            "object": {
                                "pose": identity.tolist(),
                                "source_time_s": frame / 30,
                                "available_time_s": frame / 30,
                                "evidence": "observed",
                                "reference_version": version,
                                "translation_supported": [True, True, True],
                                "rotation_supported": [True, True, True],
                            }
                        },
                    }
                )
                + "\n"
            )
    (root / "flash/v1/manifest.json").write_text(
        json.dumps({"schema_version": 1, "episodes": {"0": {"trace": trace.name, "frames": 3}}})
    )
    return bundle_path


def test_recorded_episode_compiles_to_portable_prepared_skill(tmp_path):
    reference_path = _recorded_inputs(tmp_path)
    output = prepare_recorded_episode(
        dataset=Episode(),
        dataset_root=tmp_path,
        source_repo_id="trial/recording",
        source_episode=0,
        reference_bundle_path=reference_path,
        output_dir=tmp_path / "flash/v1/prepared/from-trial",
        kinematics=Kinematics(),
        arm_joint_names=("joint",),
        gripper_joint_name="finger",
        annotations=(
            IntervalAnnotation("move", 0, 2, MotionMode.DEMONSTRATED, Relation.EE_TO_OBJECT, "object"),
        ),
    )
    prepared = PreparedFlashSkill.load(output)
    assert prepared.skill.source_repo_id == "trial/recording"
    assert prepared.skill.intervals[0].reference_version == "object-v1"
    assert [sample.source_frame_index for sample in prepared.skill.intervals[0].samples] == [0, 1, 2]
    assert (tmp_path / prepared.references[0].snippets[0].mask_artifact).read_bytes() == b"mask"


def test_recorded_episode_rejects_pose_trace_in_different_object_frame(tmp_path):
    reference_path = _recorded_inputs(tmp_path, version="other-frame")
    output = tmp_path / "flash/v1/prepared/invalid"
    with pytest.raises(ValueError, match="frame versions lack prepared references"):
        prepare_recorded_episode(
            dataset=Episode(),
            dataset_root=tmp_path,
            source_repo_id="trial/recording",
            source_episode=0,
            reference_bundle_path=reference_path,
            output_dir=output,
            kinematics=Kinematics(),
            arm_joint_names=("joint",),
            gripper_joint_name="finger",
            annotations=(
                IntervalAnnotation("move", 0, 2, MotionMode.DEMONSTRATED, Relation.EE_TO_OBJECT, "object"),
            ),
        )
    assert not output.exists()
