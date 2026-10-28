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
from lerobot.policies.flash.recorded_trial import inspect_recorded_trial, prepare_recorded_trial
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample
from lerobot.policies.flash.symmetry import AxialSymmetry


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


def _recorded_inputs(root, *, version="object-v1", symmetry=False):
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
    symmetries = (AxialSymmetry("object", [0, 0, 1], [0, 0, 0], identity),) if symmetry else ()
    bundle_path = PreparedFlashSkill(
        source_skill, (reference,), (model,), {"object": "patterned object"}, "encoder", symmetries
    ).save(root / "flash/v1/reference_bundles/episode_000000")
    trace = root / "flash/v1/episode_000000.jsonl"
    with trace.open("w") as stream:
        for frame in range(3):
            observed_pose = identity.copy()
            observed_pose[0, 3] = frame / 10
            stream.write(
                json.dumps(
                    {
                        "episode_index": 0,
                        "frame_index": frame,
                        "poses": {
                            "object": {
                                "pose": observed_pose.tolist(),
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


def test_recorded_episode_rebases_symmetry_from_measured_frame(tmp_path):
    reference_path = _recorded_inputs(tmp_path, symmetry=True)
    inputs = {
        "dataset": Episode(),
        "dataset_root": tmp_path,
        "source_repo_id": "trial/recording",
        "source_episode": 0,
        "reference_bundle_path": reference_path,
        "output_dir": tmp_path / "flash/v1/prepared/rebased",
        "kinematics": Kinematics(),
        "arm_joint_names": ("joint",),
        "gripper_joint_name": "finger",
        "annotations": (
            IntervalAnnotation("move", 0, 2, MotionMode.DEMONSTRATED, Relation.EE_TO_OBJECT, "object"),
        ),
    }
    with pytest.raises(ValueError, match="observed baseline frame"):
        prepare_recorded_episode(**inputs)
    output = prepare_recorded_episode(**inputs, symmetry_baseline_frames={"object": 1})
    symmetry = PreparedFlashSkill.load(output).axial_symmetries[0]
    assert symmetry.baseline_base_from_object[0, 3] == pytest.approx(0.1)


def test_start_relative_interval_does_not_inherit_unused_symmetry(tmp_path):
    reference_path = _recorded_inputs(tmp_path, symmetry=True)
    output = prepare_recorded_episode(
        dataset=Episode(),
        dataset_root=tmp_path,
        source_repo_id="trial/recording",
        source_episode=0,
        reference_bundle_path=reference_path,
        output_dir=tmp_path / "flash/v1/prepared/start-only",
        kinematics=Kinematics(),
        arm_joint_names=("joint",),
        gripper_joint_name="finger",
        annotations=(
            IntervalAnnotation("move", 0, 2, MotionMode.CONNECTION, Relation.EE_TO_START, "__start__"),
        ),
    )
    assert PreparedFlashSkill.load(output).axial_symmetries == ()


def test_recorded_trial_requires_a_portable_reference_snapshot(tmp_path):
    _recorded_inputs(tmp_path)
    inspection = inspect_recorded_trial(tmp_path, 0)
    assert inspection["frame_count"] == 3
    assert not inspection["ready"]
    assert "reference" in inspection["reason"] or "trial_config" in inspection["reason"]


def test_recorded_trial_editor_compiles_candidate_with_explicit_hold_event(tmp_path, monkeypatch):
    from lerobot.policies.flash import recorded_trial

    reference_path = _recorded_inputs(tmp_path)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"type": "so107_follower_gravity", "fields": {"arm": "left"}}))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"tool_point": {"offset_m": [0, 0, 0]}}))
    (tmp_path / "flash/v1/trial_config.json").write_text(
        json.dumps(
            {
                "source_reference_bundle": reference_path.relative_to(tmp_path).as_posix(),
                "output_root": str(tmp_path.resolve()),
                "output_repo_id": "trial/recording",
                "robot_profile": str(profile),
                "calibration_path": str(calibration),
                "robot_arm": "left",
            }
        )
    )

    class TrialEpisode(Episode):
        features = {
            "observation.state": {"names": ["joint.pos", "gripper.pos"]},
            "action": {"names": ["joint.pos", "gripper.pos"]},
        }

    monkeypatch.setattr(recorded_trial, "LeRobotDataset", lambda **kwargs: TrialEpisode())
    monkeypatch.setattr(recorded_trial, "MOTOR_NAMES", ("joint", "gripper"))
    monkeypatch.setattr(
        recorded_trial, "make_calibrated_so107_kinematics", lambda *args, **kwargs: Kinematics()
    )
    monkeypatch.setattr(recorded_trial, "SelectedJointKinematics", lambda inner, *args: inner)
    source = inspect_recorded_trial(tmp_path, 0)
    assert source["ready"] and source["observed_frames"]["object"] == [0, 1, 2]
    output = prepare_recorded_trial(
        tmp_path,
        0,
        "candidate",
        [
            {
                "first_frame": 0,
                "last_frame": 2,
                "mode": "demonstrated",
                "relation": "ee_to_object",
                "reference_role": "object",
                "attachment_action": "attach",
                "attachment_role": "object",
            }
        ],
    )
    prepared = PreparedFlashSkill.load(output)
    assert prepared.skill.intervals[0].samples[-1].event_id == "section-1-attach"
    assert prepared.skill.attachment_transitions[0].role == "object"
    assert json.loads((output.parent / "preparation-report.json").read_text())["review_status"] == "candidate"


def test_recorded_demo_replay_keeps_generated_route_endpoint_at_its_recorded_frame(tmp_path, monkeypatch):
    from lerobot.gui.api import run

    reference_path = _recorded_inputs(tmp_path)
    output = prepare_recorded_episode(
        dataset=Episode(),
        dataset_root=tmp_path,
        source_repo_id="trial/recording",
        source_episode=0,
        reference_bundle_path=reference_path,
        output_dir=tmp_path / "flash/v1/prepared/route",
        kinematics=Kinematics(),
        arm_joint_names=("joint",),
        gripper_joint_name="finger",
        annotations=(
            IntervalAnnotation("route", 0, 2, MotionMode.CONNECTION, Relation.EE_TO_START, "__start__"),
        ),
    )
    (tmp_path / "flash/v1/trial_config.json").write_text(json.dumps({"output_repo_id": "trial/recording"}))
    monkeypatch.setattr(
        run,
        "_flash_tracker_replay",
        lambda root, episode: {
            "kind": "rollout",
            "frames": [
                {
                    "frame_index": frame,
                    "snapshot": {"recorded_at_s": frame / 30, "layers": {}, "accepted": {}},
                }
                for frame in range(3)
            ],
        },
    )
    replay = run._flash_recorded_demo_replay(tmp_path, 0, output)
    assert replay["kind"] == "recorded_demo"
    assert replay["prepared_intervals"][0]["last_frame"] == 2
    assert "recorded_ee" in replay["frames"][0]["snapshot"]["layers"]["__gripper__"]
    assert replay["frames"][1]["snapshot"]["layers"] == {}
    assert "recorded_ee" in replay["frames"][2]["snapshot"]["layers"]["__gripper__"]
