"""A recorded trial can become a demo without changing object coordinate frames."""

import json

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.appearance import AppearanceModel
from lerobot.policies.flash.compiler import IntervalAnnotation
from lerobot.policies.flash.matched_rgbd import ObjectPointModel
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.recorded_demo import prepare_recorded_episode
from lerobot.policies.flash.recorded_trial import inspect_recorded_trial, prepare_recorded_trial
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample
from lerobot.policies.flash.surface_reference import RegisteredSurfaceModel
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


def _recorded_inputs(root, *, version="object-v1", symmetry=False, second_object=False):
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
    target_reference = ObjectReference(
        "target",
        "target-v1",
        (
            ReferenceSnippet(
                "demo/source", 0, 4, "target", identity, mask.relative_to(root).as_posix(), "cal"
            ),
        ),
    )
    target_model = AppearanceModel(
        ObjectPointModel("target", "target-v1", np.array([[0.0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])),
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
        source_skill,
        (reference, target_reference) if second_object else (reference,),
        (model, target_model) if second_object else (model,),
        {"object": "patterned object", **({"target": "blue target"} if second_object else {})},
        "encoder",
        symmetries,
        surface_models=(RegisteredSurfaceModel("object", "object-v1", (model.points.points_object_m,)),),
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
                            },
                            **(
                                {
                                    "target": {
                                        "pose": identity.tolist(),
                                        "source_time_s": frame / 30,
                                        "available_time_s": frame / 30,
                                        "evidence": "observed",
                                        "reference_version": "target-v1",
                                        "translation_supported": [True, True, True],
                                        "rotation_supported": [True, True, True],
                                    }
                                }
                                if second_object
                                else {}
                            ),
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
    np.testing.assert_allclose(
        prepared.surface_models[0].view_points_object_m[0],
        PreparedFlashSkill.load(reference_path).surface_models[0].view_points_object_m[0],
    )


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
    assert inspection["pose_counts"]["object"]["observed"] == 3
    assert "reference" in inspection["reason"] or "trial_config" in inspection["reason"]


def test_regular_recorded_demo_uses_dataset_local_source_without_trial_config(tmp_path, monkeypatch):
    from lerobot.gui.api.run import _flash_recorded_demo_replay, _flash_recordings
    from lerobot.policies.flash import recorded_trial

    dataset = LeRobotDataset.create(
        repo_id="trial/recording",
        fps=30,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["joint.pos", "gripper.pos"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["joint.pos", "gripper.pos"]},
        },
        root=tmp_path / "dataset",
        robot_type="so107_follower_gravity",
    )
    for index in range(3):
        dataset.add_frame(
            {
                "task": "move an object",
                "observation.state": np.array([index, 0], dtype=np.float32),
                "action": np.array([index, 1], dtype=np.float32),
            }
        )
    dataset.save_episode()
    dataset.finalize()
    root = dataset.root
    reference_path = _recorded_inputs(root)
    robot_dir = root / "flash/v1/robot"
    robot_dir.mkdir()
    calibration = robot_dir / "calibration.json"
    calibration.write_text(json.dumps({"tool_point": {"offset_m": [0, 0, 0]}}))
    source_dir = root / "flash/v1/recorded-sources"
    source_dir.mkdir()
    source = {
        "schema_version": 1,
        "episode": 0,
        "repo_id": "trial/recording",
        "reference_bundle": reference_path.relative_to(root).as_posix(),
        "robot": {
            "type": "so107_follower_gravity",
            "arm": "left",
            "calibration": calibration.relative_to(root).as_posix(),
        },
        "max_pose_age_s": 0.2,
    }
    source_path = source_dir / "episode_000000.json"
    source_path.write_text(json.dumps(source))
    monkeypatch.setattr(recorded_trial, "MOTOR_NAMES", ("joint", "gripper"))
    monkeypatch.setattr(
        recorded_trial, "make_calibrated_so107_kinematics", lambda *args, **kwargs: Kinematics()
    )
    monkeypatch.setattr(recorded_trial, "SelectedJointKinematics", lambda inner, *args: inner)

    assert inspect_recorded_trial(root, 0)["ready"]
    listing = _flash_recordings((tmp_path,))
    assert [(item["kind"], item["episode"]) for item in listing] == [("recorded_source_demo", 0)]
    output = prepare_recorded_trial(
        root,
        0,
        "generic-candidate",
        [
            {
                "first_frame": 0,
                "last_frame": 2,
                "mode": "demonstrated",
                "relation": "ee_to_object",
                "reference_role": "object",
            }
        ],
    )
    skill = PreparedFlashSkill.load(output).skill
    assert skill.source_repo_id == "trial/recording"
    assert skill.intervals[0].held_role is None
    assert not skill.attachment_transitions
    replay = _flash_recorded_demo_replay(root, 0, output)
    assert replay["kind"] == "recorded_demo"
    assert replay["prepared_intervals"][0]["reference_role"] == "object"

    source["reference_bundle"] = "../outside/bundle.json"
    source_path.write_text(json.dumps(source))
    inspection = inspect_recorded_trial(root, 0)
    assert not inspection["ready"] and "reference snapshot" in inspection["reason"]


def test_recorded_demo_replay_selects_rgbd_from_dataset_metadata(tmp_path):
    from lerobot.gui.api.run import _flash_tracker_replay

    extension = tmp_path / "flash/v1"
    source_dir = extension / "recorded-sources"
    source_dir.mkdir(parents=True)
    (source_dir / "episode_000000.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "episode": 0,
                "repo_id": "trial/recording",
                "rgb_feature": "observation.images.overhead",
                "depth_feature": "observation.images.depth",
            }
        )
    )
    (extension / "episode_000000.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "frame_index": 0,
                "camera_source_time_s": None,
                "poses": {},
            }
        )
        + "\n"
    )
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.overhead": {"dtype": "video", "shape": [48, 64, 3]},
                    "observation.images.wrist": {"dtype": "video", "shape": [48, 64, 3]},
                    "observation.images.depth": {
                        "dtype": "video",
                        "shape": [48, 64, 1],
                        "info": {"is_depth_map": True},
                    },
                }
            }
        )
    )
    replay = _flash_tracker_replay(tmp_path, 0)
    assert replay["camera_features"] == {
        "rgb": "observation.images.overhead",
        "depth": "observation.images.depth",
    }


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
    clip = prepare_recorded_trial(
        tmp_path,
        0,
        "clip",
        [{"first_frame": 1, "last_frame": 2, "mode": "connection", "relation": "ee_to_start"}],
    )
    assert [
        sample.source_frame_index for sample in PreparedFlashSkill.load(clip).skill.intervals[0].samples
    ] == [
        1,
        2,
    ]


def test_recorded_trial_editor_gates_a_section_on_its_observed_result(tmp_path, monkeypatch):
    from lerobot.gui.api import run
    from lerobot.policies.flash import recorded_trial

    reference_path = _recorded_inputs(tmp_path, second_object=True)
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
                "max_pose_age_s": 0.8,
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
    section = {
        "first_frame": 0,
        "last_frame": 2,
        "mode": "demonstrated",
        "relation": "ee_to_object",
        "reference_role": "object",
        "verify_relation": {
            "moving_role": "object",
            "reference_role": "target",
            "translation_tolerance_mm": 6,
            "rotation_tolerance_deg": "",
        },
    }
    output = prepare_recorded_trial(tmp_path, 0, "checked", [section])
    skill = PreparedFlashSkill.load(output).skill
    assert skill.intervals[0].completion_milestone == "section-1-result"
    assert skill.milestones[0].pose_relation.translation_tolerance_m == pytest.approx(0.006)
    assert skill.milestones[0].pose_relation.max_pose_age_s == pytest.approx(0.8)
    assert skill.milestones[0].require_post_motion_evidence
    assert json.loads((output.parent / "preparation-report.json").read_text())["result_checks"] == 1
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
    assert replay["prepared_intervals"][0]["verify_relation"] == {
        "moving_role": "object",
        "reference_role": "target",
        "translation_tolerance_mm": pytest.approx(6),
        "rotation_tolerance_deg": "",
    }
    section["task_outcome"] = True
    outcome_output = prepare_recorded_trial(tmp_path, 0, "outcome-checked", [section])
    outcome_skill = PreparedFlashSkill.load(outcome_output).skill
    assert outcome_skill.intervals[0].completion_milestone is None
    assert outcome_skill.outcome_milestone == "section-1-result"
    outcome_replay = run._flash_recorded_demo_replay(tmp_path, 0, outcome_output)
    assert outcome_replay["prepared_intervals"][0]["task_outcome"] is True
    assert outcome_replay["prepared_intervals"][0]["verify_relation"]["translation_tolerance_mm"] == 6
    section["verify_relation"]["translation_tolerance_mm"] = ""
    with pytest.raises(ValueError, match="explicit distance tolerance"):
        prepare_recorded_trial(tmp_path, 0, "invalid", [section])


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
