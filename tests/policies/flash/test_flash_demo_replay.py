"""Demo overlays report retained samples and their capture-time provenance."""

from __future__ import annotations

import json

import numpy as np

from lerobot.policies.flash.appearance import AppearanceModel
from lerobot.policies.flash.demo_replay import (
    _add_recorded_attachment_trace,
    _demo_annotation_path,
    build_demo_replay,
)
from lerobot.policies.flash.matched_rgbd import ObjectPointModel
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def test_demo_attachment_replay_uses_each_image_time_not_a_future_connection_endpoint(tmp_path) -> None:
    bundle = tmp_path / "bundle.json"
    (tmp_path / "preparation-report.json").write_text(
        json.dumps({"held_attachment_trace": "held-attachment-trace.npz"})
    )
    np.savez_compressed(
        tmp_path / "held-attachment-trace.npz",
        role=np.asarray("workpiece"),
        camera_times_s=np.array([0.25, 0.75]),
        gripper_from_object=np.stack([_pose(0.1), _pose(0.2)]),
        conflicted=np.array([False, False]),
        corrected=np.array([False, True]),
    )
    replay = {
        "frames": [
            {
                "snapshot": {
                    "recorded_at_s": time_s,
                    "held_prediction": {"role": "workpiece", "pose": _pose(1.2).tolist()},
                }
            }
            for time_s in (0.25, 0.75)
        ]
    }
    _add_recorded_attachment_trace(bundle, replay, np.array([0.0, 1.0]), np.stack([_pose(0), _pose(1)]))
    first, second = (entry["snapshot"]["held_prediction"] for entry in replay["frames"])
    assert np.isclose(first["pose"][0][3], 0.35)
    assert first["evidence"] == "predicted"
    assert np.isclose(second["pose"][0][3], 0.95)
    assert second["evidence"] == "vision_corrected_prediction"


def test_demo_attachment_replay_marks_ambiguous_visual_evidence_without_losing_gripper_pose(tmp_path):
    bundle = tmp_path / "bundle.json"
    (tmp_path / "preparation-report.json").write_text(
        json.dumps({"held_attachment_trace": "held-attachment-trace.npz"})
    )
    np.savez_compressed(
        tmp_path / "held-attachment-trace.npz",
        role=np.asarray("workpiece"),
        camera_times_s=np.array([0.25]),
        gripper_from_object=np.stack([_pose(0.1)]),
        conflicted=np.array([False]),
        visual_ambiguous=np.array([True]),
        corrected=np.array([True]),
    )
    replay = {
        "frames": [
            {
                "snapshot": {
                    "recorded_at_s": 0.25,
                    "held_prediction": {"role": "workpiece", "pose": _pose(1.2).tolist()},
                }
            }
        ]
    }
    _add_recorded_attachment_trace(bundle, replay, np.array([0.0, 1.0]), np.stack([_pose(0), _pose(1)]))
    held = replay["frames"][0]["snapshot"]["held_prediction"]
    assert np.isclose(held["pose"][0][3], 0.35)
    assert held["evidence"] == "ambiguous_visual_prediction"


def test_demo_replay_uses_the_selected_preparation_recipe(tmp_path) -> None:
    bundle = tmp_path / "flash/v1/prepared/new/bundle.json"
    bundle.parent.mkdir(parents=True)
    old = tmp_path / "flash/v1/import-annotations.json"
    old.write_text('{"grasp_reference_frame": 100}')
    assert _demo_annotation_path(tmp_path, bundle) == old
    updated = bundle.parent / "annotation.json"
    updated.write_text('{"grasp_reference_frame": 130}')
    assert _demo_annotation_path(tmp_path, bundle) == updated


def test_demo_replay_keeps_dense_samples_and_labels_sparse_endpoints() -> None:
    demonstrated = Interval(
        "contact",
        MotionMode.DEMONSTRATED,
        Relation.EE_TO_OBJECT,
        "object",
        (
            SpatialSample(0.0, _pose(0), _pose(0.2), source_frame_index=0, reference_evidence="observed"),
            SpatialSample(1.0, _pose(0.1), _pose(0.2), source_frame_index=1, reference_evidence="predicted"),
        ),
        reference_version="ref-v1",
    )
    connection = Interval(
        "carry",
        MotionMode.CONNECTION,
        Relation.EE_TO_OBJECT,
        "object",
        (
            SpatialSample(1.0, _pose(0.1), _pose(0.2), source_frame_index=1, reference_evidence="predicted"),
            SpatialSample(2.0, _pose(0.3), _pose(0.4), source_frame_index=2, reference_evidence="observed"),
        ),
        speed_m_s=0.1,
        reference_version="ref-v1",
    )
    reference = ObjectReference(
        "object", "ref-v1", (ReferenceSnippet("demo", 0, 0, "object", np.eye(4), "mask.png", "cal-v1"),)
    )
    points = ObjectPointModel("object", "ref-v1", np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]]))
    prepared = PreparedFlashSkill(
        FlashSkill("demo", 0, (demonstrated, connection)),
        (reference,),
        (AppearanceModel(points, np.ones((3, 2))),),
        {"object": "object"},
        "descriptor-v1",
    )
    replay = build_demo_replay(
        prepared,
        arm_times_s=np.array([0.0, 1.0, 1.5, 2.0, 2.5]),
        arm_tips=np.stack([_pose(x) for x in (0, 0.1, 0.2, 0.3, 0.4)]),
        camera_times_s=np.array([0.01, 0.99, 1.5, 2.01, 2.5]),
        intrinsics=np.eye(3),
        base_from_camera=np.eye(4),
        image_size=(640, 480),
        fps=30,
    )
    frames = [entry["snapshot"] for entry in replay["frames"]]
    assert frames[0]["demo"]["phase"] == "demonstrated_sample"
    assert frames[1]["accepted"]["object"]["evidence"] == "predicted"
    assert frames[2]["demo"]["phase"] == "connection_endpoint"
    assert "object" not in frames[2]["accepted"]
    assert frames[2]["demo_goal"]["role"] == "object"
    assert frames[2]["demo_goal"]["pose"][0][3] == 0.4
    assert frames[2]["demo_goal"]["source_time_s"] == 2.0
    assert frames[2]["accepted"]["__gripper__"]["pose"][0][3] == 0.2
    assert frames[2]["demo"]["camera_arm_delta_s"] == 0.0
    assert frames[2]["camera"]["frame_id"] == 2
    assert "object" not in frames[3]["accepted"]
    assert frames[3]["demo"]["phase"] == "outside_skill"
    assert replay["raw_tracker_history"] is False


def test_connection_replay_projects_held_object_from_recorded_gripper_at_camera_time() -> None:
    attachment_x = 0.5
    connection = Interval(
        "carry",
        MotionMode.CONNECTION,
        Relation.HELD_TO_TARGET,
        "receiver",
        (
            SpatialSample(1.0, _pose(0.1), _pose(1.0), _pose(0.1 + attachment_x)),
            SpatialSample(2.0, _pose(0.3), _pose(1.0), _pose(0.3 + attachment_x)),
        ),
        speed_m_s=0.1,
        reference_version="receiver-v1",
        held_reference_version="workpiece-v1",
        held_role="workpiece",
    )
    reference = ObjectReference(
        "workpiece",
        "workpiece-v1",
        (ReferenceSnippet("demo", 0, 0, "workpiece", np.eye(4), "mask.png", "cal-v1"),),
    )
    target_reference = ObjectReference(
        "receiver",
        "receiver-v1",
        (ReferenceSnippet("demo", 0, 0, "receiver", np.eye(4), "target-mask.png", "cal-v1"),),
    )
    points = ObjectPointModel("workpiece", "workpiece-v1", np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]]))
    target_points = ObjectPointModel(
        "receiver", "receiver-v1", np.array([[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]])
    )
    prepared = PreparedFlashSkill(
        FlashSkill("demo", 0, (connection,)),
        (reference, target_reference),
        (AppearanceModel(points, np.ones((3, 2))), AppearanceModel(target_points, np.ones((3, 2)))),
        {"workpiece": "workpiece", "receiver": "receiver"},
        "descriptor-v1",
    )
    replay = build_demo_replay(
        prepared,
        arm_times_s=np.array([1.0, 1.5, 2.0]),
        arm_tips=np.stack([_pose(x) for x in (0.1, 0.2, 0.3)]),
        camera_times_s=np.array([1.0, 1.5, 2.0]),
        intrinsics=np.eye(3),
        base_from_camera=np.eye(4),
        image_size=(640, 480),
        fps=30,
    )
    middle = replay["frames"][1]["snapshot"]
    assert middle["demo"]["phase"] == "connection_endpoint"
    assert np.isclose(middle["held_prediction"]["pose"][0][3], 0.7)
    assert middle["held_prediction"]["source_time_s"] == 1.5
    assert middle["held_prediction"]["estimator"] == "demo_held_recorded_gripper"
    assert "receiver" not in middle["accepted"]
    assert middle["demo_goal"]["role"] == "receiver"
    assert middle["demo_goal"]["source_time_s"] == 2.0
