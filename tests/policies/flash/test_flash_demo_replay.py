"""Demo overlays report retained samples and their capture-time provenance."""

from __future__ import annotations

import numpy as np

from lerobot.policies.flash.appearance import AppearanceModel
from lerobot.policies.flash.demo_replay import build_demo_replay
from lerobot.policies.flash.matched_rgbd import ObjectPointModel
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


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
    assert frames[2]["accepted"]["object"]["pose"][0][3] == 0.4
    assert frames[2]["accepted"]["object"]["source_time_s"] == 2.0
    assert frames[2]["accepted"]["__gripper__"]["pose"][0][3] == 0.2
    assert frames[2]["demo"]["camera_arm_delta_s"] == 0.0
    assert frames[2]["camera"]["frame_id"] == 2
    assert "object" not in frames[3]["accepted"]
    assert frames[3]["demo"]["phase"] == "outside_skill"
    assert replay["raw_tracker_history"] is False
