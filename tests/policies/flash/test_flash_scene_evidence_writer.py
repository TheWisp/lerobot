"""A prepared demo must preserve the same per-frame evidence its replay reads."""

import gzip
import hashlib
import json

import numpy as np
import pytest

from lerobot.policies.flash.demo_scene import scene_evidence_frame, scene_evidence_status
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.perception_step import PerceptionResult
from lerobot.policies.flash.pose_evidence import (
    ConditionalSurfaceInformation,
    PoseProposal,
    RegisteredSurfaceEvidence,
    SurfaceAgreement,
    VisibleSurfaceSupport,
)
from lerobot.policies.flash.scene_evidence_writer import SceneEvidenceWriter
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def test_preparation_evidence_crosses_into_demo_replay(tmp_path):
    bundle_dir = tmp_path / "prepared"
    bundle_dir.mkdir()
    bundle = bundle_dir / "bundle.json"
    annotation = bundle_dir / "annotation.json"
    bundle.write_text("{}", encoding="utf-8")
    annotation.write_text("{}", encoding="utf-8")
    identity = hashlib.sha256(bundle.read_bytes() + annotation.read_bytes()).hexdigest()
    writer = SceneEvidenceWriter(
        bundle_dir / "scene-evidence", total_frames=1, roles=("object",), source_sha256=identity
    )
    frame = RGBDFrame(
        0,
        np.zeros((4, 4, 3), np.uint8),
        np.full((4, 4), 1000, np.uint16),
        np.eye(3),
        0.001,
        np.eye(4),
        1.0,
        1.0,
        "calibration",
    )
    lost = PoseEstimate(
        "object",
        None,
        1.0,
        1.2,
        Evidence.LOST,
        "reference",
        (False,) * 3,
        (False,) * 3,
        estimator="surface_rgbd",
        source_frame_id=0,
    )
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    raw = np.eye(4)
    raw[0, 3] = 0.1
    motion_prior = np.eye(4)
    motion_prior[0, 3] = 0.02
    proposal = PoseProposal(
        0,
        "calibration",
        lost,
        raw,
        VisibleSurfaceSupport(
            4,
            principal_surface_axis_base=(1.0, 0.0, 0.0),
            principal_surface_spreads_m=(0.001, 0.002, 0.01),
            principal_surface_points=12,
        ),
        "full pose unobservable",
        motion_prior,
        ConditionalSurfaceInformation(
            (0.1, 0.2, 1.0),
            tuple(tuple(float(value) for value in row) for row in np.eye(6)),
            (0.0, 0.0, 0.0, 0.1, 0.2, 0.3),
            8,
            20,
            0.005,
        ),
        RegisteredSurfaceEvidence(
            "reference",
            2,
            SurfaceAgreement(0.002, 20, 0.003),
            SurfaceAgreement(0.001, 30, 0.004),
            1.2,
        ),
    )
    result = PerceptionResult((lost,), {}, 0, 1.0, "calibration", {"object": mask}, proposals=(proposal,))

    groups = {
        "object": {
            "verified": True,
            "measured_this_frame": True,
            "ids": np.array([7]),
            "xy": np.array([[2.0, 1.0]]),
            "base_xyz": np.array([[0.1, 0.2, 1.0]]),
            "last_group_measurement_s": 1.0,
        }
    }
    writer.record(frame, result, groups, {"object": lost})
    assert scene_evidence_status(bundle)["available"]
    assert not scene_evidence_status(bundle)["complete"]
    with pytest.raises(ValueError, match="source order"):
        writer.record(frame, result, groups, {"object": lost})
    writer.finish(source_sha256=identity)

    evidence = scene_evidence_frame(bundle, 0)
    assert evidence["available"]
    assert evidence["masks_rle"]["object"] == [[5, 2], [9, 2]]
    assert evidence["proposals"]["object"]["raw_pose"] == raw.tolist()
    assert evidence["proposals"]["object"]["motion_prior_pose"] == motion_prior.tolist()
    assert evidence["proposals"]["object"]["failure"] == "full pose unobservable"
    assert evidence["proposals"]["object"]["visible_support"]["principal_surface_points"] == 12
    assert evidence["proposals"]["object"]["conditional_surface_information"]["matched_reference_points"] == 8
    assert (
        evidence["proposals"]["object"]["registered_surface_evidence"]["all_views"]["reference_depth_pixels"]
        == 30
    )
    assert evidence["estimates"]["object"]["evidence"] == "lost"
    assert evidence["source_time_s"] == 1.0
    assert evidence["available_time_s"] == 1.0
    assert evidence["calibration_version"] == "calibration"
    assert evidence["base_from_camera"] == np.eye(4).tolist()
    assert evidence["estimates"]["object"]["source_frame_id"] == 0
    assert evidence["estimates"]["object"]["available_time_s"] == 1.2
    assert evidence["estimates"]["object"]["reference_version"] == "reference"
    assert evidence["proposals"]["object"]["frame_id"] == 0
    assert evidence["proposals"]["object"]["available_time_s"] == 1.2
    assert evidence["proposals"]["object"]["calibration_version"] == "calibration"
    assert evidence["groups"]["object"]["ids"] == [7]
    assert evidence["groups"]["object"]["base_xyz_m"] == [[0.1, 0.2, 1.0]]
    assert json.loads((bundle_dir / "scene-evidence/manifest.json").read_text())["complete"] is True


def test_demo_evidence_retains_measured_translation_with_constrained_orientation(tmp_path):
    writer = SceneEvidenceWriter(tmp_path / "scene", total_frames=1, roles=("object",))
    frame = RGBDFrame(
        0,
        np.zeros((4, 4, 3), np.uint8),
        np.full((4, 4), 1000, np.uint16),
        np.eye(3),
        0.001,
        np.eye(4),
        1.0,
        1.0,
        "calibration",
    )
    partial = PoseEstimate(
        "object",
        np.eye(4),
        1.0,
        1.0,
        Evidence.PREDICTED,
        "reference",
        (True,) * 3,
        (False,) * 3,
        direct_translation_observed=True,
    )
    mask = np.ones(frame.depth.shape, bool)
    result = PerceptionResult(
        (partial,),
        {},
        0,
        1.0,
        "calibration",
        {"object": mask},
        proposals=(PoseProposal(0, "calibration", partial, np.eye(4), VisibleSurfaceSupport(16)),),
    )
    writer.record(frame, result, {}, {"object": partial})
    writer.finish()
    with gzip.open(tmp_path / "scene/000000.json.gz", "rt") as stream:
        saved = json.load(stream)
    for record in (saved["estimates"]["object"], saved["proposals"]["object"]):
        assert record["evidence"] == "predicted"
        assert record["direct_translation_observed"] is True
        assert record["translation_supported"] == [True] * 3
        assert record["rotation_supported"] == [False] * 3
