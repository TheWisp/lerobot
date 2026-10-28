"""Prepared references remain portable derived data beside a regular episode."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from lerobot.policies.flash.appearance import AppearanceModel, ReferenceAppearanceFrame
from lerobot.policies.flash.causal_service import CausalTrackingConfig
from lerobot.policies.flash.inspect import inspect_skill
from lerobot.policies.flash.matched_rgbd import ObjectPointModel
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.prepared import PreparedFlashSkill, snapshot_prepared_skill
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample
from lerobot.policies.flash.surface_reference import RegisteredSurfaceModel, build_registered_surface_model
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def test_prepared_skill_round_trip_keeps_object_frames_and_multiview_provenance(tmp_path):
    identity = np.eye(4)
    skill = FlashSkill(
        "demo/data",
        0,
        (
            Interval(
                "approach",
                MotionMode.DEMONSTRATED,
                Relation.EE_TO_OBJECT,
                "object",
                (SpatialSample(0, identity, identity), SpatialSample(1, identity, identity)),
                reference_version="object-v1",
            ),
        ),
    )
    snippets = (
        ReferenceSnippet("demo/data", 0, 3, "object", identity, "flash/v1/masks/3.png", "cal-v1"),
        ReferenceSnippet("demo/data", 0, 9, "object", identity, "flash/v1/masks/9.png", "cal-v1"),
    )
    reference = ObjectReference("object", "object-v1", snippets)
    points = np.array([[0.0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
    model = AppearanceModel(ObjectPointModel("object", "object-v1", points), np.eye(3))
    surface = RegisteredSurfaceModel("object", "object-v1", (points, points + [0, 0, 0.02]))
    prepared = PreparedFlashSkill(
        skill,
        (reference,),
        (model,),
        {"object": "patterned object"},
        "facebook/dinov2-small",
        vision_pipeline=VisionPipelineConfig(
            registration_threshold_m=0.007, surface_reference_max_points_per_view=42
        ),
        causal_tracking=CausalTrackingConfig(60, 64, 12, 0.1, 0.005),
        surface_models=(surface,),
    )
    manifest = prepared.save(tmp_path / "flash/v1/prepared/demo-a")
    restored = PreparedFlashSkill.load(manifest)
    assert restored.skill.intervals[0].reference_version == "object-v1"
    assert [snippet.frame_index for snippet in restored.references[0].snippets] == [3, 9]
    np.testing.assert_allclose(restored.appearance_models[0].points.points_object_m, points)
    for actual, expected in zip(
        restored.surface_models[0].view_points_object_m, surface.view_points_object_m, strict=True
    ):
        np.testing.assert_allclose(actual, expected)
    assert restored.vision_pipeline == prepared.vision_pipeline
    assert restored.causal_tracking == prepared.causal_tracking
    assert json.loads(manifest.read_text())["causal_tracking"] == asdict(prepared.causal_tracking)
    assert restored.vision_pipeline.backend == "surface"
    assert restored.prompts_by_role == {"object": "patterned object"}
    assert restored.descriptor_model_id == "facebook/dinov2-small"
    assert not (manifest.parent / "rgb.mp4").exists()
    with pytest.raises(ValueError, match="registered surface views differ"):
        PreparedFlashSkill(
            skill,
            (reference,),
            (model,),
            {"object": "patterned object"},
            "encoder",
            surface_models=(RegisteredSurfaceModel("object", "object-v1", (points,)),),
        )
    from lerobot.policies.flash.live_trial import trial

    mismatched_trial = tmp_path / "mismatched-trial.json"
    mismatched_trial.write_text(
        json.dumps({"causal_tracking": asdict(CausalTrackingConfig(60, 64, 12, 0.1, 0.006))})
    )
    with pytest.raises(ValueError, match="live scene tracking differs"):
        trial(manifest, mismatched_trial, preflight=True)
    inspected = inspect_skill(str(manifest))
    assert inspected["runtime_bindings_complete"]
    assert inspected["prepared_references"] == [
        {
            "role": "object",
            "version": "object-v1",
            "views": 2,
            "model_points": 3,
            "registered_surface_views": 2,
            "registered_surface_points": 6,
        }
    ]

    data = json.loads(manifest.read_text())
    del data["vision_pipeline"]
    manifest.write_text(json.dumps(data))
    assert PreparedFlashSkill.load(manifest).vision_pipeline is None
    del data["causal_tracking"]
    manifest.write_text(json.dumps(data))
    assert PreparedFlashSkill.load(manifest).causal_tracking is None
    for entry in data["references"]:
        del entry["surface_view_count"]
    manifest.write_text(json.dumps(data))
    assert PreparedFlashSkill.load(manifest).surface_models == ()
    data["models"] = "../outside.npz"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inside the bundle"):
        PreparedFlashSkill.load(manifest)


def test_registered_surface_builder_uses_declared_object_frame_and_aligned_depth():
    depth = np.full((24, 24), 1000, dtype=np.uint16)
    mask = np.zeros(depth.shape, bool)
    mask[3:21, 3:21] = True
    k = np.array([[100, 0, 12], [0, 100, 12], [0, 0, 1]], dtype=float)
    transform = np.eye(4)
    transform[0, 3] = 0.1
    snippets = (
        ReferenceSnippet("demo/data", 0, 2, "object", np.eye(4), "flash/v1/masks/2.png", "cal"),
        ReferenceSnippet("demo/data", 0, 4, "object", transform, "flash/v1/masks/4.png", "cal"),
    )
    reference = ObjectReference("object", "object-v1", snippets)
    views = tuple(
        ReferenceAppearanceFrame(
            snippet,
            RGBDFrame(
                snippet.frame_index,
                np.zeros((*depth.shape, 3), np.uint8),
                depth,
                k,
                0.001,
                np.eye(4),
                float(index),
                float(index),
                "cal",
            ),
            mask,
        )
        for index, snippet in enumerate(snippets)
    )
    model = build_registered_surface_model(
        reference, views, observation_config=VisionPipelineConfig().observation, max_points_per_view=500
    )
    first, second = model.view_points_object_m
    assert len(first) >= 3 and len(second) >= 3
    np.testing.assert_allclose(second - first, np.tile([0.1, 0, 0], (len(first), 1)))
    with pytest.raises(ValueError, match="registered RGB-D snippet"):
        build_registered_surface_model(
            reference,
            views[::-1],
            observation_config=VisionPipelineConfig().observation,
            max_points_per_view=500,
        )


def test_trial_snapshot_keeps_reference_frames_and_masks_with_recording(tmp_path):
    identity = np.eye(4)
    source_root = tmp_path / "source"
    recording_root = tmp_path / "recording"
    mask = source_root / "flash/v1/masks/reference.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"recorded mask bytes")
    skill = FlashSkill(
        "demo/source",
        0,
        (
            Interval(
                "move",
                MotionMode.DEMONSTRATED,
                Relation.EE_TO_OBJECT,
                "item",
                (SpatialSample(0, identity, identity), SpatialSample(1, identity, identity)),
                reference_version="item-v1",
            ),
        ),
    )
    reference = ObjectReference(
        "item",
        "item-v1",
        (ReferenceSnippet("demo/source", 0, 7, "item", identity, "flash/v1/masks/reference.png", "cal-v1"),),
    )
    model = AppearanceModel(
        ObjectPointModel("item", "item-v1", np.array([[0.0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])),
        np.eye(3),
    )
    surface = RegisteredSurfaceModel("item", "item-v1", (model.points.points_object_m,))
    prepared = PreparedFlashSkill(
        skill, (reference,), (model,), {"item": "patterned item"}, "encoder", surface_models=(surface,)
    )
    path = snapshot_prepared_skill(
        prepared,
        source_dataset_root=source_root,
        recording_dataset_root=recording_root,
        episode_index=0,
    )
    mask.unlink()
    restored = PreparedFlashSkill.load(path)
    snippet = restored.references[0].snippets[0]
    assert snippet.repo_id == "demo/source"
    assert snippet.frame_index == 7
    assert snippet.object_from_camera.tolist() == identity.tolist()
    assert (recording_root / snippet.mask_artifact).read_bytes() == b"recorded mask bytes"
    np.testing.assert_allclose(
        restored.appearance_models[0].points.points_object_m, model.points.points_object_m
    )
    np.testing.assert_allclose(
        restored.surface_models[0].view_points_object_m[0], surface.view_points_object_m[0]
    )
    assert restored.skill.source_repo_id == "demo/source"
    with pytest.raises(FileExistsError, match="already exists"):
        snapshot_prepared_skill(
            prepared,
            source_dataset_root=source_root,
            recording_dataset_root=recording_root,
            episode_index=0,
        )
