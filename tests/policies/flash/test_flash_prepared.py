"""Prepared references remain portable derived data beside a regular episode."""

import json

import numpy as np
import pytest

from lerobot.policies.flash.appearance import AppearanceModel
from lerobot.policies.flash.inspect import inspect_skill
from lerobot.policies.flash.matched_rgbd import ObjectPointModel
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.skill import FlashSkill, Interval, MotionMode, Relation, SpatialSample
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
    prepared = PreparedFlashSkill(
        skill,
        (reference,),
        (model,),
        {"object": "patterned object"},
        "facebook/dinov2-small",
        vision_pipeline=VisionPipelineConfig(registration_threshold_m=0.007),
    )
    manifest = prepared.save(tmp_path / "flash/v1/prepared/demo-a")
    restored = PreparedFlashSkill.load(manifest)
    assert restored.skill.intervals[0].reference_version == "object-v1"
    assert [snippet.frame_index for snippet in restored.references[0].snippets] == [3, 9]
    np.testing.assert_allclose(restored.appearance_models[0].points.points_object_m, points)
    assert restored.vision_pipeline == prepared.vision_pipeline
    assert restored.vision_pipeline.backend == "surface"
    assert restored.prompts_by_role == {"object": "patterned object"}
    assert restored.descriptor_model_id == "facebook/dinov2-small"
    assert not (manifest.parent / "rgb.mp4").exists()
    inspected = inspect_skill(str(manifest))
    assert inspected["runtime_bindings_complete"]
    assert inspected["prepared_references"] == [
        {"role": "object", "version": "object-v1", "views": 2, "model_points": 3}
    ]

    data = json.loads(manifest.read_text())
    del data["vision_pipeline"]
    manifest.write_text(json.dumps(data))
    assert PreparedFlashSkill.load(manifest).vision_pipeline is None
    data["models"] = "../outside.npz"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inside the bundle"):
        PreparedFlashSkill.load(manifest)
