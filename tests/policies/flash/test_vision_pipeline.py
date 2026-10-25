"""Identical observations must have identical meanings in every consumer."""

from dataclasses import replace
from queue import Queue
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.offline_perception import extract_pose_trace
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.perception_step import PerceptionPipeline, PerceptionResult
from lerobot.policies.flash.perception_worker import _run_backend
from lerobot.policies.flash.tracking_debug_worker import debug_reference, make_debug_pipeline
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate
from lerobot.policies.flash.vision_pipeline import (
    FrameMaskCache,
    SurfacePoseBackend,
    VisionPipelineConfig,
    build_vision_backend,
)


def frames():
    k = np.array([[200, 0, 40], [0, 200, 32], [0, 0, 1]], dtype=float)
    return [
        RGBDFrame(
            i,
            np.zeros((64, 80, 3), np.uint8),
            np.full((64, 80), 800, np.uint16),
            k,
            0.001,
            np.eye(4),
            i / 30,
            i / 30,
            "cal",
        )
        for i in range(4)
    ]


class Masks:
    def __init__(self):
        self.calls = []

    def masks(self, frame):
        self.calls.append(frame.frame_id)
        mask = np.zeros(frame.depth.shape, bool)
        if frame.frame_id != 2:
            mask[20:44, 15 + frame.frame_id : 55 + frame.frame_id] = True
        return {"object": mask}


def reference():
    return ObjectReference(
        "object",
        "preview:0",
        (ReferenceSnippet("live-preview", 0, 0, "object", np.eye(4), "unrecorded", "cal"),),
    )


def synthetic_pairs(model, a, b, crop, previous, current, k, config):
    """Known pixel identities replace only neural inference, not metric fitting."""
    source = previous["obs"]["points"]
    old_min = np.argwhere(previous["mask"])[:, 1].min()
    new_min = np.argwhere(current["mask"])[:, 1].min()
    delta = np.array([(new_min - old_min) * 0.8 / k[0, 0], 0, 0])
    return source, source + delta


def local_backend():
    return build_vision_backend(
        VisionPipelineConfig(), Masks(), matcher=object(), local_reference=True, clock=lambda: 0.0
    )


def test_offline_spawned_worker_and_debug_consume_the_same_surface_results():
    sequence = frames()
    offline = []
    with patch("lerobot.policies.flash.surface_tracking.ufm_pairs", synthetic_pairs):
        cv2.setRNGSeed(17)
        extract_pose_trace(
            sequence, local_backend(), (reference(),), causal_tracking=None, on_vision_result=offline.append
        )
        cv2.setRNGSeed(17)
        inputs, results = Queue(), Queue()
        for frame in sequence:
            inputs.put(frame)
        inputs.put(None)
        # This is the actual spawned worker entry point, run synchronously so
        # all supplied frames arrive and the neural test double is deterministic.
        _run_backend(local_backend, (reference(),), inputs, results, True)
        worker = []
        while not results.empty():
            result = results.get()
            if isinstance(result, PerceptionResult):
                worker.append(result)
        cv2.setRNGSeed(17)
        debugger = make_debug_pipeline(VisionPipelineConfig(), Masks(), object(), sequence[0])
        debug = [debugger.process(frame) for frame in sequence]
    assert len(offline) == len(worker) == len(debug) == 4
    for a, b, c in zip(offline, worker, debug, strict=True):
        for other in (b, c):
            assert other.frame_id == a.frame_id
            assert other.source_time_s == a.source_time_s
            assert other.failures == a.failures
            np.testing.assert_array_equal(other.masks["object"], a.masks["object"])
            assert other.estimates[0].evidence == a.estimates[0].evidence
            if a.estimates[0].pose is not None:
                np.testing.assert_allclose(other.estimates[0].pose, a.estimates[0].pose, atol=1e-10)
                np.testing.assert_allclose(
                    other.details["object"]["anchors_m"], a.details["object"]["anchors_m"], atol=1e-10
                )
    assert offline[1].estimates[0].evidence == Evidence.OBSERVED
    assert offline[2].estimates[0].evidence == Evidence.LOST
    assert "camera_from_object" not in offline[2].details["object"]
    assert offline[3].details["object"]["diagnostics"]["reference_frame_id"] == 0
    assert offline[3].estimates[0].evidence == Evidence.OBSERVED
    shift = offline[3].estimates[0].pose[:3, 3] - offline[0].estimates[0].pose[:3, 3]
    np.testing.assert_allclose(shift, [0.012, 0, 0], atol=VisionPipelineConfig().shape.surface_spacing_m / 2)


def test_adding_local_object_preserves_first_reference_frame():
    class TwoMasks:
        def masks(self, frame):
            first = np.zeros(frame.depth.shape, bool)
            first[20:44, 15:45] = True
            masks = {"object": first}
            if frame.frame_id:
                second = np.zeros(frame.depth.shape, bool)
                second[20:44, 48:72] = True
                masks["object_2"] = second
            return masks

    sequence = frames()
    backend = build_vision_backend(
        VisionPipelineConfig(), TwoMasks(), matcher=object(), local_reference=True, clock=lambda: 0.0
    )
    pipeline = PerceptionPipeline(backend, (reference(),), require_masks=True)
    with patch("lerobot.policies.flash.surface_tracking.ufm_pairs", synthetic_pairs):
        pipeline.process(sequence[0])
        pipeline.add_reference(debug_reference("object_2", sequence[1]))
        output = pipeline.process(sequence[1])
    assert set(output.details) == {"object", "object_2"}
    assert output.details["object"]["diagnostics"]["reference_frame_id"] == 0
    assert output.details["object_2"]["diagnostics"]["reference_frame_id"] == 1


def test_parallel_surface_updates_match_serial_results_for_two_objects():
    class TwoMasks:
        def masks(self, frame):
            masks = {}
            for role, left in (("object", 8), ("object_2", 47)):
                mask = np.zeros(frame.depth.shape, bool)
                mask[20:44, left + frame.frame_id : left + frame.frame_id + 22] = True
                masks[role] = mask
            return masks

    outputs = []
    with patch("lerobot.policies.flash.surface_tracking.ufm_pairs", synthetic_pairs):
        for parallel in (False, True):
            backend = build_vision_backend(
                VisionPipelineConfig(parallel_surface_objects=parallel),
                TwoMasks(),
                matcher=object(),
                local_reference=True,
                clock=lambda: 0.0,
            )
            pipeline = PerceptionPipeline(
                backend,
                (reference(), debug_reference("object_2", frames()[0])),
                require_masks=True,
            )
            cv2.setRNGSeed(17)
            outputs.append([pipeline.process(frame) for frame in frames()[:2]])
    serial, parallel = outputs
    for old, new in zip(serial, parallel, strict=True):
        assert old.failures == new.failures
        assert old.masks.keys() == new.masks.keys()
        for a, b in zip(old.estimates, new.estimates, strict=True):
            assert (a.role, a.evidence) == (b.role, b.evidence)
            if a.pose is not None:
                np.testing.assert_allclose(a.pose, b.pose, atol=1e-10)
        for role in old.details:
            assert old.details[role]["status"] == new.details[role]["status"]
            np.testing.assert_allclose(
                old.details[role]["camera_from_reference"],
                new.details[role]["camera_from_reference"],
                atol=1e-10,
            )


class RegisteredInitializer:
    def __init__(self, masks, pose, *, observed=True):
        self.masks, self.pose, self.observed = masks, pose, observed
        self.calls = 0

    def initialize(self, references):
        self.reference = references[0]

    def update(self, frame):
        self.calls += 1
        self.masks.masks(frame)
        return (
            PoseEstimate(
                "object",
                self.pose if self.observed else None,
                frame.source_time_s,
                frame.available_time_s,
                Evidence.OBSERVED if self.observed else Evidence.LOST,
                self.reference.version,
                (self.observed,) * 3,
                (self.observed,) * 3,
            ),
        )


def test_registered_frame_is_transported_without_recentering_or_resegmenting():
    sequence = frames()[:2]
    base = np.eye(4)
    base[:3, 3] = [0.2, -0.1, 0.3]
    sequence = [replace(f, base_from_camera=base) for f in sequence]
    seed = np.eye(4)
    seed[:3, 3] = [0.17, -0.08, 1.1]
    source = Masks()
    masks = FrameMaskCache(source)
    initializer = RegisteredInitializer(masks, seed)
    backend = SurfacePoseBackend(
        masks, object(), VisionPipelineConfig(), initializer=initializer, clock=lambda: 0.0
    )
    pipeline = PerceptionPipeline(backend, (reference(),), require_masks=True)
    with patch("lerobot.policies.flash.surface_tracking.ufm_pairs", synthetic_pairs):
        start, moved = (pipeline.process(f).estimates[0] for f in sequence)
    np.testing.assert_allclose(start.pose, seed, atol=1e-10)
    delta = np.asarray(backend.last_details["object"]["camera_from_reference"])
    expected = base @ delta @ np.linalg.inv(base) @ seed
    np.testing.assert_allclose(moved.pose, expected, atol=1e-10)
    np.testing.assert_allclose(
        moved.pose[:3, 3] - seed[:3, 3],
        [0.004, 0, 0],
        atol=VisionPipelineConfig().shape.surface_spacing_m / 2,
    )
    assert initializer.calls == 1  # no repeated demo-to-live matching
    assert source.calls == [0, 1]  # initializer and tracker share each segmentation
    # Deliberately chosen object origin differs from display A; don't use A as
    # the prepared object's origin merely because it is convenient to display.
    assert not np.allclose(moved.pose[:3, 3], backend.last_details["object"]["anchors_m"][0])


def test_missing_cross_episode_registration_cannot_become_a_local_reference():
    masks = FrameMaskCache(Masks())
    backend = SurfacePoseBackend(
        masks,
        object(),
        VisionPipelineConfig(),
        initializer=RegisteredInitializer(masks, np.eye(4), observed=False),
    )
    pipeline = PerceptionPipeline(backend, (reference(),))
    result = pipeline.process(frames()[0])
    assert result.estimates[0].evidence == Evidence.LOST
    assert result.details["object"]["status"] == "awaiting_reference_registration"
    assert not backend.sessions


def test_failed_rgb_motion_remains_visible_for_debugging_but_not_authoritative():
    pipeline = PerceptionPipeline(local_backend(), (reference(),))
    sequence = frames()
    pipeline.process(sequence[0])
    with patch(
        "lerobot.policies.flash.surface_tracking.ufm_pairs",
        lambda *args: (np.zeros((0, 3)), np.zeros((0, 3))),
    ):
        result = pipeline.process(sequence[1])
    assert result.estimates[0].pose is None
    assert result.estimates[0].evidence == Evidence.LOST
    assert result.details["object"]["status"] == "fitted_without_rgb_motion"
    assert result.details["object"]["base_from_object"] is not None
    assert result.details["object"]["accepted"] is False


def test_pipeline_rejects_frame_or_camera_discontinuity_instead_of_corrupting_reference():
    pipeline = PerceptionPipeline(local_backend(), (reference(),))
    sequence = frames()
    pipeline.process(sequence[0])
    with pytest.raises(ValueError, match="must increase"):
        pipeline.process(sequence[0])
    with pytest.raises(ValueError, match="camera geometry"):
        pipeline.process(replace(sequence[1], intrinsics=sequence[1].intrinsics * 2))


def test_config_round_trip_and_invalid_values_are_common_to_all_consumers():
    config = VisionPipelineConfig.from_settings(
        {"registration_threshold_m": 0.007, "vision_pipeline": {"backend": "surface"}}
    )
    assert VisionPipelineConfig.from_dict(config.to_dict()) == config
    assert config.registration_threshold_m == 0.007
    for change in (
        {"backend": "typo"},
        {"motion": {"grid_step_px": 2.5}},
        {"shape": {"depth_tolerance_m": float("nan")}},
        {"observation": {"mask_margin_px": -1}},
        {"motion": {"grid_step_px": 0}},
        {"parallel_surface_objects": 1},
    ):
        with pytest.raises(ValueError):
            VisionPipelineConfig.from_dict(change)
