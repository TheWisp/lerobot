"""Shared evidence at the actual matcher, RGB-D and benchmark boundaries."""

import json
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from benchmarks.flash_experiment_groups import GroupConfig
from benchmarks.flash_tracker_experiment import run_case
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.shared_image_tracking import SharedFrameTracker, SharedMatchingConfig
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig
from tests.policies.flash.test_tracker_experiment import demo


class TranslationModel:
    """Only substitute neural inference; exercise the real UFM adapter."""

    def __init__(self, dx=0, visibility=1):
        self.dx, self.visibility, self.calls = dx, visibility, 0

    def predict_correspondences_batched(self, a, b):
        self.calls += 1
        assert a.shape == b.shape
        h, w = a.shape[:2]
        flow = torch.zeros((1, 2, h, w))
        flow[:, 0] = self.dx
        return SimpleNamespace(
            flow=SimpleNamespace(flow_output=flow),
            covisibility=SimpleNamespace(mask=torch.full((1, h, w), float(self.visibility))),
        )


def frame(index, *, camera=None):
    rgb = np.random.default_rng(42).integers(0, 256, (96, 128, 3), np.uint8)
    return RGBDFrame(
        index,
        rgb,
        np.full((96, 128), 1000, np.uint16),
        np.array([[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]]),
        0.001,
        np.eye(4) if camera is None else camera,
        index / 30,
        index / 30,
        "fixture",
    )


def masks():
    left = np.zeros((96, 128), bool)
    left[20:76, 15:60] = True
    return {"left": left, "right": np.fliplr(left).copy()}


def observation(frame, mask):
    return {
        "frame_id": frame.frame_id,
        "depth": frame.depth * frame.depth_m_per_unit,
        "mask": mask,
        "usable": mask.copy(),
        "intrinsics": frame.intrinsics,
    }


@pytest.fixture
def cpu_neural_boundary(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)


def forbidden(*args, **kwargs):
    raise AssertionError("a second matching backend was invoked")


def test_one_inference_serves_objects_groups_and_retained_reference(cpu_neural_boundary, monkeypatch):
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", forbidden)
    model = TranslationModel(dx=1)
    tracker = SharedFrameTracker(SharedMatchingConfig(max_image_size_px=64, spacing_px=3), model)
    initial = frame(0)
    first = tracker.update(initial, masks())
    camera = np.eye(4)
    camera[0, 3] = 0.1
    for i in (1, 2):
        current = frame(i, camera=camera)
        scene = tracker.update(current, masks())
        for mask in masks().values():
            evidence = {}
            a, b = tracker.pairs(
                observation(initial, mask), observation(current, mask), initial.intrinsics, evidence=evidence
            )
            assert len(a) > 8
            # One pixel at half resolution is TWO sensor pixels, i.e. 20 mm.
            np.testing.assert_allclose(b - a, np.tile([0.02 * i, 0, 0], (len(a), 1)), atol=1e-12)
            common, left, right = np.intersect1d(first.ids, scene.ids, return_indices=True)
            assert len(common) > 8
            np.testing.assert_allclose(
                scene.base_xyz[right] - first.base_xyz[left],
                np.tile([0.1 + 0.02 * i, 0, 0], (len(common), 1)),
                atol=1e-12,
            )
            assert set(evidence["track_ids"][evidence["stage"] == 3]) <= set(scene.ids)
        assert model.calls == i  # Four object queries + an older reference cost zero extra inference.


def test_empty_masks_continue_scene_and_depth_failure_does_not_invent_measurements(cpu_neural_boundary):
    model = TranslationModel()
    tracker = SharedFrameTracker(SharedMatchingConfig(), model)
    first = tracker.update(frame(0), masks())
    hidden = {key: value & False for key, value in masks().items()}
    no_depth = replace(frame(1), depth=np.zeros((96, 128), np.uint16))
    scene = tracker.update(no_depth, hidden)
    assert model.calls == 1
    assert len(scene.ids) == 0
    np.testing.assert_array_equal(tracker.ids, first.ids)  # RGB identities survive missing depth.
    scene = tracker.update(frame(2), hidden)
    assert model.calls == 2
    assert set(first.ids) <= set(scene.ids)
    # Returning depth cannot retrospectively make the missing samples measured.
    a, _ = tracker.pairs(
        observation(no_depth, masks()["left"]), observation(frame(2), masks()["left"]), frame(0).intrinsics
    )
    assert len(a) == 0


def test_rejected_tracks_reseed_new_ids_with_bounded_history_and_budget(cpu_neural_boundary):
    model = TranslationModel(visibility=0)
    config = SharedMatchingConfig(scene_points=20, object_points=12, history_frames=3)
    tracker = SharedFrameTracker(config, model)
    seen = set()
    for i in range(10):
        tracker.update(frame(i), masks())
        assert not seen.intersection(tracker.ids)
        seen.update(tracker.ids)
        assert len(tracker.ids) <= 44
        assert len(tracker.history) <= 3
    a, _ = tracker.pairs(
        observation(frame(0), masks()["left"]), observation(frame(9), masks()["left"]), frame(0).intrinsics
    )
    assert len(a) == 0
    assert model.calls == 9


def test_lk_does_not_load_or_invoke_neural_model():
    tracker = SharedFrameTracker(
        SharedMatchingConfig(backend="lk"), SimpleNamespace(predict_correspondences_batched=forbidden)
    )
    first = tracker.update(frame(0), masks())
    last = tracker.update(frame(1), masks())
    assert len(np.intersect1d(first.ids, last.ids)) > 20
    with pytest.raises(ValueError, match="increasing"):
        tracker.update(frame(1), masks())
    with pytest.raises(ValueError, match="different current frame"):
        tracker.pairs(
            observation(frame(0), masks()["left"]),
            observation(frame(2), masks()["left"]),
            frame(0).intrinsics,
        )


@pytest.mark.parametrize("backend", ["lk", "ufm"])
def test_benchmark_and_fitter_use_shared_points_without_second_tracker(
    tmp_path, monkeypatch, cpu_neural_boundary, backend
):
    from lerobot.policies.flash import causal_points, surface_frame_session as experiment, surface_tracking

    monkeypatch.setattr(experiment, "ScenePointTracker", forbidden)
    monkeypatch.setattr(causal_points.CausalRigidPointTracker, "seed", forbidden)
    monkeypatch.setattr(surface_tracking, "ufm_pairs", forbidden)
    monkeypatch.setattr(experiment, "ufm_pairs", forbidden)
    if backend == "ufm":
        monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", forbidden)
    case = demo(tmp_path)
    # Add a second object to ensure all six pose methods share the same pass.
    import gzip

    for file in (tmp_path / "masks").glob("*.gz"):
        with gzip.open(file, "rt") as stream:
            value = json.load(stream)
        value["masks_rle"]["another"] = value["masks_rle"]["item"]
        with gzip.open(file, "wt") as stream:
            json.dump(value, stream)
    case["roles"].append("another")
    model = TranslationModel()
    result = run_case(
        case,
        tmp_path / "run",
        model,
        VisionPipelineConfig(),
        None,
        None,
        shared_matching_config=SharedMatchingConfig(backend=backend),
        surface_experiment=GrowingSurfaceConfig(),
        group_experiment=GroupConfig(),
        retain_motion_reference=True,
        record_correspondences=True,
    )
    assert result["shared_matching"]["calls"] == 1
    assert model.calls == (1 if backend == "ufm" else 0)
    assert set(result["roles"]) == {"item", "another"}
    for role in result["roles"].values():
        assert role["methods"]["group_growing"]["available_after_initialization"] == 1


def test_old_points_outside_mask_cannot_exhaust_reappearing_object_budget(cpu_neural_boundary):
    config = SharedMatchingConfig(scene_points=2, object_points=20, spacing_px=3)
    tracker = SharedFrameTracker(config, TranslationModel())
    first = masks()["left"]
    tracker.update(frame(0), {"item": first})
    old_ids = set(tracker.ids)
    target = masks()["right"]
    tracker.update(frame(1), {"item": target})
    pixels = np.rint(tracker.xy).astype(int)
    inside = target[pixels[:, 1], pixels[:, 0]]
    assert np.count_nonzero(inside) == config.object_points
    assert len(set(tracker.ids) - old_ids) >= 18
    assert len(tracker.ids) <= 22


@pytest.mark.parametrize("dx", [-3, 3])
def test_boundary_pixels_are_excluded_at_both_ends_of_shared_matches(cpu_neural_boundary, dx):
    tracker = SharedFrameTracker(
        SharedMatchingConfig(max_image_size_px=128, scene_points=400, object_points=1200, spacing_px=1),
        TranslationModel(dx=dx),
    )
    preparation = SurfaceTrackingSession(
        None,
        shared_matching=tracker,
        record_correspondences=True,
        observation_config=SurfaceObservationConfig(component_gap_m=None),
    )
    mask = masks()["left"]
    for i in (0, 1):
        f = frame(i)
        tracker.update(f, {"item": mask})
        result = preparation.prepare(f.rgb, f.metric_depth().astype(np.float32), mask, f.intrinsics, i)
    pairs = result["correspondences"]
    a, b = (np.rint(pairs[key]).astype(int) for key in ("source_uv", "target_uv"))

    def interior(uv):
        # Rectangle is [15,60) x [20,76). The configured two-pixel exclusion
        # gives [17,58) x [22,74); derive expected ownership independently.
        return (uv[:, 0] >= 17) & (uv[:, 0] < 58) & (uv[:, 1] >= 22) & (uv[:, 1] < 74)

    source_inside, target_inside = interior(a), interior(b)
    assert np.any(~source_inside & target_inside), "source boundary rejection must be exercised"
    assert np.any(source_inside & ~target_inside), "target boundary rejection must be exercised"
    accepted = pairs["stage"] == 3
    assert accepted.sum() > 100, "rejecting everything must not pass"
    np.testing.assert_array_equal(accepted, source_inside & target_inside)
    assert len(result["point_pairs"][0]) == accepted.sum()
