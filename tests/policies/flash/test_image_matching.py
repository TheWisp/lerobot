"""Backend contract and golden outputs captured BEFORE the matching refactor."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from lerobot.policies.flash.image_matching import (
    LKMatchConfig,
    UFMImagePair,
    UFMMatchConfig,
    prepare_image_pair,
)
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.scene_groups import ScenePointTracker
from lerobot.policies.flash.surface_correspondence import robust_fit, ufm_pairs


class RecordedModel:
    def __init__(self, flow, visibility):
        self.calls = 0
        self.output = SimpleNamespace(
            flow=SimpleNamespace(flow_output=torch.from_numpy(flow)[None]),
            covisibility=SimpleNamespace(mask=torch.from_numpy(visibility)[None]),
        )

    def predict_correspondences_batched(self, a, b):
        self.calls += 1
        self.inputs = (a.cpu().numpy().copy(), b.cpu().numpy().copy())
        return self.output


@pytest.mark.parametrize("backend", ["lk", "ufm"])
def test_same_query_contract_keeps_order_and_rejected_rows(backend, monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    gray = np.random.default_rng(8).integers(0, 256, (96, 96), np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    shifted = cv2.warpAffine(rgb, np.float32([[1, 0, 3], [0, 1, 0]]), (96, 96))
    queries = np.array([[40, 40], [50, 30], [45, 55], [-50, -50]], float)
    flow = np.zeros((2, 96, 96), np.float32)
    flow[0] = 3
    model = RecordedModel(flow, np.ones((96, 96), np.float32))
    config = LKMatchConfig(pyramid_levels=1) if backend == "lk" else UFMMatchConfig()
    pair = prepare_image_pair(rgb, shifted, config, model=model)
    matches = pair.match(queries)
    np.testing.assert_array_equal(matches.source_xy, queries)
    np.testing.assert_array_equal(matches.accepted, [True, True, True, False])
    np.testing.assert_allclose(matches.target_xy[:3], queries[:3] + [3, 0], atol=0.03)
    assert matches.target_xy.shape == queries.shape  # No silent row deletion / ID shuffle.
    assert pair.match(np.empty((0, 2))).target_xy.shape == (0, 2)
    if backend == "ufm":
        assert model.calls == 1  # Repeated point queries reuse the pair, not inference.
        assert matches.forward_backward_error_px is None
        np.testing.assert_array_equal(model.inputs[0], rgb)
        np.testing.assert_array_equal(model.inputs[1], shifted)
    else:
        assert matches.covisibility is None
        assert model.calls == 0


def test_ufm_fractional_queries_interpolate_without_rounding_or_rejecting_target_outside():
    y, x = np.mgrid[:8, :8]
    flow = np.stack((2 * x, 3 * y)).astype(np.float32)
    visibility = np.ones((8, 8), np.float32)
    visibility[1, 1] = 0.4
    pair = UFMImagePair(flow, visibility, UFMMatchConfig())
    matches = pair.match(np.array([[1.5, 2.25], [7, 7], [1, 1], [8, 3]]))
    np.testing.assert_allclose(matches.target_xy[:3], [[4.5, 9], [21, 28], [3, 4]])
    np.testing.assert_array_equal(matches.accepted, [True, True, False, False])
    # A prediction outside the target remains available to diagnostics; the
    # existing depth/mask consumer rejects it separately. Changing that here
    # would hide erroneous raw matches in the viewer.
    assert matches.target_xy[1, 0] > 8
    assert np.isnan(matches.target_xy[3]).all()
    strict = UFMImagePair(flow, visibility, UFMMatchConfig(covisibility_min=0.3))
    assert strict.match(np.array([[1, 1]])).accepted[0]


def test_lk_configuration_reaches_opencv_and_uses_native_pixel_units():
    gray = np.random.default_rng(9).integers(0, 256, (64, 64), np.uint8)
    shifted = cv2.warpAffine(gray, np.float32([[1, 0, 1.4], [0, 1, 0.6]]), (64, 64))
    queries = np.array([[20, 20], [35, 35], [50, 12]], np.float32)
    config = LKMatchConfig(window_size_px=15, pyramid_levels=1, forward_backward_limit_px=0.01)
    before = queries.reshape(-1, 1, 2)
    after, ok, _ = cv2.calcOpticalFlowPyrLK(gray, shifted, before, None, winSize=(15, 15), maxLevel=1)
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(shifted, gray, after, None, winSize=(15, 15), maxLevel=1)
    error = np.linalg.norm((before - back).reshape(-1, 2), axis=1)
    result = prepare_image_pair(gray, shifted, config).match(queries)
    np.testing.assert_array_equal(result.target_xy, after.reshape(-1, 2))
    np.testing.assert_array_equal(result.forward_backward_error_px, error)
    np.testing.assert_array_equal(
        result.accepted, (ok.ravel() == 1) & (back_ok.ravel() == 1) & (error < 0.01)
    )


@pytest.mark.parametrize("config", [LKMatchConfig, UFMMatchConfig])
def test_reject_invalid_matching_configuration(config):
    with pytest.raises(ValueError):
        config(
            **({"forward_backward_limit_px": np.nan} if config is LKMatchConfig else {"covisibility_min": 2})
        )
    with pytest.raises(ValueError):
        prepare_image_pair(np.zeros((5, 5, 3)), np.zeros((5, 5, 3)), config())
    with pytest.raises(ValueError, match="finite Nx2"):
        pair = prepare_image_pair(np.zeros((5, 5), np.uint8), np.zeros((5, 5), np.uint8), LKMatchConfig())
        pair.match(np.array([[np.nan, 0]]))


@pytest.mark.parametrize("name", ["gamepad_168", "dowel_240"])
def test_recorded_matches_depth_fit_and_scene_ids_equal_original(name, monkeypatch):
    """Expected arrays came from frozen old source, not from this adapter."""
    fixture = Path(__file__).parent / "fixtures" / f"image_matching_{name}.npz"
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    with np.load(fixture, allow_pickle=False) as data:
        model = RecordedModel(data["flow"], data["visibility"])
        a, b = (
            {key: data[f"{side}_{key}"] for key in ("depth", "mask", "usable")}
            for side in ("source", "target")
        )
        evidence = {}
        source, target = ufm_pairs(
            model,
            data["source_rgb"],
            data["target_rgb"],
            data["crop"],
            a,
            b,
            data["intrinsics"],
            evidence=evidence,
        )
        for key in ("source_uv", "target_uv", "stage"):
            np.testing.assert_array_equal(evidence[key], data[f"expected_{key}"])
        np.testing.assert_array_equal(source, data["expected_source_xyz"])
        np.testing.assert_array_equal(target, data["expected_target_xyz"])
        step, _, _ = robust_fit(source, target)
        np.testing.assert_array_equal(step, data["expected_step"])
        assert model.calls == 1
        np.testing.assert_array_equal(model.inputs[0], data["source_rgb"])
        np.testing.assert_array_equal(model.inputs[1], data["target_rgb"])
        assert len(source) > 6
        assert np.any(evidence["stage"] < 3)  # Actually exercises rejection.
        assert np.max(np.linalg.norm(target - source, axis=1)) > 0

        tracker = ScenePointTracker()
        for key in ("_gray", "_uv", "_ids", "_next_id", "_last_time_s"):
            value = data["scene" + key]
            setattr(tracker, key, value.item() if value.ndim == 0 else value.copy())
        frame = RGBDFrame(
            1,
            data["scene_rgb"],
            data["scene_depth"],
            data["intrinsics"],
            float(data["scene_depth_m_per_unit"]),
            data["scene_base_from_camera"],
            float(data["scene_time_s"]),
            float(data["scene_time_s"]),
            "recorded-test",
        )
        tracks = tracker.update(frame)
        for value, key in ((tracks.ids, "ids"), (tracks.xy, "xy"), (tracks.base_xyz, "xyz")):
            np.testing.assert_array_equal(value, data["expected_scene_" + key])
        assert len(tracks.ids) > 100


def test_ufm_integer_grid_preserves_native_threshold_comparison():
    # NumPy compares the original float32 field at float32 precision, including
    # when the configured Python threshold is between adjacent float32 values.
    visibility = np.full((4, 4), 0.5, np.float32)
    threshold = 0.50000001
    pair = UFMImagePair(np.zeros((2, 4, 4), np.float32), visibility, UFMMatchConfig(threshold))
    result = pair.match(np.array([[1, 1]]))
    np.testing.assert_array_equal(result.accepted, visibility[[1], [1]] >= threshold)
