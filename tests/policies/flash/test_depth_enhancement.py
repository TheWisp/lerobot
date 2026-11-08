"""Geometry and provenance checks for opt-in learned depth sources."""

import numpy as np
import pytest

from lerobot.policies.flash import depth_enhancement
from lerobot.policies.flash.depth_enhancement import (
    DepthInput,
    FoundationStereoDepth,
    LiteAnyStereoV2Depth,
    StereoInput,
    color_aligned_stereo_depth,
    fill_projection_gaps,
    make_depth_source,
    needs_stereo_pair,
)


def test_stereo_depth_projects_to_color_with_nearest_visible_sample():
    left = np.zeros((3, 3), np.float32)
    left[1, 1] = 1.0
    left[1, 2] = 2.0
    k = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    color_from_left = np.eye(4)
    color_from_left[0, 3] = -1.0
    actual = color_aligned_stereo_depth(left, k, k, color_from_left, (4, 4))
    # Nearer point is shifted one pixel; farther one shifts half a pixel.
    assert actual[1, 0] == pytest.approx(1.0)
    assert actual[1, 1] == pytest.approx(1.0)
    assert actual[0, 0] == 0


def test_projection_fills_regular_resampling_gap_without_crossing_depth_edge():
    depth = np.full((9, 9), 0.4, np.float32)
    depth[:, 5:] = 0.6
    depth[4, 2] = 0  # upsampling gap surrounded by one continuous surface
    depth[4, 4] = 0  # hole exactly at the 20 cm depth step
    depth[1:5, 6:9] = 0  # larger missing patch is not one-pixel resampling
    result = fill_projection_gaps(depth)
    assert result[4, 2] == pytest.approx(0.4)
    assert result[4, 4] == 0
    assert result[3, 7] == 0
    np.testing.assert_array_equal(result[depth > 0], depth[depth > 0])


def test_foundation_stereo_requires_pair_and_preserves_sensor_provenance():
    model = FoundationStereoDepth.__new__(FoundationStereoDepth)

    class Session:
        def run(self, _, inputs):
            assert inputs["left_image"].shape == (1, 3, 2, 32)
            return [np.full((1, 1, 2, 32), 2.0, np.float32)]

    model.session = Session()
    k = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    sensor = np.array([[0.0, 0.4], [0.4, 0.0]], np.float32)
    stereo = StereoInput(np.ones((2, 2), np.uint8), np.ones((2, 2), np.uint8), k, np.eye(4), 0.5)
    result = model.predict(DepthInput(np.zeros((2, 2, 3), np.uint8), sensor, k, stereo))
    assert result.depth_m.shape == sensor.shape
    assert result.depth_m[0, 0] == pytest.approx(0.5)
    np.testing.assert_array_equal(result.sensor_measured, sensor > 0)
    with pytest.raises(ValueError, match="requires synchronized"):
        model.predict(DepthInput(np.zeros((2, 2, 3), np.uint8), sensor, k))


def test_fast_foundation_stereo_does_not_silently_use_slow_dynamic_onnx(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LEROBOT_FOUNDATION_STEREO_MODEL", raising=False)
    monkeypatch.delenv("LEROBOT_FOUNDATION_STEREO_ONNX", raising=False)
    old = tmp_path / ".cache/flash-foundationstereo/fast-foundationstereo-dynhw.onnx"
    old.parent.mkdir(parents=True)
    old.touch()
    monkeypatch.setattr(depth_enhancement, "_cuda_onnx_session", lambda path: pytest.fail("used slow ONNX"))
    with pytest.raises(FileNotFoundError, match="fixed 480x864 TensorRT engine"):
        make_depth_source("foundation_stereo")

    engine = tmp_path / ".cache/flash-foundationstereo/fixed-480x864-d192-i8/fast_foundationstereo.engine"
    engine.parent.mkdir(parents=True)
    engine.touch()
    monkeypatch.setattr(depth_enhancement, "_TensorRtSession", lambda path: path)
    assert make_depth_source("foundation_stereo").session == engine


def test_fast_foundation_stereo_allows_explicit_onnx_trial(tmp_path, monkeypatch):
    model = tmp_path / "trial.onnx"
    model.touch()
    monkeypatch.setenv("LEROBOT_FOUNDATION_STEREO_MODEL", str(model))
    monkeypatch.setattr(depth_enhancement, "_cuda_onnx_session", lambda path: path)
    assert make_depth_source("foundation_stereo").session == model


def test_fast_foundation_stereo_pads_ir_and_applies_external_imagenet_normalization():
    image = np.full((480, 848), 255, np.uint8)
    image[0, 0] = 0
    prepared = FoundationStereoDepth._image(image)
    assert prepared.shape == (1, 3, 480, 864)
    np.testing.assert_allclose(
        prepared[0, :, 0, 0], -np.array([0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225], rtol=1e-6
    )
    np.testing.assert_allclose(
        prepared[0, :, 0, -1], (1 - np.array([0.485, 0.456, 0.406])) / [0.229, 0.224, 0.225], rtol=1e-6
    )


def test_las2_h_uses_symmetric_ir_padding_and_shared_depth_alignment():
    model = LiteAnyStereoV2Depth.__new__(LiteAnyStereoV2Depth)

    class Session:
        def run(self, _, inputs):
            assert inputs["left"].shape == (1, 3, 480, 864)
            assert inputs["right"].shape == (1, 3, 480, 864)
            assert inputs["left"][0, 0, 200, 0] == inputs["left"][0, 0, 200, 8]
            assert inputs["left"][0, 0, 200, -1] == inputs["left"][0, 0, 200, -9]
            return [np.full((1, 1, 480, 864), 2.0, np.float32)]

    model.session = Session()
    shape = (480, 848)
    k = np.eye(3, dtype=np.float32)
    stereo = StereoInput(np.ones(shape, np.uint8), np.ones(shape, np.uint8), k, np.eye(4), 0.5)
    output = model.predict(DepthInput(np.zeros((*shape, 3), np.uint8), np.ones(shape, np.float32), k, stereo))
    assert output.source == "lite_any_stereo_v2"
    assert output.depth_m[200, 400] == pytest.approx(0.25)


def test_sensor_source_has_no_model_loading():
    assert make_depth_source("sensor") is None
    with pytest.raises(ValueError, match="Unknown depth source"):
        make_depth_source("guess")


@pytest.mark.parametrize(
    "size,source",
    [
        ("s", "lite_any_stereo_v2_s"),
        ("m", "lite_any_stereo_v2_m"),
        ("l", "lite_any_stereo_v2_l"),
        ("h", "lite_any_stereo_v2"),
    ],
)
def test_las2_sizes_use_distinct_models_and_shared_stereo_input(tmp_path, monkeypatch, size, source):
    checkpoint = tmp_path / f"las2-{size}-480x864.onnx"
    checkpoint.touch()
    monkeypatch.setattr(depth_enhancement, "_cuda_onnx_session", lambda path: path)
    model = LiteAnyStereoV2Depth(checkpoint, model_size=size)
    assert model.source == source
    assert model.session == checkpoint
    assert needs_stereo_pair(source)
    monkeypatch.setattr(depth_enhancement, "LiteAnyStereoV2Depth", lambda **kwargs: kwargs)
    if size != "h":
        assert make_depth_source(source) == {"model_size": size}
    else:
        assert make_depth_source(source) == {}
