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


def _reference_color_projection(left_depth, left_k, color_k, color_from_left, color_shape):
    """Original 2D-scatter projection, retained as an equivalence oracle."""
    y, x = np.indices(left_depth.shape, dtype=np.float32)
    z = left_depth.ravel()
    good = np.isfinite(z) & (z > 0)
    if not np.any(good):
        return np.zeros(color_shape, np.float32)
    x, y, z = x.ravel()[good], y.ravel()[good], z[good]
    points = np.stack(
        ((x - left_k[0, 2]) * z / left_k[0, 0], (y - left_k[1, 2]) * z / left_k[1, 1], z),
        axis=0,
    )
    moved = color_from_left[:3, :3] @ points + color_from_left[:3, 3:4]
    cz = moved[2]
    good = np.isfinite(cz) & (cz > 0)
    u = color_k[0, 0] * moved[0, good] / cz[good] + color_k[0, 2]
    v = color_k[1, 1] * moved[1, good] / cz[good] + color_k[1, 2]
    cz = cz[good].astype(np.float32)
    out = np.full(color_shape, np.inf, np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            pixel_x, pixel_y = np.floor(u).astype(np.int32) + dx, np.floor(v).astype(np.int32) + dy
            inside = (pixel_x >= 0) & (pixel_y >= 0) & (pixel_x < color_shape[1]) & (pixel_y < color_shape[0])
            np.minimum.at(out, (pixel_y[inside], pixel_x[inside]), cz[inside])
    out[~np.isfinite(out)] = 0
    return fill_projection_gaps(out)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_flat_scatter_matches_original_stereo_projection(seed):
    rng = np.random.default_rng(seed)
    left = rng.uniform(0.3, 0.9, (32, 48)).astype(np.float32)
    left[rng.random(left.shape) < 0.15] = 0
    left[2, 3] = np.nan
    left[12:20, 20:35] = 0.35  # a foreground patch over farther samples
    left_k = np.array([[35, 0, 24], [0, 35, 16], [0, 0, 1]], float)
    color_k = np.array([[54, 0, 35], [0, 54, 22], [0, 0, 1]], float)
    color_from_left = np.eye(4)
    color_from_left[:3, :3] = np.array([[0.9998, 0, 0.02], [0, 1, 0], [-0.02, 0, 0.9998]], float)
    color_from_left[:3, 3] = [0.015, -0.005, 0]
    args = (left, left_k, color_k, color_from_left, (48, 72))
    np.testing.assert_array_equal(color_aligned_stereo_depth(*args), _reference_color_projection(*args))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gpu_stereo_projection_matches_original_depth_and_holes(seed):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA stereo projection requires a GPU")
    rng = np.random.default_rng(seed)
    left = rng.uniform(0.3, 0.9, (32, 48)).astype(np.float32)
    left[rng.random(left.shape) < 0.15] = 0
    left[2, 3] = np.nan
    left[12:20, 20:35] = 0.35
    left_k = np.array([[35, 0, 24], [0, 35, 16], [0, 0, 1]], float)
    color_k = np.array([[54, 0, 35], [0, 54, 22], [0, 0, 1]], float)
    color_from_left = np.eye(4)
    color_from_left[:3, :3] = [[0.9998, 0, 0.02], [0, 1, 0], [-0.02, 0, 0.9998]]
    color_from_left[:3, 3] = [0.015, -0.005, 0]
    args = (left_k, color_k, color_from_left, (48, 72))
    expected = _reference_color_projection(left, *args)
    actual = depth_enhancement.color_aligned_stereo_depth_gpu(torch.from_numpy(left).cuda(), *args)
    np.testing.assert_array_equal(actual > 0, expected > 0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_gpu_stereo_projection_preserves_all_missing_depth():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA stereo projection requires a GPU")
    left = torch.zeros((3, 4), device="cuda")
    actual = depth_enhancement.color_aligned_stereo_depth_gpu(left, np.eye(3), np.eye(3), np.eye(4), (5, 6))
    np.testing.assert_array_equal(actual, np.zeros((5, 6), np.float32))


def test_dynamic_onnx_output_uses_runtime_allocated_fallback():
    class Output:
        name = "disparity"
        type = "tensor(float)"
        shape = [1, 1, "height", "width"]

    class Session:
        def io_binding(self):
            pytest.fail("Dynamic output cannot bind a fixed CUDA buffer")

        def get_outputs(self):
            return [Output()]

    assert depth_enhancement._stereo_disparity_gpu(Session(), {}) is None


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


@pytest.mark.parametrize(
    "model_class,shape", [(FoundationStereoDepth, (2, 2)), (LiteAnyStereoV2Depth, (480, 848))]
)
def test_stereo_adapters_use_gpu_output_without_copying_disparity_to_cpu(model_class, shape):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA stereo projection requires a GPU")
    model = model_class.__new__(model_class)

    class Session:
        def run_gpu(self, _, inputs):
            name = "left_image" if model_class is FoundationStereoDepth else "left"
            assert inputs[name].shape[-2:] == (shape[0], 864 if shape[1] == 848 else 32)
            return [torch.full((1, 1, shape[0], inputs[name].shape[-1]), 2.0, device="cuda")]

        def run(self, _, inputs):
            pytest.fail("Stereo disparity was copied to CPU before projection")

    model.session = Session()
    k = np.eye(3, dtype=np.float32)
    stereo = StereoInput(np.ones(shape, np.uint8), np.ones(shape, np.uint8), k, np.eye(4), 0.5)
    sensor = np.zeros(shape, np.float32)
    sensor[0, 0] = 0.4
    output = model.predict(DepthInput(np.zeros((*shape, 3), np.uint8), sensor, k, stereo))
    assert output.depth_m[shape[0] // 2, shape[1] // 2] == pytest.approx(0.25)
    np.testing.assert_array_equal(output.sensor_measured, sensor > 0)


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
