"""Deployment parity, configuration and lifetime of the native CPU path."""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation
from lerobot.policies.flash.vision_config import VisionPipelineConfig


def recorded_fit(name):
    data = np.load(Path(__file__).with_name("fixtures") / f"{name}.npz")
    is_ring = "source_surface" in data
    points = data["source_surface" if is_ring else "model_points"]
    depth = data["target_depth_mm" if is_ring else "depth"].astype(np.float32) * 0.001
    mask = data["target_mask" if is_ring else "mask"].astype(bool)
    k = data["intrinsics"]
    obs = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    current = {"obs": obs, "mask": mask, "depth": depth, "all_surface_points": obs["points"]}
    initial = np.eye(4) if is_ring else data["candidate"]
    return WholeSurfaceFit(points, current, k, ShapeConfig()), initial


@pytest.mark.parametrize("name", ["ring_visible_motion_20261016", "dowel_held_frame_315"])
def test_deployed_fitter_matches_independent_numpy_path(name):
    pytest.importorskip("numba")
    baseline, initial = recorded_fit(name)
    kernels = compiled_kernels()
    assert kernels is not None
    expected, old = baseline.correct(initial)
    with SearchPool(8) as pool:
        fit = WholeSurfaceFit(
            baseline.reference,
            baseline.observation,
            baseline.k,
            baseline.config,
            fit_kernels=kernels,
            search_pool=pool,
        )
        actual, new = fit.correct(initial)
    np.testing.assert_array_equal(actual, expected)
    assert new == old


def test_missing_optional_compiler_retains_numpy_path(monkeypatch, caplog):
    compiled_kernels.cache_clear()
    try:
        with monkeypatch.context() as context:
            context.setitem(sys.modules, "numba", None)
            assert compiled_kernels() is None
            assert "using NumPy" in caplog.text
    finally:
        compiled_kernels.cache_clear()


@pytest.mark.parametrize("key", ["compiled_surface_fit", "reuse_surface_search"])
def test_native_options_reject_non_boolean_values(key):
    with pytest.raises(ValueError, match="booleans"):
        VisionPipelineConfig.from_dict({key: 1})
