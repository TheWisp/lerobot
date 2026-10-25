"""Deployment parity, configuration and lifetime of the native CPU path."""

import sys
from unittest.mock import patch

import numpy as np
import pytest
from test_native_acceleration_candidates import recorded_fit
from test_vision_pipeline import Masks, frames, reference, synthetic_pairs

from lerobot.policies.flash.perception_step import PerceptionPipeline, PerceptionRunner
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend


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


def test_default_shared_pipeline_matches_disabled_path_and_closes_search_pool():
    pytest.importorskip("numba")
    outputs = []
    for enabled in (False, True):
        config = VisionPipelineConfig.from_dict(
            {"compiled_surface_fit": enabled, "reuse_surface_search": enabled}
        )
        assert VisionPipelineConfig.from_dict(config.to_dict()) == config
        backend = build_vision_backend(
            config, Masks(), matcher=object(), local_reference=True, clock=lambda: 0.0
        )
        pool = backend._search_pool
        pipeline = PerceptionPipeline(backend, (reference(),))
        source = iter(frames())
        results = []
        with (
            patch("lerobot.policies.flash.surface_tracking.ufm_pairs", synthetic_pairs),
            PerceptionRunner(pipeline) as runner,
        ):
            while (item := runner.process_next(lambda source=source: next(source, None))) is not None:
                results.append(item[1])
        if enabled:
            assert pool.closed
            assert results[1].details["object"]["diagnostics"]["compiled_surface_fit"]
            assert results[1].details["object"]["diagnostics"]["reused_search_workers"] > 0
        outputs.append(results)
    for old, new in zip(*outputs, strict=True):
        assert (old.frame_id, old.source_time_s, old.failures) == (
            new.frame_id,
            new.source_time_s,
            new.failures,
        )
        a, b = old.estimates[0], new.estimates[0]
        assert a.evidence == b.evidence
        assert (a.pose is None) == (b.pose is None)
        if a.pose is not None:
            np.testing.assert_array_equal(a.pose, b.pose)
        np.testing.assert_array_equal(old.masks["object"], new.masks["object"])
    assert outputs[1][2].estimates[0].pose is None
    assert outputs[1][3].estimates[0].pose is not None
    assert (
        np.linalg.norm(outputs[1][3].estimates[0].pose[:3, 3] - outputs[1][0].estimates[0].pose[:3, 3]) > 0.01
    )


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
