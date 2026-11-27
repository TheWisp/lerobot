"""Uncertainty cannot overrule observed contradictions or freeze actual motion."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.flash_pipeline_settings import catalog, defaults, from_recording, validate
from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    PoseHint,
)
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, SurfacePoseUncertainty
from tests.policies.flash.test_experimental_surface_session import prepared_frames, transform


def setup_surface(monkeypatch):
    prepare, rgb, depth, mask, k = prepared_frames(monkeypatch)
    prepared = prepare.prepare(rgb, depth, mask, k, 0)
    geometry = WholeSurfaceFit(
        prepared["current"]["obs"]["points"], prepared["current"], k, prepare.shape_config
    )
    return prepare, rgb, depth, mask, k, geometry


@pytest.mark.parametrize("correlated_error", [0.0, 0.002, 1.0])
def test_bad_depth_and_tilt_rejected_even_with_arbitrarily_large_uncertainty(monkeypatch, correlated_error):
    *_, geometry = setup_surface(monkeypatch)
    estimator = SurfacePoseUncertainty(
        geometry, np.eye(4), PoseUncertaintyConfig(correlated_error_m=correlated_error)
    )
    tilted = np.eye(4)
    tilted[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    pivot = geometry.reference.mean(0)
    tilted[:3, 3] = pivot - tilted[:3, :3] @ pivot
    for pose in (transform(z=0.03), tilted):
        result = estimator.check(pose)
        assert not result["accepted"]
        assert not result["covariance_tested"]
        assert result["reason"] == "visible_surface_contradiction"


def test_plane_has_unobserved_directions_and_does_not_report_zero_variance(monkeypatch):
    *_, geometry = setup_surface(monkeypatch)
    estimator = SurfacePoseUncertainty(geometry, np.eye(4), PoseUncertaintyConfig())
    assert estimator.rank == 3
    assert estimator.covariance is None
    assert estimator.check(transform(x=0.001))["accepted"]
    assert not estimator.check(transform(z=0.004))["accepted"]


def test_observed_motion_rejects_stale_group_and_empty_mask_still_predicts(monkeypatch):
    prepare, rgb, depth, mask, k, _ = setup_surface(monkeypatch)
    tracker = GrowingSurfaceSession(
        prepare.shape_config, GrowingSurfaceConfig(uncertainty=PoseUncertaintyConfig(enabled=True))
    )
    tracker.update(prepare.prepare(rgb, depth, mask, k, 1), 0.1)
    model = tracker.store.points.copy()
    hidden = tracker.update(
        prepare.prepare(rgb, depth, mask & False, k, 2),
        0.2,
        hint=PoseHint(transform(z=0.02), "synthetic support", 0.2, 0.1, 1),
    )
    assert hidden["status"] == "predicted"
    np.testing.assert_allclose(hidden["camera_from_reference"], transform(z=0.02))
    np.testing.assert_array_equal(model, tracker.store.points)
    # A real measured 30mm change must defeat the stale group. RGB is textureless;
    # use the real depth/mask optimizer, not a supplied correct motion.
    result = tracker.update(
        prepare.prepare(rgb, depth + 0.03, mask, k, 3),
        0.3,
        hint=PoseHint(np.eye(4), "stale support", 0.3, 0.1, 1),
    )
    assert result["status"] == "observed_candidate"
    assert abs(np.array(result["camera_from_reference"])[2, 3] - 0.03) < 0.005


def test_no_valid_candidate_does_not_return_rejected_prediction(monkeypatch):
    prepare, rgb, depth, mask, k, _ = setup_surface(monkeypatch)
    tracker = GrowingSurfaceSession(
        prepare.shape_config,
        GrowingSurfaceConfig(uncertainty=PoseUncertaintyConfig(enabled=True), retain_unobserved_pose=True),
    )
    tracker.update(prepare.prepare(rgb, depth, mask, k, 1), 0.1)
    version = tracker.store.version
    # A smoothly corrugated observation cannot fit the stored plane. The supplied hint is also wrong; neither may be published.
    depth += (0.04 * np.sin(np.linspace(0, 4 * np.pi, depth.shape[1])))[None, :]
    result = tracker.update(
        prepare.prepare(rgb, depth, mask, k, 2),
        0.2,
        hint=PoseHint(transform(z=0.20), "bad prediction", 0.2, 0.1, 1),
    )
    assert result["camera_from_reference"] is None
    assert tracker.store.version == version


def test_nested_json_roundtrip_reaches_real_consumer_and_cache():
    config = GrowingSurfaceConfig(uncertainty=PoseUncertaintyConfig(enabled=True, correlated_error_m=0.004))
    restored = GrowingSurfaceConfig(**asdict(config))
    session = GrowingSurfaceSession(ShapeConfig(), restored)
    assert session.config.uncertainty.correlated_error_m == 0.004
    value = defaults()
    value["surface_experiment"] = asdict(config)
    normalized = validate(value)
    assert normalized["surface_experiment"]["uncertainty"]["enabled"]
    assert normalized != defaults()
    paths = {row["path"] for row in catalog()}

    def leaves(values, prefix=""):
        for key, value in values.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                yield from leaves(value, path)
            else:
                yield path

    assert paths == set(leaves(defaults()))
    with pytest.raises(ValueError):
        replace(config, surface_priority=False)
    with pytest.raises(TypeError):
        GrowingSurfaceConfig(uncertainty={"enabledd": True})


def test_settings_reference_covers_the_actual_configuration_catalog():
    from pathlib import Path

    import lerobot.policies.flash as flash

    reference = Path(flash.__file__).parent / "docs" / "pipeline_settings.md"
    rows = [
        [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        for line in reference.read_text().splitlines()
        if line.startswith("| `")
    ]
    import json

    assert rows == [[row["path"], json.dumps(row["default"]), row["description"]] for row in catalog()]


def test_recorded_group_evidence_weight_moves_to_object_estimator():
    old = defaults()
    old["surface_experiment"].pop("group_point_evidence_weight")
    old["group_experiment"]["point_evidence_weight"] = 0.4
    restored = validate(old)
    assert restored["surface_experiment"]["group_point_evidence_weight"] == 0.4
    assert "point_evidence_weight" not in restored["group_experiment"]


def test_object_group_association_minimum_is_validated_separately():
    assert GrowingSurfaceConfig(group_association_min_points=4).group_association_min_points == 4
    with pytest.raises(ValueError):
        GrowingSurfaceConfig(group_association_min_points=2)


@pytest.mark.parametrize(
    "values",
    [
        {"independent_error_m": 0},
        {"max_measurements": True},
        {"compatibility_probability": float("nan")},
        {"max_visible_outlier_fraction": 1},
        {"enabled": 1},
    ],
)
def test_invalid_uncertainty_settings_rejected(values):
    with pytest.raises(ValueError):
        PoseUncertaintyConfig(**values)


def test_old_recordings_keep_filter_off_while_new_sessions_use_kalman():
    settings = defaults()
    session = GrowingSurfaceSession(ShapeConfig(), GrowingSurfaceConfig(**settings["surface_experiment"]))
    assert session.config.uncertainty.enabled
    assert session.config.uncertainty.mode == "kalman"
    assert session.config.uncertainty.max_visible_outlier_fraction == 0.25
    old = {**settings, "surface_experiment": dict(settings["surface_experiment"])}
    del old["surface_experiment"]["uncertainty"]
    restored = from_recording(old, ["group_growing"])
    legacy = GrowingSurfaceSession(ShapeConfig(), GrowingSurfaceConfig(**restored["surface_experiment"]))
    assert not legacy.config.uncertainty.enabled
    assert "uncertainty" not in old["surface_experiment"]  # Loading cannot rewrite provenance.
    # Explicit historical tuning wins over the new defaults too.
    old["surface_experiment"]["uncertainty"] = asdict(
        PoseUncertaintyConfig(enabled=True, mode="kalman", max_visible_outlier_fraction=0.1)
    )
    restored = from_recording(old, ["group_growing"])
    assert restored["surface_experiment"]["uncertainty"] == old["surface_experiment"]["uncertainty"]
