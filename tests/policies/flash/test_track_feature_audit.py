"""Physical invariants and known blind spots of the offline drift measurements."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.flash_track_feature_audit import grasp_proxy, measure


def test_rigid_motion_with_visibility_changes_has_no_deformation():
    source = np.random.default_rng(4).normal(size=(30, 3)) * 0.04
    points = np.array(
        [
            source @ Rotation.from_euler("xyz", [t, -t / 2, t / 3]).as_matrix().T + [t, 0, 1]
            for t in np.linspace(0, 1, 20)
        ]
    )
    support = np.ones(points.shape[:2], dtype=bool)
    support[3:9, :4] = False
    report, per_id = measure(points, support)
    assert report["all_frames_pair_distance_change_mm"]["p90"] < 1e-9
    assert np.max(per_id) < 1e-9
    assert report["continuous"]["count"] == 26


def test_one_drifting_point_is_visible_in_per_track_scores():
    source = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (-0.02, 0.02)])
    points = np.repeat(source[None], 4, axis=0)
    points[1:, 0, 0] -= 0.03
    _, per_id = measure(points, np.ones(points.shape[:2], dtype=bool))
    assert per_id[0] > 15
    assert np.max(per_id[1:]) < 1e-9


def test_coherent_wrong_translation_needs_an_independent_reference():
    source = np.random.default_rng(8).normal(size=(20, 3)) * 0.02
    points = np.repeat(source[None], 5, axis=0)
    points[:, :, 0] += np.linspace(0, 0.02, 5)[:, None]
    support = np.ones(points.shape[:2], dtype=bool)
    report, _ = measure(points, support)
    assert report["all_frames_pair_distance_change_mm"]["p90"] < 1e-9
    proxy = grasp_proxy(points, support, np.repeat(np.eye(4)[None], 5, axis=0))["shared"]
    assert proxy["error_mm"]["median"] == pytest.approx(20)
    assert proxy["coherence_ratio"] == pytest.approx(1)
    assert proxy["fraction_squared_error_explained_by_common_translation"] == pytest.approx(1)
