"""Acceptance requirements before changing recovery; no pose/fitter is mocked.

Known failures are strict xfails: an unexpected pass must be reviewed, not
silently absorbed. Run --runxfail to see the original failing assertions.
"""

from functools import cache

import numpy as np
import pytest

from benchmarks.flash_kalman_recovery_control import HIDDEN, MODES, WARMUP, run_scene
from benchmarks.flash_recovery_fixture import run_mug
from benchmarks.flash_recovery_matrix import WorkProbe, scenarios
from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    SurfaceView,
)
from lerobot.policies.flash.surface_grounding import ShapeConfig
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession


@pytest.fixture(scope="module", params=MODES)
def mug_result(request):
    with WorkProbe() as probe:
        result = run_mug(request.param)
    return result, probe.result()


def test_recorded_false_recovery_is_not_published(mug_result, request):
    result, _ = mug_result
    if result["mode"] == "current_reset":
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True, reason="Current automatic reset republishes the recorded wrong mug alignment"
            )
        )
    assert not result["available"], result["error"]
    if result["mode"] != "current_reset":
        k = result["diagnostics"]["kalman"]
        assert k["filter_reason"] == "innovation_conflict"
        assert k["innovation_squared"] > k["innovation_limit"]


def test_rejected_mug_recovery_does_not_grow_or_confirm_the_surface(mug_result):
    result, _ = mug_result
    assert result["surface_unchanged"]
    assert not result["diagnostics"]["map_events"]
    if result["mode"] != "current_reset":
        assert "growth" not in result["diagnostics"]


def test_one_frame_has_bounded_search_and_only_one_filter_correction(mug_result):
    _, work = mug_result
    assert 1 <= work["calls"]["fit_refinement"] <= 6
    assert work["max_refine_evaluations"] <= ShapeConfig().max_evaluations
    assert work["calls"]["kalman_predict"] == 1
    assert work["calls"]["kalman_correct"] == 1


@pytest.fixture(scope="module")
def rendered_case():
    @cache
    def run(name):
        return run_scene(scenarios()[name], "strict_broader_motion", grow=name == "corrupted_then_clear")

    return run


def assert_accurate(rows):
    for row in rows:
        assert row["available"], (row["index"], row["reason"])
        assert row["error"]["median_point_error_mm"] < 5, row["error"]
        assert row["error"]["rotation_error_deg"] < 1, row["error"]


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(
            "current_reset",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Current reset follows motion but exceeds the unchanged 5mm accuracy budget",
            ),
        ),
        pytest.param(
            "strict",
            marks=pytest.mark.xfail(
                strict=True, reason="Narrow prior rejects real visible motion and loses tracking"
            ),
        ),
        "strict_broader_motion",
    ],
)
def test_fully_visible_movement_does_not_freeze_or_lag(rendered_case, mode):
    result = (
        rendered_case("visible_motion")
        if mode == "strict_broader_motion"
        else run_scene(scenarios()["visible_motion"], mode)
    )
    rows = result["rows"]
    assert_accurate(rows)
    assert rows[-1]["pose"][0][3] > 0.03


def test_stationary_hidden_object_stays_still_and_uncertainty_grows(rendered_case):
    rows = rendered_case("hidden_stationary")["rows"]
    assert_accurate(rows)
    before = np.array(rows[WARMUP - 1]["covariance"])
    hidden = np.array(rows[WARMUP + HIDDEN - 1]["covariance"])
    assert np.linalg.eigvalsh(hidden - before).min() > 0
    assert len({row["map_version"] for row in rows}) == 1


def test_visible_strip_does_not_invent_motion(rendered_case):
    rows = rendered_case("partial_stationary")["rows"]
    assert all(row["measurement_count"] for row in rows[WARMUP : WARMUP + HIDDEN])
    assert_accurate(rows)


def test_supporting_motion_and_camera_movement_are_composed_once(rendered_case):
    rows = rendered_case("group_and_camera")["rows"]
    assert_accurate(rows)
    # World object moves +40mm; camera moves -20mm. Returning only either
    # component, or applying the camera change twice, fails against rendered truth.
    assert rows[WARMUP + HIDDEN - 1]["pose"][0][3] == pytest.approx(0.06, abs=0.001)
    assert len({row["map_version"] for row in rows}) == 1


def test_bad_observations_do_not_poison_surface_and_clear_frames_recover(rendered_case):
    rows = rendered_case("corrupted_then_clear")["rows"]
    before = rows[WARMUP - 1]["map_version"]
    assert all(row["map_version"] == before for row in rows[WARMUP : WARMUP + HIDDEN])
    assert all(not row["available"] for row in rows[WARMUP : WARMUP + HIDDEN])
    assert_accurate(rows[-3:])


@pytest.mark.parametrize("object_count", [1, 3])
def test_long_hidden_sequence_keeps_memory_and_work_bounded(object_count):
    from benchmarks.flash_kalman_recovery_control import scene

    first, hidden = scene(0)[0], scene(0)[WARMUP]
    config = GrowingSurfaceConfig(
        partial_surface_matching=True,
        retain_unobserved_pose=True,
        max_patches=3,
        max_pending=2,
        max_points=256,
        points_per_patch=128,
    )
    sessions = [GrowingSurfaceSession(ShapeConfig(), config) for _ in range(object_count)]

    def prepare(frame):
        return SurfaceTrackingSession(None).prepare(
            frame["rgb"], frame["depth"], frame["mask"], frame["intrinsics"], frame["index"]
        )

    initial, missing = prepare(first), prepare(hidden)
    for session in sessions:
        session.update(initial, 0.0)
        # Populate more than capacity using actual retained RGB-D measurements.
        # This exercises the storage budget independently of fitting success.
        for i in range(1, 12):
            session.store.add(
                initial["current"]["obs"]["points"],
                session.pose,
                -i,
                active=i % 2 == 0,
                view=SurfaceView.from_observation(
                    initial["current"],
                    initial["intrinsics"],
                    session.growth_shape_config,
                    retain_measurement=True,
                ),
            )

    def retained_bytes():
        return sum(
            a.nbytes
            for s in sessions
            for p in s.store.patches.values()
            for a in (p.measured, p.view.depth, p.view.mask, p.view.measured_points)
        )

    before = retained_bytes()
    covariance = [s.kalman.covariance.copy() for s in sessions]
    with WorkProbe() as probe:
        for i in range(1, 301):
            for session in sessions:
                result = session.update({**missing, "frame_id": i}, i / 30)
                assert result["camera_from_reference"] is not None
                assert len(session.store.patches) <= config.max_patches + config.max_pending
                assert len(session.store.points) <= config.max_points
                assert not session.store.events
    assert retained_bytes() == before
    assert probe.calls["kalman_predict"] == object_count * 300
    assert probe.calls["kalman_correct"] == probe.calls["fit_refinement"] == 0
    for session, previous in zip(sessions, covariance, strict=True):
        assert np.linalg.eigvalsh(session.kalman.covariance - previous).min() > 0
