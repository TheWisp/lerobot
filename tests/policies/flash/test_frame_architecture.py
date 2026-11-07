"""Exercise the diagram through actual RGB-D fitting, filtering and map growth."""

import numpy as np

from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    PoseHint,
)
from lerobot.policies.flash.surface_grounding import WholeSurfaceFit
from lerobot.policies.flash.surface_kalman import MeasuredGroupMotion
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_session_config import defaults
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def test_confirmed_growth_is_used_after_occlusion_without_changing_object_coordinates(monkeypatch):
    settings = defaults()
    shape = VisionPipelineConfig.from_dict(settings["config"]).shape
    track = GrowingSurfaceSession(shape, GrowingSurfaceConfig(**settings["surface_experiment"]))
    intrinsics = np.array([[800.0, 0, 70], [0, 800.0, 50], [0, 0, 1]])

    def observation(index, *, occluded=False, shift_px=0):
        mask = np.zeros((100, 140), bool)
        mask[25:75, 35 + shift_px : (90 if index == 0 else 100) + shift_px] = True
        depth = np.where(mask, 0.8, 1.2).astype(np.float32)
        if index == 0:
            # A nearer object hides a strip which becomes visible in frame 1.
            depth[25:75, 90:100] = 0.6
        if occluded:
            mask[:] = False
            depth[:] = 0.6
        # No image matches are provided: this checks the geometry/map/filter
        # path independently of a neural model or a textureless LK estimate.
        return SurfaceTrackingSession(None, shape_config=shape).prepare(
            np.zeros((*mask.shape, 3), np.uint8), depth, mask, intrinsics, index
        )

    fitted_surfaces = []
    solve = WholeSurfaceFit._solve

    def record_solve(fitter, *args, **kwargs):
        fitted_surfaces.append(fitter.reference.copy())
        return solve(fitter, *args, **kwargs)

    monkeypatch.setattr(WholeSurfaceFit, "_solve", record_solve)
    assert track.update(observation(0), 0.0)["status"] == "initialized"
    initial = track.store.points.copy()
    assert not fitted_surfaces

    first = track.update(observation(1), 0.1)
    assert first["status"] == "observed_candidate"
    assert first["diagnostics"]["growth"]["admitted_points"] > 0
    np.testing.assert_array_equal(track.store.points, initial)  # Addition is still pending.

    confirmed = track.update(observation(2), 0.2)
    assert confirmed["status"] == "observed_candidate"
    grown = track.store.points.copy()
    assert len(grown) > len(initial)
    for fitted in fitted_surfaces:
        np.testing.assert_array_equal(fitted, initial)  # Confirmation happens after fitting.

    support = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.06, 0.06) for z in (0.6, 0.7)])
    pose = np.eye(4)
    pose[0, 3] = 0.01
    hint = PoseHint(
        pose,
        "measured_group",
        0.3,
        0.2,
        0,
        motion=MeasuredGroupMotion(support, support + [0.01, 0, 0], 0.2, 0.3, "measured_group"),
    )
    covariance = track.kalman.covariance.copy()
    hidden = track.update(observation(3, occluded=True), 0.3, hint=hint)
    assert hidden["status"] == "predicted"
    np.testing.assert_allclose(hidden["camera_from_reference"], pose, atol=1e-5)
    assert np.trace(track.kalman.covariance) > np.trace(covariance)
    np.testing.assert_array_equal(track.store.points, grown)
    assert track.accepted_frame == 2
    assert len(fitted_surfaces) == 2  # No solve without a usable object measurement.

    # 10 pixels at 0.8 m and fx=800 is exactly the measured 10 mm group motion.
    returned = track.update(observation(4, shift_px=10), 0.4)
    assert returned["status"] == "observed_candidate"
    assert len(fitted_surfaces) == 3  # One solve on each visible frame, no retry loop.
    np.testing.assert_array_equal(fitted_surfaces[-1], grown)
    assert returned["diagnostics"]["surface_frames_used"] == [0, 1]
    assert returned["diagnostics"]["accepted_frame_before"] == 2
    assert track.accepted_frame == 4
    assert track.reference_frame == 0
    error = move(grown, np.asarray(returned["camera_from_reference"])) - move(grown, pose)
    assert np.max(np.linalg.norm(error, axis=1)) < 0.003
