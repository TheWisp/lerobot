"""Saved live clip: a static controller vetoed by one older view's rim depth.

Frames 430 -> 436 of saved clip ``saved-d6d3d3b38e7944e49c5b80be224f8cb4``
(green game controller, hand approaching the right edge). Neither object nor
camera moved: the group prediction and the saved 430 pose place the surface
centre 0.02 mm apart. Fitting against the current mask and depth alone passes.
Adding the retained 430 depth view rejected the prediction live, because 22 of
2754 current points, all inside both masks and 3-7 px from the silhouette, met
frame-430 depth that read 4-10 mm farther than frame 436 and the saved map.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.surface_grounding import ShapeConfig, SurfaceFitView, WholeSurfaceFit
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_config import VisionPipelineConfig

FIXTURE = Path(__file__).parent / "fixtures/green_controller_retained_view_20261107.npz"


@pytest.fixture(scope="module")
def clip():
    with np.load(FIXTURE) as saved:
        arrays = {key: saved[key] for key in saved.files}
    settings = json.loads(str(arrays["settings_json"]))
    vision = VisionPipelineConfig.from_dict(settings["config"])
    surface = settings["surface_experiment"]

    def observation(name, frame_id):
        session = SurfaceTrackingSession(
            None,
            observation_config=vision.observation,
            shape_config=vision.shape,
            motion_config=vision.motion,
        )
        return session.prepare(
            arrays[f"{name}_rgb"],
            arrays[f"{name}_depth"],
            arrays[f"{name}_mask"],
            arrays["intrinsics"],
            frame_id,
        )["current"]

    retained = observation("retained", 430)
    current = observation("lost", 436)
    view = SurfaceFitView(retained, arrays["intrinsics"], arrays["retained_pose"])
    one = WholeSurfaceFit(arrays["surface_used"], current, arrays["intrinsics"], vision.shape)
    two = WholeSurfaceFit(
        arrays["surface_used"], current, arrays["intrinsics"], vision.shape, reference_views=(view,)
    )

    def check(fit, pose):
        return fit.check_pose(
            np.asarray(pose),
            max_outlier_fraction=surface["uncertainty"]["max_visible_outlier_fraction"],
            agreement_m=surface["agreement_m"],
            min_overlap=surface["min_overlap"],
        )

    return {"arrays": arrays, "one": one, "two": two, "check": check}


def shifted(pose, delta_m):
    moved = np.array(pose, float)
    moved[:3, 3] += delta_m
    return moved


def test_static_prediction_fits_current_depth_alone(clip):
    verdict = clip["check"](clip["one"], clip["arrays"]["prediction_pose"])
    assert verdict["accepted"]
    assert verdict["observed_overlap"] > 0.98
    assert verdict["surface_median_mm"] < 2.0
    assert not verdict["scene_depth_coherent_protrusion"]


def test_static_prediction_passes_the_retained_view_check(clip):
    verdict = clip["check"](clip["two"], clip["arrays"]["prediction_pose"])
    views = {v["direction"]: v for v in verdict["scene_depth_views"]}
    assert not views["map_to_current"]["coherent_protrusion"]
    assert not views["current_to_retained"]["coherent_protrusion"]
    assert verdict["accepted"]
    assert verdict["observed_overlap"] > 0.98


@pytest.mark.parametrize(
    "delta_mm",
    [(0, 0, -10), (0, 0, 10), (8, 0, 0), (0, 8, 0)],
    ids=["toward_camera", "away", "lateral_x", "lateral_y"],
)
def test_protruding_poses_still_rejected_with_the_retained_view(clip, delta_mm):
    pose = shifted(clip["arrays"]["prediction_pose"], np.asarray(delta_mm) / 1000)
    verdict = clip["check"](clip["two"], pose)
    assert not verdict["accepted"]
    assert verdict["scene_depth_coherent_protrusion"]


def test_recent_measured_surface_can_refute_its_own_noisy_depth_before_map_confirmation():
    # These three measured points came from the accepted view but have not
    # entered the confirmed accumulated map. A stale depth reading in that
    # same view cannot independently prove that the points occupy empty space.
    k = np.array([[100.0, 0, 20.0], [0, 100.0, 20.0], [0, 0, 1.0]])
    mask = np.zeros((48, 48), bool)
    mask[12:30, 12:30] = True
    uv = np.array([[16.0, 20.0], [20.0, 20.0], [24.0, 20.0]])
    points = np.column_stack(((uv[:, 0] - 20) * 0.005, np.zeros(3), np.full(3, 0.5)))
    current_depth = np.full(mask.shape, 0.5, np.float32)
    retained_depth = np.full(mask.shape, 0.52, np.float32)
    current = {
        "obs": {"points": points},
        "depth": current_depth,
        "scene_depth": current_depth,
        "mask": mask,
        "frame_id": 2,
    }
    retained = {
        "obs": {"points": points.copy()},
        "depth": retained_depth,
        "scene_depth": retained_depth,
        "mask": mask,
        "frame_id": 1,
    }
    # A different patch occupies the map. The just-accepted points are still
    # pending its strict growth/confirmation rule.
    map_points = points + [0.1, 0, 0]
    fit = WholeSurfaceFit(
        map_points, current, k, ShapeConfig(), reference_views=(SurfaceFitView(retained, k, np.eye(4)),)
    )
    good = fit.scene_depth_check(np.eye(4))
    assert not good["scene_depth_views"][1]["coherent_protrusion"]
    wrong = np.eye(4)
    wrong[0, 3] = 0.03
    assert fit.scene_depth_check(wrong)["scene_depth_coherent_protrusion"]
