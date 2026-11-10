"""Regression: a previously published cube occupies measured air.

Saved clip saved-3b9b9f3d70a74fbda3d65772f3f32b9c, frame 3011, after replaying
from initialization with small_gicp + LK and missing-motion rotation noise 10.
The fixture stores the FINAL post-Kalman pose, its 242-point pre-growth map,
the actual accepted 3004 reference pose, and both frames' recorded selected
Fast-FoundationStereo depths, RGB and SAM masks. Crop x=330:530, y=380:560;
intrinsics are shifted accordingly. No mixed-backend source alignment.

The recorded acceptance is not ground truth. Both implementations of the
geometry check must reject the independent background contradictions here.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.surface_acceleration import compiled_kernels
from lerobot.policies.flash.surface_grounding import SurfaceFitView, WholeSurfaceFit, project
from lerobot.policies.flash.surface_observation import move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


@pytest.fixture(scope="module")
def published_cube():
    with np.load(Path(__file__).parent / "fixtures/cube_published_geometry_20261110.npz") as f:
        data = {key: f[key] for key in f.files}
    settings = json.loads(str(data["settings_json"]))
    vision = VisionPipelineConfig.from_dict(settings["config"])

    def observation(prefix, frame):
        builder = SurfaceTrackingSession(
            None,
            observation_config=vision.observation,
            shape_config=vision.shape,
            motion_config=vision.motion,
        )
        return builder.prepare(
            data[prefix + "rgb"],
            data[prefix + "depth"] * float(data[prefix + "depth_m_per_unit"]),
            data[prefix + "mask"],
            data["intrinsics"],
            frame,
        )["current"]

    return data, settings, vision, observation("", 3011), observation("source_", 3004)


def geometry(fixture, kernels=None):
    data, _, vision, current, source = fixture
    return WholeSurfaceFit(
        data["surface"],
        current,
        data["intrinsics"],
        vision.shape,
        reference_views=(SurfaceFitView(source, data["intrinsics"], data["source_pose"]),),
        fit_kernels=kernels,
    )


def test_recorded_pose_has_reliable_background_contradictions(published_cube):
    data, _, _, _, _ = published_cube
    fit = geometry(published_cube)
    points = move(data["surface"], data["pose"])
    uv = np.rint(project(points, data["intrinsics"])).astype(int)
    # This map is already cleaned and conservatively eroded around boundaries.
    # Several projected samples are well outside the mask, over 20 mm in front
    # of valid background readings. They are neither holes nor nearer occluders.
    near = fit.scene_nearest_depth[uv[:, 1], uv[:, 0]]
    outside = fit.outside_mask[uv[:, 1], uv[:, 0]]
    clear_air = (near > 0) & (near - points[:, 2] > 0.020) & (outside > 8)
    assert np.count_nonzero(clear_air) >= 3


@pytest.mark.parametrize("compiled", [False, True])
def test_final_published_cube_must_fail_geometry(published_cube, compiled):
    data, settings, _, _, _ = published_cube
    kernels = compiled_kernels() if compiled else None
    if compiled and kernels is None:
        pytest.skip("optional compiled kernels unavailable")
    fit = geometry(published_cube, kernels)
    config = settings["surface_experiment"]
    verdict = fit.check_pose(
        data["pose"],
        max_outlier_fraction=config["uncertainty"]["max_visible_outlier_fraction"],
        agreement_m=config["agreement_m"],
        min_overlap=config["min_overlap"],
    )
    assert not verdict["accepted"], verdict
