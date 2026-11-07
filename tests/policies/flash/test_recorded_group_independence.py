"""Actual saved ring RGB-D -> shared LK -> groups -> independent surface check."""

from pathlib import Path

import numpy as np

from lerobot.policies.flash.motion_groups import GroupConfig, SceneMotionGroups
from lerobot.policies.flash.object_group_association import ObjectGroupAssociation
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.shared_image_tracking import SharedFrameTracker, SharedMatchingConfig
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


def test_recorded_ring_geometry_rejection_cannot_erase_measured_scene_support():
    fixture = np.load(Path(__file__).with_name("fixtures") / "ring_group_rejection_20261107.npz")
    shared = SharedFrameTracker(SharedMatchingConfig(backend="lk"))
    # This two-frame fixture tests the rejection boundary, not merge duration
    # (which is separately tested across source rates and measured trajectories).
    groups = SceneMotionGroups(GroupConfig(support_confirmation_s=0))
    source = ObjectGroupAssociation(groups.config)
    prepare = SurfaceTrackingSession(None, shared_matching=shared)
    for index in range(2):
        time = float(fixture["times"][index])
        frame = RGBDFrame(
            index,
            fixture[f"rgb{index}"],
            fixture[f"depth{index}"],
            fixture["intrinsics"],
            1.0,
            np.eye(4),
            time,
            time,
            "recorded",
        )
        mask = fixture[f"mask{index}"]
        scene = shared.update(frame, {"ring": mask})
        groups.update(scene)
        prepared = prepare.prepare(frame.rgb, frame.depth, mask, frame.intrinsics, index)
        if index == 0:
            source.record_observation(
                frame,
                scene,
                mask,
                {
                    "status": "initialized",
                    "camera_from_reference": fixture["pose0"],
                    "diagnostics": {"reference_frame": 0},
                },
                fixture["reference"],
                object_usable=prepared["current"]["usable"],
            )
    hint, details = source.query(
        groups.frame, frame, scene, mask, 0, object_usable=prepared["current"]["usable"]
    )
    assert hint is not None
    assert details["object_inliers"] >= 12 and details["environment_inliers"] >= 100
    before = {gid: (g.ids.copy(), g.base_xyz.copy()) for gid, g in groups.frame.groups.items()}
    geometry = WholeSurfaceFit(fixture["reference"], prepared["current"], frame.intrinsics, ShapeConfig())
    check = check_surface_contradiction(
        geometry, hint.camera_from_reference, PoseUncertaintyConfig(enabled=True)
    )
    assert not check["accepted"], "Must actually exercise a surface rejection"
    source.record_observation(
        frame,
        scene,
        mask,
        {
            "status": "lost",
            "camera_from_reference": None,
            "diagnostics": {"reference_frame": 0, "prediction_geometry": check},
        },
        fixture["reference"],
    )
    after, after_details = source.query(
        groups.frame, frame, scene, mask, 0, object_usable=prepared["current"]["usable"]
    )
    assert after is not None and after_details["environment_inliers"] == details["environment_inliers"]
    for gid, (ids, xyz) in before.items():
        np.testing.assert_array_equal(groups.frame.groups[gid].ids, ids)
        np.testing.assert_array_equal(groups.frame.groups[gid].base_xyz, xyz)
    np.testing.assert_allclose(after.camera_from_reference, hint.camera_from_reference)
    assert shared.calls == 1  # No second point tracker in groups or object association.
