"""A partial direct view cannot silently replace a verified group pose."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lerobot.policies.flash.causal_service import CausalPoseService, _Anchor
from lerobot.policies.flash.perception import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.pose_disagreement import PoseComparisonVerdict
from lerobot.policies.flash.scene_groups import SceneTracks
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def test_real_partial_dowel_view_keeps_ambiguous_direct_pose_out_of_group_anchor():
    fixture = Path(__file__).parent / "fixtures" / "dowel_group_direct_ambiguous_20261020.npz"
    with np.load(fixture) as recorded:
        depth = recorded["depth"]
        mask = recorded["mask"].astype(bool)
        intrinsics = recorded["intrinsics"]
        base_from_camera = recorded["base_from_camera"]
        points = recorded["points"]
        direct_pose = recorded["direct_pose"]
        group_pose = recorded["group_pose"]
    frame = RGBDFrame(
        170,
        np.zeros((*depth.shape, 3), dtype=np.uint8),
        depth,
        intrinsics,
        0.001,
        base_from_camera,
        10.0,
        10.0,
        "camera-v1",
    )
    scene = SceneTracks(170, 10.0, np.array([], dtype=np.int64), np.empty((0, 2)), np.empty((0, 3)))
    direct = PoseEstimate(
        "dowel", direct_pose, 10.0, 10.0, Evidence.OBSERVED, "dowel-v1", (True,) * 3, (True,) * 3
    )
    group = PoseEstimate(
        "dowel",
        group_pose,
        10.01,
        10.01,
        Evidence.PREDICTED,
        "dowel-v1",
        (True,) * 3,
        (True,) * 3,
        estimator="environment_group_motion",
        last_direct_observation_s=9.0,
    )
    store = LatestPoseStore()
    store.publish((group,))
    comparisons = []
    service = CausalPoseService(
        store,
        fit_threshold_m=0.005,
        object_centers_by_role={"dowel": points.mean(axis=0)},
        object_points_by_role={"dowel": points},
        on_comparison=lambda role, result: comparisons.append((role, result)),
    )
    service.history.append(frame)
    service.scene_history.append(scene)
    service.relationships["dowel"] = SimpleNamespace(
        anchor_time_s=9.0,
        predict=lambda _scene, **_kwargs: group_pose,
    )

    service._anchor(_Anchor(170, (direct,), {"dowel": mask}, 10.0, "camera-v1"))

    assert store.snapshot()[1]["dowel"] is group
    assert service.anchor_outcomes_by_role["dowel"]["direct_group_unresolved"] == 1
    assert comparisons[0][0] == "dowel"
    assert comparisons[0][1].verdict == PoseComparisonVerdict.AMBIGUOUS
    assert comparisons[0][1].object_center_difference_m > 0.01
