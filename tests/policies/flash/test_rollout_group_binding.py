"""The recorded first ring drop crosses real scene groups and object binding.

Fixture: main b567 rollout, reference plus 28 processed frames through 251.
Contains production image tracks, usable ring IDs, and saved accepted surface
publications. It deliberately holds surface fitting fixed to isolate binding.
The fixture metadata records the source run, frame times, and group settings.
"""

import json
from pathlib import Path

import numpy as np

from lerobot.policies.flash.motion_groups import GroupConfig, SceneMotionGroups
from lerobot.policies.flash.object_group_association import ObjectGroupAssociation
from lerobot.policies.flash.observations import RGBDFrame
from lerobot.policies.flash.scene_groups import SceneTracks


def test_recorded_first_ring_drop_retains_supported_group_for_never_joined_point():
    path = Path(__file__).with_name("fixtures") / "rollout_b567_group_binding.npz"
    with np.load(path) as data:
        meta = json.loads(str(data["metadata"]))
        groups = SceneMotionGroups(GroupConfig.from_dict(meta["config"]))
        binding = ObjectGroupAssociation(groups.config, min_association_points=meta["min_association_points"])
        checked = False
        for n, item in enumerate(meta["frames"]):
            fid, time = item["frame"], item["time"]
            scene = SceneTracks(fid, time, data[f"ids_{n}"], data[f"xy_{n}"], data[f"xyz_{n}"])
            snapshot = groups.update(scene)
            h, w = item["shape"]
            frame = RGBDFrame(
                fid,
                np.zeros((h, w, 3), np.uint8),
                np.ones((h, w), np.float32),
                np.eye(3),
                1.0,
                data[f"camera_{n}"],
                time,
                time,
                "saved-rollout",
            )
            mask = np.zeros((h, w), bool)
            xy = scene.xy[np.isin(scene.ids, data[f"own_{n}"])].astype(int)
            mask[xy[:, 1], xy[:, 0]] = True
            if n:
                hint, details = binding.query(snapshot, frame, scene, mask, meta["reference_frame"])
                if fid == 246:
                    assert set(data[f"own_{n}"]) & binding.anchors.keys() == {2633}
                    assert all(2633 not in g.ids for g in snapshot.groups.values())
                    group = snapshot.groups[0]
                    assert group.confirmed and group.supported
                    # Group refinement may classify a few marginal readings
                    # as predicted, but the established carrier must retain
                    # broad measured support and the object's hidden IDs.
                    assert group.measured.sum() >= 500
                    remembered_members = set(binding.anchors) & {
                        int(i) for g in snapshot.groups.values() for i in g.ids
                    }
                    assert len(remembered_members) >= 38
                    assert hint is not None  # The old code loses the ring here.
                    assert details["group_id"] == 0
                    assert details["association_check"] == {
                        "group_id": 0,
                        "persistent_members": len(remembered_members),
                        "measured_members": 0,
                        "predicted_members": len(remembered_members),
                        "unassigned_points": 1,
                    }
                    assert "association_contradiction" not in details
                    checked = True
            binding.record_observation(
                frame,
                scene,
                mask,
                {
                    "status": item["status"],
                    "camera_from_reference": item["pose"],
                    "diagnostics": {"reference_frame": item["reference_frame"]},
                },
                data["reference"],
            )
        assert checked
