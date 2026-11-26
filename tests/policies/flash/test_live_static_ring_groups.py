"""The ring and tray stay one physical group in the 9103 live replay."""

from pathlib import Path

import numpy as np

from lerobot.policies.flash.motion_groups import GroupConfig, SceneMotionGroups
from lerobot.policies.flash.scene_groups import SceneTracks
from lerobot.policies.flash.surface_observation import move

REPLAY = Path(__file__).with_name("fixtures") / "live_9103_static_ring_groups_20261116.npz"


def test_established_ring_and_tray_edge_points_never_separate_or_disappear():
    """Check every established ID, including ones hidden by later frames.

    The fixture is the actual 152-update scene-point input to the group core.
    At frame 233 the green ring and the right wall of the black tray are
    separately visible. Select established tracks from those two regions;
    later masks and object poses never enter the group update.
    """
    tracker = SceneMotionGroups(GroupConfig())
    checked = []
    carriers = set()
    hidden_predictions = 0
    with np.load(REPLAY) as saved:
        for row, frame_id in enumerate(saved["frame_ids"]):
            start, stop = saved["offsets"][row : row + 2]
            scene = SceneTracks(
                int(frame_id),
                float(saved["times"][row]),
                saved["ids"][start:stop].astype(np.int64),
                saved["xy"][start:stop],
                saved["xyz"][start:stop].astype(np.float64),
            )
            snapshot = tracker.update(scene)
            for group in snapshot.groups.values():
                if group.measured.any():
                    predicted = move(group.local_xyz[group.measured], group.base_from_group)
                    residual = np.linalg.norm(predicted - group.base_xyz[group.measured], axis=1)
                    assert residual.max() <= tracker.config.fit_threshold_m + 1e-9, (
                        f"group {group.id} labels a mismatched point measured at frame {frame_id}"
                    )
            if frame_id == 233:
                ring_candidates = set(map(int, saved["ring_landmark_ids"]))
                # Right tray wall in the recorded 848x480 image. The previous
                # "tray rim" fixture points were actually on the green ring.
                tray_candidates = {
                    int(point_id)
                    for point_id, (x, y) in zip(scene.ids, scene.xy, strict=True)
                    if 730 <= x < 780 and 120 <= y < 270
                }
                carrier = max(
                    snapshot.groups.values(), key=lambda group: len(set(group.ids) & ring_candidates)
                )
                ring = ring_candidates & set(map(int, carrier.ids))
                tray = tray_candidates & set(map(int, carrier.ids))
                assert len(ring) >= 50 and len(tray) >= 10
            if frame_id < 233:
                continue
            by_id = {int(i): group.id for group in snapshot.groups.values() for i in group.ids}
            owners = {by_id.get(i) for i in ring | tray}
            assert len(owners) == 1 and None not in owners, (
                f"established ring/tray-edge points separated or disappeared at frame {frame_id}: {owners}"
            )
            group_id = owners.pop()
            carriers.add(group_id)
            group = snapshot.groups[group_id]
            measured = dict(zip(map(int, group.ids), group.measured, strict=True))
            missing = ring - set(map(int, scene.ids))
            assert all(not measured[i] for i in missing)
            if frame_id == 276:
                assert 0 < sum(measured[i] for i in ring) < len(ring)
            if frame_id == 296:
                assert not any(measured[i] for i in ring)
            hidden_predictions += len(missing)
            checked.append(int(frame_id))
    assert checked[0] == 233 and checked[-1] == 865 and len(checked) >= 130
    assert 243 in checked and 276 in checked and 296 in checked and 806 in checked
    assert len(carriers) > 1  # The tray really separated from its old scene group.
    assert hidden_predictions > 0  # Hidden physical IDs remain in the group.


def test_held_out_ring_points_still_predict_the_recorded_positions():
    """Hide real ring measurements; score its carried points against those withheld measurements."""
    with np.load(REPLAY) as saved:
        scenes = []
        ring_each_frame = []
        for row, frame_id in enumerate(saved["frame_ids"]):
            start, stop = saved["offsets"][row : row + 2]
            scenes.append(
                SceneTracks(
                    int(frame_id),
                    float(saved["times"][row]),
                    saved["ids"][start:stop].astype(np.int64),
                    saved["xy"][start:stop],
                    saved["xyz"][start:stop].astype(np.float64),
                )
            )
            start, stop = saved["ring_offsets"][row : row + 2]
            ring_each_frame.append(set(map(int, saved["ring_ids_each_frame"][start:stop])))

    comparison_counts = []
    # Under the plunger after 271, repeated image IDs acquire corrupted 3D
    # readings (up to 98 mm away); they cannot serve as withheld position truth.
    for first_frame in (233, 530, 806):
        first_row = next(row for row, scene in enumerate(scenes) if scene.frame_id == first_frame)
        tracker = SceneMotionGroups(GroupConfig())
        for scene in scenes[: first_row + 1]:
            snapshot = tracker.update(scene)
        carrier = max(
            snapshot.groups.values(),
            key=lambda group: len(set(map(int, group.ids)) & ring_each_frame[first_row]),
        )
        established = set(map(int, carrier.ids)) & ring_each_frame[first_row]
        assert len(established) >= 16
        errors = []
        for row in range(first_row + 1, first_row + 9):
            original = scenes[row]
            keep = np.asarray([int(point_id) not in ring_each_frame[row] for point_id in original.ids])
            withheld = SceneTracks(
                original.frame_id,
                original.source_time_s,
                original.ids[keep],
                original.xy[keep],
                original.base_xyz[keep],
            )
            snapshot = tracker.update(withheld)
            predicted = {
                int(point_id): group.base_xyz[index]
                for group in snapshot.groups.values()
                for index, point_id in enumerate(group.ids)
            }
            assert established <= predicted.keys(), f"ring points disappeared at frame {original.frame_id}"
            measured = dict(zip(map(int, original.ids), original.base_xyz, strict=True))
            for point_id in established & ring_each_frame[row] & measured.keys():
                errors.append(np.linalg.norm(predicted[point_id] - measured[point_id]))
        assert len(errors) >= 100
        assert np.percentile(errors, 95) < 0.010
        comparison_counts.append(len(errors))
    assert sum(comparison_counts) >= 1000
