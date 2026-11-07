"""Saved and live group drawings keep links by scene point identity."""

from types import SimpleNamespace

import numpy as np

from lerobot.gui.flash_group_overlay import SceneGroupOverlay
from lerobot.policies.flash.surface_frame_session import scene_group_record


def test_scene_group_links_persist_through_reordering_and_new_points():
    overlay = SceneGroupOverlay()
    base = np.array([(x, y) for y in range(5) for x in range(6)], float)
    extra = np.array([(x + 0.5, y + 0.5) for y in range(4) for x in range(5)], float)

    def record(ids, locations):
        return {
            "id": 7,
            "ids": ids,
            "pixels": (locations * 14 + 100).tolist(),
            "camera_xyz": np.column_stack((locations * 0.012, np.full(len(ids), 0.5))).tolist(),
        }

    original = record(list(range(30)), base)
    for _ in range(12):
        drawn = overlay.update([original])[0]
    before = {tuple(sorted((original["ids"][a], original["ids"][b]))) for a, b in drawn["links"]}
    assert len(before) >= 10

    order = list(range(29, -1, -1))
    reordered = record(order, base[order])
    drawn = overlay.update([reordered])[0]
    after = {tuple(sorted((reordered["ids"][a], reordered["ids"][b]))) for a, b in drawn["links"]}
    assert len(before & after) >= 0.8 * len(before)

    extended = record(order + list(range(30, 50)), np.vstack((base[order], extra)))
    drawn = overlay.update([extended])[0]
    after = {tuple(sorted((extended["ids"][a], extended["ids"][b]))) for a, b in drawn["links"]}
    assert len(before & after) >= 0.7 * len(before)


def test_scene_group_links_never_connect_different_groups():
    overlay = SceneGroupOverlay()
    a = {
        "id": 1,
        "ids": [1, 2, 3, 4],
        "pixels": [[10, 10], [20, 10], [10, 20], [20, 20]],
        "camera_xyz": [[0, 0, 1], [0.01, 0, 1], [0, 0.01, 1], [0.01, 0.01, 1]],
    }
    b = {
        "id": 2,
        "ids": [5, 6, 7, 8],
        "pixels": [[30, 10], [40, 10], [30, 20], [40, 20]],
        "camera_xyz": [[0.02, 0, 1], [0.03, 0, 1], [0.02, 0.01, 1], [0.03, 0.01, 1]],
    }
    for _ in range(8):
        result = overlay.update([a, b])
    assert any(group["links"] for group in result)
    assert all(
        0 <= index < len(group["ids"]) for group in result for pair in group["links"] for index in pair
    )


def test_large_stationary_group_draws_local_geometry_not_scene_spanning_lines():
    rng = np.random.default_rng(5)
    xyz = np.column_stack((rng.uniform(-0.25, 0.25, 700), rng.uniform(-0.15, 0.15, 700), np.full(700, 0.5)))
    pixels = np.column_stack((600 * xyz[:, 0] / xyz[:, 2] + 320, 600 * xyz[:, 1] / xyz[:, 2] + 240))
    ids = np.arange(len(xyz))
    group = SimpleNamespace(
        id=0,
        ids=ids,
        base_xyz=xyz,
        measured=np.ones(len(ids), bool),
        confirmed=True,
        supported=True,
        residual_m=0.0,
    )
    snapshot = SimpleNamespace(groups={0: group})
    scene = SimpleNamespace(ids=ids, xy=pixels, base_xyz=xyz)
    frame = SimpleNamespace(
        base_from_camera=np.eye(4), intrinsics=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], float)
    )
    record = scene_group_record(snapshot, scene, frame)[0]
    assert len(record["ids"]) == 700
    overlay = SceneGroupOverlay()
    for _ in range(12):
        displayed = overlay.update([record], frame_size=(640, 480))[0]
    lengths = [np.linalg.norm(xyz[record["ids"][a]] - xyz[record["ids"][b]]) for a, b in displayed["links"]]
    pixel_lengths = [
        np.linalg.norm(np.asarray(record["pixels"][a]) - record["pixels"][b]) for a, b in displayed["links"]
    ]
    assert len(lengths) > 100
    assert max(lengths) < 0.1  # No line should span a large fraction of this 0.5 m tray.
    assert max(pixel_lengths) <= 48  # The 480 px image's local-link display bound.
