"""Small geometry checks for the replay-only SDF input experiment."""

from __future__ import annotations

import numpy as np
from sdf_boundary_occlusion_filters import (
    inside_mask_margin,
    old_surface_depth_image,
    old_voxels_in_front_of_new_depth,
    reject_nearer_than_old_surface,
)


def test_boundary_margin_removes_only_the_requested_inner_rim():
    mask = np.zeros((7, 7), dtype=bool)
    mask[1:6, 1:6] = True
    assert inside_mask_margin(mask, 0).sum() == 25
    assert inside_mask_margin(mask, 1).sum() == 9
    assert inside_mask_margin(mask, 2).sum() == 1


def test_nearer_new_depth_requires_projected_old_surface_and_a_real_gap():
    class Volume:
        gpu_mode = False
        _vol_origin = np.array([0.0, 0.0, 1.0])
        _voxel_size = 0.1
        _tsdf_vol_cpu = np.ones((1, 1, 1), dtype=np.float32)
        _weight_vol_cpu = np.ones((1, 1, 1), dtype=np.float32)

    Volume._tsdf_vol_cpu[0, 0, 0] = 0
    intrinsics = np.array([[10, 0, 2], [0, 10, 2], [0, 0, 1]], dtype=float)
    old = old_surface_depth_image(Volume(), np.eye(4), intrinsics, (5, 5), 0.15)
    assert old[2, 2] == 1
    assert np.isinf(old[2, 3])

    new = np.zeros((5, 5), dtype=np.float32)
    new[2, 3] = 0.95
    rejected, comparable = reject_nearer_than_old_surface(new, old, radius_px=1, gap_m=0.02)
    assert comparable[2, 3] and rejected[2, 3]
    rejected, comparable = reject_nearer_than_old_surface(new, old, radius_px=0, gap_m=0.02)
    assert not comparable[2, 3] and not rejected[2, 3]
    rejected, _ = reject_nearer_than_old_surface(new, old, radius_px=1, gap_m=0.06)
    assert not rejected[2, 3]

    new[2, 3] = 1.05
    selected, surface_count = old_voxels_in_front_of_new_depth(
        Volume(),
        new,
        np.eye(4),
        intrinsics,
        radius_px=1,
        gap_m=0.02,
        surface_tsdf_max=0.15,
    )
    assert selected.tolist() == [[0, 0, 0]] and surface_count == 1
    selected, _ = old_voxels_in_front_of_new_depth(
        Volume(),
        new,
        np.eye(4),
        intrinsics,
        radius_px=1,
        gap_m=0.06,
        surface_tsdf_max=0.15,
    )
    assert len(selected) == 0


def test_deeper_measurement_does_not_delete_the_far_side_of_the_old_body():
    class Volume:
        gpu_mode = False
        _vol_origin = np.array([0.0, 0.0, 1.0])
        _voxel_size = 0.2
        _tsdf_vol_cpu = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        _weight_vol_cpu = np.ones((1, 1, 3), dtype=np.float32)

    intrinsics = np.array([[10, 0, 2], [0, 10, 2], [0, 0, 1]], dtype=float)
    new = np.zeros((5, 5), dtype=np.float32)
    new[2, 2] = 1.5
    selected, _ = old_voxels_in_front_of_new_depth(
        Volume(),
        new,
        np.eye(4),
        intrinsics,
        radius_px=0,
        gap_m=0.02,
        surface_tsdf_max=0.15,
    )
    assert selected.tolist() == [[0, 0, 0]]
