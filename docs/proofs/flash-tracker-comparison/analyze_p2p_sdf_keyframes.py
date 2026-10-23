"""Explain dowel TSDF contamination using the exact masked keyframe depths.

Run with the pinned Point2Pose checkout on ``PYTHONPATH``. The optional trim is
an oracle ablation using the first-view dowel shaft; it is not a runtime rule.
"""

from pathlib import Path

import numpy as np
from point2pose.modules.reconstruction.sdf_builder import TSDFVolume

HERE = Path(__file__).parent
SHAFT_OUTLIER_M = 0.04
NEAR_SURFACE_TSDF = 0.15


def shaft_radius(points: np.ndarray, center: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.cross(points - center, axis), axis=1)


def masked_camera_points(depth: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.nonzero(depth > 0)
    z = depth[y, x]
    points = np.stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z), axis=1)
    return points, y, x


def near_surface_counts(
    volume: TSDFVolume, origin: np.ndarray, voxel_size: float, center: np.ndarray, axis: np.ndarray
) -> tuple[int, int, int]:
    surface = (volume._weight_vol_cpu > 0) & (np.abs(volume._tsdf_vol_cpu) < NEAR_SURFACE_TSDF)
    points = origin + np.column_stack(np.nonzero(surface)) * voxel_size
    radius = shaft_radius(points, center, axis)
    return len(points), int(np.count_nonzero(radius > 0.04)), int(np.count_nonzero(radius > 0.08))


def main() -> None:
    snapshots = np.load(HERE / "p2p_dowel_sdf_release_snapshots.npz")
    keyframes = np.load(HERE / "p2p_dowel_sdf_keyframe_depths.npz")
    reference = np.load(HERE / "pose_traces.npz")
    center = reference["dowel_reference_center_camera_m"]
    axis = reference["dowel_reference_shaft_camera"]
    origin = snapshots["origin"]
    voxel_size = float(snapshots["voxel_size"])

    def volume_from_first_keyframe() -> TSDFVolume:
        volume = TSDFVolume(snapshots["bounds"].copy(), voxel_size, use_gpu=False)
        volume._tsdf_vol_cpu[:] = snapshots["tsdf_100"]
        volume._weight_vol_cpu[:] = snapshots["weight_100"]
        return volume

    full = volume_from_first_keyframe()
    without_outlier_pixels = volume_from_first_keyframe()
    print("source,integrated,valid_depth,far_40mm,far_80mm,full_sdf_far_40mm,trimmed_sdf_far_40mm")
    for index in range(1, len(keyframes["source_frames"])):
        depth = keyframes["depth_masked_m"][index]
        k = keyframes["intrinsics"][index]
        pose = keyframes["poses"][index]
        points, y, x = masked_camera_points(depth, k)
        in_object_frame = (points - pose[:3, 3]) @ pose[:3, :3]
        radii = shaft_radius(in_object_frame, center, axis)
        retained_depth = depth.copy()
        retained_depth[y[radii > SHAFT_OUTLIER_M], x[radii > SHAFT_OUTLIER_M]] = 0
        color_unused = np.zeros((*depth.shape, 3), dtype=np.uint8)
        for volume, selected_depth in ((full, depth), (without_outlier_pixels, retained_depth)):
            volume.integrate(color_unused, selected_depth, k, np.linalg.inv(pose), obs_weight=1)
        print(
            f"{keyframes['source_frames'][index]},{keyframes['integration_frames'][index]},"
            f"{len(points)},{int(np.count_nonzero(radii > 0.04))},"
            f"{int(np.count_nonzero(radii > 0.08))},"
            f"{near_surface_counts(full, origin, voxel_size, center, axis)[1]},"
            f"{near_surface_counts(without_outlier_pixels, origin, voxel_size, center, axis)[1]}"
        )
        if keyframes["integration_frames"][index] == 505:
            expected = snapshots["tsdf_540"]
            expected_weight = snapshots["weight_540"]
            assert np.array_equal(full._weight_vol_cpu, expected_weight)
            assert np.allclose(full._tsdf_vol_cpu, expected, rtol=0, atol=1e-5)
            assert near_surface_counts(full, origin, voxel_size, center, axis)[1] == 85
            assert near_surface_counts(without_outlier_pixels, origin, voxel_size, center, axis)[1] == 0


if __name__ == "__main__":
    main()
