"""Replay exact P2P SDF keyframes with a provisional temporal-support filter.

The filter uses only earlier masked-depth keyframes transformed by the saved
Point2Pose poses. The first view is fused unchanged. This is an offline map
ablation, not a tracker rerun or a proposed runtime threshold.

Run with PYTHONPATH pointing at the pinned Point2Pose checkout.
"""

from collections import deque
from pathlib import Path

import numpy as np
from analyze_p2p_sdf_keyframes import masked_camera_points, near_surface_counts
from point2pose.modules.reconstruction.sdf_builder import TSDFVolume

HERE = Path(__file__).parent


def main() -> None:
    snapshots = np.load(HERE / "p2p_dowel_sdf_release_snapshots.npz")
    keyframes = np.load(HERE / "p2p_dowel_sdf_keyframe_depths.npz")
    reference = np.load(HERE / "pose_traces.npz")
    center = reference["dowel_reference_center_camera_m"]
    axis = reference["dowel_reference_shaft_camera"]
    origin = snapshots["origin"]
    voxel_size = float(snapshots["voxel_size"])

    def initial_volume() -> TSDFVolume:
        volume = TSDFVolume(snapshots["bounds"].copy(), voxel_size, use_gpu=False)
        volume._tsdf_vol_cpu[:] = snapshots["tsdf_100"]
        volume._weight_vol_cpu[:] = snapshots["weight_100"]
        return volume

    full = initial_volume()
    filtered = initial_volume()
    voxel_m = 0.010
    prior_frames = deque(maxlen=3)
    first_depth = keyframes["depth_masked_m"][0]
    first_k = keyframes["intrinsics"][0]
    first_pose = keyframes["poses"][0]
    first_points = masked_camera_points(first_depth, first_k)[0]
    first_object_points = (first_points - first_pose[:3, 3]) @ first_pose[:3, :3]
    prior_frames.append({tuple(v) for v in np.floor(first_object_points / voxel_m).astype(np.int32)})

    print("source,valid_depth,retained,full_surface,full_far40,filtered_surface,filtered_far40")
    for index in range(1, len(keyframes["source_frames"])):
        depth = keyframes["depth_masked_m"][index]
        k = keyframes["intrinsics"][index]
        pose = keyframes["poses"][index]
        points, y, x = masked_camera_points(depth, k)
        object_points = (points - pose[:3, 3]) @ pose[:3, :3]
        voxels = np.floor(object_points / voxel_m).astype(np.int32)
        prior = set().union(*prior_frames)
        retained = np.array([tuple(voxel) in prior for voxel in voxels], dtype=bool)
        filtered_depth = depth.copy()
        filtered_depth[y[~retained], x[~retained]] = 0
        color_unused = np.zeros((*depth.shape, 3), dtype=np.uint8)
        for volume, selected_depth in ((full, depth), (filtered, filtered_depth)):
            volume.integrate(color_unused, selected_depth, k, np.linalg.inv(pose), obs_weight=1)
        full_counts = near_surface_counts(full, origin, voxel_size, center, axis)
        filtered_counts = near_surface_counts(filtered, origin, voxel_size, center, axis)
        print(
            f"{keyframes['source_frames'][index]},{len(points)},{int(retained.sum())},"
            f"{full_counts[0]},{full_counts[1]},{filtered_counts[0]},{filtered_counts[1]}"
        )
        if keyframes["integration_frames"][index] == 505:
            assert np.array_equal(full._weight_vol_cpu, snapshots["weight_540"])
            assert np.allclose(full._tsdf_vol_cpu, snapshots["tsdf_540"], rtol=0, atol=1e-5)
        prior_frames.append({tuple(voxel) for voxel in voxels})


if __name__ == "__main__":
    main()
