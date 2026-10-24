"""Preflight the causal view-confirmation ablation on saved dowel SDF inputs.

The 20 mm shaft radius is used only to score retained pixels. Filtering uses
the prior SDF, earlier accepted keyframes, and the SDF voxel size; it never
uses the shaft reference. Run with the pinned Point2Pose checkout on PYTHONPATH.
"""

from collections import deque
from pathlib import Path

import numpy as np
from analyze_p2p_sdf_keyframes import masked_camera_points, near_surface_counts, shaft_radius
from compare_p2p_sdf_input_filters import largest_depth_component
from point2pose.modules.reconstruction.sdf_builder import TSDFVolume
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
from sdf_boundary_occlusion_filters import old_surface_depth_image

HERE = Path(__file__).parent
DEPTH_EDGE_M = 0.005  # The existing depth-layer control.
SUPPORT_WINDOW = 3  # Earlier *accepted* keyframes, excluding the current view.
PROJECTED_RADIUS_PX = 3  # The existing sparse-SDF projection control.
DEPTH_AGREEMENT_M = 0.020  # The existing two-way depth-gap control.


def main() -> None:
    snapshots = np.load(HERE / "p2p_dowel_sdf_release_snapshots.npz")
    keyframes = np.load(HERE / "p2p_dowel_sdf_keyframe_depths.npz")
    reference = np.load(HERE / "pose_traces.npz")
    center = reference["dowel_reference_center_camera_m"]
    axis = reference["dowel_reference_shaft_camera"]
    voxel_size = float(snapshots["voxel_size"])

    def initial_volume() -> TSDFVolume:
        volume = TSDFVolume(snapshots["bounds"].copy(), voxel_size, use_gpu=False)
        volume._tsdf_vol_cpu[:] = snapshots["tsdf_100"]
        volume._weight_vol_cpu[:] = snapshots["weight_100"]
        return volume

    baseline = initial_volume()
    confirmed = initial_volume()
    first_points, _, _ = masked_camera_points(keyframes["depth_masked_m"][0], keyframes["intrinsics"][0])
    first_pose = keyframes["poses"][0]
    first_object_points = (first_points - first_pose[:3, 3]) @ first_pose[:3, :3]
    prior_views = deque([cKDTree(first_object_points)], maxlen=SUPPORT_WINDOW)
    totals = np.zeros(4, dtype=np.int64)
    zero_color = np.zeros((*keyframes["depth_masked_m"][0].shape, 3), dtype=np.uint8)

    print("source,depth_component_pixels,confirmed_pixels,within_20mm_kept,beyond_20mm_kept")
    for index in range(1, len(keyframes["source_frames"])):
        depth = keyframes["depth_masked_m"][index]
        k = keyframes["intrinsics"][index]
        pose = keyframes["poses"][index]
        camera_points, y, x = masked_camera_points(depth, k)
        object_points = (camera_points - pose[:3, 3]) @ pose[:3, :3]
        selected = largest_depth_component(depth, DEPTH_EDGE_M)[y, x]

        old_depth = old_surface_depth_image(confirmed, pose, k, depth.shape, 0.15)
        nearby_old = minimum_filter(
            old_depth,
            size=2 * PROJECTED_RADIUS_PX + 1,
            mode="constant",
            cval=np.inf,
        )[y, x]
        agrees_with_map = np.isfinite(nearby_old) & (
            np.abs(camera_points[:, 2] - nearby_old) <= DEPTH_AGREEMENT_M
        )
        confirmed_by_view = np.zeros(len(object_points), dtype=bool)
        for old_view in prior_views:
            confirmed_by_view |= old_view.query(object_points)[0] <= voxel_size * np.sqrt(3)
        accepted = selected & (agrees_with_map | confirmed_by_view)

        # This first-view shaft check scores the experiment; it is not read by the filter.
        within_shaft = shaft_radius(object_points, center, axis) <= 0.020
        totals += (
            np.count_nonzero(selected & within_shaft),
            np.count_nonzero(selected & ~within_shaft),
            np.count_nonzero(accepted & within_shaft),
            np.count_nonzero(accepted & ~within_shaft),
        )
        print(
            f"{keyframes['source_frames'][index]},{selected.sum()},{accepted.sum()},"
            f"{np.count_nonzero(accepted & within_shaft)},{np.count_nonzero(accepted & ~within_shaft)}"
        )
        for volume, keep in ((baseline, selected), (confirmed, accepted)):
            chosen_depth = depth.copy()
            chosen_depth[y[~keep], x[~keep]] = 0
            volume.integrate(zero_color, chosen_depth, k, np.linalg.inv(pose), obs_weight=1)
        if accepted.any():
            prior_views.append(cKDTree(object_points[accepted]))

    print("depth_component_within20,beyond20,confirmed_within20,beyond20:", *totals)
    for name, volume in (("depth_component", baseline), ("confirmed", confirmed)):
        print(
            name,
            "near_surface,total,beyond40,beyond80:",
            *near_surface_counts(volume, snapshots["origin"], voxel_size, center, axis),
        )


if __name__ == "__main__":
    main()
