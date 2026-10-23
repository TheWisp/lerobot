"""Compare diagnostic mask/depth filters on the dowel's SDF inputs.

The first-view shaft and 40 mm radius classify known impossible surfaces only
for this offline ablation. None of these filter thresholds is a runtime policy.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = Path(__file__).parent


def largest_depth_component(depth: np.ndarray, edge_limit_m: float) -> np.ndarray:
    valid = depth > 0
    height, width = depth.shape
    ids = np.full((height, width), -1, dtype=np.int32)
    ids[valid] = np.arange(int(valid.sum()))
    horizontal = valid[:, :-1] & valid[:, 1:] & (np.abs(depth[:, :-1] - depth[:, 1:]) <= edge_limit_m)
    vertical = valid[:-1, :] & valid[1:, :] & (np.abs(depth[:-1, :] - depth[1:, :]) <= edge_limit_m)
    source = np.concatenate((ids[:, :-1][horizontal], ids[:-1, :][vertical]))
    target = np.concatenate((ids[:, 1:][horizontal], ids[1:, :][vertical]))
    graph = coo_matrix(
        (np.ones(2 * len(source)), (np.r_[source, target], np.r_[target, source])),
        shape=(int(valid.sum()), int(valid.sum())),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    result = np.zeros(valid.shape, dtype=bool)
    result[valid] = labels == np.bincount(labels).argmax()
    return result


def main() -> None:
    data = np.load(HERE / "p2p_dowel_sdf_keyframe_depths.npz")
    reference = np.load(HERE / "pose_traces.npz")
    center = reference["dowel_reference_center_camera_m"]
    axis = reference["dowel_reference_shaft_camera"]
    initial_depth = data["depth_masked_m"][0]
    initial_y, initial_x = np.nonzero(initial_depth > 0)
    initial_z = initial_depth[initial_y, initial_x]
    initial_k = data["intrinsics"][0]
    initial_points = np.stack(
        (
            (initial_x - initial_k[0, 2]) * initial_z / initial_k[0, 0],
            (initial_y - initial_k[1, 2]) * initial_z / initial_k[1, 1],
            initial_z,
        ),
        axis=1,
    )
    first_view_surface = cKDTree(initial_points)
    aggregate: dict[str, list[int]] = {}
    print("frame,good_pixels,far_pixels,cc5_good,cc5_far,cc10_good,cc10_far")
    for index in range(1, len(data["source_frames"])):
        depth = data["depth_masked_m"][index]
        y, x = np.nonzero(depth > 0)
        z = depth[y, x]
        k = data["intrinsics"][index]
        pose = data["poses"][index]
        points = np.stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z), axis=1)
        object_points = (points - pose[:3, 3]) @ pose[:3, :3]
        good = np.linalg.norm(np.cross(object_points - center, axis), axis=1) <= 0.04
        distance_to_first_view = first_view_surface.query(object_points)[0]
        median_distance = np.median(distance_to_first_view)
        median_absolute_deviation = np.median(np.abs(distance_to_first_view - median_distance))
        logits = data["mask_logit_at_depth"][index][y, x]
        methods = {
            "SAM2 logit > 10": logits > 10,
            "mask erode 3x3": cv2.erode((depth > 0).astype(np.uint8), np.ones((3, 3), np.uint8))[y, x].astype(
                bool
            ),
            "mask erode 5x5": cv2.erode((depth > 0).astype(np.uint8), np.ones((5, 5), np.uint8))[y, x].astype(
                bool
            ),
            "depth component 5 mm": largest_depth_component(depth, 0.005)[y, x],
            "depth component 10 mm": largest_depth_component(depth, 0.010)[y, x],
            "first-view surface median + 7 MAD": distance_to_first_view
            <= median_distance + 7 * median_absolute_deviation,
        }
        outcome = {}
        for name, kept in methods.items():
            selected_good = int(np.count_nonzero(kept & good))
            selected_far = int(np.count_nonzero(kept & ~good))
            counts = aggregate.setdefault(name, [0, 0, 0, 0, 0])
            counts[0] += selected_good
            counts[1] += int(np.count_nonzero(good))
            counts[2] += selected_far
            counts[3] += int(np.count_nonzero(~good))
            counts[4] = max(counts[4], selected_far)
            outcome[name] = (selected_good, selected_far)
        cc5, cc10 = outcome["depth component 5 mm"], outcome["depth component 10 mm"]
        print(
            f"{data['source_frames'][index]},{int(good.sum())},{int((~good).sum())},"
            f"{cc5[0]},{cc5[1]},{cc10[0]},{cc10[1]}"
        )
    for name, (good_kept, all_good, far_kept, all_far, max_far) in aggregate.items():
        print(
            f"{name}: good retained {good_kept}/{all_good}={good_kept / all_good:.1%}; "
            f"far rejected {all_far - far_kept}/{all_far}={(all_far - far_kept) / all_far:.1%}; "
            f"worst keyframe far retained {max_far}"
        )


if __name__ == "__main__":
    main()
