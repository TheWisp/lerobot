"""Diagnostic causal keyframe support for the dowel's masked SDF inputs.

These voxel sizes and the first-view shaft label only measure this recording.
They are not parameters for a runtime object-fusion policy. Keyframe spacing is
irregular, so this is not equivalent to checking consecutive camera frames.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def object_points(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(depth > 0)
    z = depth[y, x]
    camera = np.stack(
        (
            (x - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (y - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ),
        axis=1,
    )
    return (camera - pose[:3, 3]) @ pose[:3, :3]


def main() -> None:
    data = np.load(HERE / "p2p_dowel_sdf_keyframe_depths.npz")
    reference = np.load(HERE / "pose_traces.npz")
    center = reference["dowel_reference_center_camera_m"]
    axis = reference["dowel_reference_shaft_camera"]
    points = [
        object_points(depth, intrinsics, pose)
        for depth, intrinsics, pose in zip(
            data["depth_masked_m"], data["intrinsics"], data["poses"], strict=True
        )
    ]
    plausible = [np.linalg.norm(np.cross(frame - center, axis), axis=1) <= 0.04 for frame in points]

    print("voxel_mm,prior_keyframes,plausible_retained,off_axis_rejected")
    for voxel_m in (0.005, 0.010):
        voxelized = [np.floor(frame / voxel_m).astype(np.int32) for frame in points]
        for prior_keyframes in (1, 3, 5):
            plausible_kept = plausible_total = off_axis_kept = off_axis_total = 0
            for index in range(1, len(points)):
                prior = {
                    tuple(voxel)
                    for earlier in voxelized[max(0, index - prior_keyframes) : index]
                    for voxel in earlier
                }
                supported = np.array([tuple(voxel) in prior for voxel in voxelized[index]], dtype=bool)
                good = plausible[index]
                plausible_kept += int(np.count_nonzero(supported & good))
                plausible_total += int(np.count_nonzero(good))
                off_axis_kept += int(np.count_nonzero(supported & ~good))
                off_axis_total += int(np.count_nonzero(~good))
            print(
                f"{voxel_m * 1000:.0f},{prior_keyframes},"
                f"{plausible_kept / plausible_total:.1%},"
                f"{1 - off_axis_kept / off_axis_total:.1%}"
            )


if __name__ == "__main__":
    main()
