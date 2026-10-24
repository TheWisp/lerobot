"""Show the saved Point2Pose SDF surface before and after depth filtering.

Only observed voxels close to the TSDF zero level are drawn. These are cells
from the tracker volume, not a separately reconstructed mesh or a pose line.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def surface_points(tsdf: np.ndarray, weight: np.ndarray, origin: np.ndarray, voxel: float) -> np.ndarray:
    near_surface = (weight > 0) & (np.abs(tsdf) < 0.15)
    return origin + np.argwhere(near_surface) * voxel


def main() -> None:
    original = np.load(HERE / "p2p_dowel_sdf_release_snapshots.npz")
    filtered = np.load(HERE / "p2p_dowel_sdf_depth_component_540.npz")
    reference = np.load(HERE / "pose_traces.npz")
    center = reference["dowel_reference_center_camera_m"]
    along = reference["dowel_reference_shaft_camera"]
    along /= np.linalg.norm(along)
    camera_up = np.array([0.0, 0.0, 1.0])
    side = np.cross(along, camera_up if abs(along @ camera_up) < 0.95 else [0, 1, 0])
    side /= np.linalg.norm(side)
    up = np.cross(along, side)
    basis = np.column_stack((along, side, up))

    views = (
        (
            "Original Point2Pose",
            surface_points(
                original["tsdf_540"],
                original["weight_540"],
                original["origin"],
                float(original["voxel_size"]),
            ),
        ),
        (
            "After separating depth surfaces",
            surface_points(
                filtered["tsdf"], filtered["weight"], filtered["origin"], float(filtered["voxel_size"])
            ),
        ),
    )
    fig = plt.figure(figsize=(12.5, 9.0), layout="constrained")
    fig.suptitle(
        "The dowel surface stored by Point2Pose · same demo frame 540\n"
        "Blue: within 20 mm of first shaft · orange: 20–40 mm · red: farther than 40 mm",
        fontsize=16,
    )
    for column, (title, points) in enumerate(views):
        coords = (points - center) @ basis * 1000
        along_mm, side_mm, up_mm = coords.T
        radius_mm = np.hypot(side_mm, up_mm)
        near = radius_mm <= 20
        moderate = (radius_mm > 20) & (radius_mm <= 40)
        far = radius_mm > 40
        panel_3d = fig.add_subplot(2, 2, column + 1, projection="3d")
        panel_3d.scatter(along_mm[near], side_mm[near], up_mm[near], s=17, c="#087da5", alpha=0.85)
        panel_3d.scatter(along_mm[moderate], side_mm[moderate], up_mm[moderate], s=22, c="#eda33f", alpha=0.9)
        panel_3d.scatter(along_mm[far], side_mm[far], up_mm[far], s=22, c="#d94831", alpha=0.9)
        panel_3d.plot([-60, 60], [0, 0], [0, 0], color="#333333", linewidth=2, alpha=0.6)
        panel_3d.set(
            xlim=(-110, 160),
            ylim=(-115, 115),
            zlim=(-115, 115),
            xlabel="Along dowel (mm)",
            ylabel="Across (mm)",
            zlabel="Across (mm)",
            title=f"{title}\n{len(points)} surface cells · {moderate.sum()} orange · {far.sum()} red",
        )
        panel_3d.set_box_aspect((270, 230, 230))
        panel_3d.view_init(elev=26, azim=-69)
        panel_3d.set_proj_type("ortho")

        side_view = fig.add_subplot(2, 2, column + 3)
        side_view.scatter(along_mm[near], radius_mm[near], s=21, c="#087da5", alpha=0.8)
        side_view.scatter(along_mm[moderate], radius_mm[moderate], s=25, c="#eda33f", alpha=0.9)
        side_view.scatter(along_mm[far], radius_mm[far], s=25, c="#d94831", alpha=0.9)
        side_view.axhspan(0, 20, color="#dcf3e7", zorder=-1)
        side_view.axhline(20, color="#eda33f", linewidth=1, linestyle="--", alpha=0.6)
        side_view.axhline(40, color="#d94831", linewidth=1, linestyle="--", alpha=0.6)
        side_view.set(
            xlim=(-110, 160),
            ylim=(0, 120),
            xlabel="Position along dowel (mm)",
            ylabel="Distance from dowel centerline (mm)" if column == 0 else "",
            title="Side view of the same 3D cells",
        )
        side_view.grid(alpha=0.15)
    fig.savefig(HERE / "p2p_sdf_before_after.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
