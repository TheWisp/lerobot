"""Plot observed Point2Pose SDF near-surface voxels in the demo shaft frame.

The snapshot file comes from a bit-identical configured Point2Pose replay.
Near-surface voxels are a visualization of the stored TSDF, not a new mesh fit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).parent
FRAMES = (100, 340, 540)


def main() -> None:
    snapshot = np.load(HERE / "p2p_dowel_sdf_release_snapshots.npz")
    reference = np.load(HERE / "pose_traces.npz")
    center = reference["dowel_reference_center_camera_m"]
    axis = reference["dowel_reference_shaft_camera"]
    origin = snapshot["origin"]
    voxel = float(snapshot["voxel_size"])
    fig, plots = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True, sharey=True, constrained_layout=True)
    for plot, frame in zip(plots, FRAMES, strict=True):
        tsdf = snapshot[f"tsdf_{frame}"]
        weight = snapshot[f"weight_{frame}"]
        # TSDF truncation is five voxels. +/-0.15 is a 3.75 mm band
        # around the reconstructed zero level set at this 5 mm resolution.
        near_surface = (weight > 0) & (np.abs(tsdf) < 0.15)
        ijk = np.column_stack(np.nonzero(near_surface))
        points = origin + ijk * voxel
        relative = points - center
        axial = relative @ axis
        radius = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
        beyond = radius > 0.04

        plot.add_patch(Rectangle((-60, 0), 120, 20, facecolor="#d7f4e3", edgecolor="#4aa776", linewidth=1.5))
        plot.axhline(10, color="#4aa776", linestyle="--", linewidth=1)
        plot.scatter(axial[~beyond] * 1e3, radius[~beyond] * 1e3, s=19, color="#177ca7", alpha=0.7)
        plot.scatter(axial[beyond] * 1e3, radius[beyond] * 1e3, s=22, color="#d95139", alpha=0.85)
        plot.set(
            title=f"Frame {frame}: {int(beyond.sum())}/{len(points)} far from shaft",
            xlabel="Position along original shaft (mm)",
            xlim=(-110, 160),
            ylim=(0, 120),
        )
        plot.grid(alpha=0.15)
    plots[0].set_ylabel("Distance from original shaft axis (mm)")
    fig.suptitle("Point2Pose TSDF: initial rod develops stray surface branches", fontsize=15)
    fig.savefig(HERE / "p2p_dowel_sdf_release.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
