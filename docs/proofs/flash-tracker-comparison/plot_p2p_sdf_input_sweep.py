"""Show the saved P2P map and emitted-pose fit for selected replay filters."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "p2p_sdf_input_sweep.png")
    args = parser.parse_args()
    summary = {
        row["setting"]: row for row in csv.DictReader((HERE / "p2p_sdf_input_ablation_summary.csv").open())
    }
    reference = np.load(HERE / "pose_traces.npz")
    center = reference["dowel_reference_center_camera_m"]
    axis = reference["dowel_reference_shaft_camera"]
    axis /= np.linalg.norm(axis)
    columns = (
        ("baseline", "Original", HERE / "p2p_dowel_sdf_release_snapshots.npz"),
        ("border2", "2 px mask margin", args.trial_dir / "border2_sdf.npz"),
        ("occ3gap10", "Nearer-new check", args.trial_dir / "occ3gap10_sdf.npz"),
        ("depth", "Depth layers", HERE / "p2p_dowel_sdf_depth_component_540.npz"),
        ("strict_depth3gap20", "Depth + two-way", args.trial_dir / "strict_depth3gap20_sdf.npz"),
    )
    fig, axes = plt.subplots(
        1, len(columns), figsize=(17, 4.8), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (key, title, path) in zip(axes, columns, strict=True):
        saved = np.load(path)
        if key == "baseline":
            tsdf, weight = saved["tsdf_540"], saved["weight_540"]
        else:
            tsdf, weight = saved["tsdf"], saved["weight"]
        points = saved["origin"] + np.argwhere((weight > 0) & (np.abs(tsdf) < 0.15)) * saved["voxel_size"]
        along = (points - center) @ axis * 1000
        radius = np.linalg.norm(np.cross(points - center, axis), axis=1) * 1000
        for selection, color in (
            (radius <= 20, "#007da3"),
            ((radius > 20) & (radius <= 40), "#d7912a"),
            (radius > 40, "#cc3a32"),
        ):
            ax.scatter(along[selection], radius[selection], s=14, c=color, alpha=0.83)
        ax.axhline(20, color="#d7912a", linestyle="--", linewidth=0.8)
        ax.axhline(40, color="#cc3a32", linestyle="--", linewidth=0.8)
        ax.set(
            xlim=(-110, 160),
            ylim=(0, 130),
            title=f"{title}\n{summary[key]['axis_error_514_550_median_deg']}° shaft error",
        )
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Distance from first-view shaft (mm)")
    fig.supxlabel("Position along first-view shaft (mm)")
    fig.suptitle("Stored dowel surface at frame 540  ·  same recorded RGB-D sequence", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
