"""Plot the first two contaminated dowel keyframes against their original RGB.

The viewports only frame the relevant region of the recorded overhead image;
the radial split is an analysis diagnostic, not a tracking threshold.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FRAMES_AND_CROPS = ((249, (185, 375, 50, 325)), (276, (130, 325, 80, 365)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", required=True, type=Path, help="Original demo recording/rgb directory")
    parser.add_argument("--output", type=Path, default=HERE / "p2p_dowel_sdf_bad_keyframes.png")
    args = parser.parse_args()
    keyframes = np.load(HERE / "p2p_dowel_sdf_keyframe_depths.npz")
    reference = np.load(HERE / "pose_traces.npz")
    center = reference["dowel_reference_center_camera_m"]
    axis = reference["dowel_reference_shaft_camera"]
    fig, plots = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, (frame, (y0, y1, x0, x1)) in enumerate(FRAMES_AND_CROPS):
        index = int(np.flatnonzero(keyframes["source_frames"] == frame)[0])
        depth = keyframes["depth_masked_m"][index]
        k = keyframes["intrinsics"][index]
        pose = keyframes["poses"][index]
        bgr = cv2.imread(str(args.rgb_dir / f"{frame:06d}.jpg"))
        if bgr is None:
            raise FileNotFoundError(args.rgb_dir / f"{frame:06d}.jpg")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        y, x = np.nonzero(depth > 0)
        z = depth[y, x]
        points = np.stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z), axis=1)
        object_points = (points - pose[:3, 3]) @ pose[:3, :3]
        bad = np.linalg.norm(np.cross(object_points - center, axis), axis=1) > 0.04
        for ax in plots[row]:
            ax.imshow(rgb[y0:y1, x0:x1])
            ax.set(xlim=(0, x1 - x0), ylim=(y1 - y0, 0))
            ax.axis("off")
        plots[row, 0].set_title(f"Frame {frame}: RGB")
        plots[row, 1].scatter(x[~bad] - x0, y[~bad] - y0, s=1.5, c="#17e56f", alpha=0.68)
        plots[row, 1].scatter(x[bad] - x0, y[bad] - y0, s=3, c="#ff3c30", alpha=0.84)
        plots[row, 1].set_title(f"SAM2-masked depth: {bad.sum()} / {len(bad)} inconsistent")
    fig.suptitle(
        "False surfaces admitted during dowel SDF fusion\n"
        "Green: within 4 cm of first-view shaft after pose alignment; red: farther away",
        fontsize=15,
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
