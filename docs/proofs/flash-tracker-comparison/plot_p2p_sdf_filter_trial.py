"""Draw the controlled SDF-filter pose branches against the current RGB-D view.

The cyan line is a visible-depth shaft estimate, not ground truth; the orange
line is the full first-view shaft transported by each emitted P2P pose.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from analyze_dowel_release_geometry import cloud_and_pixels, shaft_middle

HERE = Path(__file__).parent
DATASET = Path.home() / ".cache/huggingface/lerobot/demos/insertion/recording"
MASKS = Path.home() / ".cache/flash-tracker-compare/sam3_masks"


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    return points[:, :2] / points[:, 2, None] @ k[:2, :2].T + k[:2, 2]


def main() -> None:
    reference = np.load(HERE / "pose_traces.npz")
    k = reference["intrinsics"]
    center = reference["dowel_reference_center_camera_m"]
    axis = reference["dowel_reference_shaft_camera"]
    offsets = reference["dowel_reference_end_offsets_m"]
    shaft = center + offsets[:, None] * axis
    traces = {
        "Original P2P": np.load(HERE / "p2p_dowel_internal_control_100_550.npz"),
        "3-keyframe window": np.load(HERE / "p2p_dowel_sdf_window_100_550.npz"),
        "Depth layer": np.load(HERE / "p2p_dowel_sdf_depth_component_100_550.npz"),
    }
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, frame in enumerate((505, 540)):
        rgb = cv2.imread(str(DATASET / "rgb" / f"{frame:06d}.jpg"))
        if rgb is None:
            raise FileNotFoundError(frame)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        points, pixels, mask = cloud_and_pixels(DATASET, MASKS, frame, k)
        middle, _, visible_axis, _ = shaft_middle(points, pixels)
        visible_center = middle.mean(axis=0)
        visible_line = project(visible_center + np.array([-0.05, 0.05])[:, None] * visible_axis, k)
        ys, xs = np.nonzero(mask)
        x0, x1 = max(0, int(xs.min()) - 70), min(rgb.shape[1], int(xs.max()) + 70)
        y0, y1 = max(0, int(ys.min()) - 70), min(rgb.shape[0], int(ys.max()) + 70)
        for column, (name, trace) in enumerate(traces.items()):
            pose = trace["poses"][frame - 100]
            predicted = project(shaft @ pose[:3, :3].T + pose[:3, 3], k)
            ax = axes[row, column]
            ax.imshow(rgb)
            ax.contour(mask, levels=[127], colors=["#4ae09d"], linewidths=1.5)
            ax.plot(visible_line[:, 0], visible_line[:, 1], color="#17cbef", linewidth=3)
            ax.plot(predicted[:, 0], predicted[:, 1], color="#ff963d", linewidth=3)
            ax.scatter(predicted[:, 0], predicted[:, 1], color="#ff963d", s=22)
            ax.set(xlim=(x0, x1), ylim=(y1, y0), title=f"Frame {frame} · {name}")
            ax.axis("off")
    figure.suptitle("Green: SAM3 mask · cyan: current depth axis · orange: emitted P2P shaft", fontsize=14)
    figure.savefig(HERE / "p2p_sdf_filter_trial.png", dpi=150)


if __name__ == "__main__":
    main()
