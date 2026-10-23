"""Check the late dowel pose against independent segmented RGB-D geometry.

This is a diagnostic, not a cylinder-specific FLASH runtime estimator. It uses
the recorded demo, its SAM3 masks, and the bit-reproduced Point2Pose trace.
The shaft radius is learned from the first clear demo view (frame 100). The
middle of the segmented shaft avoids its end caps and the insertion contact.

Run from this directory with::

    python analyze_dowel_release_geometry.py \
      --dataset ~/.cache/huggingface/lerobot/demos/insertion/recording \
      --masks ~/.cache/flash-tracker-compare/sam3_masks \
      --diagnostics ~/.cache/flash-tracker-compare/p2p_release_probe/p2p_dowel_release_probe_configured_diagnostics
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).parent
FRAMES = (490, 495, 500, 505, 510, 520, 530, 535, 540, 545, 550)


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(unit(a) @ unit(b)), 0, 1))))


def cloud_and_pixels(
    dataset: Path, masks: Path, frame: int, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = cv2.imread(str(dataset / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(masks / "dowel" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None or depth.dtype != np.uint16 or depth.shape != mask.shape:
        raise ValueError(f"missing aligned depth or mask for frame {frame}")
    depth_m = depth.astype(np.float32) * 0.001
    lower = cv2.erode(depth_m, np.ones((3, 3), np.uint8))
    upper = cv2.dilate(depth_m, np.ones((3, 3), np.uint8))
    # Exclude depth holes and discontinuities before inferring a surface axis.
    valid = (cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0) & (lower > 0) & (upper - lower <= 0.010)
    y, x = np.nonzero(valid)
    z = depth_m[y, x]
    xyz = np.column_stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z))
    return xyz, np.column_stack((x, y)), mask


def shaft_middle(points: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
    rough_axis = eigenvectors[:, -1]
    coordinate = (points - points.mean(axis=0)) @ rough_axis
    lo, hi = np.quantile(coordinate, (0.2, 0.8))
    keep = (coordinate >= lo) & (coordinate <= hi)
    middle = points[keep]
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(middle.T))
    return middle, pixels[keep], eigenvectors[:, -1], float(eigenvalues[-1] / eigenvalues[-2])


def cross_section(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = unit(axis)
    transverse = unit(np.cross(axis, (0, 0, 1) if abs(axis[2]) < 0.9 else (0, 1, 0)))
    return np.column_stack((points @ transverse, points @ np.cross(axis, transverse)))


def fit_circle(points: np.ndarray, axis: np.ndarray, radius: float | None) -> tuple[float, np.ndarray]:
    xy = cross_section(points, axis)
    center = xy.mean(axis=0)
    initial_radius = float(np.median(np.linalg.norm(xy - center, axis=1)))
    if radius is None:
        fit = least_squares(
            lambda q: np.linalg.norm(xy - q[:2], axis=1) - q[2],
            np.r_[center, initial_radius],
            loss="soft_l1",
            f_scale=0.002,
        )
        radius = float(fit.x[2])
        center = fit.x[:2]
    else:
        fit = least_squares(
            lambda q: np.linalg.norm(xy - q, axis=1) - radius,
            center,
            loss="soft_l1",
            f_scale=0.002,
        )
        center = fit.x
    return radius, np.abs(np.linalg.norm(xy - center, axis=1) - radius)


def held_out_circle_error(
    points: np.ndarray, pixels: np.ndarray, axis: np.ndarray, radius: float
) -> np.ndarray:
    """Fit the center on checkerboard half A; score only half B."""
    train = (pixels[:, 0] + pixels[:, 1]) % 2 == 0
    xy_train = cross_section(points[train], axis)
    xy_test = cross_section(points[~train], axis)
    fit = least_squares(
        lambda center: np.linalg.norm(xy_train - center, axis=1) - radius,
        xy_train.mean(axis=0),
        loss="soft_l1",
        f_scale=0.002,
    )
    return np.abs(np.linalg.norm(xy_test - fit.x, axis=1) - radius)


def projected_axis(axis: np.ndarray, point: np.ndarray, k: np.ndarray) -> np.ndarray:
    ends = point + np.stack((-0.03 * axis, 0.03 * axis))
    pixels = ends[:, :2] / ends[:, 2, None] @ k[:2, :2].T + k[:2, 2]
    return unit(pixels[1] - pixels[0])


def draw_frame(
    ax,
    dataset: Path,
    mask: np.ndarray,
    frame: int,
    k: np.ndarray,
    depth_axis: np.ndarray,
    p2p_axis: np.ndarray,
    centroid: np.ndarray,
    fit_uv: np.ndarray | None,
    fit_inlier: np.ndarray | None,
) -> None:
    rgb = cv2.imread(str(dataset / "rgb" / f"{frame:06d}.jpg"))
    if rgb is None:
        raise ValueError(f"cannot read RGB frame {frame}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    ys, xs = np.nonzero(mask)
    x0, x1 = max(0, int(xs.min()) - 45), min(rgb.shape[1], int(xs.max()) + 45)
    y0, y1 = max(0, int(ys.min()) - 45), min(rgb.shape[0], int(ys.max()) + 45)
    ax.imshow(rgb)
    ax.contour(mask, levels=[127], colors=["#54e7a4"], linewidths=1.4)
    center = np.array([xs.mean(), ys.mean()])
    length = 0.43 * max(xs.max() - xs.min(), ys.max() - ys.min())
    for direction, color, label in (
        (depth_axis, "#2fc8ff", "depth shaft"),
        (p2p_axis, "#ff9b35", "P2P shaft"),
    ):
        v = projected_axis(direction, centroid, k)
        line = np.stack((center - length * v, center + length * v))
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=3, label=label)
    if fit_uv is not None:
        ax.scatter(
            fit_uv[fit_inlier, 0],
            fit_uv[fit_inlier, 1],
            c="white",
            s=7,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.8,
            label="P2P fit inliers",
        )
    ax.set(xlim=(x0, x1), ylim=(y1, y0), title=f"Demo frame {frame}")
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "dowel_release_geometry.png")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration = np.load(HERE / "pose_traces.npz")
    k = calibration["intrinsics"]
    p2p = np.load(HERE / "p2p_dowel_internal_control_100_550.npz")
    reference_axis = calibration["dowel_reference_shaft_camera"]

    source_xyz, source_uv, _ = cloud_and_pixels(args.dataset, args.masks, 100, k)
    source_middle, _, source_depth_axis, _ = shaft_middle(source_xyz, source_uv)
    source_radius, _ = fit_circle(source_middle, source_depth_axis, None)
    print(f"First-view learned shaft radius: {source_radius * 1000:.2f} mm")
    rows = []
    pictured = {}
    for frame in FRAMES:
        xyz, pixels, mask = cloud_and_pixels(args.dataset, args.masks, frame, k)
        middle, middle_pixels, depth_axis, ratio = shaft_middle(xyz, pixels)
        p2p_axis = p2p["poses"][frame - 100][:3, :3] @ reference_axis
        # Fit the transverse circle center separately for each axis, holding
        # the shaft radius learned from the clear first view fixed.
        train = (middle_pixels[:, 0] + middle_pixels[:, 1]) % 2 == 0
        _, train_vectors = np.linalg.eigh(np.cov(middle[train].T))
        depth_axis = train_vectors[:, -1]
        depth_error = held_out_circle_error(middle, middle_pixels, depth_axis, source_radius)
        p2p_error = held_out_circle_error(middle, middle_pixels, p2p_axis, source_radius)
        ys, xs = np.nonzero(mask)
        mask_xy = np.column_stack((xs, ys))
        _, vectors = np.linalg.eigh(np.cov(mask_xy.T))
        image_axis = vectors[:, -1]
        image_error = angle_degrees(image_axis, projected_axis(p2p_axis, middle.mean(axis=0), k))
        row = (
            frame,
            len(middle),
            ratio,
            angle_degrees(depth_axis, p2p_axis),
            image_error,
            np.median(depth_error) * 1000,
            np.quantile(depth_error, 0.9) * 1000,
            np.median(p2p_error) * 1000,
            np.quantile(p2p_error, 0.9) * 1000,
        )
        rows.append(row)
        if frame in (495, 505, 540):
            diag = np.load(args.diagnostics / f"{frame:06d}.npz") if args.diagnostics else None
            pictured[frame] = (
                mask,
                depth_axis,
                p2p_axis,
                middle.mean(axis=0),
                None if diag is None else diag["fit_uv"],
                None if diag is None else diag["fit_inlier"],
            )
    with args.output.with_suffix(".csv").open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(
            (
                "frame",
                "middle_depth_points",
                "axial_eigen_ratio",
                "p2p_vs_depth_axis_deg",
                "p2p_vs_mask_axis_deg",
                "depth_circle_median_mm",
                "depth_circle_p90_mm",
                "p2p_circle_median_mm",
                "p2p_circle_p90_mm",
            )
        )
        writer.writerows(rows)

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.2, 0.8))
    for i, frame in enumerate((495, 505, 540)):
        ax = fig.add_subplot(grid[0, i])
        draw_frame(ax, args.dataset, *pictured[frame][:1], frame, k, *pictured[frame][1:])
    ax = fig.add_subplot(grid[1, :2])
    frames = [r[0] for r in rows]
    ax.plot(frames, [r[3] for r in rows], "o-", color="#186fa0", label="P2P vs observed depth shaft (3D)")
    ax.plot(frames, [r[4] for r in rows], "s-", color="#d36b20", label="P2P vs image mask shaft (2D)")
    ax.axvline(514, color="gray", linestyle="--", alpha=0.7, label="gripper release")
    ax.set(xlabel="Demo frame", ylabel="Unoriented shaft-angle mismatch (degrees)", ylim=(0, 50))
    ax.annotate(
        "500: contact occlusion; depth axis unstable",
        xy=(500, 46),
        xytext=(0, 0),
        textcoords="offset points",
        fontsize=8,
        color="#555555",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    ax = fig.add_subplot(grid[1, 2])
    late = [r for r in rows if r[0] >= 530]
    ax.plot([r[0] for r in late], [r[6] for r in late], "o-", color="#186fa0", label="Depth-derived axis")
    ax.plot([r[0] for r in late], [r[8] for r in late], "s-", color="#d36b20", label="P2P axis")
    ax.set(xlabel="Demo frame", ylabel="Held-out cylinder residual p90 (mm)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.suptitle("Dowel release: visible geometry contradicts P2P's shaft angle", fontsize=15)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    for row in rows:
        print(" ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row))


if __name__ == "__main__":
    main()
