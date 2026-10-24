"""Check causal pose candidates against measured RGB-D free space.

For each projected reference pixel, keep the nearest modeled surface. A closer
measured depth can be an occluder; a farther measured depth contradicts the
candidate surface. This is a diagnostic, not a full visibility likelihood or
an independently measured 6D pose label.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROLES = ("dowel", "ring")


def reference_points(
    dataset: Path,
    masks: Path,
    role: str,
    intrinsics: np.ndarray,
    *,
    mask_erosion_px: int = 1,
) -> np.ndarray:
    if mask_erosion_px < 0:
        raise ValueError("mask erosion must be nonnegative")
    depth = cv2.imread(str(dataset / "depth" / "000100.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(masks / role / "000100.png"), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None or depth.shape != mask.shape:
        raise ValueError(f"missing first-view RGB-D/mask for {role}")
    depth_m = depth.astype(np.float32) * 0.001
    eroded = cv2.erode(mask, np.ones((2 * mask_erosion_px + 1, 2 * mask_erosion_px + 1), dtype=np.uint8)) > 0
    y, x = np.nonzero(eroded & (depth_m > 0))
    if len(x) > 4000:
        selected = np.linspace(0, len(x) - 1, 4000, dtype=int)
        x, y = x[selected], y[selected]
    z = depth_m[y, x]
    return np.column_stack(
        ((x - intrinsics[0, 2]) * z / intrinsics[0, 0], (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z)
    )


def score_rays(
    reference: np.ndarray,
    pose: np.ndarray,
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    depth_tolerance_m: float,
) -> tuple[int, int, int, int, int, int]:
    """Return modeled, valid, supported, free-space, occluded and masked counts."""
    height, width = depth_m.shape
    camera_points = reference @ pose[:3, :3].T + pose[:3, 3]
    camera_points = camera_points[camera_points[:, 2] > 0]
    x = np.rint(camera_points[:, 0] / camera_points[:, 2] * intrinsics[0, 0] + intrinsics[0, 2]).astype(int)
    y = np.rint(camera_points[:, 1] / camera_points[:, 2] * intrinsics[1, 1] + intrinsics[1, 2]).astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    flat = y[inside] * width + x[inside]
    nearest = np.full(height * width, np.inf, dtype=float)
    np.minimum.at(nearest, flat, camera_points[inside, 2])
    pixels = np.flatnonzero(np.isfinite(nearest))
    measured = depth_m.flat[pixels]
    valid = depth_valid.flat[pixels]
    delta = measured - nearest[pixels]
    current_mask = mask.flat[pixels]
    supported = valid & (np.abs(delta) <= depth_tolerance_m) & current_mask
    free_space = valid & (delta > depth_tolerance_m)
    occluded = valid & (delta < -depth_tolerance_m)
    return (
        len(pixels),
        int(valid.sum()),
        int(supported.sum()),
        int(free_space.sum()),
        int(occluded.sum()),
        int((valid & current_mask).sum()),
    )


def phase(frame: int) -> str:
    if frame <= 149:
        return "pregrasp"
    if frame <= 248:
        return "occlusion"
    if frame <= 280:
        return "reacquisition"
    if frame <= 399:
        return "carry"
    if frame <= 500:
        return "insertion"
    return "retreat"


def plot_free_space(rows: list[tuple], output: Path, tolerance_mm: float) -> None:
    colors = {"p2p_sam2": "#dc7d18", "p2p_sam3": "#d74391", "bundlesdf": "#1677a8"}
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for axis, role in zip(axes, ROLES, strict=True):
        for candidate, color in colors.items():
            subset = [
                row
                for row in rows
                if row[0] == role and row[3] == candidate and row[4] == tolerance_mm and row[6] > 0
            ]
            axis.plot(
                [row[1] for row in subset],
                [row[8] / row[6] for row in subset],
                color=color,
                linewidth=1.2,
                label=candidate,
            )
        axis.axvspan(161, 275, color="#888888", alpha=0.1)
        axis.axvspan(400, 500, color="#81a0c3", alpha=0.08)
        axis.set(title=role.capitalize(), ylabel="Measured free-space contradiction", ylim=(0, 1.03))
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)
    axes[-1].set(xlabel="Demo frame", xlim=(100, 550))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--depth-tolerances-mm", type=float, nargs="+", default=(3.0, 5.0, 10.0))
    parser.add_argument("--local-depth-span-mm", type=float, default=10.0)
    args = parser.parse_args()
    if any(value <= 0 for value in args.depth_tolerances_mm) or args.local_depth_span_mm <= 0:
        parser.error("depth tolerances and local span must be positive")
    root = Path(__file__).parent
    baseline = np.load(root / "pose_traces.npz")
    intrinsics = baseline["intrinsics"]
    rows = []
    for role in ROLES:
        reference = reference_points(args.dataset, args.masks, role, intrinsics)
        sam2_name = "internal_control" if role == "dowel" else "sam2"
        sam2 = np.load(root / f"p2p_{role}_{sam2_name}_100_550.npz")
        sam3 = np.load(root / f"p2p_{role}_sam3_100_550.npz")
        sources = {
            "p2p_sam2": (sam2["poses"], sam2["lost"]),
            "p2p_sam3": (sam3["poses"], sam3["lost"]),
            "bundlesdf": (baseline[f"bundlesdf_{role}_delta"], baseline[f"bundlesdf_{role}_failed"]),
        }
        for frame in range(100, 551):
            stem = f"{frame:06d}"
            depth = cv2.imread(str(args.dataset / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(args.masks / role / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            if depth is None or mask is None or depth.shape != mask.shape:
                raise ValueError(f"missing or misaligned RGB-D/mask at {role} frame {frame}")
            depth_m = depth.astype(np.float32) * 0.001
            neighborhood = np.ones((3, 3), dtype=np.uint8)
            lower = cv2.erode(depth_m, neighborhood)
            upper = cv2.dilate(depth_m, neighborhood)
            depth_valid = (depth_m > 0) & (lower > 0) & (upper - lower <= args.local_depth_span_mm / 1000)
            for name, (poses, lost) in sources.items():
                if lost[frame - 100]:
                    continue
                for tolerance_mm in args.depth_tolerances_mm:
                    counts = score_rays(
                        reference,
                        poses[frame - 100],
                        depth_m,
                        depth_valid,
                        mask > 0,
                        intrinsics,
                        depth_tolerance_m=tolerance_mm / 1000,
                    )
                    rows.append((role, frame, phase(frame), name, tolerance_mm, *counts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "role",
                "frame",
                "phase",
                "candidate",
                "depth_tolerance_mm",
                "modeled_pixels",
                "valid_depth_pixels",
                "supported_pixels",
                "free_space_contradictions",
                "occluded_pixels",
                "masked_depth_pixels",
            )
        )
        writer.writerows(rows)
    if args.figure is not None:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        plot_free_space(rows, args.figure, args.depth_tolerances_mm[len(args.depth_tolerances_mm) // 2])
    for frame in (130, 249, 260, 263, 276, 300, 400, 450):
        selected = [row for row in rows if row[0] == "dowel" and row[1] == frame and row[4] == 5]
        for row in selected:
            valid = max(row[6], 1)
            print(frame, row[3], "free-space fraction", round(row[8] / valid, 3), "valid pixels", row[6])


if __name__ == "__main__":
    main()
