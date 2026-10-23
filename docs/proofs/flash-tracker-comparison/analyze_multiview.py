"""Reproduce the causal multiview/depth-edge ablation on the insertion demo.

The model uses only source frames preceding the scored 400–500 interval.
These scores check consistency with segmented RGB-D, not 6D ground truth.

Example:
    python analyze_multiview.py --dataset /path/to/insertion/recording \
        --masks /path/to/sam3_masks --output /tmp/flash-multiview
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROLES = ("dowel", "ring")
SOURCE_FRAMES = (100, 300, 340, 380)
SCORE_FRAMES = range(400, 501)


def frame_points(
    dataset: Path,
    masks: Path,
    role: str,
    frame: int,
    intrinsics: np.ndarray,
    *,
    edge_limit_m: float | None = None,
    max_points: int | None = None,
) -> tuple[np.ndarray, int]:
    depth = cv2.imread(str(dataset / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(masks / role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if depth is None or mask is None or depth.shape != mask.shape or depth.dtype != np.uint16:
        raise ValueError(f"missing or incompatible RGB-D/mask frame {frame} for {role}")
    depth_m = depth.astype(np.float32) * 0.001
    # Same one-pixel mask erosion used by the original cross-tracker score.
    mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0)
    if edge_limit_m is not None:
        if edge_limit_m <= 0:
            raise ValueError("depth-edge limit must be positive")
        # A valid centre pixel next to missing depth is uncertain as well.
        neighborhood = np.ones((3, 3), dtype=np.uint8)
        lower = cv2.erode(depth_m, neighborhood)
        upper = cv2.dilate(depth_m, neighborhood)
        valid &= (lower > 0) & np.isfinite(lower) & np.isfinite(upper) & (upper - lower <= edge_limit_m)
    y, x = np.nonzero(valid)
    count = len(x)
    if max_points is not None and count > max_points:
        select = np.linspace(0, count - 1, max_points, dtype=int)
        x, y = x[select], y[select]
    z = depth_m[y, x]
    return np.column_stack(
        (
            (x - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (y - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    ), count


def move(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def surface_p90_mm(live: np.ndarray, model: np.ndarray) -> float:
    if not len(live) or not len(model):
        return float("nan")
    return float(1e3 * np.quantile(cKDTree(model).query(live)[0], 0.9))


def build_models(
    dataset: Path,
    masks: Path,
    role: str,
    intrinsics: np.ndarray,
    poses: np.ndarray,
    lost: np.ndarray,
    support_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    clouds = []
    for frame in SOURCE_FRAMES:
        index = frame - SOURCE_FRAMES[0]
        if lost[index]:
            raise ValueError(f"{role} source pose missing at frame {frame}")
        camera_points, _ = frame_points(dataset, masks, role, frame, intrinsics, max_points=4000)
        clouds.append(move(camera_points, np.linalg.inv(poses[index])))
    first_view = clouds[0]
    confirmed = [first_view]
    for index, cloud in enumerate(clouds[1:], start=1):
        neighbors = np.concatenate(
            [other for other_index, other in enumerate(clouds) if other_index != index]
        )
        distances = cKDTree(neighbors).query(cloud)[0]
        confirmed.append(cloud[distances <= support_radius_m])
    return first_view, np.concatenate(confirmed), sum(map(len, clouds))


def write_csv(path: Path, columns: tuple[str, ...], rows: list[tuple]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def plot_scores(path: Path, rows: list[tuple], edge_limit_mm: float) -> None:
    import matplotlib.pyplot as plt

    dowel = [row for row in rows if row[0] == "dowel" and row[2] == "p2p"]
    ring = [row for row in rows if row[0] == "ring" and row[2] == "p2p"]
    bundle_ring = [row for row in rows if row[0] == "ring" and row[2] == "bundle"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    for axis, subset, title in (
        (axes[0, 0], dowel, "Dowel: all insertion frames"),
        (axes[0, 1], [r for r in dowel if r[1] <= 420], "Dowel: early contact"),
    ):
        frames = [r[1] for r in subset]
        axis.plot(frames, [r[5] for r in subset], color="#777777", label="single view, all depth")
        axis.plot(frames, [r[6] for r in subset], color="#db5f24", label="confirmed views, all depth")
        axis.plot(frames, [r[8] for r in subset], color="#1677a8", label="confirmed views, interior depth")
        axis.set(title=title, ylabel="Surface residual p90 (mm)", xlabel="Demo frame")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].plot([r[1] for r in ring], [r[8] for r in ring], label="Point2Pose", color="#1677a8")
    axes[1, 0].plot(
        [r[1] for r in bundle_ring], [r[8] for r in bundle_ring], label="BundleSDF", color="#db5f24"
    )
    axes[1, 0].set(
        title="Ring: confirmed views, interior depth", ylabel="Surface residual p90 (mm)", xlabel="Demo frame"
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)
    for subset, role in ((dowel, "dowel"), (ring, "ring")):
        axes[1, 1].plot([r[1] for r in subset], [r[4] / r[3] for r in subset], label=role)
    axes[1, 1].set(
        title=f"Depth retained after {edge_limit_mm:g} mm local-span check",
        ylabel="Fraction",
        xlabel="Demo frame",
        ylim=(0, 1.02),
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--support-radius-mm", type=float, default=3.0)
    parser.add_argument("--edge-limit-mm", type=float, default=10.0)
    args = parser.parse_args()
    if args.support_radius_mm <= 0 or args.edge_limit_mm <= 0:
        parser.error("support radius and edge limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    traces = np.load(Path(__file__).with_name("pose_traces.npz"))
    intrinsics = traces["intrinsics"]
    rows: list[tuple] = []
    models = {}
    for role in ROLES:
        first, confirmed, raw_count = build_models(
            args.dataset,
            args.masks,
            role,
            intrinsics,
            traces[f"p2p_{role}_delta"],
            traces[f"p2p_{role}_lost"],
            args.support_radius_mm / 1e3,
        )
        models[role] = (first, confirmed)
        print(role, "source count", raw_count, "first view", len(first), "confirmed", len(confirmed))

    for role in ROLES:
        first, confirmed = models[role]
        for frame in SCORE_FRAMES:
            index = frame - SOURCE_FRAMES[0]
            if traces[f"p2p_{role}_lost"][index]:
                continue
            raw, raw_count = frame_points(args.dataset, args.masks, role, frame, intrinsics)
            interior, interior_count = frame_points(
                args.dataset,
                args.masks,
                role,
                frame,
                intrinsics,
                edge_limit_m=args.edge_limit_mm / 1e3,
            )
            p2p = traces[f"p2p_{role}_delta"][index]
            for candidate, pose in (
                ("p2p", p2p),
                ("bundle", traces[f"bundlesdf_{role}_delta"][index]),
            ):
                if candidate == "bundle" and traces[f"bundlesdf_{role}_failed"][index]:
                    continue
                rows.append(
                    (
                        role,
                        frame,
                        candidate,
                        raw_count,
                        interior_count,
                        surface_p90_mm(raw, move(first, pose)),
                        surface_p90_mm(raw, move(confirmed, pose)),
                        surface_p90_mm(interior, move(first, pose)),
                        surface_p90_mm(interior, move(confirmed, pose)),
                    )
                )
    write_csv(
        args.output / "multiview_depth_ablation.csv",
        (
            "role",
            "frame",
            "candidate",
            "raw_depth_points",
            "interior_depth_points",
            "first_view_raw_p90_mm",
            "confirmed_raw_p90_mm",
            "first_view_interior_p90_mm",
            "confirmed_interior_p90_mm",
        ),
        rows,
    )
    plot_scores(args.output / "multiview_depth_ablation.png", rows, args.edge_limit_mm)
    for role in ROLES:
        for candidate in ("p2p", "bundle"):
            values = np.asarray([r[8] for r in rows if r[0] == role and r[2] == candidate])
            print(
                role,
                candidate,
                "interior+confirmed median/p90 mm",
                np.median(values),
                np.quantile(values, 0.9),
            )


if __name__ == "__main__":
    main()
