"""Compare identical Point2Pose fitting with SAM2 versus causal SAM3 masks.

The two pose streams share the same frame-100 mask, RGB-D, fitter and config.
The saved same-mask adapter replay must reproduce the internal SAM2 stream
exactly before comparing the alternative per-frame masks.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from analyze_multiview import build_models, frame_points, move, surface_p90_mm

ROLES = ("dowel", "ring")


def projected_axis_error_deg(
    mask: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray, center: np.ndarray, axis: np.ndarray
) -> tuple[float, float]:
    y, x = np.nonzero(mask)
    if len(x) < 3:
        return float("nan"), float("nan")
    eig, basis = np.linalg.eigh(np.cov(np.column_stack((x, y)).T))
    image_axis = basis[:, -1]
    pixels = []
    for distance in (-0.03, 0.03):
        point = move((center + distance * axis)[None, :], pose)[0]
        if point[2] <= 0:
            return float("nan"), float(eig[-1] / max(eig[-2], np.finfo(float).eps))
        pixels.append(
            (
                intrinsics[0, 0] * point[0] / point[2] + intrinsics[0, 2],
                intrinsics[1, 1] * point[1] / point[2] + intrinsics[1, 2],
            )
        )
    direction = np.subtract(*pixels[::-1])
    direction /= np.linalg.norm(direction)
    error = np.degrees(np.arccos(np.clip(abs(direction @ image_axis), -1, 1)))
    return float(error), float(eig[-1] / max(eig[-2], np.finfo(float).eps))


def projection_figure(
    dataset: Path,
    masks: Path,
    output: Path,
    intrinsics: np.ndarray,
    traces: dict[str, np.lib.npyio.NpzFile],
    center: np.ndarray,
    axis: np.ndarray,
    ends: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 5), constrained_layout=True)
    for panel, frame in zip(axes, (400, 450), strict=True):
        bgr = cv2.imread(str(dataset / "rgb" / f"{frame:06d}.jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(masks / "dowel" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        y, x = np.nonzero(mask)
        x0, x1 = max(0, x.min() - 75), min(rgb.shape[1], x.max() + 75)
        y0, y1 = max(0, y.min() - 55), min(rgb.shape[0], y.max() + 55)
        panel.imshow(rgb[y0:y1, x0:x1])
        panel.contour(mask[y0:y1, x0:x1], levels=[0.5], colors="cyan", linewidths=1)
        for backend, color in (("sam2", "#ffa31a"), ("sam3", "#ec3f91")):
            pose = traces[backend]["poses"][frame - 100]
            pixels = []
            for end in ends:
                point = move((center + end * axis)[None, :], pose)[0]
                pixels.append(
                    (
                        intrinsics[0, 0] * point[0] / point[2] + intrinsics[0, 2] - x0,
                        intrinsics[1, 1] * point[1] / point[2] + intrinsics[1, 2] - y0,
                    )
                )
            panel.plot(
                [p[0] for p in pixels],
                [p[1] for p in pixels],
                color=color,
                linewidth=2,
                label=backend.upper(),
            )
            panel.scatter([p[0] for p in pixels], [p[1] for p in pixels], color=color, s=22)
        panel.set_title(f"Frame {frame}: same RGB-D, different mask history")
        panel.axis("off")
    axes[0].legend(loc="lower left", fontsize=8)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def cross_score_dowel_models(
    dataset: Path,
    masks: Path,
    intrinsics: np.ndarray,
    traces: dict[str, np.lib.npyio.NpzFile],
    edge_limit_m: float,
    support_radius_m: float,
) -> None:
    """Check whether the alternate pose can validate itself with its own earlier-view map."""
    models = {
        backend: build_models(
            dataset, masks, "dowel", intrinsics, trace["poses"], trace["lost"], support_radius_m
        )[1]
        for backend, trace in traces.items()
    }
    for model_backend, model in models.items():
        for pose_backend, trace in traces.items():
            scores = []
            for frame in range(400, 501):
                index = frame - 100
                if trace["lost"][index]:
                    continue
                live, _ = frame_points(dataset, masks, "dowel", frame, intrinsics, edge_limit_m=edge_limit_m)
                scores.append(surface_p90_mm(live, move(model, trace["poses"][index])))
            print(
                "dowel multiview map",
                model_backend,
                "pose",
                pose_backend,
                "insertion median surface p90 mm",
                round(float(np.median(scores)), 2),
                "map points",
                len(model),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path, help="saved causal SAM3 masks")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--edge-limit-mm", type=float, default=10.0)
    parser.add_argument("--map-support-radius-mm", type=float, default=3.0)
    args = parser.parse_args()
    if args.edge_limit_mm <= 0 or args.map_support_radius_mm <= 0:
        parser.error("edge limit and map support radius must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parent
    with (
        np.load(root / "p2p_dowel_internal_control_100_550.npz") as internal,
        np.load(root / "p2p_dowel_same_mask_control_100_550.npz") as adapted,
    ):
        for key in ("poses", "lost", "fit_points", "fit_inliers", "mask_pixels"):
            if not np.array_equal(internal[key], adapted[key]):
                raise ValueError(f"same-mask adapter failed to reproduce {key}")
    print("same-mask adapter replay: all 451 poses, flags and fit counts identical")
    traces = {
        "dowel": {
            "sam2": np.load(root / "p2p_dowel_internal_control_100_550.npz"),
            "sam3": np.load(root / "p2p_dowel_sam3_100_550.npz"),
        },
        "ring": {
            "sam2": np.load(root / "p2p_ring_sam2_100_550.npz"),
            "sam3": np.load(root / "p2p_ring_sam3_100_550.npz"),
        },
    }
    features = np.load(root / "pose_traces.npz")
    intrinsics = features["intrinsics"]
    center = features["dowel_reference_center_camera_m"]
    axis = features["dowel_reference_shaft_camera"]
    ends = features["dowel_reference_end_offsets_m"]
    rows = []
    for role in ROLES:
        first, _ = frame_points(args.dataset, args.masks, role, 100, intrinsics, max_points=4000)
        for frame in range(100, 551):
            live, live_count = frame_points(
                args.dataset,
                args.masks,
                role,
                frame,
                intrinsics,
                edge_limit_m=args.edge_limit_mm / 1000,
            )
            mask = cv2.imread(str(args.masks / role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
            for backend, trace in traces[role].items():
                index = frame - 100
                lost = bool(trace["lost"][index])
                pose = trace["poses"][index]
                axis_error, elongation = (
                    projected_axis_error_deg(mask, pose, intrinsics, center, axis)
                    if role == "dowel" and not lost
                    else (float("nan"), float("nan"))
                )
                rows.append(
                    (
                        role,
                        frame,
                        backend,
                        lost,
                        int(trace["fit_inliers"][index]),
                        live_count,
                        surface_p90_mm(live, move(first, pose)) if not lost else float("nan"),
                        axis_error,
                        elongation,
                    )
                )
    with (args.output / "p2p_mask_ablation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "role",
                "frame",
                "mask_backend",
                "lost",
                "fit_inliers",
                "interior_depth_points",
                "first_view_surface_p90_mm",
                "sam3_mask_axis_error_deg",
                "sam3_mask_elongation",
            )
        )
        writer.writerows(rows)
    projection_figure(
        args.dataset,
        args.masks,
        args.output / "p2p_mask_pose_projection.png",
        intrinsics,
        traces["dowel"],
        center,
        axis,
        ends,
    )
    for role in ROLES:
        print(role)
        for backend in ("sam2", "sam3"):
            subset = [row for row in rows if row[0] == role and 400 <= row[1] <= 500 and row[2] == backend]
            scores = [row[6] for row in subset if not row[3]]
            print(
                backend,
                "insertion poses",
                len(scores),
                "median surface p90 mm",
                round(float(np.median(scores)), 2),
                "median projected axis error deg",
                round(float(np.nanmedian([row[7] for row in subset])), 2) if role == "dowel" else "n/a",
            )
    cross_score_dowel_models(
        args.dataset,
        args.masks,
        intrinsics,
        traces["dowel"],
        args.edge_limit_mm / 1000,
        args.map_support_radius_mm / 1000,
    )


if __name__ == "__main__":
    main()
