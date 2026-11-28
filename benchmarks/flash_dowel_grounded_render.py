"""Render a common-mask/depth point-and-pose comparison for the held dowel clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_surface_benchmark import _load, lift_measured_depth, validate_prediction
from lerobot.policies.flash.registration import transform_points
from lerobot.policies.flash.visible_points import surface_support
from lerobot.policies.flash.visible_rigid_tracking import (
    VisibleRGBDFrame,
    fit_visible_trace,
    visible_surface_cloud,
)

NAMES = ("LiteTracker", "CoTracker3", "Optical flow", "BootsTAPIR", "TrackEverything", "TAPNext")
FILES = ("litetracker", "cotracker3", "lk", "bootstapir", "trackeverything", "tapnext")
PANEL_W, PANEL_H = 340, 430
PAD = 12
CROP = (148, 50, 318, 265)  # same world-camera crop for every panel and frame


def project(points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 0)
    xy = np.full((len(points), 2), np.nan)
    xy[valid] = points[valid, :2] / z[valid, None]
    xy[valid] = xy[valid] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    return xy, valid


def draw_panel(
    bgr: np.ndarray,
    mask: np.ndarray,
    k: np.ndarray,
    xy: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
    pose: np.ndarray | None,
    source_surface: np.ndarray,
    basis: np.ndarray,
    name: str,
) -> np.ndarray:
    crop_x0, crop_y0, crop_x1, crop_y1 = CROP
    view = bgr.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (255, 255, 255), 2)
    if pose is not None:
        cloud = transform_points(pose, source_surface)
        projected, finite = project(cloud, k)
        footprint = projected[finite]
        footprint = footprint[
            (footprint[:, 0] >= crop_x0)
            & (footprint[:, 0] < crop_x1)
            & (footprint[:, 1] >= crop_y0)
            & (footprint[:, 1] < crop_y1)
        ]
        if len(footprint) >= 3:
            hull = cv2.convexHull(np.rint(footprint).astype(np.int32))
            cv2.polylines(view, [hull], True, (255, 255, 0), 2)
        center = source_surface.mean(axis=0)
        origin = transform_points(pose, center[None])[0]
        ends = transform_points(pose, center[None] + basis * 0.025)
        axis_pixels, axis_valid = project(np.vstack((origin, ends)), k)
        if axis_valid.all():
            start = tuple(np.rint(axis_pixels[0]).astype(int))
            for i, color in enumerate(((60, 60, 255), (70, 225, 70), (255, 100, 80))):
                cv2.arrowedLine(
                    view,
                    start,
                    tuple(np.rint(axis_pixels[i + 1]).astype(int)),
                    color,
                    2,
                    tipLength=0.25,
                )
    for index, position in enumerate(xy):
        if not visible[index] or not np.isfinite(position).all():
            continue
        x, y = np.rint(position).astype(int)
        if not (crop_x0 <= x < crop_x1 and crop_y0 <= y < crop_y1):
            continue
        cv2.circle(view, (x, y), 2, (65, 230, 80) if supported[index] else (60, 65, 255), -1)
    body = view[crop_y0:crop_y1, crop_x0:crop_x1]
    body = cv2.resize(body, (PANEL_W, PANEL_H - 48), interpolation=cv2.INTER_CUBIC)
    panel = np.full((PANEL_H, PANEL_W, 3), (24, 24, 24), dtype=np.uint8)
    panel[48:] = body
    cv2.putText(panel, name, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    if pose is None:
        cv2.putText(
            panel, "POSE LOST", (120, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 255), 2, cv2.LINE_AA
        )
    return panel


def render(experiment: Path, output: Path, still: Path) -> None:
    manifest = json.loads((experiment / "manifest.json").read_text())
    reference = _load(experiment / "queries.npz")
    frames, k = reference["frames"], reference["intrinsics"]
    bgrs, depths, masks, surfaces, rgbd = [], [], [], [], []
    for frame in frames:
        bgr = cv2.imread(str(Path(manifest["recording"]) / "rgb" / f"{frame:06d}.jpg"))
        depth_mm = cv2.imread(str(Path(manifest["recording"]) / "depth" / f"{frame:06d}.png"), -1)
        mask = cv2.imread(str(Path(manifest["mask_dir"]) / f"{frame:06d}.png"), 0) > 0
        depth = depth_mm.astype(float) * 0.001
        surface = visible_surface_cloud(depth, mask, k, edge_pixels=2, max_points=1000)
        bgrs.append(bgr)
        depths.append(depth)
        masks.append(mask)
        surfaces.append(surface)
        rgbd.append(VisibleRGBDFrame(surface, mask, depth, k))
    source = surfaces[0]
    _, _, principal = np.linalg.svd(source - source.mean(axis=0), full_matrices=False)
    basis = principal.copy()
    if np.linalg.det(basis) < 0:
        basis[-1] *= -1
    tracks = {}
    for label, filename in zip(NAMES, FILES, strict=True):
        prediction = _load(experiment / f"{filename}.npz")
        validate_prediction(reference, prediction)
        xyz = np.stack([lift_measured_depth(prediction["xy"][i], depths[i], k) for i in range(len(frames))])
        supported = np.stack(
            [
                surface_support(xyz[i], prediction["visible"][i].astype(float), masks[i], depths[i], k, 0.01)[
                    0
                ]
                for i in range(len(frames))
            ]
        )
        fit = fit_visible_trace(
            xyz,
            supported,
            tolerance_m=0.005,
            min_inliers=12,
            trials=100,
            reference_surface=source,
            rgbd_frames=rgbd,
            depth_m_per_unit=1.0,
            surface_tolerance_m=0.01,
            icp_iterations=20,
            trim_fraction=0.75,
        )
        tracks[label] = prediction, supported, fit
    output.parent.mkdir(parents=True, exist_ok=True)
    still.parent.mkdir(parents=True, exist_ok=True)
    mosaic_w = 3 * PANEL_W + 4 * PAD
    mosaic_h = 2 * PANEL_H + 3 * PAD + 44
    stream = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (mosaic_w, mosaic_h))
    if not stream.isOpened():
        raise RuntimeError("could not open video writer")
    last_frame = None
    try:
        for index, frame in enumerate(frames):
            canvas = np.full((mosaic_h, mosaic_w, 3), (15, 17, 19), dtype=np.uint8)
            for cell, name in enumerate(NAMES):
                prediction, supported, fit = tracks[name]
                pose = None if fit["lost"][index] else fit["poses"][index]
                panel = draw_panel(
                    bgrs[index],
                    masks[index],
                    k,
                    prediction["xy"][index],
                    prediction["visible"][index],
                    supported[index],
                    pose,
                    source,
                    basis,
                    name,
                )
                col, row = cell % 3, cell // 3
                x, y = PAD + col * (PANEL_W + PAD), PAD + row * (PANEL_H + PAD)
                canvas[y : y + PANEL_H, x : x + PANEL_W] = panel
            legend = "WHITE mask   CYAN source footprint   GREEN accepted   RED rejected"
            cv2.putText(
                canvas,
                legend,
                (18, mosaic_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"frame {frame}",
                (mosaic_w - 145, mosaic_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            stream.write(canvas)
            last_frame = canvas
    finally:
        stream.release()
    if last_frame is None:
        raise ValueError("empty interval")
    cv2.imwrite(str(still), last_frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--still", type=Path, required=True)
    args = parser.parse_args()
    render(args.experiment, args.output, args.still)


if __name__ == "__main__":
    main()
