"""Offline TrackEverything bridge from a FLASH demo view to a recorded trial.

This deliberately treats the demo frame and the trial frames as one video. It
measures partial surface correspondence, without asking the tracker to identify
an annotated tip or assuming that all query points survive the cut.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch


def _image(data: bytes, mode: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), mode)
    if image is None:
        raise ValueError("recorded image could not be decoded")
    return image


def _queries(mask: np.ndarray, depth_mm: np.ndarray, count: int) -> np.ndarray:
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    valid = interior & (depth_mm > 0)
    ys, xs = np.nonzero(valid)
    if len(xs) < count:
        raise ValueError("too few depth-backed pixels inside source mask")
    side = int(np.ceil(np.sqrt(count)))
    chosen: set[int] = set()
    for y in np.linspace(ys.min(), ys.max(), side):
        for x in np.linspace(xs.min(), xs.max(), side):
            chosen.add(int(np.argmin((xs - x) ** 2 + (ys - y) ** 2)))
    return np.column_stack((xs[sorted(chosen)], ys[sorted(chosen)])).astype(np.float32)[:count]


def _project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (k[0, 0] * points[:, 0] / points[:, 2] + k[0, 2], k[1, 1] * points[:, 1] / points[:, 2] + k[1, 2])
    )


def _live_masks(root: Path, frames: list[int], shape: tuple[int, int], role: str) -> dict[int, np.ndarray]:
    episode = [json.loads(line) for line in (root / "flash/v1/episode_000000.jsonl").open()]
    frames_by_camera: dict[int, list[int]] = {}
    for frame in frames:
        frames_by_camera.setdefault(episode[frame]["camera_frame_id"], []).append(frame)
    masks = {}
    for line in (root / "flash/v1/tracker_episode_000000.jsonl").open():
        record = json.loads(line)
        boundary = record.get("observed_masks", {}).get(role, {})
        camera_id = boundary.get("frame_id")
        if camera_id in frames_by_camera:
            mask = np.zeros(shape, np.uint8)
            polygon = np.asarray(boundary["boundary_xy"], np.int32)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                continue
            cv2.fillPoly(mask, [polygon], 1)
            for frame in frames_by_camera[camera_id]:
                masks[frame] = mask.astype(bool)
    return masks


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_center, dst_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - src_center).T @ (target - dst_center))
    fix = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ fix @ u.T
    return rotation, dst_center - rotation @ src_center


def _rigid_consensus(source: np.ndarray, target: np.ndarray, threshold_m: float, trials: int) -> dict:
    if len(source) < 3:
        return {"inliers": 0, "inlier_fraction": 0.0, "median_residual_mm": None}
    rng = np.random.default_rng(0)
    best = np.zeros(len(source), dtype=bool)
    for _ in range(trials):
        sample = rng.choice(len(source), 3, replace=False)
        if np.linalg.matrix_rank(source[sample] - source[sample].mean(axis=0)) < 2:
            continue
        rotation, translation = _rigid_fit(source[sample], target[sample])
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        candidate = residual <= threshold_m
        if candidate.sum() > best.sum():
            best = candidate
    if best.sum() >= 3:
        rotation, translation = _rigid_fit(source[best], target[best])
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        best = residual <= threshold_m
        return {
            "inliers": int(best.sum()),
            "inlier_fraction": float(best.mean()),
            "median_residual_mm": float(np.median(residual[best]) * 1000) if best.any() else None,
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
        }
    return {"inliers": 0, "inlier_fraction": 0.0, "median_residual_mm": None}


def run(args: argparse.Namespace) -> dict:
    if args.query_count < 3 or args.fit_trials < 1 or args.live_frames < 1:
        raise ValueError("query count, fit trials, and live span must be positive")
    demo_rgb = _image(
        (args.demo_recording / "rgb" / f"{args.demo_frame:06d}.jpg").read_bytes(), cv2.IMREAD_COLOR
    )
    demo_depth = _image(
        (args.demo_recording / "depth" / f"{args.demo_frame:06d}.png").read_bytes(), cv2.IMREAD_UNCHANGED
    )
    source_mask = (
        _image((args.demo_masks / f"{args.demo_frame:06d}.png").read_bytes(), cv2.IMREAD_GRAYSCALE) > 0
    )
    query_xy = _queries(source_mask, demo_depth, args.query_count)
    table = pq.read_table(
        args.live_root / "data/chunk-000/file-000.parquet",
        columns=["observation.images.overhead", "observation.images.overhead_depth"],
    )
    frames = list(range(args.live_frame, args.live_frame + args.live_frames))
    if frames[-1] >= len(table):
        raise ValueError("live span exceeds recording")
    rgbs = [cv2.cvtColor(demo_rgb, cv2.COLOR_BGR2RGB)]
    depths = [demo_depth.astype(np.float32) * 0.001]
    for frame in frames:
        rgb = _image(table["observation.images.overhead"][frame].as_py()["bytes"], cv2.IMREAD_COLOR)
        depth = _image(
            table["observation.images.overhead_depth"][frame].as_py()["bytes"], cv2.IMREAD_UNCHANGED
        )
        rgbs.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depths.append(depth.astype(np.float32) * 0.001)
    k = np.loadtxt(args.demo_recording / "cam_K.txt").reshape(3, 3)
    query = np.column_stack((np.zeros(len(query_xy)), query_xy))
    sys.path.insert(0, str(args.model_repo.parent))
    from trackeverything.demo_hf import TrackEverything  # noqa: PLC0415

    model = TrackEverything.from_pretrained(str(args.checkpoint), device="cuda")
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.infer(
            video=np.stack(rgbs),
            depths=np.stack(depths),
            intrinsics=np.repeat(k[None], len(rgbs), axis=0),
            extrinsics=np.repeat(np.eye(4)[None], len(rgbs), axis=0),
            query_points_2d=query,
            mode="sparse",
        )
    elapsed = time.perf_counter() - start
    points = out.query_coords[0].float().cpu().numpy() * out.norm_factor
    visibility = out.query_visibs[0].float().cpu().numpy()
    source = points[0]
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "tracks.npz", query_xy=query_xy, points=points, visibility=visibility)
    results = []
    live_masks = _live_masks(args.live_root, frames, depths[1].shape, args.role)
    for index, frame in enumerate(frames, 1):
        mask = live_masks.get(frame)
        if mask is None:
            continue
        predicted = points[index]
        pixel = _project(predicted, k)
        xy = np.rint(pixel).astype(int)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
        mask_hit = np.zeros(len(xy), dtype=bool)
        surface_error = np.full(len(xy), np.nan)
        mask_hit[inside] = mask[xy[inside, 1], xy[inside, 0]]
        observed_depth = depths[index][xy[inside, 1], xy[inside, 0]]
        surface_error[inside] = np.abs(observed_depth - predicted[inside, 2])
        supported = (visibility[index] > 0) & mask_hit & (surface_error <= args.surface_threshold_m)
        consensus = _rigid_consensus(
            source[supported], predicted[supported], args.rigid_threshold_m, args.fit_trials
        )
        result = {
            "live_frame": frame,
            "visibility_positive": int((visibility[index] > 0).sum()),
            "mask_hit": int(mask_hit.sum()),
            "mask_and_depth_supported": int(supported.sum()),
            "median_surface_error_mm_on_mask": float(np.nanmedian(surface_error[mask_hit]) * 1000)
            if mask_hit.any()
            else None,
            "rigid_consensus": consensus,
        }
        results.append(result)
        overlay = cv2.cvtColor(rgbs[index], cv2.COLOR_RGB2BGR)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        for j, (x, y) in enumerate(xy):
            if inside[j]:
                color = (0, 255, 0) if supported[j] else (255, 0, 255) if mask_hit[j] else (255, 255, 0)
                cv2.circle(overlay, (int(x), int(y)), 2, color, -1)
        cv2.imwrite(str(args.output / f"live-{frame:06d}.jpg"), overlay)
    report = {
        "demo_frame": args.demo_frame,
        "role": args.role,
        "live_start": args.live_frame,
        "query_count": len(query),
        "model_seconds": elapsed,
        "live_frames_evaluated": results,
        "thresholds_m": {"surface": args.surface_threshold_m, "rigid": args.rigid_threshold_m},
        "fit_trials": args.fit_trials,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-recording", type=Path, required=True)
    parser.add_argument("--demo-masks", type=Path, required=True)
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--demo-frame", type=int, required=True)
    parser.add_argument("--live-frame", type=int, required=True)
    parser.add_argument("--live-frames", type=int, default=15)
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surface-threshold-m", type=float, default=0.01)
    parser.add_argument("--rigid-threshold-m", type=float, default=0.01)
    parser.add_argument("--fit-trials", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args()
    parsed.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(parsed), indent=2))
