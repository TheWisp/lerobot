"""Find a demonstrated object-pair configuration in a saved RGB-D trial.

This is an offline diagnostic, not a FLASH runtime policy. It fits one rigid
scene transform to two *labelled* measured clouds, so a candidate must explain
both objects at once. The cloud-distance radius is only a benchmark setting;
coverage is not a probability or a contact-success certificate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from flash_cross_episode_trackeverything import _image, _live_masks, _project, _rigid_fit
from flash_rgbd_registration import apply, cloud, pca_hypotheses
from scipy.spatial import cKDTree


def sample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    return points[np.linspace(0, len(points) - 1, max_points, dtype=int)]


def measured_cloud(
    depth: np.ndarray, mask: np.ndarray, k: np.ndarray, edge_pixels: int, max_points: int
) -> np.ndarray | None:
    try:
        return sample(cloud(depth, mask, k, edge_pixels), max_points)
    except ValueError:
        return None


def fit_pair(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    iterations: int,
    trim_fraction: float,
    proximity_m: float,
) -> list[dict]:
    labels = tuple(source)
    source_all = np.vstack([source[label] for label in labels])
    target_all = np.vstack([target[label] for label in labels])
    trees = {label: cKDTree(target[label]) for label in labels}
    candidates = []
    for index, (rotation, translation) in enumerate(pca_hypotheses(source_all, target_all)):
        for _ in range(iterations):
            kept_source = []
            kept_target = []
            for label in labels:
                distances, nearest = trees[label].query(
                    apply(source[label], rotation, translation), workers=1
                )
                keep = distances <= np.quantile(distances, trim_fraction)
                kept_source.append(source[label][keep])
                kept_target.append(target[label][nearest[keep]])
            next_rotation, next_translation = _rigid_fit(np.vstack(kept_source), np.vstack(kept_target))
            change = np.linalg.norm(next_rotation - rotation) + np.linalg.norm(next_translation - translation)
            rotation, translation = next_rotation, next_translation
            if change < 1e-6:
                break
        evidence = {}
        for label in labels:
            moved = apply(source[label], rotation, translation)
            forward = trees[label].query(moved, workers=1)[0]
            reverse = cKDTree(moved).query(target[label], workers=1)[0]
            evidence[label] = {
                "source_to_target_coverage": float(np.mean(forward < proximity_m)),
                "target_to_source_coverage": float(np.mean(reverse < proximity_m)),
                "source_to_target_median_mm": float(np.median(forward) * 1000),
                "target_to_source_median_mm": float(np.median(reverse) * 1000),
            }
        quality = min(
            min(
                item["source_to_target_coverage"],
                item["target_to_source_coverage"],
            )
            for item in evidence.values()
        )
        candidates.append(
            {
                "initializer": f"pca_{index}",
                "balanced_coverage": float(quality),
                "objects": evidence,
                "rotation": rotation.tolist(),
                "translation_m": translation.tolist(),
            }
        )
    candidates.sort(key=lambda candidate: candidate["balanced_coverage"], reverse=True)
    return candidates


def comparison_image(
    demo_rgb: np.ndarray,
    demo_masks: dict[str, np.ndarray],
    live_rgb: np.ndarray,
    live_masks: dict[str, np.ndarray],
    live_depth: np.ndarray,
    source: dict[str, np.ndarray],
    candidate: dict,
    k: np.ndarray,
    proximity_m: float,
    demo_frame: int,
    live_frame: int,
    output: Path,
) -> dict:
    demo = demo_rgb.copy()
    live = live_rgb.copy()
    diagnostic = live_rgb.copy()
    colors = {"held": (255, 0, 255), "target": (255, 255, 0)}
    pixel_evidence = {}
    for label, color in colors.items():
        for image, masks in ((demo, demo_masks), (live, live_masks), (diagnostic, live_masks)):
            contours, _ = cv2.findContours(
                masks[label].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, color, 2)
        moved = apply(
            source[label], np.asarray(candidate["rotation"]), np.asarray(candidate["translation_m"])
        )
        pixel = np.rint(_project(moved, k)).astype(int)
        x, y = pixel[:, 0], pixel[:, 1]
        inside = (moved[:, 2] > 0) & (x >= 0) & (x < live.shape[1]) & (y >= 0) & (y < live.shape[0])
        x_safe = np.clip(x, 0, live.shape[1] - 1)
        y_safe = np.clip(y, 0, live.shape[0] - 1)
        observed_m = live_depth[y_safe, x_safe].astype(np.float64) * 0.001
        has_depth = inside & (observed_m > 0)
        mask_hit = inside & live_masks[label][y_safe, x_safe]
        depth_delta = moved[:, 2] - observed_m
        depth_agree = has_depth & (np.abs(depth_delta) <= proximity_m)
        supported = mask_hit & depth_agree
        occluded = has_depth & (depth_delta > proximity_m)
        free_space = has_depth & (depth_delta < -proximity_m)
        pixel_evidence[label] = {
            "source_points": int(len(moved)),
            "projected_in_frame": int(inside.sum()),
            "inside_live_mask": int(mask_hit.sum()),
            "mask_and_depth_supported": int(supported.sum()),
            "mask_and_depth_supported_fraction": float(supported.mean()),
            "behind_measured_surface_possible_occlusion": int(occluded.sum()),
            "in_front_of_measured_surface_contradiction": int(free_space.sum()),
            "in_front_of_measured_surface_fraction": float(free_space.mean()),
            "outside_mask_with_matching_depth": int((inside & ~mask_hit & depth_agree).sum()),
            "no_live_depth": int((inside & ~has_depth).sum()),
        }
        # Only render points independently supported by this recorded RGB-D frame.
        # Surface agreement does not establish a persistent point correspondence.
        for px, py in pixel[supported][::4]:
            cv2.circle(live, (int(px), int(py)), 1, color, -1)
        # The separate diagnostic retains projections contradicted by depth.
        for selected, diagnostic_color in (
            (occluded, (150, 150, 150)),
            (free_space, (0, 0, 255)),
            (supported, color),
        ):
            for px, py in pixel[selected][::4]:
                cv2.circle(diagnostic, (int(px), int(py)), 1, diagnostic_color, -1)
    for image, title in (
        (demo, f"Demo frame {demo_frame}"),
        (live, f"Live frame {live_frame}: supported projections only"),
    ):
        cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(image, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    combined = cv2.hconcat([demo, live])
    cv2.imwrite(str(output), combined, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.rectangle(diagnostic, (0, 0), (diagnostic.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        diagnostic,
        "Red: in front of depth   Grey: possibly occluded",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.imwrite(
        str(output.with_name("projection_diagnostics.jpg")), diagnostic, [cv2.IMWRITE_JPEG_QUALITY, 88]
    )
    return pixel_evidence


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    k = np.loadtxt(args.camera_k or args.demo_recording / "cam_K.txt").reshape(3, 3)
    table = pq.read_table(
        args.live_root / "data/chunk-000/file-000.parquet",
        columns=["observation.images.overhead", "observation.images.overhead_depth"],
    )
    live_rgb = _image(
        table["observation.images.overhead"][args.live_frame].as_py()["bytes"], cv2.IMREAD_COLOR
    )
    live_depth = _image(
        table["observation.images.overhead_depth"][args.live_frame].as_py()["bytes"],
        cv2.IMREAD_UNCHANGED,
    )
    live_masks = {
        "held": _live_masks(args.live_root, [args.live_frame], live_depth.shape, args.held_role).get(
            args.live_frame
        ),
        "target": _live_masks(args.live_root, [args.live_frame], live_depth.shape, args.target_role).get(
            args.live_frame
        ),
    }
    live_clouds = {
        label: measured_cloud(live_depth, mask, k, args.edge_pixels, args.max_points)
        if mask is not None
        else None
        for label, mask in live_masks.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "live_frame": args.live_frame,
        "live_points": {
            label: len(points) if points is not None else 0 for label, points in live_clouds.items()
        },
        "settings": {
            "demo_start": args.demo_start,
            "demo_end": args.demo_end,
            "demo_stride": args.demo_stride,
            "edge_pixels": args.edge_pixels,
            "max_points": args.max_points,
            "icp_iterations": args.icp_iterations,
            "trim_fraction": args.trim_fraction,
            "proximity_m": args.proximity_m,
        },
        "frames": [],
    }
    if any(points is None for points in live_clouds.values()):
        result["status"] = "no_measured_object_pair"
    else:
        result["status"] = "measured_object_pair"
        for frame in range(args.demo_start, args.demo_end + 1, args.demo_stride):
            masks = {}
            for label, path in (("held", args.held_masks), ("target", args.target_masks)):
                mask_path = path / f"{frame:06d}.png"
                masks[label] = (
                    _image(mask_path.read_bytes(), cv2.IMREAD_GRAYSCALE) > 0 if mask_path.exists() else None
                )
            if any(mask is None for mask in masks.values()):
                result["frames"].append({"demo_frame": frame, "status": "missing_demo_mask"})
                continue
            depth_path = args.demo_recording / "depth" / f"{frame:06d}.png"
            if not depth_path.exists():
                result["frames"].append({"demo_frame": frame, "status": "missing_demo_depth"})
                continue
            depth = _image(depth_path.read_bytes(), cv2.IMREAD_UNCHANGED)
            source = {
                label: measured_cloud(depth, mask, k, args.edge_pixels, args.max_points)
                for label, mask in masks.items()
            }
            if any(points is None for points in source.values()):
                result["frames"].append({"demo_frame": frame, "status": "no_measured_object_pair"})
                continue
            candidates = fit_pair(
                source,
                live_clouds,
                iterations=args.icp_iterations,
                trim_fraction=args.trim_fraction,
                proximity_m=args.proximity_m,
            )
            result["frames"].append(
                {
                    "demo_frame": frame,
                    "status": "measured_object_pair",
                    "source_points": {label: len(points) for label, points in source.items()},
                    "candidates": candidates[: args.show_top],
                }
            )
        supported = [frame for frame in result["frames"] if frame["status"] == "measured_object_pair"]
        if supported:
            best = max(supported, key=lambda frame: frame["candidates"][0]["balanced_coverage"])
            result["best_demo_frame"] = best["demo_frame"]
            demo_rgb = _image(
                (args.demo_recording / "rgb" / f"{best['demo_frame']:06d}.jpg").read_bytes(),
                cv2.IMREAD_COLOR,
            )
            demo_depth = _image(
                (args.demo_recording / "depth" / f"{best['demo_frame']:06d}.png").read_bytes(),
                cv2.IMREAD_UNCHANGED,
            )
            demo_masks = {
                label: _image((path / f"{best['demo_frame']:06d}.png").read_bytes(), cv2.IMREAD_GRAYSCALE) > 0
                for label, path in (("held", args.held_masks), ("target", args.target_masks))
            }
            source = {
                label: cloud(demo_depth, mask, k, args.edge_pixels) for label, mask in demo_masks.items()
            }
            result["best_pixel_evidence"] = comparison_image(
                demo_rgb,
                demo_masks,
                live_rgb,
                live_masks,
                live_depth,
                source,
                best["candidates"][0],
                k,
                args.proximity_m,
                best["demo_frame"],
                args.live_frame,
                args.output / "comparison.jpg",
            )
    result["seconds"] = time.perf_counter() - start
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-recording", type=Path, required=True)
    parser.add_argument("--held-masks", type=Path, required=True)
    parser.add_argument("--target-masks", type=Path, required=True)
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--held-role", required=True)
    parser.add_argument("--target-role", required=True)
    parser.add_argument("--camera-k", type=Path)
    parser.add_argument("--live-frame", type=int, required=True)
    parser.add_argument("--demo-start", type=int, required=True)
    parser.add_argument("--demo-end", type=int, required=True)
    parser.add_argument("--demo-stride", type=int, default=5)
    parser.add_argument("--edge-pixels", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=700)
    parser.add_argument("--icp-iterations", type=int, default=25)
    parser.add_argument("--trim-fraction", type=float, default=0.8)
    parser.add_argument("--proximity-m", type=float, default=0.006)
    parser.add_argument("--show-top", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.demo_stride <= 0 or args.max_points < 3 or not 0 < args.trim_fraction <= 1:
        parser.error("stride, maximum points, and trim fraction must be valid")
    report = run(args)
    print(report["status"], report.get("best_demo_frame"), round(report["seconds"], 2))
