"""Probe whether a demo pixel landmark transfers to one recorded live frame.

This is an offline diagnostic, not a FLASH runtime estimator. A mask-derived
endpoint is only a geometric reference: it cannot identify which physical end
of a uniform dowel is which after the scene has been reset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from flash_cross_episode_trackeverything import _live_masks, _project, _rigid_consensus


def _decode(data: bytes, mode: int) -> np.ndarray:
    result = cv2.imdecode(np.frombuffer(data, np.uint8), mode)
    if result is None:
        raise ValueError("could not decode recorded image")
    return result


def _mask_endpoints(mask: np.ndarray, tail_fraction: float) -> list[np.ndarray]:
    ys, xs = np.nonzero(mask)
    xy = np.column_stack((xs, ys))
    if len(xy) < 3:
        raise ValueError("live object mask is empty")
    center = xy.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov(xy.T))
    along = (xy - center) @ axes[:, -1]
    return [
        xy[along <= np.quantile(along, tail_fraction)].mean(axis=0),
        xy[along >= np.quantile(along, 1 - tail_fraction)].mean(axis=0),
    ]


def _mask_and_depth_support(
    predictions: np.ndarray,
    visibility: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    k: np.ndarray,
    tolerance_m: float,
) -> np.ndarray:
    pixels = np.rint(_project(predictions, k)).astype(int)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < mask.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < mask.shape[0])
    )
    supported = np.zeros(len(predictions), dtype=bool)
    x, y = pixels[inside, 0], pixels[inside, 1]
    supported[inside] = (
        (visibility[inside] > 0) & mask[y, x] & (np.abs(depth[y, x] - predictions[inside, 2]) <= tolerance_m)
    )
    return supported


def run(args: argparse.Namespace) -> dict:
    if not 0 < args.endpoint_tail_fraction < 0.5:
        raise ValueError("endpoint tail fraction must be between zero and one half")
    sys.path.insert(0, str(args.model_repo.parent))
    from TrackEverything.demo_hf import TrackEverything  # noqa: PLC0415

    demo_rgb = _decode(
        (args.demo_recording / "rgb" / f"{args.demo_frame:06d}.jpg").read_bytes(), cv2.IMREAD_COLOR
    )
    demo_depth = _decode(
        (args.demo_recording / "depth" / f"{args.demo_frame:06d}.png").read_bytes(), cv2.IMREAD_UNCHANGED
    )
    table = pq.read_table(
        args.live_root / "data/chunk-000/file-000.parquet",
        columns=["observation.images.overhead", "observation.images.overhead_depth"],
    )
    live_rgb = _decode(
        table["observation.images.overhead"][args.live_frame].as_py()["bytes"], cv2.IMREAD_COLOR
    )
    live_depth = _decode(
        table["observation.images.overhead_depth"][args.live_frame].as_py()["bytes"], cv2.IMREAD_UNCHANGED
    )
    k = np.loadtxt(args.demo_recording / "cam_K.txt").reshape(3, 3)
    model = TrackEverything.from_pretrained(str(args.checkpoint), device=args.device)
    with torch.inference_mode():
        output = model.infer(
            video=np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in (demo_rgb, live_rgb)]),
            depths=np.stack([demo_depth, live_depth]).astype(np.float32) * 0.001,
            intrinsics=np.stack([k, k]),
            extrinsics=np.stack([np.eye(4), np.eye(4)]),
            query_points_2d=np.array([[0, *args.landmark_xy]], dtype=np.float32),
            mode="sparse",
        )
    points = output.query_coords[0].float().cpu().numpy() * output.norm_factor
    visibility = output.query_visibs[0].float().cpu().numpy()
    predicted = points[1, 0]
    pixel = _project(predicted[None], k)[0]
    depth_m = live_depth.astype(np.float32) * 0.001
    mask = _live_masks(args.live_root, [args.live_frame], live_depth.shape, args.role)[args.live_frame]
    xi, yi = np.rint(pixel).astype(int)
    in_image = 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]
    ends = _mask_endpoints(mask, args.endpoint_tail_fraction)
    result = {
        "demo_frame": args.demo_frame,
        "demo_landmark_xy": args.landmark_xy,
        "live_frame": args.live_frame,
        "direct_prediction_xy": pixel.tolist(),
        "direct_prediction_depth_m": float(predicted[2]),
        "visibility_logit": float(visibility[1, 0]),
        "inside_live_mask": bool(mask[yi, xi]) if in_image else False,
        "live_surface_depth_error_mm": float(abs(depth_m[yi, xi] - predicted[2]) * 1000)
        if in_image
        else None,
        "visible_mask_endpoints_xy": [end.tolist() for end in ends],
        "nearest_visible_endpoint_error_px": float(min(np.linalg.norm(pixel - end) for end in ends)),
        "note": "Mask endpoints are approximate; neither the mask nor an unmarked dowel establishes physical end identity.",
    }
    if args.surface_tracks:
        tracks = np.load(args.surface_tracks)
        source, target, query_xy = tracks["points"][0], tracks["points"][1], tracks["query_xy"]
        supported = _mask_and_depth_support(
            target, tracks["visibility"][1], mask, depth_m, k, args.surface_tolerance_m
        )
        source_mask = _decode(args.demo_mask.read_bytes(), cv2.IMREAD_GRAYSCALE) > 0
        ys, xs = np.nonzero(source_mask)
        mask_xy = np.column_stack((xs, ys))
        center = mask_xy.mean(axis=0)
        _, axes = np.linalg.eigh(np.cov(mask_xy.T))
        axis = axes[:, -1]
        tip_direction = np.sign((np.asarray(args.landmark_xy) - center) @ axis)
        axial = (query_xy - center) @ axis * tip_direction
        extent = np.ptp((mask_xy - center) @ axis)
        # Hold out the end fifth, including the marked landmark, from the fit.
        supported &= axial <= 0.3 * extent
        fit = _rigid_consensus(source[supported], target[supported], args.rigid_tolerance_m, 1000)
        result["held_out_fit"] = {
            "supported_queries": int(supported.sum()),
            "inliers": fit["inliers"],
            "median_inlier_residual_mm": fit["median_residual_mm"],
        }
        if fit["inliers"] >= 3:
            source_depth = (
                float(
                    np.median(
                        demo_depth[
                            max(0, args.landmark_xy[1] - 2) : args.landmark_xy[1] + 3,
                            max(0, args.landmark_xy[0] - 2) : args.landmark_xy[0] + 3,
                        ]
                    )
                )
                * 0.001
            )
            x, y = args.landmark_xy
            landmark_xyz = np.array(
                [
                    (x - k[0, 2]) * source_depth / k[0, 0],
                    (y - k[1, 2]) * source_depth / k[1, 1],
                    source_depth,
                ]
            )
            rigid_xyz = np.asarray(fit["rotation"]) @ landmark_xyz + np.asarray(fit["translation_m"])
            rigid_xy = _project(rigid_xyz[None], k)[0]
            result["held_out_fit"].update(
                {
                    "prediction_xy": rigid_xy.tolist(),
                    "nearest_visible_endpoint_error_px": float(
                        min(np.linalg.norm(rigid_xy - end) for end in ends)
                    ),
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    overlay = live_rgb.copy()
    for end in ends:
        cv2.circle(overlay, tuple(np.rint(end).astype(int)), 7, (0, 180, 0), 2)
    cv2.circle(overlay, (xi, yi), 7, (0, 0, 255), 2)
    if "prediction_xy" in result.get("held_out_fit", {}):
        rigid_xy = result["held_out_fit"]["prediction_xy"]
        cv2.circle(overlay, tuple(np.rint(rigid_xy).astype(int)), 7, (255, 0, 0), 2)
    cv2.imwrite(str(args.output / "live-overlay.jpg"), overlay)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-recording", type=Path, required=True)
    parser.add_argument("--demo-mask", type=Path, required=True)
    parser.add_argument("--demo-frame", type=int, required=True)
    parser.add_argument("--landmark-xy", type=int, nargs=2, required=True)
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--live-frame", type=int, required=True)
    parser.add_argument("--role", default="dowel")
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surface-tracks", type=Path)
    parser.add_argument("--surface-tolerance-m", type=float, default=0.01)
    parser.add_argument("--rigid-tolerance-m", type=float, default=0.01)
    parser.add_argument("--endpoint-tail-fraction", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
