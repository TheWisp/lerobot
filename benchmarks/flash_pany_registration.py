"""Score the published PANY RGB-D reference pipeline on saved FLASH data.

PANY is optional and remains in its own checkout. This adapter follows its
single-reference inference path and evaluates the resulting demo-to-live rigid
transform against the same raw masked depth used by flash_rgbd_registration.
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
from flash_cross_episode_trackeverything import _image, _live_masks, _project
from flash_rgbd_registration import apply, cloud, score, trimmed_icp


def run(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(args.pany_repo))
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from inference_utils.model import load_model  # noqa: PLC0415
    from inference_utils.pose_estimation import robust_umeyama  # noqa: PLC0415
    from inference_utils.utils import center_crop, crop_input, to_tensor  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from vggt.utils.geometry import (  # noqa: PLC0415
        depth_to_cam_coords_points,
        unproject_depth_map_to_point_map,
    )

    started = time.perf_counter()
    source_rgb = cv2.cvtColor(
        _image((args.demo_recording / "rgb" / f"{args.demo_frame:06d}.jpg").read_bytes(), cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    source_depth = _image(
        (args.demo_recording / "depth" / f"{args.demo_frame:06d}.png").read_bytes(), cv2.IMREAD_UNCHANGED
    )
    source_mask = (
        _image((args.demo_masks / f"{args.demo_frame:06d}.png").read_bytes(), cv2.IMREAD_GRAYSCALE) > 0
    )
    table = pq.read_table(
        args.live_root / "data/chunk-000/file-000.parquet",
        columns=["observation.images.overhead", "observation.images.overhead_depth"],
    )
    target_rgb = cv2.cvtColor(
        _image(table["observation.images.overhead"][args.live_frame].as_py()["bytes"], cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    target_depth = _image(
        table["observation.images.overhead_depth"][args.live_frame].as_py()["bytes"],
        cv2.IMREAD_UNCHANGED,
    )
    target_mask = _live_masks(args.live_root, [args.live_frame], target_depth.shape, args.role)[
        args.live_frame
    ]
    k = np.loadtxt(args.demo_recording / "cam_K.txt").reshape(3, 3)
    anchor_image, anchor_mask, anchor_depth, anchor_k, anchor_pose = center_crop(
        source_rgb,
        source_mask,
        source_depth.astype(np.float32) * 0.001,
        k,
        np.eye(4),
        target_image_shape=(518, 518),
    )
    query_image, query_mask, query_depth, query_k, valid = crop_input(
        target_rgb,
        target_mask,
        target_depth.astype(np.float32) * 0.001,
        k,
        target_size=(518, 518),
    )
    if not valid:
        raise ValueError("PANY query object mask is empty")
    anchor_points = unproject_depth_map_to_point_map(anchor_depth[None], anchor_pose[None], anchor_k[None])[0]
    images = (
        torch.stack([to_tensor(Image.fromarray(anchor_image.astype(np.uint8))), query_image])
        .unsqueeze(0)
        .to(args.device)
    )
    with initialize_config_dir(version_base=None, config_dir=str(args.pany_repo / "training")):
        config = compose(config_name="pany_model")
    prepared = time.perf_counter()
    model = load_model(config, str(args.checkpoint)).to(args.device).eval()
    loaded = time.perf_counter()
    with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16):
        tokens, patch_start = model.aggregator(images)
        point_maps, confidences = model.point_head(tokens, images, patch_start)
    inferred = time.perf_counter()
    points = point_maps.squeeze(0).float().cpu().numpy()
    confidence = confidences.squeeze(0).float().cpu().numpy()

    reference_mask = anchor_mask > 0
    reference_predicted = points[0][reference_mask]
    reference_measured = anchor_points[reference_mask]
    reference_confidence = confidence[0][reference_mask]
    valid = (
        np.isfinite(reference_predicted).all(axis=1)
        & np.isfinite(reference_measured).all(axis=1)
        & (reference_measured[:, 2] > 0)
    )
    scale, reference_rotation, reference_translation = robust_umeyama(
        reference_predicted[valid],
        reference_measured[valid],
        reference_confidence[valid],
        reference_confidence[valid],
        with_scaling=True,
    )
    query_aligned = scale * (points[1] @ reference_rotation.T) + reference_translation
    query_measured = depth_to_cam_coords_points(query_depth * query_mask, query_k)
    query_valid = (query_mask > 0) & (query_depth > 0)
    predicted = query_aligned[query_valid]
    measured = query_measured[query_valid]
    weights = confidence[1][query_valid]
    finite = np.isfinite(predicted).all(axis=1) & np.isfinite(measured).all(axis=1)
    _, rotation, translation = robust_umeyama(
        predicted[finite], measured[finite], weights[finite], weights[finite], with_scaling=False
    )
    fitted = time.perf_counter()
    source = cloud(source_depth, source_mask, k, args.edge_pixels)
    target = cloud(target_depth, target_mask, k, args.edge_pixels)
    raw_score = score(source, target, target_mask, target_depth, k, rotation, translation, args.proximity_m)
    refined_rotation, refined_translation = trimmed_icp(
        source,
        target,
        rotation,
        translation,
        iterations=args.icp_iterations,
        trim_fraction=args.trim_fraction,
    )
    refined_score = score(
        source,
        target,
        target_mask,
        target_depth,
        k,
        refined_rotation,
        refined_translation,
        args.proximity_m,
    )
    result = {
        "method": "PANY official RGB-D single-reference pointmap, then optional trimmed ICP",
        "demo_frame": args.demo_frame,
        "live_frame": args.live_frame,
        "target_role": args.role,
        "source_points": len(source),
        "target_points": len(target),
        "reference_scale": float(scale),
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "score": raw_score,
        "refined_rotation": refined_rotation.tolist(),
        "refined_translation_m": refined_translation.tolist(),
        "refined_score": refined_score,
        "timings_s": {
            "preprocess": prepared - started,
            "model_load": loaded - prepared,
            "model_inference": inferred - loaded,
            "reference_and_query_fit": fitted - inferred,
            "score_and_refine": time.perf_counter() - fitted,
            "total": time.perf_counter() - started,
        },
    }
    if args.landmark_xy is not None:
        x, y = args.landmark_xy
        z = source_depth[y, x] * 0.001
        if z > 0:
            point = np.array([(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z])
            result["landmark_before_xy"] = _project(apply(point[None], rotation, translation), k)[0].tolist()
            result["landmark_after_xy"] = _project(
                apply(point[None], refined_rotation, refined_translation), k
            )[0].tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-recording", type=Path, required=True)
    parser.add_argument("--demo-masks", type=Path, required=True)
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--pany-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--demo-frame", type=int, required=True)
    parser.add_argument("--live-frame", type=int, required=True)
    parser.add_argument("--landmark-xy", type=int, nargs=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--edge-pixels", type=int, default=1)
    parser.add_argument("--proximity-m", type=float, default=0.006)
    parser.add_argument("--icp-iterations", type=int, default=40)
    parser.add_argument("--trim-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    output = run(parser.parse_args())
    print(
        json.dumps(
            {
                "score": output["score"],
                "refined_score": output["refined_score"],
                "timings_s": output["timings_s"],
            },
            indent=2,
        )
    )
