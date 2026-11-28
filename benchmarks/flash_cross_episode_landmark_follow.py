"""Follow a cross-episode landmark seed through adjacent live RGB-D frames.

Use the two-frame result from flash_cross_episode_landmark.py as --seed-result.
This does not certify physical point identity after occlusion or self-symmetry.
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
from flash_cross_episode_trackeverything import _live_masks, _project


def run(args: argparse.Namespace) -> dict:
    if args.frame_count < 1:
        raise ValueError("frame count must be positive")
    sys.path.insert(0, str(args.model_repo.parent))
    from TrackEverything.demo_hf import TrackEverything  # noqa: PLC0415

    seed = (
        json.loads(args.seed_result.read_text())["direct_prediction_xy"] if args.seed_result else args.seed_xy
    )
    table = pq.read_table(
        args.live_root / "data/chunk-000/file-000.parquet",
        columns=["observation.images.overhead", "observation.images.overhead_depth"],
    )
    frames = list(range(args.live_frame, args.live_frame + args.frame_count))
    if frames[-1] >= len(table):
        raise ValueError("requested frame span exceeds recording")
    rgb, depths = [], []
    for frame in frames:
        color = cv2.imdecode(
            np.frombuffer(table["observation.images.overhead"][frame].as_py()["bytes"], np.uint8),
            cv2.IMREAD_COLOR,
        )
        depth = cv2.imdecode(
            np.frombuffer(table["observation.images.overhead_depth"][frame].as_py()["bytes"], np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if color is None or depth is None:
            raise ValueError(f"could not decode recorded frame {frame}")
        rgb.append(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        depths.append(depth.astype(np.float32) * 0.001)
    k = np.loadtxt(args.demo_recording / "cam_K.txt").reshape(3, 3)
    model = TrackEverything.from_pretrained(str(args.checkpoint), device=args.device)
    with torch.inference_mode():
        output = model.infer(
            video=np.stack(rgb),
            depths=np.stack(depths),
            intrinsics=np.repeat(k[None], len(frames), axis=0),
            extrinsics=np.repeat(np.eye(4)[None], len(frames), axis=0),
            query_points_2d=np.array([[0, *seed]], dtype=np.float32),
            mode="sparse",
        )
    points = output.query_coords[0].float().cpu().numpy() * output.norm_factor
    visibility = output.query_visibs[0].float().cpu().numpy()
    pixels = _project(points[:, 0], k)
    masks = _live_masks(args.live_root, frames, depths[0].shape, args.role)
    evidence = []
    for index, frame in enumerate(frames):
        mask = masks.get(frame)
        if mask is None:
            continue
        x, y = np.rint(pixels[index]).astype(int)
        in_image = 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]
        mask_hit = bool(mask[y, x]) if in_image else False
        surface_error = abs(depths[index][y, x] - points[index, 0, 2]) * 1000 if in_image else None
        evidence.append(
            {
                "live_frame": frame,
                "prediction_xy": pixels[index].tolist(),
                "visibility_logit": float(visibility[index, 0]),
                "mask_hit": mask_hit,
                "surface_depth_error_mm": float(surface_error) if surface_error is not None else None,
            }
        )
    supported = [
        row
        for row in evidence
        if row["mask_hit"] and row["surface_depth_error_mm"] <= args.surface_tolerance_m * 1000
    ]
    report = {
        "seed_live_xy": seed,
        "live_frame_start": args.live_frame,
        "frame_count": len(frames),
        "exact_camera_frame_masks": len(evidence),
        "mask_hits": sum(row["mask_hit"] for row in evidence),
        "mask_and_depth_supported": len(supported),
        "surface_tolerance_m": args.surface_tolerance_m,
        "positive_visibility": sum(row["visibility_logit"] > 0 for row in evidence),
        "caveat": "Surface agreement cannot establish that a reappearing prediction has retained material-point identity.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "landmark-track.npz", xyz=points, visibility=visibility, pixel=pixels)
    (args.output / "frame-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-recording", type=Path, required=True, help="source of calibrated camera intrinsics"
    )
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--live-frame", type=int, default=0)
    parser.add_argument("--frame-count", type=int, required=True)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--seed-result", type=Path)
    seeds.add_argument("--seed-xy", type=float, nargs=2)
    parser.add_argument("--role", default="dowel")
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surface-tolerance-m", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
