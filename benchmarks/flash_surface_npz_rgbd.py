"""Prepare an unlabelled calibrated RGB-D NPZ for visual point-tracker review.

Requires an explicit first-frame seed polygon and a declared playback FPS.
Missing source timestamps and per-point reference labels are not invented.
Later masks are empty; they are not claimed as segmentation evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_surface_benchmark import _digest, prepare_local


def build(source: Path, output: Path, polygon: list[int], fps: float, points: int) -> dict:
    if fps <= 0 or len(polygon) < 6 or len(polygon) % 2:
        raise ValueError("positive playback rate and at least three polygon vertices required")
    with np.load(source, allow_pickle=False) as saved:
        required = {"video", "depths", "intrinsics", "extrinsics"}
        if not required.issubset(saved.files):
            raise ValueError(f"RGB-D NPZ missing {sorted(required - set(saved.files))}")
        video, depth, intrinsics, extrinsics = (
            saved[key] for key in ("video", "depths", "intrinsics", "extrinsics")
        )
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8 or depth.shape != video.shape[:3]:
        raise ValueError("expected RGB uint8 [T,H,W,3] and depth metres [T,H,W]")
    if intrinsics.shape != (len(video), 3, 3) or extrinsics.shape != (len(video), 4, 4):
        raise ValueError("expected per-frame 3x3 intrinsics and 4x4 extrinsics")
    if not np.allclose(intrinsics, intrinsics[0]) or not np.allclose(extrinsics, extrinsics[0]):
        raise ValueError("single-calibration benchmark adapter requires a static camera")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("depth contains invalid negative or non-finite values")
    h, w = video.shape[1:3]
    vertices = np.asarray(polygon, int).reshape(-1, 2)
    if (
        np.any(vertices[:, 0] < 0)
        or np.any(vertices[:, 0] >= w)
        or np.any(vertices[:, 1] < 0)
        or np.any(vertices[:, 1] >= h)
    ):
        raise ValueError("initial seed polygon exceeds image")
    recording, masks = output / "recording", output / "masks"
    for directory in (recording / "rgb", recording / "depth", masks):
        directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(recording / "cam_K.txt", intrinsics[0], fmt="%.9f")
    np.savetxt(recording / "times.txt", np.arange(len(video)) / fps, fmt="%.9f")
    first_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(first_mask, [vertices.astype(np.int32)], 255)
    for index in range(len(video)):
        bgr = cv2.cvtColor(video[index], cv2.COLOR_RGB2BGR)
        valid = (depth[index] > 0) & (depth[index] < np.iinfo(np.uint16).max * 0.001)
        raw = np.zeros((h, w), np.uint16)
        raw[valid] = np.rint(depth[index, valid] * 1000).astype(np.uint16)
        cv2.imwrite(str(recording / "rgb" / f"{index:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(recording / "depth" / f"{index:06d}.png"), raw)
        cv2.imwrite(str(masks / f"{index:06d}.png"), first_mask if index == 0 else np.zeros((h, w), np.uint8))
    manifest = prepare_local(recording, masks, 0, len(video) - 1, points, output)
    manifest.update(
        {
            "source_npz": str(source.resolve()),
            "source_npz_sha256": _digest(source),
            "source_depth_unit": "float32 metres, rounded to uint16 millimetres for common adapters",
            "timestamp_provenance": f"no source timestamps; nominal playback rate explicitly supplied as {fps:g} fps",
            "initial_seed_polygon_xy": vertices.tolist(),
            "mask_scope": "manually supplied initial seed polygon only; later masks intentionally empty",
            "ground_truth": "unlabelled second public RGB-D source; no identity accuracy score",
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-polygon", nargs="+", type=int, required=True, metavar="XY")
    parser.add_argument("--playback-fps", type=float, required=True)
    parser.add_argument("--points", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source, args.output, args.seed_polygon, args.playback_fps, args.points), indent=2
        )
    )


if __name__ == "__main__":
    main()
