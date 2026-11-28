"""TAPIP3D full-clip RGB-D adapter for frozen FLASH point queries.

Uses the official checkpoint, sensor depths, calibrated intrinsics, identity
camera extrinsics for the static benchmark cameras, and the upstream inference
preprocessing. Its full-clip throughput is not a causal 30 Hz rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    experiment: Path,
    repository: Path,
    checkpoint: Path,
    output: Path,
    resolution_factor: float = 2.0,
    support_grid: int = 16,
    iterations: int = 6,
    depth_filter: str = "upstream",
) -> dict:
    if (
        resolution_factor <= 0
        or support_grid < 0
        or iterations < 1
        or depth_filter not in {"upstream", "none"}
    ):
        raise ValueError("bad TAPIP3D inference options")
    manifest = json.loads((experiment / "manifest.json").read_text())
    query_path = experiment / "queries.npz"
    if manifest["schema"] != "flash-visible-points-v1" or _sha256(query_path) != manifest["queries_sha256"]:
        raise ValueError("benchmark query manifest changed")
    if not (repository / "utils" / "inference_utils.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("TAPIP3D source or checkpoint missing")
    with np.load(query_path, allow_pickle=False) as saved:
        frames, queries, k, z = (
            saved["frames"],
            saved["queries"],
            saved["intrinsics"],
            saved["initial_depth_m"],
        )
    if np.any(queries[:, 0] < 0) or np.any(queries[:, 0] >= len(frames)):
        raise ValueError("TAPIP3D query birth is outside the source clip")
    recording = Path(manifest["recording"])
    if _sha256(recording / "rgb" / f"{frames[0]:06d}.jpg") != manifest["first_rgb_sha256"]:
        raise ValueError("source RGB changed")
    sys.path.insert(0, str(repository))
    from datasets.data_ops import _filter_one_depth  # noqa: PLC0415
    from utils.inference_utils import inference, load_model, resize_depth_bilinear  # noqa: PLC0415

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = load_model(str(checkpoint)).cuda()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    native_h, native_w = model.image_size
    target_h = int(native_h * np.sqrt(resolution_factor))
    target_w = int(native_w * np.sqrt(resolution_factor))
    started = time.perf_counter()
    rgbs, depths = [], []
    for frame in frames:
        bgr = cv2.imread(str(recording / "rgb" / f"{frame:06d}.jpg"), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None or raw.dtype != np.uint16 or raw.shape != bgr.shape[:2]:
            raise ValueError(f"unaligned RGB-D at frame {frame}")
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        depths.append(raw.astype(np.float32) * 0.001)
    video = np.stack(rgbs)
    depth = np.stack(depths)
    decode_s = time.perf_counter() - started
    source_h, source_w = video.shape[1:3]
    intrinsics = np.repeat(k.astype(np.float32)[None], len(frames), axis=0)
    intrinsics[:, 0, :] *= (target_w - 1) / (source_w - 1)
    intrinsics[:, 1, :] *= (target_h - 1) / (source_h - 1)
    started = time.perf_counter()
    resized_rgb = np.stack(
        [cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for rgb in video]
    )
    resized_depth = np.stack([resize_depth_bilinear(d, (target_w, target_h)) for d in depth])
    resize_s = time.perf_counter() - started
    started = time.perf_counter()
    filtered_depth = (
        np.stack(
            [_filter_one_depth(d, 0.08, 15, ki) for d, ki in zip(resized_depth, intrinsics, strict=True)]
        )
        if depth_filter == "upstream"
        else resized_depth
    )
    filter_s = time.perf_counter() - started
    started = time.perf_counter()
    rgb_t = torch.from_numpy(resized_rgb).permute(0, 3, 1, 2).float().div_(255).cuda()
    depth_t = torch.from_numpy(filtered_depth).float().cuda()
    k_t = torch.from_numpy(intrinsics).float().cuda()
    extrinsics_t = torch.eye(4, dtype=torch.float32, device="cuda").repeat(len(frames), 1, 1)
    # Each depth belongs to its query's birth frame. The static benchmark
    # camera makes camera and world coordinates coincide across births.
    source_xyz = np.column_stack(
        ((queries[:, 1] - k[0, 2]) * z / k[0, 0], (queries[:, 2] - k[1, 2]) * z / k[1, 1], z)
    )
    query_world = torch.from_numpy(np.column_stack((queries[:, 0], source_xyz))).float().cuda()
    transfer_and_queries_s = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        coords, is_visible = inference(
            model=model,
            video=rgb_t,
            depths=depth_t,
            intrinsics=k_t,
            extrinsics=extrinsics_t,
            query_point=query_world,
            num_iters=iterations,
            grid_size=support_grid,
        )
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - started
    xyz = coords.float().cpu().numpy()
    visible = is_visible.cpu().numpy().astype(bool)
    if xyz.shape != (len(frames), len(queries), 3) or visible.shape != (len(frames), len(queries)):
        raise ValueError(f"TAPIP3D result shape mismatch: {xyz.shape}, {visible.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = xyz[..., :2] / xyz[..., 2:3]
    xy = xy * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=frames,
        queries=queries,
        xy=xy.astype(np.float32),
        xyz=xyz.astype(np.float32),
        visible=visible,
        geometry_source=np.asarray("model_3d"),
    )
    report = {
        "model": "TAPIP3D",
        "mode": "full-clip model; no causal publication rate measured",
        "frames": len(frames),
        "queries": len(queries),
        "support_grid": support_grid,
        "iterations": iterations,
        "depth_filter": depth_filter,
        "resolution_factor": resolution_factor,
        "source_hw": [source_h, source_w],
        "inference_hw": [target_h, target_w],
        "checkpoint_sha256": _sha256(checkpoint),
        "repository": str(repository.resolve()),
        "hardware": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "stage_s": {
            "model_load": load_s,
            "decode": decode_s,
            "resize": resize_s,
            "depth_filter": filter_s,
            "transfer_and_queries": transfer_and_queries_s,
            "inference": infer_s,
        },
        "unique_source_frame_fps_model_only": len(frames) / infer_s,
        "unique_source_frame_fps_decode_plus_model": len(frames)
        / (decode_s + resize_s + filter_s + transfer_and_queries_s + infer_s),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    output.with_suffix(".timing.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-factor", type=float, default=2.0)
    parser.add_argument("--support-grid", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--depth-filter", choices=("upstream", "none"), default="upstream")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.experiment,
                args.repository,
                args.checkpoint,
                args.output,
                args.resolution_factor,
                args.support_grid,
                args.iterations,
                args.depth_filter,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
