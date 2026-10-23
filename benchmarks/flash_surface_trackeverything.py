"""Run one frozen FLASH visible-surface query set through TrackEverything.

Run in the TrackEverything environment. This is full-clip, windowed inference;
its throughput is not a causal publication rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(experiment: Path, repository: Path, checkpoint: Path, output: Path) -> dict:
    manifest = json.loads((experiment / "manifest.json").read_text())
    if manifest["schema"] != "flash-visible-points-v1":
        raise ValueError("unsupported benchmark schema")
    if not (repository / "demo_hf.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("TrackEverything source or checkpoint missing")
    queries_path = experiment / "queries.npz"
    if _sha256(queries_path) != manifest["queries_sha256"]:
        raise ValueError("benchmark queries changed after manifest was frozen")
    with np.load(queries_path, allow_pickle=False) as data:
        frames, queries, intrinsics = data["frames"], data["queries"], data["intrinsics"]
    recording = Path(manifest["recording"])
    if _sha256(recording / "rgb" / f"{frames[0]:06d}.jpg") != manifest["first_rgb_sha256"]:
        raise ValueError("recording's initial RGB has changed")
    if _sha256(recording / "depth" / f"{frames[0]:06d}.png") != manifest["first_depth_sha256"]:
        raise ValueError("recording's initial depth has changed")
    start = time.perf_counter()
    rgbs, depths = [], []
    for frame in frames:
        bgr = cv2.imread(str(recording / "rgb" / f"{frame:06d}.jpg"), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None or bgr.shape[:2] != raw.shape or raw.dtype != np.uint16:
            raise ValueError(f"bad aligned RGB-D frame {frame}")
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        depths.append(raw.astype(np.float32) * 0.001)
    video = np.stack(rgbs)
    depth = np.stack(depths)
    decode_s = time.perf_counter() - start
    sys.path.insert(0, str(repository.parent))
    from trackeverything.demo_hf import TrackEverything  # noqa: PLC0415

    start = time.perf_counter()
    model = TrackEverything.from_pretrained(str(checkpoint), device="cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - start
    # The recorded overhead camera is static. No estimated geometry is injected.
    cameras = np.repeat(np.eye(4, dtype=np.float32)[None], len(frames), axis=0)
    intrinsics_per_frame = np.repeat(intrinsics.astype(np.float32)[None], len(frames), axis=0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        tracked = model.infer(
            video=video,
            depths=depth,
            intrinsics=intrinsics_per_frame,
            extrinsics=cameras,
            query_points_2d=queries,
            mode="dense",
        )
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - start
    start = time.perf_counter()
    xyz = tracked.query_coords[0].float().cpu().numpy() * tracked.norm_factor
    logits = tracked.query_visibs[0].float().cpu().numpy()
    if xyz.shape != (len(frames), len(queries), 3) or logits.shape != (len(frames), len(queries)):
        raise ValueError(f"unexpected output shapes: {xyz.shape}, {logits.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = xyz[..., :2] / xyz[..., 2:3]
    xy = xy * [intrinsics[0, 0], intrinsics[1, 1]] + [intrinsics[0, 2], intrinsics[1, 2]]
    extract_s = time.perf_counter() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    np.savez_compressed(
        output,
        frames=frames,
        queries=queries,
        xy=xy.astype(np.float32),
        xyz=xyz.astype(np.float32),
        visible=(logits > 0),
        visibility_logits=logits.astype(np.float32),
        geometry_source=np.asarray("model_3d"),
    )
    serialize_s = time.perf_counter() - start
    report = {
        "model": "TrackEverything",
        "mode": "full-clip, windowed, may use future frames",
        "frames": len(frames),
        "queries": len(queries),
        "checkpoint_sha256": _sha256(checkpoint),
        "repository": str(repository.resolve()),
        "prediction": str(output.resolve()),
        "hardware": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "pid": os.getpid(),
        "stage_s": {
            "decode_and_stack": decode_s,
            "model_load": load_s,
            "model_infer_cuda_synchronized": inference_s,
            "extract_and_project": extract_s,
            "serialize": serialize_s,
        },
        "unique_source_frame_fps_model_only": len(frames) / inference_s,
        "unique_source_frame_fps_decode_plus_model": len(frames) / (decode_s + inference_s),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "process_rss_bytes": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss
        * 1024,
    }
    output.with_suffix(".timing.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.repository, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
