"""CoTracker3 official offline RGB point baseline for frozen FLASH clips.

This model sees the full video and may use future frames. Its 2D output is
lifted with aligned sensor depth for separately labelled 3D scoring. It is
not a measured streaming publication rate.
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

from benchmarks.flash_surface_benchmark import lift_measured_depth


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(experiment: Path, repository: Path, checkpoint: Path, output: Path) -> dict:
    manifest = json.loads((experiment / "manifest.json").read_text())
    query_path = experiment / "queries.npz"
    if manifest["schema"] != "flash-visible-points-v1" or _sha256(query_path) != manifest["queries_sha256"]:
        raise ValueError("benchmark query manifest changed")
    if not (repository / "cotracker" / "predictor.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("CoTracker3 source or checkpoint missing")
    with np.load(query_path, allow_pickle=False) as saved:
        frames, queries, k = saved["frames"], saved["queries"], saved["intrinsics"]
    if np.any(queries[:, 0] < 0) or np.any(queries[:, 0] >= len(frames)):
        raise ValueError("query birth outside clip")
    recording = Path(manifest["recording"])
    if _sha256(recording / "rgb" / f"{frames[0]:06d}.jpg") != manifest["first_rgb_sha256"]:
        raise ValueError("source RGB changed")
    started = time.perf_counter()
    rgbs, depths = [], []
    for frame in frames:
        bgr = cv2.imread(str(recording / "rgb" / f"{frame:06d}.jpg"), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None or raw.dtype != np.uint16 or bgr.shape[:2] != raw.shape:
            raise ValueError(f"unaligned RGB-D at frame {frame}")
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        depths.append(raw.astype(np.float32) * 0.001)
    video = np.stack(rgbs)
    depth = np.stack(depths)
    decode_s = time.perf_counter() - started
    sys.path.insert(0, str(repository))
    from cotracker.predictor import CoTrackerPredictor  # noqa: PLC0415

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = CoTrackerPredictor(checkpoint=str(checkpoint)).cuda().eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    video_t = torch.from_numpy(video).permute(0, 3, 1, 2).float()[None].cuda()
    queries_t = torch.from_numpy(queries).float()[None].cuda()
    torch.cuda.synchronize()
    transfer_s = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        tracks, visibility = model(video_t, queries=queries_t, backward_tracking=False)
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - started
    xy = tracks[0].float().cpu().numpy()
    visible = visibility[0].cpu().numpy().astype(bool)
    if xy.shape != (len(frames), len(queries), 2) or visible.shape != (len(frames), len(queries)):
        raise ValueError("CoTracker3 result shape mismatch")
    visible[np.arange(len(frames))[:, None] < queries[:, 0]] = False
    started = time.perf_counter()
    xyz = np.array([lift_measured_depth(xy[i], depth[i], k) for i in range(len(frames))])
    lift_s = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=frames,
        queries=queries,
        xy=xy,
        xyz=xyz,
        visible=visible,
        geometry_source=np.asarray("sensor_at_prediction"),
    )
    report = {
        "model": "CoTracker3 scaled offline",
        "mode": "full-clip offline; may use future frames",
        "frames": len(frames),
        "queries": len(queries),
        "checkpoint_sha256": _sha256(checkpoint),
        "repository": str(repository.resolve()),
        "hardware": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "stage_s": {
            "decode": decode_s,
            "model_load": load_s,
            "transfer": transfer_s,
            "inference": infer_s,
            "lift": lift_s,
        },
        "unique_source_frame_fps_model_only": len(frames) / infer_s,
        "unique_source_frame_fps_decode_plus_model": len(frames) / (decode_s + transfer_s + infer_s + lift_s),
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
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.repository, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
