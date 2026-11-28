"""SpaTrackerV2 RGB-D full-clip adapter for frozen visible-surface queries.

The official RGB-D example loads an RGB-only VGGT front model before choosing
the RGB-D branch. This adapter calls the same checkpoint/predictor directly so
that unrelated model load is excluded from the RGB-D measurement.
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


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(experiment: Path, repository: Path, output: Path, *, iterations: int = 4) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    manifest = json.loads((experiment / "manifest.json").read_text())
    query_path = experiment / "queries.npz"
    if manifest["schema"] != "flash-visible-points-v1" or _digest(query_path) != manifest["queries_sha256"]:
        raise ValueError("benchmark query manifest changed")
    if not (repository / "models" / "SpaTrackV2" / "models" / "predictor.py").is_file():
        raise FileNotFoundError("SpaTrackerV2 source missing")
    with np.load(query_path, allow_pickle=False) as saved:
        frames = saved["frames"]
        queries = saved["queries"]
        intrinsics = saved["intrinsics"].astype(np.float32)
    if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
        raise ValueError("SpaTrackerV2 adapter requires consecutive source frames")
    if np.any(queries[:, 0] < 0) or np.any(queries[:, 0] >= len(frames)):
        raise ValueError("query birth is outside the source clip")
    recording = Path(manifest["recording"])
    if _digest(recording / "rgb" / f"{frames[0]:06d}.jpg") != manifest["first_rgb_sha256"]:
        raise ValueError("source RGB changed")
    sys.path.insert(0, str(repository))
    from models.SpaTrackV2.models.predictor import Predictor  # noqa: PLC0415

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    model.eval()
    # Upstream Predictor.to() does not return self.
    model.to("cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started

    started = time.perf_counter()
    rgb_frames, depth_frames = [], []
    for frame in frames:
        bgr = cv2.imread(str(recording / "rgb" / f"{frame:06d}.jpg"), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None or raw.dtype != np.uint16 or raw.shape != bgr.shape[:2]:
            raise ValueError(f"unaligned RGB-D at frame {frame}")
        rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        depth_frames.append(raw.astype(np.float32) * 0.001)
    video = torch.from_numpy(np.stack(rgb_frames)).permute(0, 3, 1, 2).float()
    depth = np.stack(depth_frames)
    intrs = np.repeat(intrinsics[None], len(frames), axis=0)
    extrs = np.repeat(np.eye(4, dtype=np.float32)[None], len(frames), axis=0)
    decode_s = time.perf_counter() - started

    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        c2w, _, _, _, track3d, track2d, visibility, confidence, _ = model.forward(
            video,
            depth=depth,
            intrs=intrs,
            extrs=extrs,
            queries=queries,
            fps=1,
            # Upstream's default false drops externally supplied queries at
            # depth edges and pads random duplicates into their slots. That
            # destroys the frozen query identity contract of this benchmark.
            full_point=True,
            iters_track=iterations,
            query_no_BA=True,
            fixed_cam=False,
            stage=1,
            unc_metric=None,
            support_frame=len(frames) - 1,
            replace_ratio=0.2,
        )
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - started
    xy = track2d[..., :2].float().cpu().numpy()
    xyz = torch.einsum("tij,tnj->tni", c2w[:, :3, :3], track3d[..., :3].cpu())
    xyz += c2w[:, None, :3, 3]
    xyz = xyz.float().numpy()
    visible = visibility.squeeze(-1).float().cpu().numpy() > 0.5
    confident = confidence.squeeze(-1).float().cpu().numpy()
    if xy.shape != (len(frames), len(queries), 2) or xyz.shape != (len(frames), len(queries), 3):
        raise ValueError(f"SpaTrackerV2 result shape mismatch: {xy.shape}, {xyz.shape}")
    visible &= np.isfinite(xy).all(axis=-1) & np.isfinite(xyz).all(axis=-1)
    visible &= np.arange(len(frames))[:, None] >= queries[:, 0][None]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=frames,
        queries=queries,
        xy=xy.astype(np.float32),
        xyz=xyz.astype(np.float32),
        visible=visible,
        confidence=confident.astype(np.float32),
        geometry_source=np.asarray("model_3d"),
    )
    first = queries[:, 0].astype(int)
    source_error = np.linalg.norm(xy[first, np.arange(len(queries))] - queries[:, 1:3], axis=1)
    report = {
        "model": "SpaTrackerV2-Offline",
        "mode": "full-clip windowed RGB-D model; no causal publication rate measured",
        "frames": len(frames),
        "queries": len(queries),
        "iterations": iterations,
        "repository": str(repository.resolve()),
        "checkpoint_revision": "76e275b00f9c57dab71d46544df5255d4538106d",
        "hardware": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "model_load_s": load_s,
        "decode_s_total": decode_s,
        "tracker_s_total": infer_s,
        "model_only_fps": len(frames) / infer_s,
        "decode_track_lift_fps": len(frames) / (decode_s + infer_s),
        "query_birth_error_px_median": float(np.median(source_error)),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    output.with_suffix(".timing.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(run(args.experiment, args.repository, args.output, iterations=args.iterations), indent=2)
    )


if __name__ == "__main__":
    main()
