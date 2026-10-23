"""Point4D RGB-video baseline projected onto frozen RGB-D point queries.

Point4D predicts its own scene scale and geometry. This adapter scores only
projected 2D point identity; its 3D output is deliberately not compared with
metric sensor-depth candidates or fed to the common metric rigid fitter.
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


def project_model_tracks(
    trajectory: np.ndarray, geometry: dict, source_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Project frame-0 world tracks with Point4D's own per-frame cameras."""
    t, n, _ = trajectory.shape
    rotation = geometry["cam_R"]
    translation = geometry["cam_t"]
    intrinsics = geometry["intrinsics"]
    if rotation.shape != (t, 3, 3) or translation.shape != (t, 3) or intrinsics.shape != (t, 3, 3):
        raise ValueError("Point4D geometry has wrong camera dimensions")
    camera = np.einsum("tnj,tjk->tnk", trajectory - translation[:, None, :], rotation)
    uvw = np.einsum("tnj,tkj->tnk", camera, intrinsics)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[..., :2] / uvw[..., 2:3]
    model_h, model_w = geometry["image_hw"]
    source_h, source_w = source_hw
    xy = (uv + 0.5) * [source_w / model_w, source_h / model_h] - 0.5
    inside = (
        np.isfinite(xy).all(axis=-1)
        & np.isfinite(camera).all(axis=-1)
        & (camera[..., 2] > 0)
        & (xy[..., 0] >= 0)
        & (xy[..., 0] < source_w)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] < source_h)
    )
    if xy.shape != (t, n, 2):
        raise ValueError("Point4D projection lost query dimensions")
    return xy.astype(np.float32), inside


def run(experiment: Path, repository: Path, checkpoint: Path, output: Path) -> dict:
    manifest = json.loads((experiment / "manifest.json").read_text())
    query_path = experiment / "queries.npz"
    if manifest["schema"] != "flash-visible-points-v1" or _digest(query_path) != manifest["queries_sha256"]:
        raise ValueError("benchmark query manifest changed")
    if not (repository / "point4d" / "inference" / "longtrack.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Point4D source or checkpoint missing")
    with np.load(query_path, allow_pickle=False) as saved:
        frames, queries = saved["frames"], saved["queries"]
    if np.any(queries[:, 0] != 0):
        raise ValueError("Point4D released long-video API only accepts first-frame queries")
    recording = Path(manifest["recording"])
    if _digest(recording / "rgb" / f"{frames[0]:06d}.jpg") != manifest["first_rgb_sha256"]:
        raise ValueError("source RGB changed")
    sys.path.insert(0, str(repository))
    from point4d.inference import LongTrackConfig, track_long_video  # noqa: PLC0415
    from point4d.loader import load_model  # noqa: PLC0415

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = load_model(checkpoint_path=str(checkpoint), device="cuda")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    images = []
    for frame in frames:
        bgr = cv2.imread(str(recording / "rgb" / f"{frame:06d}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"missing RGB at frame {frame}")
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    images = np.stack(images)
    source_hw = images.shape[1:3]
    decode_s = time.perf_counter() - started
    config = LongTrackConfig()
    model_h, model_w = config.image_hw
    source_h, source_w = source_hw
    query_uv = (queries[:, 1:3] + 0.5) * [model_w / source_w, model_h / source_h] - 0.5
    started = time.perf_counter()
    with torch.inference_mode():
        trajectory, confidence, geometry = track_long_video(
            model, images, query_uv.astype(np.float32), config, return_geometry=True
        )
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - started
    if trajectory.shape != (len(frames), len(queries), 3) or confidence.shape != (len(frames), len(queries)):
        raise ValueError("Point4D result lost source frame or query identity")
    geometry["image_hw"] = config.image_hw
    xy, in_view = project_model_tracks(trajectory, geometry, source_hw)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=frames,
        queries=queries,
        xy=xy,
        visible=in_view,
        confidence=confidence.astype(np.float32),
        model_xyz_unscaled=trajectory.astype(np.float32),
        geometry_source=np.asarray("model_3d_unscaled_projected_2d"),
    )
    source_error = np.linalg.norm(xy[0] - queries[:, 1:3], axis=1)
    report = {
        "model": "Point4D",
        "mode": "RGB full-clip model with predicted camera geometry; in-view, not occlusion visibility",
        "frames": len(frames),
        "queries": len(queries),
        "repository": str(repository.resolve()),
        "checkpoint_sha256": _digest(checkpoint),
        "hardware": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "config": vars(config),
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.repository, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
