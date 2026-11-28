"""Track-On-R streaming point adapter for frozen FLASH RGB-D query clips.

Run in an isolated environment with official Track-On-R, DINOv3 and a CUDA
compatible MMCV build. The published 2D locations are lifted with aligned
sensor depth for separate 3D scoring; no object mask corrects predictions.
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
from benchmarks.flash_surface_pacing import PacedConsumer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(experiment: Path, repository: Path, checkpoint: Path, output: Path, paced: bool = False) -> dict:
    manifest = json.loads((experiment / "manifest.json").read_text())
    query_path = experiment / "queries.npz"
    if manifest["schema"] != "flash-visible-points-v1" or _sha256(query_path) != manifest["queries_sha256"]:
        raise ValueError("benchmark query manifest changed")
    if not (repository / "model" / "trackon_predictor.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Track-On-R source or checkpoint missing")
    with np.load(query_path, allow_pickle=False) as saved:
        frames, queries, k, capture_s = (
            saved["frames"],
            saved["queries"],
            saved["intrinsics"],
            saved["capture_s"],
        )
    if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
        raise ValueError("Track-On-R adapter requires consecutive source frames")
    if paced and np.any(queries[:, 0] != 0):
        raise ValueError("paced latest-frame replay cannot preserve queries born on skipped frames")
    recording = Path(manifest["recording"])
    if _sha256(recording / "rgb" / f"{frames[0]:06d}.jpg") != manifest["first_rgb_sha256"]:
        raise ValueError("source RGB changed")
    sys.path.insert(0, str(repository))
    from model.trackon_predictor import Predictor  # noqa: PLC0415

    started = time.perf_counter()
    model = Predictor(checkpoint_path=str(checkpoint), support_grid_size=0).cuda().eval()
    model.reset()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - started
    t, n = len(frames), len(queries)
    xy = np.full((t, n, 2), np.nan, np.float32)
    xyz = np.full((t, n, 3), np.nan, np.float32)
    visible = np.zeros((t, n), bool)
    decode_s, inference_s, lift_s = (np.zeros(t) for _ in range(3))
    clock = PacedConsumer(capture_s) if paced else None
    index = 0
    with torch.inference_mode():
        while index < t:
            if clock is not None:
                index = clock.next_source(index)
            frame = frames[index]
            started = time.perf_counter()
            bgr = cv2.imread(str(recording / "rgb" / f"{frame:06d}.jpg"), cv2.IMREAD_COLOR)
            raw = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
            if bgr is None or raw is None or raw.dtype != np.uint16 or raw.shape != bgr.shape[:2]:
                raise ValueError(f"bad aligned RGB-D at frame {frame}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb).permute(2, 0, 1).float()[None].cuda()
            decode_s[index] = time.perf_counter() - started
            born = np.flatnonzero(queries[:, 0] == index)
            new = torch.from_numpy(queries[born, 1:3]).float().cuda() if len(born) else None
            started = time.perf_counter()
            coords, is_visible = model.forward_frame(image, new_queries=new)
            torch.cuda.synchronize()
            inference_s[index] = time.perf_counter() - started
            known = np.flatnonzero(queries[:, 0] <= index)
            if coords.shape != (len(known), 2) or is_visible.shape != (len(known),):
                raise ValueError("Track-On-R returned wrong point identities")
            xy[index, known] = coords.float().cpu().numpy()
            visible[index, known] = is_visible.cpu().numpy()
            started = time.perf_counter()
            xyz[index, known] = lift_measured_depth(xy[index, known], raw.astype(np.float32) * 0.001, k)
            lift_s[index] = time.perf_counter() - started
            if clock is not None:
                clock.publish(index)
            index += 1
    pacing = clock.finish() if clock is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=frames,
        queries=queries,
        xy=xy,
        xyz=xyz,
        visible=visible,
        geometry_source=np.asarray("sensor_at_prediction"),
        decode_s=decode_s,
        tracker_s=inference_s,
        lift_s=lift_s,
        **(
            {
                "publication_source_index": np.asarray(pacing["publication_source_index"]),
                "publication_s": np.asarray(pacing["publication_s"]),
                "consumer_schedule_s": np.asarray(pacing["consumer_schedule_s"]),
                "consumer_actual_s": np.asarray(pacing["consumer_actual_s"]),
                "consumer_source_index": np.asarray(pacing["consumer_source_index"]),
            }
            if pacing is not None
            else {}
        ),
    )
    processed = int(np.count_nonzero(inference_s))
    report = {
        "model": "Track-On-R",
        "mode": "paced causal streaming" if paced else "causal streaming, all source frames",
        "frames": t,
        "processed": processed,
        "queries": n,
        "checkpoint_sha256": _sha256(checkpoint),
        "repository": str(repository.resolve()),
        "hardware": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "model_load_s": load_s,
        "decode_s_total": float(decode_s.sum()),
        "tracker_s_total": float(inference_s.sum()),
        "lift_s_total": float(lift_s.sum()),
        "model_only_fps": processed / inference_s.sum(),
        "decode_track_lift_fps": processed / (decode_s.sum() + inference_s.sum() + lift_s.sum()),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "pacing": pacing,
    }
    output.with_suffix(".timing.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paced", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(run(args.experiment, args.repository, args.checkpoint, args.output, args.paced), indent=2)
    )


if __name__ == "__main__":
    main()
