"""Ablate native 3D predictions with the same RGB point tracks lifted by sensor depth."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_surface_benchmark import _load, lift_measured_depth, validate_prediction


def run(experiment: Path, prediction_path: Path, output: Path) -> dict:
    manifest = json.loads((experiment / "manifest.json").read_text())
    reference, prediction = _load(experiment / "queries.npz"), _load(prediction_path)
    validate_prediction(reference, prediction)
    frames, xy = reference["frames"], prediction["xy"]
    intrinsics = reference["intrinsics"]
    recording = Path(manifest["recording"])
    xyz = np.full((*xy.shape[:2], 3), np.nan, np.float32)
    lift_s = np.zeros(len(frames))
    for index, frame in enumerate(frames):
        raw = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.shape != tuple(manifest["source_hw"]):
            raise ValueError(f"bad aligned metric depth at frame {frame}")
        started = time.perf_counter()
        xyz[index] = lift_measured_depth(xy[index], raw.astype(np.float32) * 0.001, intrinsics)
        lift_s[index] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=frames,
        queries=reference["queries"],
        xy=xy,
        xyz=xyz,
        visible=prediction["visible"],
        geometry_source=np.asarray("sensor_at_prediction"),
    )
    source_timing = json.loads(prediction_path.with_suffix(".timing.json").read_text())
    result = {
        **source_timing,
        "model": f"{source_timing['model']} + measured-depth ablation",
        "source_prediction": prediction_path.name,
        "geometry_source": "sensor_at_prediction",
        "lift_s_total": float(lift_s.sum()),
        "decode_track_lift_fps": len(frames)
        / (len(frames) / source_timing["decode_track_lift_fps"] + lift_s.sum()),
    }
    output.with_suffix(".timing.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.prediction, args.output), indent=2))


if __name__ == "__main__":
    main()
