"""Replay a camera debugger's aligned RGB-D/mask trace through the shared FLASH backend.

Run with the same UFM-capable Python environment used by the live debugger.
The first saved frame defines the local reference; start capture when selecting
the object if pose numbers must match a live session exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.perception_step import PerceptionPipeline
from lerobot.policies.flash.surface_tracking import load_ufm
from lerobot.policies.flash.tracking_debug_worker import debug_reference
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend


class RecordedMasks:
    def __init__(self, trace: Path, roles: tuple[str, ...]):
        self.trace = trace
        self.roles = roles

    def masks(self, frame: RGBDFrame) -> dict[str, np.ndarray]:
        return {
            role: cv2.imread(str(self.trace / "masks" / role / f"{frame.frame_id:06d}.png"), 0) > 0
            for role in self.roles
        }


def load_frames(trace: Path):
    rows = [json.loads(line) for line in (trace / "frames.jsonl").read_text().splitlines()]
    for row in rows:
        stem = f"{row['frame_id']:06d}.png"
        rgb = cv2.cvtColor(cv2.imread(str(trace / "rgb" / stem)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(trace / "depth" / stem), -1)
        yield (
            row,
            RGBDFrame(
                row["frame_id"],
                rgb,
                depth,
                np.asarray(row["intrinsics"]),
                row["depth_m_per_unit"],
                np.eye(4),
                row["source_time_s"],
                row["source_time_s"],
                row["calibration_version"],
            ),
        )


def replay(trace: Path) -> list[dict]:
    frames = list(load_frames(trace))
    if not frames:
        raise ValueError("diagnostic trace has no frames")
    roles = tuple(frames[0][0]["objects"])
    config = VisionPipelineConfig.from_dict(
        json.loads((trace / "config.json").read_text())["vision_pipeline"]
    )
    backend = build_vision_backend(
        config, RecordedMasks(trace, roles), matcher=load_ufm(), local_reference=True
    )
    pipeline = PerceptionPipeline(
        backend, tuple(debug_reference(role, frames[0][1]) for role in roles), require_masks=True
    )
    result = []
    for row, frame in frames:
        step = pipeline.process(frame)
        for role in roles:
            detail = step.details[role]
            result.append(
                {
                    "frame_id": frame.frame_id,
                    "role": role,
                    "status": detail["status"],
                    "accepted": detail["accepted"],
                    "surface_p90_mm": detail["diagnostics"].get("surface_p90_mm"),
                    "inliers": detail["diagnostics"].get("inliers"),
                    "camera_from_reference": detail.get("camera_from_reference"),
                    "recorded": row["objects"][role],
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = replay(args.trace)
    encoded = json.dumps(rows, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
