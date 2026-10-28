"""Probe the shared visible-surface tracker on a regular external RGB-D episode.

Masks are supplied separately so a tracker test does not confuse segmentation
quality with pose quality. This produces diagnostic evidence only, never a skill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession, load_ufm


def probe(dataset_root: Path, mask_dir: Path, output: Path, *, mask_kind: str) -> dict:
    dataset_root = dataset_root.expanduser().resolve()
    source = json.loads((dataset_root / "flash" / "v1" / "source.json").read_text())
    intrinsics = np.asarray(source["camera_intrinsics"], dtype=np.float64)
    dataset = LeRobotDataset(source["repo_id"], root=dataset_root)
    session = SurfaceTrackingSession(load_ufm())
    rows = []
    for index in range(dataset.num_frames):
        row = dataset[index]
        rgb = (row["observation.images.rgb"].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        depth = row["observation.images.depth"].numpy().squeeze().astype(np.float32) * 0.001
        mask = cv2.imread(str(mask_dir / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != depth.shape:
            raise ValueError(f"missing or misaligned mask for frame {index}")
        result = session.update(rgb, depth, mask != 0, intrinsics, index)
        rows.append(
            {
                "frame_index": index,
                "source_frame_index": int(row["flash.source_frame_index"].item()),
                "source_time_s": float(row["flash.source_time_s"].item()),
                "status": result.get("status"),
                "camera_from_reference": np.asarray(result["camera_from_reference"]).tolist()
                if result.get("camera_from_reference") is not None
                else None,
                "diagnostics": result.get("diagnostics", {}),
            }
        )
        if (index + 1) % 10 == 0:
            print(f"Tracked {index + 1}/{dataset.num_frames} external frames", flush=True)
    report = {
        "dataset_root": str(dataset_root),
        "mask_source": str(mask_dir),
        "mask_kind": mask_kind,
        "tracking": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("mask_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask-kind", required=True, help="provenance of the masks supplied to this run")
    args = parser.parse_args()
    report = probe(args.dataset_root, args.mask_dir, args.output)
    print(f"Saved {len(report['tracking'])} frames to {args.output}")


if __name__ == "__main__":
    main()
