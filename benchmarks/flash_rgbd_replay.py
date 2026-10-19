"""Replay recorded aligned RGB-D through FLASH's first appearance/pose backend.

This is an offline diagnostic, not a ground-truth accuracy benchmark. It keeps
the original temporal order for SAM3's mask tracker and reports why each pose
was accepted or lost. Example::

    PYTHONPATH=src python benchmarks/flash_rgbd_replay.py \
      /path/to/stage --prompt "wooden dowel" --reference-frame 100 \
      --start 100 --stop 320 --stride 5 --output /tmp/dowel.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.appearance import (
    AppearancePixelMatcher,
    ReferenceAppearanceFrame,
    Sam3MaskSource,
    build_appearance_model,
)
from lerobot.policies.flash.dino_features import DinoPatchExtractor
from lerobot.policies.flash.matched_rgbd import MatchedRGBDPoseBackend
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.registration import transform_points, valid_depth_pixels
from lerobot.policies.flash.trajectory import Evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path, help="directory containing rgb/, depth/, and cam_K.txt")
    parser.add_argument("--prompt", required=True, help="SAM3 text for one tracked object")
    parser.add_argument("--reference-frame", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True, help="inclusive last frame")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--role", default="object")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam-resolution", type=int, default=448)
    parser.add_argument("--depth-m-per-unit", type=float, default=0.001)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--similarity", type=float, default=0.75)
    parser.add_argument("--threshold-mm", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    if args.start < args.reference_frame or args.stop < args.start or args.stride < 1 or args.fps <= 0:
        parser.error("use reference-frame <= start <= stop, positive fps and stride")
    return args


def main() -> None:
    args = parse_args()
    intrinsics = np.loadtxt(args.stage / "cam_K.txt")

    def frame(index: int) -> RGBDFrame:
        rgb_path = args.stage / "rgb" / f"{index:06d}.jpg"
        depth_path = args.stage / "depth" / f"{index:06d}.png"
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None:
            raise FileNotFoundError(f"missing RGB or depth at frame {index}")
        return RGBDFrame(
            index,
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            depth,
            intrinsics,
            args.depth_m_per_unit,
            np.eye(4),
            index / args.fps,
            index / args.fps,
            "recorded",
        )

    mask_source = Sam3MaskSource(
        {args.role: args.prompt}, camera_id="recorded", device=args.device, resolution=args.sam_resolution
    )
    extractor = DinoPatchExtractor(device=args.device)
    reference_frame = frame(args.reference_frame)
    mask = mask_source.masks(reference_frame).get(args.role)
    if mask is None or not mask.any():
        raise ValueError(f"SAM3 found no {args.role!r} in reference frame {args.reference_frame}")
    snippet = ReferenceSnippet("recorded", 0, args.reference_frame, args.role, np.eye(4), "sam3", "recorded")
    reference = ObjectReference(args.role, "reference-v1", (snippet,))
    model = build_appearance_model(
        reference, (ReferenceAppearanceFrame(snippet, reference_frame, mask),), extractor
    )

    class MeasuredMasks:
        area = 0

        def masks(self, observation: RGBDFrame) -> dict[str, np.ndarray]:
            found = mask_source.masks(observation)
            self.area = int(found[args.role].sum()) if args.role in found else 0
            return found

    class MeasuredMatches:
        count = 0
        with_depth = 0

        def __init__(self, matcher: AppearancePixelMatcher) -> None:
            self.matcher = matcher

        def matches(self, observation: RGBDFrame):
            found = self.matcher.matches(observation)
            self.count = len(found[0].model_indices) if found else 0
            self.with_depth = (
                int(valid_depth_pixels(observation.depth, found[0].live_xy).sum()) if found else 0
            )
            return found

    measured_masks = MeasuredMasks()
    measured_matches = MeasuredMatches(
        AppearancePixelMatcher((model,), extractor, measured_masks, min_similarity=args.similarity)
    )
    backend = MatchedRGBDPoseBackend(
        (model.points,),
        measured_matches,
        threshold_m=args.threshold_mm / 1000,
        min_inliers=args.min_inliers,
        trials=args.trials,
    )
    backend.initialize((reference,))
    reference_center = model.points.points_object_m.mean(axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(args.start, args.stop + 1, args.stride):
        observation = frame(index)
        began = time.monotonic()
        (estimate,) = backend.update(observation)
        duration_ms = 1000 * (time.monotonic() - began)
        center = (
            transform_points(estimate.pose, reference_center[None])[0]
            if estimate.evidence == Evidence.OBSERVED
            else [float("nan")] * 3
        )
        rows.append(
            {
                "frame": index,
                "evidence": estimate.evidence.value,
                "mask_pixels": measured_masks.area,
                "matched_pixels": measured_matches.count,
                "depth_matches": measured_matches.with_depth,
                "center_x_m": center[0],
                "center_y_m": center[1],
                "center_z_m": center[2],
                "duration_ms": round(duration_ms, 2),
                "failure": backend.last_failure.get(args.role, ""),
            }
        )
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    observed = sum(row["evidence"] == Evidence.OBSERVED.value for row in rows)
    print(
        f"{observed}/{len(rows)} observed; median pose-update {np.median([r['duration_ms'] for r in rows]):.1f} ms; "
        f"{len(model.descriptors)} reference patches; {args.output}"
    )


if __name__ == "__main__":
    main()
