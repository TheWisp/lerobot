"""Cache causal SAM3 masks for every recorded RGB-D frame, including empty results."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.appearance import Sam3MaskSource
from lerobot.policies.flash.perception import RGBDFrame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--resolution", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    intrinsics = np.loadtxt(args.recording / "cam_K.txt").reshape(3, 3)
    timestamps = np.atleast_1d(np.loadtxt(args.recording / "times.txt"))
    source = Sam3MaskSource(
        {"object": args.prompt}, camera_id="mask-depth-offline", resolution=args.resolution
    )
    cv2.setNumThreads(1)
    with (args.output / "timing.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("frame", "mask_pixels", "sam_ms"))
        for index, timestamp in enumerate(timestamps):
            rgb = cv2.imread(str(args.recording / "rgb" / f"{index:06d}.jpg"))
            depth = cv2.imread(str(args.recording / "depth" / f"{index:06d}.png"), -1)
            if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
                raise ValueError(f"missing aligned RGB-D at frame {index}")
            frame = RGBDFrame(
                index,
                cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                depth,
                intrinsics,
                0.001,
                np.eye(4),
                float(timestamp),
                float(timestamp),
                "recorded",
            )
            start = time.perf_counter()
            masks = source.masks(frame)
            elapsed = (time.perf_counter() - start) * 1000
            mask = masks.get("object", np.zeros(depth.shape, dtype=bool))
            cv2.imwrite(str(args.output / f"{index:06d}.png"), mask.astype(np.uint8) * 255)
            writer.writerow((index, int(mask.sum()), elapsed))
            if index % 100 == 0:
                stream.flush()
                print(index, int(mask.sum()), f"{elapsed:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
