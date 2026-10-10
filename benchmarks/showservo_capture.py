# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Capture RGB + aligned depth scenes from a RealSense, for the showservo real bench.

One capture per Enter press. Each scene lands as its own directory:

    <out>/intrinsics.json                  fx/fy/cx/cy of the color stream (written once)
    <out>/scene_000/rgb.png                color frame, HxWx3
    <out>/scene_000/depth.npy              aligned depth, HxW float32 METRES (0 = no data)
    <out>/scene_000/preview.jpg            rgb + depth colormap side by side, to eyeball

Depth is the median over several frames: cheap, and it removes single-frame speckle
without inventing anything. Two camera filters are deliberately DISABLED against their
library defaults:

* decimation — halves depth resolution AFTER alignment, silently breaking the
  pixel-for-pixel correspondence with color that deprojection depends on;
* hole filling — invents depth exactly at object boundaries, which is the flying-pixel
  class the pipeline's nearest-neighbour sampling exists to avoid. A missing value is
  honest; a fabricated one is not.

Usage (find the serial with `lerobot-find-cameras realsense`):
    python benchmarks/showservo_capture.py --serial <SN> --out captures/ring_session
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def capture_session(serial: str, out: pathlib.Path, *, fps: int, width: int, height: int) -> None:
    import cv2

    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    config = RealSenseCameraConfig(
        serial_number_or_name=serial,
        fps=fps,
        width=width,
        height=height,
        use_depth=True,
        enable_decimation=False,  # keep depth on the color pixel grid
        enable_hole_filling=False,  # a hole is honest; an invented edge depth is not
    )
    camera = RealSenseCamera(config)
    camera.connect()
    try:
        out.mkdir(parents=True, exist_ok=True)
        intr = camera.color_intrinsics()
        (out / "intrinsics.json").write_text(json.dumps(intr, indent=2))
        print(f"intrinsics: fx={intr['fx']:.1f} fy={intr['fy']:.1f} cx={intr['cx']:.1f} cy={intr['cy']:.1f}")
        print("Enter = capture a scene, q+Enter = quit.")

        index = 0
        while True:
            key = input(f"[scene_{index:03d}] ready> ").strip().lower()
            if key == "q":
                break
            rgb, depth_mm = camera.read_color_and_aligned_depth()
            depths = [depth_mm]
            for _ in range(4):
                _, d = camera.read_color_and_aligned_depth()
                depths.append(d)
            depth_m = (np.median(np.stack(depths), axis=0) / 1000.0).astype(np.float32)
            assert depth_m.shape == rgb.shape[:2], (
                f"depth {depth_m.shape} does not match color {rgb.shape[:2]} — is decimation really off?"
            )

            scene = out / f"scene_{index:03d}"
            scene.mkdir(exist_ok=True)
            cv2.imwrite(str(scene / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(scene / "depth.npy", depth_m)
            valid = depth_m > 0
            colored = cv2.applyColorMap(
                cv2.convertScaleAbs(np.clip(depth_m, 0, 1.5), alpha=170), cv2.COLORMAP_TURBO
            )
            colored[~valid] = 0
            preview = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colored], axis=1)
            cv2.imwrite(str(scene / "preview.jpg"), preview)
            print(
                f"  saved scene_{index:03d}: depth valid {valid.mean():.0%}, "
                f"median {np.median(depth_m[valid]):.3f} m"
            )
            index += 1
    finally:
        camera.disconnect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True, help="serial from `lerobot-find-cameras realsense`")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()
    capture_session(args.serial, args.out, fps=args.fps, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
