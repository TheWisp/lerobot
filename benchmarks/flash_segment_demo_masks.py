"""Save causal SAM3 object masks for offline FLASH point-track review.

This runs outside the GUI process so the model cannot block the event loop or
hold GPU memory while TrackEverything is loaded in its separate environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.appearance import Sam3MaskSource
from lerobot.policies.flash.perception import RGBDFrame


def run(
    recording: Path, specs: list[dict], prompts: dict[str, str], output: Path, *, resolution: int | None
) -> None:
    selected = {spec["role"]: prompts[spec["role"]] for spec in specs}
    source = Sam3MaskSource(selected, camera_id="flash-demo-point-growth", resolution=resolution)
    intrinsics = np.loadtxt(recording / "cam_K.txt").reshape(3, 3)
    first = min(int(spec["frame_index"]) for spec in specs)
    last = len(np.atleast_1d(np.loadtxt(recording / "times.txt"))) - 1
    directory_by_role = {spec["role"]: output / str(index) for index, spec in enumerate(specs)}
    for directory in directory_by_role.values():
        directory.mkdir(parents=True, exist_ok=False)
    for index in range(first, last + 1):
        stem = f"{index:06d}"
        rgb = cv2.imread(str(recording / "rgb" / f"{stem}.jpg"))
        depth = cv2.imread(str(recording / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
            raise ValueError(f"missing aligned RGB-D frame {index}")
        frame = RGBDFrame(
            index,
            cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
            depth,
            intrinsics,
            0.001,
            np.eye(4),
            float(index),
            float(index),
            "recorded",
        )
        masks = source.masks(frame)
        for role in selected:
            mask = masks.get(role)
            if mask is not None and mask.shape == depth.shape:
                cv2.imwrite(str(directory_by_role[role] / f"{stem}.png"), mask.astype(np.uint8) * 255)
    # The reviewed preparation's reference mask is authoritative at its own frame.
    for spec in specs:
        role = spec["role"]
        target = directory_by_role[role] / f"{int(spec['frame_index']):06d}.png"
        reference = cv2.imread(str(spec["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise ValueError(f"missing prepared mask for {role}")
        cv2.imwrite(str(target), reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--reference-specs", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int)
    args = parser.parse_args()
    run(
        args.recording,
        json.loads(args.reference_specs.read_text()),
        json.loads(args.prompts.read_text()),
        args.output,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
