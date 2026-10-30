"""Verify the dish's depth discrepancy against the original cached source video."""

import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = (
    Path.home()
    / ".cache/huggingface/lerobot/hub/datasets--CaryxAI--everyday-manipulation-3d/snapshots/3c2c3bca2283b3e4b3953063d2971e668ebb9661"
)
out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
ds = LeRobotDataset(
    "CaryxAI/everyday-manipulation-3d",
    root=root,
    episodes=[90],
    cameras=["observation.images.ego", "observation.images.depth"],
    depth_output_unit="m",
    video_backend="pyav",
)
rows = []
for i in (0, 1, 10):
    row = ds[i]
    z = row["observation.images.depth"].numpy().squeeze()
    rgb = np.rint(row["observation.images.ego"].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
    rgb = cv2.resize(rgb, (192, 256), interpolation=cv2.INTER_AREA)
    ours = np.load(out / f"dish_{i:03d}.npz")
    mask = ours["mask"]
    item = {
        "frame": i,
        "source_to_import_depth_mm_percentiles": np.percentile(
            np.abs(z[mask] - ours["depth"][mask]) * 1000, [50, 90, 99, 100]
        ).tolist(),
        "rgb_abs_difference": float(np.median(np.abs(rgb.astype(float) - ours["rgb"].astype(float)))),
    }
    rows.append(item)
    np.savez(out / f"dish_source_{i:03}.npz", depth=z, rgb=rgb)
    print(item, flush=True)
(out / "dish_import_check.json").write_text(json.dumps(rows, indent=2))
