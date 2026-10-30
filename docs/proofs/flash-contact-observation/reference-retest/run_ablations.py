"""Rerun the shared pipeline with explicit diagnostic settings and unchanged masks."""

import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.observation_processing import extract

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
out.mkdir(exist_ok=True)
demos = Path.home() / ".cache/huggingface/lerobot/demos"
maskdir = out / "dish_masks"
maskdir.mkdir(exist_ok=True)
for p in sorted(Path("/tmp/flash_dish_click_evidence_20261029").glob("*.json.gz")):  # nosec B108: offline scratch artifact
    v = json.loads(gzip.decompress(p.read_bytes()))
    mask = np.zeros(v["image_size"], np.uint8)
    for start, n in v["masks_rle"]["object_b"]:
        mask.flat[start : start + n] = 255
    cv2.imwrite(str(maskdir / f"{v['frame_index']:06d}.png"), mask)
for name, root, role, masks, disable_span in [
    ("dish_allpatch", demos / "flash_benchmark_caryx_090", "object_b", maskdir, False),
    (
        "mug_allpatch",
        demos / "flash_benchmark_caryx_002",
        "object_a",
        demos / "flash_benchmark_caryx_002/flash/v1/imported-masks/object_a",
        False,
    ),
    ("dish_no_span", demos / "flash_benchmark_caryx_090", "object_b", maskdir, True),
]:
    cfg = json.loads((root / "flash/v1/observation-config.json").read_text())
    cfg["vision_pipeline"]["observation"]["component_gap_m"] = None
    if disable_span:
        # Diagnostic: every valid source depth lies inside this range, so the local
        # span rule cannot reject it. Zero readings and the mask boundary still do.
        cfg["vision_pipeline"]["observation"]["depth_span_m"] = json.loads(
            (root / "flash/v1/source.json").read_text()
        )["output_depth_max_m"]
    path = out / f"{name}_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    print(extract(root, {role: masks}, path, output=out / f"{name}_evidence"), flush=True)
