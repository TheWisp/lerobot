"""Measure mask/depth removal and source time gaps, without fitting or pose labels."""

import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.surface_observation import observation

ROOT = Path.home() / ".cache/flash-failure-sweep-20261029"
out = []
for directory in sorted(ROOT.iterdir()):
    if not (directory / "completed.json").exists():
        continue
    provenance = json.loads((directory / "provenance.json").read_text())
    cfg = provenance["config"]["vision_pipeline"]["observation"]
    source = Path(provenance["source"])
    k = np.array(provenance["intrinsics"])
    rows = []
    for i, name in enumerate(provenance["names"]):
        entry = json.loads(gzip.decompress((directory / "evidence" / f"{i:06d}.json.gz").read_bytes()))
        depth = cv2.imread(str(source / "depth" / name), -1).astype(np.float32) * 0.001
        mask = np.zeros(depth.shape, bool)
        for start, n in entry["masks_rle"]["object"]:
            mask.flat[start : start + n] = True
        measured = observation(depth, mask, k, cfg)
        groups = entry["groups"].get("object", {})
        rows.append(
            {
                "frame": i,
                "state": entry["estimates"]["object"]["evidence"],
                "mask_pixels": int(mask.sum()),
                "mask_components": cv2.connectedComponents(mask.astype("uint8"))[0] - 1,
                "raw_depth": measured["raw_count"],
                "boundary_removed": measured["boundary_count"],
                "depth_span_removed": measured["edge_count"],
                "clean_before_sampling": measured["clean_count"],
                "detached_removed": measured["detached_count"],
                "sampled_retained": len(measured["points"]),
                "group_points": len(groups.get("ids", [])),
                "dt_s": None
                if i == 0
                else provenance["source_times_s"][i] - provenance["source_times_s"][i - 1],
            }
        )
    gaps = [
        r
        for r in rows
        if r["dt_s"] is not None and r["dt_s"] > provenance["config"]["causal_tracking"]["max_frame_gap_s"]
    ]
    out.append({"sequence": directory.name, "gaps_over_group_limit": gaps, "frames": rows})
Path(__file__).with_name("input_audit.json").write_text(json.dumps(out, indent=2, allow_nan=False))
for case in out:
    print(
        case["sequence"],
        "initial",
        case["frames"][0],
        "gaps",
        [(r["frame"], round(r["dt_s"], 3), r["state"]) for r in case["gaps_over_group_limit"]],
    )
