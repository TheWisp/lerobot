"""Reconstruct selection and rejection on the clicked episode-90 dish; no changes to data."""

import gzip
import json
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flash.surface_observation import observation

root = Path.home() / ".cache/huggingface/lerobot/demos/flash_benchmark_caryx_090"
evidence = Path("/tmp/flash_dish_click_evidence_20261029")  # nosec B108: offline scratch artifact
out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
out.mkdir(exist_ok=True)
s = json.loads((root / "flash/v1/source.json").read_text())
cfg = json.loads((root / "flash/v1/observation-config.json").read_text())["vision_pipeline"]["observation"]
k = np.array(s["camera_intrinsics"])
ds = LeRobotDataset(s["repo_id"], root=root, video_backend="pyav")
rows = []
for i in (0, 1, 10, 50, 55, 56, 100, 244):
    e = json.loads(gzip.decompress((evidence / f"{i:06d}.json.gz").read_bytes()))
    row = ds[i]
    depth = row["observation.images.depth"].numpy().squeeze().astype(np.float32) * 0.001
    rgb = np.rint(row["observation.images.rgb"].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
    mask = np.zeros(depth.shape, bool)
    for start, n in e["masks_rle"]["object_b"]:
        mask.flat[start : start + n] = True
    obs = observation(depth, mask, k, cfg)
    whole = observation(depth, mask, k, {**cfg, "component_gap_m": None})
    pix = obs["pixels"]
    selected = np.zeros(mask.shape, bool)
    selected[pix[:, 1], pix[:, 0]] = True
    prop = e["proposals"]["object_b"]
    group = e["groups"].get("object_b", {})
    distances = cKDTree(obs["points"]).query(whole["points"])[0] * 1000
    item = {
        "frame": i,
        "mask_pixels": int(mask.sum()),
        "valid_depth": obs["raw_count"],
        "boundary_removed": obs["boundary_count"],
        "depth_jump_removed": obs["edge_count"],
        "clean_pixels": obs["clean_count"],
        "sampled_points_before_patch_selection": len(whole["points"]),
        "retained_points": len(obs["points"]),
        "same_frame_disagreement_p90_mm": float(np.percentile(distances, 90)),
        "pose_state": e["estimates"]["object_b"]["evidence"],
        "rejection": prop["failure"],
        "support": prop["visible_support"],
        "group_points": len(group.get("ids", [])),
        "group_measured": group.get("measured_this_frame"),
    }
    if i == 0:
        yy, xx = np.indices(depth.shape)
        xyz = np.stack(((xx - k[0, 2]) * depth / k[0, 0], (yy - k[1, 2]) * depth / k[1, 1], depth), axis=-1)
        blocks = (
            sliding_window_view(xyz, (3, 3), axis=(0, 1))
            .transpose(0, 1, 3, 4, 2)
            .reshape(depth.shape[0] - 2, depth.shape[1] - 2, 9, 3)
        )
        centered = blocks - blocks.mean(axis=2, keepdims=True)
        eigenvalues = np.linalg.eigvalsh(np.einsum("...ni,...nj->...ij", centered, centered))
        rms_mm = np.sqrt(np.maximum(eigenvalues[..., 0], 0) / 9) * 1000
        margin = cfg["mask_margin_px"]
        interior = cv2.erode(mask.astype("uint8"), np.ones((2 * margin + 1,) * 2, "uint8")) > 0
        span = cv2.dilate(depth, np.ones((3, 3), "uint8")) - cv2.erode(depth, np.ones((3, 3), "uint8"))
        rejected = interior[1:-1, 1:-1] & (span[1:-1, 1:-1] > cfg["depth_span_m"])
        item["rejected_neighborhood_plane_rms_mm"] = dict(
            zip(
                ("median", "p90", "max"),
                map(float, np.percentile(rms_mm[rejected], [50, 90, 100])),
                strict=False,
            )
        )
        item["rejected_neighborhoods_plane_rms_under_1mm_fraction"] = float(np.mean(rms_mm[rejected] < 1))
    rows.append(item)
    np.savez(
        out / f"dish_{i:03d}.npz",
        rgb=rgb,
        depth=depth,
        mask=mask,
        valid=obs["valid_image"],
        selected=selected,
        points=whole["points"],
        k=k,
    )
    print(json.dumps({x: y for x, y in item.items() if x != "support"}), flush=True)
(out / "dish.json").write_text(json.dumps(rows, indent=2))
