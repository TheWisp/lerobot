"""Replay the real stray-point counterexample and legitimate split-cup surface.

Compare image connectivity, 3D connectivity and mask erosion without changing
any production setting. Point counts are sampling-specific, not object truth.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from lerobot.policies.flash.surface_observation import largest_connected_surface, observation

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
repo = Path(__file__).resolve().parents[4]
base = Path.home() / ".cache/huggingface/lerobot/demos/insertion/recording"
rgb = cv2.cvtColor(cv2.imread(str(base / "rgb/000280.jpg")), cv2.COLOR_BGR2RGB)
depth = cv2.imread(str(base / "depth/000280.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
mask = (
    cv2.imread(
        str(Path.home() / ".cache/flash-mask-depth-experiment/sam3-full/000280.png"), cv2.IMREAD_GRAYSCALE
    )
    > 0
)
k = np.loadtxt(base / "cam_K.txt")
cfg = json.loads((repo / "docs/proofs/flash-mask-depth-tracker/config.json").read_text())
mug = np.load("/tmp/mug_000_audit.npz")  # nosec B108: offline scratch artifact
mug_root = Path.home() / ".cache/huggingface/lerobot/demos/flash_benchmark_caryx_002"
mug_cfg = json.loads((mug_root / "flash/v1/observation-config.json").read_text())["vision_pipeline"][
    "observation"
]
mug_k = np.array(json.loads((mug_root / "flash/v1/source.json").read_text())["camera_intrinsics"])
rows = []
for name, case_rgb, case_depth, case_mask, case_k, case_cfg in [
    ("dowel", rgb, depth, mask, k, cfg),
    ("mug", mug["rgb"], mug["depth"], mug["mask"], mug_k, mug_cfg),
]:
    case_cfg = {**case_cfg, "component_gap_m": None}
    n, labels, stats, _ = cv2.connectedComponentsWithStats(case_mask.astype(np.uint8), 8)
    two_d_largest = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    for margin in (0, 1, 2, 3, 4, 5, 6):
        settings = {**case_cfg, "mask_margin_px": margin}
        obs = observation(case_depth, case_mask, case_k, settings)
        points = obs["points"]
        px = obs["pixels"]
        keep = largest_connected_surface(points, 0.01)
        pairs = cKDTree(points).query_pairs(0.01, output_type="ndarray")
        _, ids = connected_components(
            coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points))),
            directed=False,
        )
        item = {
            "case": name,
            "margin_px": margin,
            "mask_component_areas_px": sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True),
            "sampled_points": len(points),
            "kept_by_largest_3d": int(keep.sum()),
            "kept_by_largest_2d": int(two_d_largest[px[:, 1], px[:, 0]].sum()),
            "component_sizes_3d": sorted(np.bincount(ids).tolist(), reverse=True),
        }
        if name == "dowel":
            # The historical stray cluster is 80--110 mm behind the main dowel.
            item["points_deeper_than_400mm"] = int((points[:, 2] > 0.4).sum())
            item["far_points_in_largest_2d_mask"] = int(
                np.sum((points[:, 2] > 0.4) & two_d_largest[px[:, 1], px[:, 0]])
            )
        rows.append(item)
        print(item, flush=True)
        if margin == 2:
            np.savez(
                out / f"{name}_stray_audit.npz",
                rgb=case_rgb,
                depth=case_depth,
                mask=case_mask,
                k=case_k,
                points=points,
                pixels=px,
                keep=keep,
                valid=obs["valid_image"],
            )
(out / "stray_cleanup.json").write_text(json.dumps(rows, indent=2))
