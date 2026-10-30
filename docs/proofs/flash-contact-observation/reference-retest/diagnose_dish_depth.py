"""Separate ray sampling, surface disagreement and pose change on the clicked dish.

Reads completed diagnostic reruns, without changing observations or tracker gates.
Depth interpolation is evaluated only where all four neighbors are valid; local
planes are measured diagnostics, not a replacement acceptance rule.
"""

import gzip
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from lerobot.policies.flash.surface_grounding import project
from lerobot.policies.flash.surface_observation import move, observation, unproject

out = Path("/tmp/flash_case_retest_20261029")  # nosec B108: offline scratch artifact
cfg = json.loads((out / "dish_no_span_config.json").read_text())["vision_pipeline"]["observation"]


def evidence(i):
    return json.loads(gzip.decompress((out / f"dish_no_span_evidence/{i:06d}.json.gz").read_bytes()))


def pct(a):
    return (
        dict(zip(("median", "p90", "max"), map(float, np.percentile(a, [50, 90, 100])), strict=False))
        if len(a)
        else None
    )


zero = np.load(out / "dish_000.npz")
k = zero["k"]
ref = observation(zero["depth"], zero["mask"], k, cfg)["points"]
e0 = evidence(0)
c0 = np.array(e0["base_from_camera"])
p0 = np.array(e0["proposals"]["object_b"]["raw_pose"])
rows = []
for i in (1, 10, 50, 55, 56, 100, 244):
    data = np.load(out / f"dish_{i:03d}.npz")
    depth = data["depth"]
    mask = data["mask"]
    obs = observation(depth, mask, k, cfg)
    xyz = unproject(depth, k)
    e = evidence(i)
    cam = np.array(e["base_from_camera"])
    prop = e["proposals"]["object_b"]
    for name, world in [("fitted", prop["raw_pose"]), ("image_motion", prop.get("motion_prior_pose"))]:
        if world is None:
            continue
        transform = np.linalg.inv(cam) @ np.array(world) @ np.linalg.inv(p0) @ c0
        pred = move(ref, transform)
        uv = project(pred, k)
        ix = np.rint(uv).astype(int)
        good = (
            (ix[:, 0] >= 1)
            & (ix[:, 0] < depth.shape[1] - 1)
            & (ix[:, 1] >= 1)
            & (ix[:, 1] < depth.shape[0] - 1)
        )
        pred, uv, ix = pred[good], uv[good], ix[good]
        good = obs["valid_image"][ix[:, 1], ix[:, 0]]
        pred, uv, ix = pred[good], uv[good], ix[good]
        # Same nearest-pixel selection as the published reverse-depth check.
        linear = ix[:, 1] * depth.shape[1] + ix[:, 0]
        order = np.lexsort((pred[:, 2], linear))
        _, first = np.unique(linear[order], return_index=True)
        chosen = order[first]
        pred, uv, ix = pred[chosen], uv[chosen], ix[chosen]
        raw = np.abs(depth[ix[:, 1], ix[:, 0]] - pred[:, 2]) * 1000
        x0, y0 = np.floor(uv).astype(int).T
        four = (
            obs["valid_image"][y0, x0]
            & obs["valid_image"][y0, x0 + 1]
            & obs["valid_image"][y0 + 1, x0]
            & obs["valid_image"][y0 + 1, x0 + 1]
        )
        interp = 1 / map_coordinates(1 / np.maximum(depth, 1e-9), uv.T[::-1], order=1)
        smooth = np.abs(interp - pred[:, 2]) * 1000
        neighborhoods = np.stack(
            [xyz[ix[:, 1] + dy, ix[:, 0] + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1
        )
        means = neighborhoods.mean(1)
        centered = neighborhoods - means[:, None]
        values, vecs = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered))
        normal = vecs[:, :, 0]
        plane = np.abs(np.sum((pred - means) * normal, axis=1)) * 1000
        local_rms = np.sqrt(np.maximum(values[:, 0], 0) / 9) * 1000
        plane_clean = np.all(
            np.stack(
                [obs["valid_image"][ix[:, 1] + dy, ix[:, 0] + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            ),
            axis=0,
        )
        nearest = cKDTree(obs["points"]).query(pred)[0] * 1000
        item = {
            "frame": i,
            "proposal": name,
            "points": len(pred),
            "nearest_pixel_z_mm": pct(raw),
            "same_subset_nearest_z_mm": pct(raw[four]),
            "interpolated_inverse_depth_z_mm": pct(smooth[four]),
            "normal_distance_mm": pct(plane[plane_clean]),
            "local_plane_rms_mm": pct(local_rms[plane_clean]),
            "nearest_surface_mm": pct(nearest),
            "current_to_reference_mm": pct(cKDTree(move(ref, transform)).query(obs["points"])[0] * 1000),
        }
        rows.append(item)
        if i in (1, 10, 100):
            np.savez(
                out / f"dish_projection_{i:03d}_{name}.npz",
                rgb=data["rgb"],
                depth=depth,
                mask=mask,
                k=k,
                pred=pred,
                uv=uv,
                ix=ix,
                raw=raw,
                interp=interp,
                smooth=smooth,
                plane=plane,
                local_rms=local_rms,
                plane_clean=plane_clean,
                four=four,
            )
        print(json.dumps(item), flush=True)
(out / "dish_projection.json").write_text(json.dumps(rows, indent=2))
