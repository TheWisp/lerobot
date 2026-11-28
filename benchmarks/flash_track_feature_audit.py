"""Compare spatially stratified query choices on recorded RGB-D.

All query cohorts share an inference call. A change in distance between two
accepted tracks is a rigidity violation; a perfectly coherent wrong motion is
invisible to this metric. No RANSAC or ICP discards disagreements in this audit.
Pair-distance metrics are computed before fitting any pose.
Configuration contains capture paths and analysis settings, never robot commands.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr

from lerobot.policies.flash.registration import rigid_fit, transform_points
from lerobot.policies.flash.visible_points import surface_support


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return (
        dict(
            zip(
                ("p25", "median", "p75", "p90"),
                np.quantile(values, [0.25, 0.5, 0.75, 0.9]).tolist(),
                strict=True,
            )
        )
        if len(values)
        else None
    )


def feature_queries(rgb, depth, mask, settings):
    """Same per-cell quota; vary only ranking within each cell of the object bbox.

    'Corner' and 'edge' mean biased toward the respective structure-tensor score,
    not verified physical corners/edges. In particular a highlight can score high.
    """
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    valid = (distance > settings["edge_exclusion_px"]) & np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(valid)
    if len(x) < settings["points_per_cohort"]:
        raise ValueError("too few eligible pixels for equal-size cohorts")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    eigen = cv2.cornerEigenValsAndVecs(gray, settings["feature_block_px"], 3)
    large, small = eigen[y, x, 0], eigen[y, x, 1]
    grid = settings["strata_per_axis"]
    cells = ((x - x.min()) * grid // (np.ptp(x) + 1)) + grid * ((y - y.min()) * grid // (np.ptp(y) + 1))
    rng = np.random.default_rng(settings["seed"])
    rankings = {"uniform": rng.random(len(x)), "corner": small, "edge": large - small}
    queries, groups, scores = [], [], []
    for label, ranking in rankings.items():
        buckets = [
            ids[np.argsort(-ranking[ids], kind="stable")]
            for cell in np.unique(cells)
            if len(ids := np.flatnonzero(cells == cell))
        ]
        selected = []
        for rank in range(max(map(len, buckets))):
            for bucket in buckets:
                if rank < len(bucket):
                    selected.append(bucket[rank])
                if len(selected) == settings["points_per_cohort"]:
                    break
            if len(selected) == settings["points_per_cohort"]:
                break
        selected = np.asarray(selected)
        queries.extend(np.column_stack((np.zeros(len(selected)), x[selected], y[selected])))
        groups.extend([label] * len(selected))
        scores.extend(np.column_stack((small[selected], large[selected], distance[y[selected], x[selected]])))
    return np.asarray(queries, dtype=np.float32), np.asarray(groups), np.asarray(scores)


def measure(points, support):
    """Per-ID deformation against all co-visible reference IDs, across all frames."""
    reference_distances = cdist(points[0], points[0])
    scores = np.full(support.shape, np.nan)
    all_errors = []
    for index in range(1, len(points)):
        ids = np.flatnonzero(support[0] & support[index])
        if len(ids) < 3:
            continue
        change = (
            np.abs(cdist(points[index, ids], points[index, ids]) - reference_distances[np.ix_(ids, ids)])
            * 1000
        )
        all_errors.extend(change[np.triu_indices(len(ids), 1)])
        np.fill_diagonal(change, np.nan)
        scores[index, ids] = np.nanmedian(change, axis=1)
    per_id = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in scores.T])
    report = {
        "queries": int(points.shape[1]),
        "supported_fraction": float(support.mean()),
        "support_count": quantiles(support.sum(axis=1)),
        "per_track_median_deformation_mm": quantiles(per_id),
        "all_frames_pair_distance_change_mm": quantiles(all_errors),
    }
    for name, ids in (("endpoint_shared", support[0] & support[-1]), ("continuous", support.all(axis=0))):
        record = {"count": int(ids.sum())}
        if ids.sum() >= 3:
            source, target = points[0, ids], points[-1, ids]
            fit = rigid_fit(source, target)
            residual = transform_points(fit, source) - target
            record["pair_distance_change_mm"] = quantiles(np.abs(pdist(source) - pdist(target)) * 1000)
            record["residual_after_best_rigid_motion_mm"] = quantiles(np.linalg.norm(residual, axis=1) * 1000)
        report[name] = record
    return report, per_id


def grasp_proxy(points, support, camera_from_gripper):
    """Separate common motion from deformation, conditional on a rigid grasp.

    Calibration, timing and real grasp slip can all create common motion. This
    is not ground-truth object error. Pair-distance deformation needs none of
    those assumptions.
    """
    first = transform_points(np.linalg.inv(camera_from_gripper[0]), points[0])
    last = transform_points(np.linalg.inv(camera_from_gripper[-1]), points[-1])
    report = {}
    for label, ids in (("shared", support[0] & support[-1]), ("continuous", support.all(axis=0))):
        if ids.sum() < 3:
            continue
        error = last[ids] - first[ids]
        common = error.mean(axis=0)
        norms = np.linalg.norm(error, axis=1)
        norm_common = np.linalg.norm(common)
        energy = np.sum(error**2)
        fit = rigid_fit(first[ids], last[ids])
        residual = last[ids] - transform_points(fit, first[ids])
        record = {
            "count": int(ids.sum()),
            "error_mm": quantiles(norms * 1000),
            "mean_vector_mm": (common * 1000).tolist(),
        }
        record["coherence_ratio"] = float(norm_common / norms.mean()) if norms.mean() > 0 else None
        if norm_common > 0:
            cosine = error[norms > 0] @ common / (norms[norms > 0] * norm_common)
            record["angle_to_mean_deg"] = quantiles(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        if energy > 0:
            record["fraction_squared_error_explained_by_common_translation"] = float(
                1 - np.sum((error - common) ** 2) / energy
            )
            record["fraction_squared_error_explained_by_common_rigid_motion"] = float(
                1 - np.sum(residual**2) / energy
            )
        record["remaining_nonrigid_mm"] = quantiles(np.linalg.norm(residual, axis=1) * 1000)
        report[label] = record
    return report


def analyze_saved(path):
    """Reanalyze an inference artifact without loading a model."""
    with np.load(path) as saved:
        points, support, groups, features = (saved[k] for k in ("points", "support", "groups", "features"))
        report = json.loads(str(saved["metadata"]))
        per_id_all = np.full(len(groups), np.nan)
        report["cohorts"] = {}
        for group in np.unique(groups):
            ids = groups == group
            result, per_id = measure(points[:, ids], support[:, ids])
            per_id_all[ids] = per_id
            result["initial_corner_score"] = quantiles(features[ids, 0])
            if "camera_from_gripper" in saved:
                result["rigid_grasp_proxy"] = grasp_proxy(
                    points[:, ids], support[:, ids], saved["camera_from_gripper"]
                )
            report["cohorts"][str(group)] = result
        # Uniform cohort only: do not count duplicated IDs across cohorts as
        # independent evidence, or correlate scores designed by selection.
        ids = (groups == "uniform") & np.isfinite(per_id_all)
        report["uniform_corner_deformation_spearman"] = (
            float(spearmanr(features[ids, 0], per_id_all[ids]).statistic) if ids.sum() >= 3 else None
        )
        report["per_id_deformation_mm"] = [float(v) if np.isfinite(v) else None for v in per_id_all]
        report["per_id_support_fraction"] = support.mean(axis=0).tolist()
        if "comparison_points" in saved:
            # Identical IDs, frames and support in both experiments: avoids
            # attributing a change in the survivor set to appearance quality.
            report["matched_comparison"] = {}
            for group in np.unique(groups):
                ids = groups == group
                shared = (support & saved["comparison_support"])[:, ids]
                report["matched_comparison"][str(group)] = {
                    "original": measure(points[:, ids], shared)[0],
                    "comparison": measure(saved["comparison_points"][:, ids], shared)[0],
                }
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run(args):
    config = json.loads(args.config.read_text())
    settings = config["settings"]
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repository.parent))
    from trackeverything.demo_hf import TrackEverything  # noqa: PLC0415

    model = TrackEverything.from_pretrained(str(args.checkpoint), device="cuda")
    print("Model default image size:", model.model.image_size, flush=True)
    for clip in config["clips"]:
        if args.clip and clip["name"] not in args.clip:
            continue
        recording, mask_dir = Path(clip["recording"]), Path(clip["mask_dir"])
        frames = np.arange(clip["first_frame"], clip["last_frame"] + 1)
        rgb, depth, masks = [], [], []
        for frame in frames:
            rgb.append(
                cv2.cvtColor(cv2.imread(str(recording / "rgb" / f"{frame:06d}.jpg")), cv2.COLOR_BGR2RGB)
            )
            depth.append(
                cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), -1).astype(np.float32) * 0.001
            )
            masks.append(cv2.imread(str(mask_dir / f"{frame:06d}.png"), 0) > 0)
        rgb, depth, masks = np.asarray(rgb), np.asarray(depth), np.asarray(masks)
        k = np.loadtxt(recording / "cam_K.txt").reshape(3, 3)
        queries, groups, features = feature_queries(rgb[0], depth[0], masks[0], settings)
        if clip.get("appearance") == "flat_object":
            # Controlled synthetic ablation: keep original query locations,
            # depth, camera, masks and motion, but replace all object appearance
            # (texture, shading and highlights) with a fixed source-frame color.
            # This is not a physical painted-object experiment or ground truth.
            color = np.median(rgb[0, masks[0]], axis=0).astype(np.uint8)
            rgb[masks] = color
        for resolution in config["resolutions"]:
            path = args.output / f"{clip['name']}_{resolution}.npz"
            if path.exists():
                analyze_saved(path)
                continue
            torch.manual_seed(settings["seed"])
            inference_rgb, inference_depth = rgb, depth
            if resolution == "padded_native":
                # This checkpoint's multiscale adapter errors on 480x848 even
                # though the public API accepts multiples of 4. Right/bottom
                # padding to its 32-pixel pyramid preserves all real pixels,
                # their intrinsics and query coordinates. Padding has no depth.
                pad_y, pad_x = (-rgb.shape[1]) % 32, (-rgb.shape[2]) % 32
                inference_rgb = np.pad(rgb, ((0, 0), (0, pad_y), (0, pad_x), (0, 0)), mode="edge")
                inference_depth = np.pad(depth, ((0, 0), (0, pad_y), (0, pad_x)))
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                tracked = model.infer(
                    video=inference_rgb,
                    depths=inference_depth,
                    intrinsics=np.repeat(k[None], len(frames), axis=0),
                    extrinsics=np.repeat(np.eye(4)[None], len(frames), axis=0),
                    query_points_2d=queries,
                    mode="dense",
                    use_input_resolution=resolution != "default",
                )
            points = tracked.query_coords[0].float().cpu().numpy() * tracked.norm_factor
            logits = tracked.query_visibs[0].float().cpu().numpy()
            elapsed = time.perf_counter() - start
            support = np.asarray(
                [
                    surface_support(p, v, m, d, k, settings["support_tolerance_m"])[0]
                    for p, v, m, d in zip(points, logits, masks, depth, strict=True)
                ]
            )
            metadata = {
                "clip": clip,
                "settings": settings,
                "resolution": resolution,
                "model_default_hw": list(model.model.image_size),
                "input_hw": list(rgb.shape[1:3]),
                "model_input_hw": list(inference_rgb.shape[1:3])
                if resolution != "default"
                else list(model.model.image_size),
                "inference_s": elapsed,
                "method": "No rigid fitting or outlier rejection in pair-distance metrics; masks and depth only.",
            }
            np.savez_compressed(
                path,
                frames=frames,
                points=points,
                support=support,
                logits=logits,
                queries=queries,
                groups=groups,
                features=features,
                intrinsics=k,
                metadata=json.dumps(metadata),
            )
            report = analyze_saved(path)
            print(
                clip["name"],
                resolution,
                elapsed,
                {g: (r["endpoint_shared"], r["supported_fraction"]) for g, r in report["cohorts"].items()},
                flush=True,
            )
            del tracked
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("config", "repository", "checkpoint", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--clip", action="append")
    run(parser.parse_args())
