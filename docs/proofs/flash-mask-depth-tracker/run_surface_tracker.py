"""Causal mask/depth-only surface tracking experiment; no learned point motion.

Fit a rigid transform with robust nearest-surface point-to-plane ICP, using
the previous cloud or a bounded growing map. Original geometry stays fixed;
novel patches require repeated accepted observations before map admission.
The acceptance check compares the pre-update map with current mask/depth.
All solver and preprocessing settings are explicit in the recorded JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def move(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def unproject(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    y, x = np.indices(depth.shape)
    return np.stack(
        (
            (x - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (y - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    )


def voxel_sample(points: np.ndarray, spacing: float) -> np.ndarray:
    if not len(points):
        return np.zeros(0, dtype=int)
    _, selected = np.unique(np.floor(points / spacing).astype(np.int32), axis=0, return_index=True)
    return np.sort(selected)


def largest_connected_surface(points: np.ndarray, gap_m: float) -> np.ndarray:
    """Study ablation: use one connected patch, withholding all detached patches.

    This can discard legitimate visible pieces across an occluder. It is not an
    object-ownership oracle, and connected contamination can still survive.
    """
    if not len(points):
        return np.zeros(0, dtype=bool)
    pairs = cKDTree(points).query_pairs(gap_m, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    _, labels = connected_components(graph, directed=False)
    return labels == np.argmax(np.bincount(labels))


def observation(depth: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray, config: dict) -> dict:
    xyz = unproject(depth, intrinsics)
    raw = mask & (depth > 0) & np.isfinite(depth)
    margin = int(config["mask_margin_px"])
    interior = cv2.erode(mask.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    neighborhood = np.ones((config["depth_neighborhood_px"],) * 2, np.uint8)
    lower = cv2.erode(depth, neighborhood)
    upper = cv2.dilate(depth, neighborhood)
    reliable = (lower > 0) & (upper - lower <= config["depth_span_m"])
    # Smoothed local gradients define tangent planes only; fitted XYZ remains measured depth.
    smoothed = cv2.bilateralFilter(
        depth, config["normal_bilateral_diameter_px"], config["huber_m"], config["normal_bilateral_sigma_px"]
    )
    normal_xyz = unproject(smoothed, intrinsics)
    horizontal = np.zeros_like(xyz)
    vertical = np.zeros_like(xyz)
    horizontal[:, 1:-1] = normal_xyz[:, 2:] - normal_xyz[:, :-2]
    vertical[1:-1] = normal_xyz[2:] - normal_xyz[:-2]
    normals = np.cross(horizontal, vertical)
    norm = np.linalg.norm(normals, axis=-1)
    valid = raw & interior & reliable & (norm > np.finfo(float).eps)
    normals /= np.maximum(norm[..., None], np.finfo(float).eps)
    normals[np.sum(normals * xyz, axis=-1) > 0] *= -1
    y, x = np.nonzero(valid)
    selected = voxel_sample(xyz[y, x], config["surface_spacing_m"])
    y, x = y[selected], x[selected]
    detached = 0
    if config.get("component_gap_m") is not None:
        keep = largest_connected_surface(xyz[y, x], config["component_gap_m"])
        detached = int((~keep).sum())
        y, x = y[keep], x[keep]
    # Spatially interleaved held-out pixels, not independent ground-truth labels.
    tile = config["validation_tile_px"]
    fit = ((x // tile + y // tile) % 2) == 0
    return {
        "points": xyz[y, x],
        "normals": normals[y, x],
        "pixels": np.column_stack((x, y)),
        "fit": fit,
        "valid_image": valid,
        "depth_reliable": reliable,
        "raw_count": int(raw.sum()),
        "boundary_count": int((raw & ~interior).sum()),
        "edge_count": int((raw & interior & ~reliable).sum()),
        "clean_count": int(valid.sum()),
        "detached_count": detached,
    }


def skew(points: np.ndarray) -> np.ndarray:
    result = np.zeros((len(points), 3, 3))
    result[:, 0, 1] = -points[:, 2]
    result[:, 0, 2] = points[:, 1]
    result[:, 1, 0] = points[:, 2]
    result[:, 1, 2] = -points[:, 0]
    result[:, 2, 0] = -points[:, 1]
    result[:, 2, 1] = points[:, 0]
    return result


def fit_surface(
    model: np.ndarray, model_normals: np.ndarray, observed: dict, initial: np.ndarray, config: dict
) -> tuple[np.ndarray, dict]:
    pose = initial.copy()
    tree = cKDTree(model)
    points = observed["points"][observed["fit"]]
    normals = observed["normals"][observed["fit"]]
    radius = max(
        float(np.sqrt(np.mean(np.sum((model - model.mean(axis=0)) ** 2, axis=1)))),
        config["surface_spacing_m"],
    )
    match_count, eigenvalues, steps = 0, np.zeros(6), 0
    for distance_limit in config["correspondence_distances_m"]:
        for _ in range(config["iterations_per_level"]):
            local = (points - pose[:3, 3]) @ pose[:3, :3]
            distances, indices = tree.query(local)
            normal_camera = model_normals[indices] @ pose[:3, :3].T
            valid = (distances < distance_limit) & (np.sum(normal_camera * normals, axis=1) > 0)
            match_count = int(valid.sum())
            if match_count < 6:  # Six rigid-motion unknowns; not an accuracy/confidence threshold.
                break
            predicted = move(model[indices[valid]], pose)
            target = points[valid]
            normal = normals[valid]
            difference = predicted - target
            residual = np.sum(normal * difference, axis=1)
            weights = np.minimum(1, config["huber_m"] / np.maximum(distances[valid], 1e-12))
            centered = predicted - pose[:3, 3]
            jacobian = np.column_stack((np.cross(centered, normal) / radius, normal))
            hessian = jacobian.T @ (weights[:, None] * jacobian)
            rhs = -jacobian.T @ (weights * residual)
            eigenvalues = np.linalg.eigvalsh(hessian)
            # Small Euclidean term anchors weak tangent directions to nearby surface samples.
            point_j = np.concatenate(
                (-skew(centered) / radius, np.tile(np.eye(3), (len(centered), 1, 1))), axis=2
            )
            point_w = weights * config["point_regularizer"]
            hessian += np.einsum("nki,n,nkj->ij", point_j, point_w, point_j)
            rhs -= np.einsum("nki,n,nk->i", point_j, point_w, difference)
            hessian += np.eye(6) * config["solver_damping"] * max(float(np.trace(hessian) / 6), 1)
            delta = np.linalg.solve(hessian, rhs)
            angle = delta[:3] / radius
            ratio = max(
                1,
                np.linalg.norm(angle) / config["max_rotation_step_rad"],
                np.linalg.norm(delta[3:]) / config["max_translation_step_m"],
            )
            delta /= ratio
            rotation = Rotation.from_rotvec(delta[:3] / radius).as_matrix()
            pose[:3, :3] = rotation @ pose[:3, :3]
            pose[:3, 3] += delta[3:]
            steps += 1
            if np.linalg.norm(delta) < config["step_convergence_m"]:
                break
    return pose, {
        "pairs": match_count,
        "steps": steps,
        "information_ratio": float(eigenvalues[0] / max(eigenvalues[-1], 1e-12)),
    }


def evidence(
    model: np.ndarray,
    pose: np.ndarray,
    observed: dict,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    config: dict,
) -> dict:
    transformed = move(model, pose)
    camera_points = transformed[transformed[:, 2] > 0]
    uv = np.rint(
        camera_points[:, :2] / camera_points[:, 2, None] * np.diag(intrinsics)[:2] + intrinsics[:2, 2]
    ).astype(int)
    height, width = depth.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, camera_points = uv[inside], camera_points[inside]
    zbuffer = np.full(height * width, np.inf)
    np.minimum.at(zbuffer, uv[:, 1] * width + uv[:, 0], camera_points[:, 2])
    pixels = np.flatnonzero(np.isfinite(zbuffer) & observed["depth_reliable"].ravel())
    gaps = depth.ravel()[pixels] - zbuffer[pixels]
    supported = (np.abs(gaps) <= config["depth_agreement_m"]) & mask.ravel()[pixels]
    contradicted = gaps > config["depth_agreement_m"]
    test_points = observed["points"][~observed["fit"]]
    distances = cKDTree(transformed).query(test_points)[0] if len(test_points) else np.array([np.nan])
    return {
        "supported": int(supported.sum()),
        "contradicted": int(contradicted.sum()),
        "valid_rays": len(pixels),
        "heldout_median_mm": float(np.nanmedian(distances) * 1000),
        "heldout_p90_mm": float(np.nanquantile(distances, 0.9) * 1000),
    }


def initial_landmarks(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    _, _, basis = np.linalg.svd(points - center, full_matrices=False)
    basis = basis.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    first = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    second = int(np.argmax(np.linalg.norm(points - points[first], axis=1)))
    axis = points[second] - points[first]
    axis /= np.linalg.norm(axis)
    residual = points - points[first]
    residual -= (residual @ axis)[:, None] * axis
    third = int(np.argmax(np.linalg.norm(residual, axis=1)))
    return points[[first, second, third]], basis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("previous", "growing"), default="growing")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "recording": str(args.recording),
                "masks": str(args.masks),
                "mode": args.mode,
                "start": args.start,
                "stop": args.stop,
                "pose_inputs": "only mask and depth; no P2P, gripper, or future frame",
            },
            indent=2,
        )
        + "\n"
    )
    intrinsics = np.loadtxt(args.recording / "cam_K.txt").reshape(3, 3)
    timestamps = np.atleast_1d(np.loadtxt(args.recording / "times.txt"))
    last = len(timestamps) if args.stop is None else min(len(timestamps), args.stop + 1)
    cv2.setNumThreads(1)
    model, model_normals = np.zeros((0, 3)), np.zeros((0, 3))
    pose, pending = np.eye(4), {}
    rows, poses, displays, states = [], [], [], []
    initial_points = None
    limits = config["display_point_limit"]
    for frame in range(args.start, last):
        load_start = time.perf_counter()
        raw_depth = cv2.imread(str(args.recording / "depth" / f"{frame:06d}.png"), -1)
        raw_mask = cv2.imread(str(args.masks / f"{frame:06d}.png"), 0)
        if raw_depth is None or raw_mask is None or raw_depth.shape != raw_mask.shape:
            raise ValueError(f"missing aligned depth/mask at frame {frame}")
        depth = raw_depth.astype(np.float32) * config["depth_scale_m"]
        mask = raw_mask > 0
        compute_start = time.perf_counter()
        observed = observation(depth, mask, intrinsics, config)
        preprocess_end = time.perf_counter()
        obs_points = observed["points"]
        stats = {"pairs": 0, "steps": 0, "information_ratio": float("nan")}
        checks = {
            "supported": 0,
            "contradicted": 0,
            "valid_rays": 0,
            "heldout_median_mm": float("nan"),
            "heldout_p90_mm": float("nan"),
        }
        added, withheld = 0, 0
        fit_end = preprocess_end
        check_end = preprocess_end
        if len(obs_points) < 6:
            state = "no_geometry"
        elif not len(model):
            pose[:3, 3] = obs_points.mean(axis=0)
            model = obs_points - pose[:3, 3]
            model_normals = observed["normals"].copy()
            initial_points = model.copy()
            landmarks, basis = initial_landmarks(model)
            seed_frame = frame
            state = "initialized"
        else:
            candidate, stats = fit_surface(model, model_normals, observed, pose, config)
            fit_end = time.perf_counter()
            checks = evidence(model, candidate, observed, depth, mask, intrinsics, config)
            check_end = time.perf_counter()
            # Positive same-frame evidence must exceed contradiction before publication/fusion.
            if stats["pairs"] >= 6 and checks["supported"] > checks["contradicted"]:
                pose = candidate
                state = "surface_supported"
                local = (obs_points - pose[:3, 3]) @ pose[:3, :3]
                local_normals = observed["normals"] @ pose[:3, :3]
                if args.mode == "previous":
                    model, model_normals = local, local_normals
                else:
                    distances, _ = cKDTree(model).query(local)
                    novel = (distances > config["surface_spacing_m"]) & (
                        distances < config["correspondence_distances_m"][0]
                    )
                    withheld = int((distances >= config["correspondence_distances_m"][0]).sum())
                    keys = np.floor(local / config["surface_spacing_m"]).astype(int)
                    seen = set()
                    to_add, normals_to_add = [], []
                    for point, normal, key_array in zip(
                        local[novel], local_normals[novel], keys[novel], strict=True
                    ):
                        key = tuple(key_array)
                        if key in seen:
                            continue
                        seen.add(key)
                        count, total, total_normal, previous_frame = pending.get(
                            key, (0, np.zeros(3), np.zeros(3), frame)
                        )
                        if frame - previous_frame > config["pending_lifetime_frames"]:
                            count, total, total_normal = 0, np.zeros(3), np.zeros(3)
                        count += 1
                        total = total + point
                        total_normal = total_normal + normal
                        if (
                            count >= config["promotion_observations"]
                            and len(model) + len(to_add) < config["maximum_map_points"]
                        ):
                            to_add.append(total / count)
                            normals_to_add.append(total_normal / max(np.linalg.norm(total_normal), 1e-12))
                            pending.pop(key, None)
                        else:
                            pending[key] = (count, total, total_normal, frame)
                    if to_add:
                        added = len(to_add)
                        model = np.concatenate((model, to_add))
                        model_normals = np.concatenate((model_normals, normals_to_add))
                    pending = {
                        key: value
                        for key, value in pending.items()
                        if frame - value[3] <= config["pending_lifetime_frames"]
                    }
            else:
                state = "pose_rejected"
        finish = time.perf_counter()
        display = np.full((limits, 3), np.nan, dtype=np.float32)
        if len(model):
            selection = np.linspace(0, len(model) - 1, min(limits, len(model)), dtype=int)
            display[: len(selection)] = move(model[selection], pose)
        displays.append(display)
        poses.append(pose.copy())
        states.append(state)
        row = {
            "frame": frame,
            "state": state,
            "raw_pixels": observed["raw_count"],
            "boundary_removed": observed["boundary_count"],
            "depth_edge_removed": observed["edge_count"],
            "detached_samples_removed": observed["detached_count"],
            "clean_pixels": observed["clean_count"],
            "fit_surface_points": int(observed["fit"].sum()),
            "map_points": len(model),
            "map_added": added,
            "far_points_withheld": withheld,
            **stats,
            **checks,
            "load_ms": (compute_start - load_start) * 1000,
            "preprocess_ms": (preprocess_end - compute_start) * 1000,
            "fit_ms": (fit_end - preprocess_end) * 1000,
            "check_ms": (check_end - fit_end) * 1000,
            "map_ms": (finish - check_end) * 1000,
            "compute_ms": (finish - compute_start) * 1000,
        }
        rows.append(row)
        if frame % 100 == 0:
            print(
                frame,
                state,
                len(model),
                f"{checks['heldout_p90_mm']:.1f}mm",
                f"{row['compute_ms']:.1f}ms",
                flush=True,
            )
    if initial_points is None:
        raise ValueError("no frame supplied enough geometry to initialize")
    np.savez_compressed(
        args.output / "trace.npz",
        frames=np.arange(args.start, last),
        poses=poses,
        state=np.asarray(states),
        display_points_camera=displays,
        intrinsics=intrinsics,
        landmarks_object=landmarks,
        basis_object=basis,
        initial_points=initial_points,
        final_map=model,
        final_normals=model_normals,
        seed_frame=seed_frame,
    )
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "frames": len(rows),
        "seed_frame": seed_frame,
        "states": {state: states.count(state) for state in sorted(set(states))},
        "compute_ms_median": float(np.median([r["compute_ms"] for r in rows])),
        "compute_ms_p95": float(np.percentile([r["compute_ms"] for r in rows], 95)),
        "final_map_points": len(model),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
