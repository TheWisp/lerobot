"""Compare direct RGB-D object registration across a demo and a trial.

This is an offline diagnostic. No video tracker output is used as a point
correspondence. All input points are back-projected from recorded mask/depth.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from flash_cross_episode_trackeverything import _image, _live_masks, _project, _rigid_fit
from scipy.spatial import cKDTree


def cloud(depth_mm: np.ndarray, mask: np.ndarray, k: np.ndarray, edge_pixels: int) -> np.ndarray:
    interior = cv2.erode(mask.astype(np.uint8), np.ones((2 * edge_pixels + 1,) * 2, np.uint8)) > 0
    ys, xs = np.nonzero(interior & (depth_mm > 0))
    if len(xs) < 3:
        raise ValueError("object mask has fewer than three depth-backed interior pixels")
    z = depth_mm[ys, xs].astype(np.float64) * 0.001
    return np.column_stack(((xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z))


def cloud_colors(rgb: np.ndarray, depth_mm: np.ndarray, mask: np.ndarray, edge_pixels: int) -> np.ndarray:
    interior = cv2.erode(mask.astype(np.uint8), np.ones((2 * edge_pixels + 1,) * 2, np.uint8)) > 0
    ys, xs = np.nonzero(interior & (depth_mm > 0))
    return rgb[ys, xs, ::-1].astype(np.float64) / 255


def apply(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - center).T))
    basis = vectors[:, ::-1]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return center, basis


def pca_hypotheses(source: np.ndarray, target: np.ndarray):
    src_center, src_basis = axes(source)
    dst_center, dst_basis = axes(target)
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            mapping = np.zeros((3, 3))
            mapping[np.arange(3), perm] = signs
            if np.linalg.det(mapping) < 0:
                continue
            rotation = dst_basis @ mapping @ src_basis.T
            yield rotation, dst_center - rotation @ src_center


def trimmed_icp(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    iterations: int,
    trim_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(target)
    for _ in range(iterations):
        moved = apply(source, rotation, translation)
        distances, nearest = tree.query(moved, workers=-1)
        keep = distances <= np.quantile(distances, trim_fraction)
        next_rotation, next_translation = _rigid_fit(source[keep], target[nearest[keep]])
        change = np.linalg.norm(next_translation - translation) + np.linalg.norm(next_rotation - rotation)
        rotation, translation = next_rotation, next_translation
        if change < 1e-6:
            break
    return rotation, translation


def feature_hypothesis(
    source_rgb: np.ndarray,
    source_depth: np.ndarray,
    source_mask: np.ndarray,
    target_rgb: np.ndarray,
    target_depth: np.ndarray,
    target_mask: np.ndarray,
    k: np.ndarray,
    name: str,
    *,
    trials: int,
    tolerance_m: float,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, dict]:
    if name == "sift":
        detector = cv2.SIFT_create(nfeatures=2000)
        norm = cv2.NORM_L2
    elif name == "orb":
        detector = cv2.ORB_create(nfeatures=2000, edgeThreshold=5, fastThreshold=5)
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(name)
    source_keypoints, source_descriptors = detector.detectAndCompute(
        source_rgb, source_mask.astype(np.uint8) * 255
    )
    target_keypoints, target_descriptors = detector.detectAndCompute(
        target_rgb, target_mask.astype(np.uint8) * 255
    )
    info = {"source_keypoints": len(source_keypoints), "target_keypoints": len(target_keypoints)}
    if (
        source_descriptors is None
        or target_descriptors is None
        or len(source_descriptors) < 2
        or len(target_descriptors) < 2
    ):
        return None, info
    matcher = cv2.BFMatcher(norm)
    matches = [
        a
        for a, b in matcher.knnMatch(source_descriptors, target_descriptors, k=2)
        if a.distance < 0.8 * b.distance
    ]
    source_xyz, target_xyz = [], []
    for match in matches:
        a = np.rint(source_keypoints[match.queryIdx].pt).astype(int)
        b = np.rint(target_keypoints[match.trainIdx].pt).astype(int)
        if not (0 <= a[0] < source_mask.shape[1] and 0 <= a[1] < source_mask.shape[0]):
            continue
        if not (0 <= b[0] < target_mask.shape[1] and 0 <= b[1] < target_mask.shape[0]):
            continue
        if not (source_mask[a[1], a[0]] and target_mask[b[1], b[0]]):
            continue
        za, zb = source_depth[a[1], a[0]] * 0.001, target_depth[b[1], b[0]] * 0.001
        if za <= 0 or zb <= 0:
            continue
        source_xyz.append([(a[0] - k[0, 2]) * za / k[0, 0], (a[1] - k[1, 2]) * za / k[1, 1], za])
        target_xyz.append([(b[0] - k[0, 2]) * zb / k[0, 0], (b[1] - k[1, 2]) * zb / k[1, 1], zb])
    src, dst = np.asarray(source_xyz), np.asarray(target_xyz)
    info["depth_backed_matches"] = len(src)
    if len(src) < 3:
        return None, info
    return fit_correspondences(src, dst, info, trials=trials, tolerance_m=tolerance_m)


def fit_correspondences(
    src: np.ndarray, dst: np.ndarray, info: dict, *, trials: int, tolerance_m: float
) -> tuple[tuple[np.ndarray, np.ndarray] | None, dict]:
    rng = np.random.default_rng(0)
    best = np.zeros(len(src), dtype=bool)
    for _ in range(trials):
        selected = rng.choice(len(src), 3, replace=False)
        if np.linalg.matrix_rank(src[selected] - src[selected].mean(axis=0)) < 2:
            continue
        rotation, translation = _rigid_fit(src[selected], dst[selected])
        residual = np.linalg.norm(apply(src, rotation, translation) - dst, axis=1)
        inliers = residual <= tolerance_m
        if inliers.sum() > best.sum():
            best = inliers
    info["rigid_inliers"] = int(best.sum())
    if best.sum() < 3:
        return None, info
    return _rigid_fit(src[best], dst[best]), info


def dino_hypothesis(
    source_rgb: np.ndarray,
    source_depth: np.ndarray,
    source_mask: np.ndarray,
    target_rgb: np.ndarray,
    target_depth: np.ndarray,
    target_mask: np.ndarray,
    k: np.ndarray,
    model_path: Path,
    device: str,
    *,
    trials: int,
    tolerance_m: float,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, dict]:
    import torch  # noqa: PLC0415
    from transformers import Dinov2Model  # noqa: PLC0415

    def crop(image, mask):
        ys, xs = np.nonzero(mask)
        center_x, center_y = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = int(np.ceil(max(np.ptp(xs), np.ptp(ys)) * 1.2))
        side = max(side, 14)
        x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
        x1, y1 = x0 + side, y0 + side
        left, top = max(0, -x0), max(0, -y0)
        right, bottom = max(0, x1 - image.shape[1]), max(0, y1 - image.shape[0])
        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(127, 127, 127)
        )
        padded_mask = cv2.copyMakeBorder(mask.astype(np.uint8), top, bottom, left, right, cv2.BORDER_CONSTANT)
        piece = padded[y0 + top : y1 + top, x0 + left : x1 + left].copy()
        piece_mask = padded_mask[y0 + top : y1 + top, x0 + left : x1 + left] > 0
        piece[~piece_mask] = 127
        return cv2.cvtColor(cv2.resize(piece, (224, 224)), cv2.COLOR_BGR2RGB), x0, y0, side

    inputs = [crop(source_rgb, source_mask), crop(target_rgb, target_mask)]
    model = Dinov2Model.from_pretrained(model_path, local_files_only=True).to(device).eval()
    images = torch.from_numpy(np.stack([entry[0] for entry in inputs])).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    with torch.inference_mode():
        features = model(
            pixel_values=(images.permute(0, 3, 1, 2).float() / 255 - mean) / std
        ).last_hidden_state[:, 1:]
        features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
    assert features.shape[1] == 16 * 16
    valid_points, descriptors = [], []
    for index, (_, x0, y0, side) in enumerate(inputs):
        depth, mask = (source_depth, source_mask) if index == 0 else (target_depth, target_mask)
        xyz, desc = [], []
        for patch in range(16 * 16):
            u = x0 + ((patch % 16 + 0.5) / 16) * side
            v = y0 + ((patch // 16 + 0.5) / 16) * side
            x, y = int(round(u)), int(round(v))
            if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] and depth[y, x] > 0):
                continue
            z = depth[y, x] * 0.001
            xyz.append([(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z])
            desc.append(features[index, patch])
        valid_points.append(np.asarray(xyz))
        descriptors.append(np.asarray(desc))
    info = {"source_patches": len(valid_points[0]), "target_patches": len(valid_points[1])}
    if min(len(points) for points in valid_points) < 3:
        return None, info
    similarity = descriptors[0] @ descriptors[1].T
    src_best = similarity.argmax(axis=1)
    dst_best = similarity.argmax(axis=0)
    matches = [(a, b) for a, b in enumerate(src_best) if dst_best[b] == a]
    info["mutual_matches"] = len(matches)
    if len(matches) < 3:
        return None, info
    src = np.asarray([valid_points[0][a] for a, _ in matches])
    dst = np.asarray([valid_points[1][b] for _, b in matches])
    return fit_correspondences(src, dst, info, trials=trials, tolerance_m=tolerance_m)


def geometry_hypotheses(source: np.ndarray, target: np.ndarray, voxel_m: float):
    import open3d as o3d  # noqa: PLC0415

    def prepare(points):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud = cloud.voxel_down_sample(voxel_m)
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 4, max_nn=30))
        features = o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 8, max_nn=100)
        )
        return cloud, features

    a, af = prepare(source)
    b, bf = prepare(target)
    r = o3d.pipelines.registration
    distance = voxel_m * 3
    output = {}
    try:
        fit = r.registration_ransac_based_on_feature_matching(
            a,
            b,
            af,
            bf,
            True,
            distance,
            r.TransformationEstimationPointToPoint(False),
            3,
            [r.CorrespondenceCheckerBasedOnEdgeLength(0.9), r.CorrespondenceCheckerBasedOnDistance(distance)],
            r.RANSACConvergenceCriteria(100000, 0.999),
        )
        output["fpfh_ransac"] = (fit.transformation[:3, :3], fit.transformation[:3, 3])
    except Exception as error:
        output["fpfh_ransac_error"] = str(error)
    try:
        fit = r.registration_fgr_based_on_feature_matching(
            a, b, af, bf, r.FastGlobalRegistrationOption(maximum_correspondence_distance=distance)
        )
        output["fpfh_fgr"] = (fit.transformation[:3, :3], fit.transformation[:3, 3])
    except Exception as error:
        output["fpfh_fgr_error"] = str(error)
    return output


def surface_refinements(
    source: np.ndarray,
    target: np.ndarray,
    source_colors: np.ndarray,
    target_colors: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    voxel_m: float,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str]]:
    import open3d as o3d  # noqa: PLC0415

    clouds = []
    for points, colors in ((source, source_colors), (target, target_colors)):
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        point_cloud = point_cloud.voxel_down_sample(voxel_m)
        point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * 4, max_nn=30))
        clouds.append(point_cloud)
    initial = np.eye(4)
    initial[:3, :3], initial[:3, 3] = rotation, translation
    registration = o3d.pipelines.registration
    output = {}
    errors = {}
    methods = {
        "point_to_plane_icp": registration.TransformationEstimationPointToPlane(),
        "colored_icp": registration.TransformationEstimationForColoredICP(),
    }
    for name, estimator in methods.items():
        try:
            function = (
                registration.registration_colored_icp
                if name == "colored_icp"
                else registration.registration_icp
            )
            result = function(
                clouds[0],
                clouds[1],
                voxel_m * 3,
                initial,
                estimator,
                registration.ICPConvergenceCriteria(max_iteration=20),
            )
            output[name] = (result.transformation[:3, :3], result.transformation[:3, 3])
        except Exception as error:
            errors[name] = re.sub(r"\x1b\[[0-9;]*m", "", str(error))
    return output, errors


def projection_evidence(
    moved: np.ndarray,
    target_mask: np.ndarray,
    target_depth: np.ndarray,
    k: np.ndarray,
    proximity_m: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    pixels = np.full((len(moved), 2), -1, dtype=int)
    in_front_of_camera = moved[:, 2] > 0
    pixels[in_front_of_camera] = np.rint(_project(moved[in_front_of_camera], k)).astype(int)
    inside = (
        in_front_of_camera
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < target_mask.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < target_mask.shape[0])
    )
    x = np.clip(pixels[:, 0], 0, target_mask.shape[1] - 1)
    y = np.clip(pixels[:, 1], 0, target_mask.shape[0] - 1)
    observed_depth = target_depth[y, x].astype(np.float64) * 0.001
    valid_depth = inside & (observed_depth > 0)
    mask_hit = inside & target_mask[y, x]
    depth_delta = moved[:, 2] - observed_depth
    supported = mask_hit & valid_depth & (np.abs(depth_delta) <= proximity_m)
    front = valid_depth & (depth_delta < -proximity_m)
    behind = valid_depth & (depth_delta > proximity_m)
    return (
        {
            "projection_mask_fraction": float(mask_hit.mean()),
            "projection_mask_and_depth_fraction": float(supported.mean()),
            "projection_front_of_depth_fraction": float(front.mean()),
            "projection_behind_depth_fraction": float(behind.mean()),
            "projection_surface_median_mm": float(
                np.median(np.abs(depth_delta[mask_hit & valid_depth])) * 1000
            )
            if (mask_hit & valid_depth).any()
            else None,
        },
        pixels,
        supported,
    )


def score(
    source: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    target_depth: np.ndarray,
    k: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    proximity_m: float,
) -> dict:
    moved = apply(source, rotation, translation)
    source_to_target = cKDTree(target).query(moved)[0]
    target_to_source = cKDTree(moved).query(target)[0]
    projection, _, _ = projection_evidence(moved, target_mask, target_depth, k, proximity_m)
    return {
        "source_to_target_median_mm": float(np.median(source_to_target) * 1000),
        "target_to_source_median_mm": float(np.median(target_to_source) * 1000),
        "source_supported_fraction": float(np.mean(source_to_target < proximity_m)),
        "target_supported_fraction": float(np.mean(target_to_source < proximity_m)),
        **projection,
    }


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    demo_rgb = _image(
        (args.demo_recording / "rgb" / f"{args.demo_frame:06d}.jpg").read_bytes(), cv2.IMREAD_COLOR
    )
    demo_depth = _image(
        (args.demo_recording / "depth" / f"{args.demo_frame:06d}.png").read_bytes(), cv2.IMREAD_UNCHANGED
    )
    demo_mask = (
        _image((args.demo_masks / f"{args.demo_frame:06d}.png").read_bytes(), cv2.IMREAD_GRAYSCALE) > 0
    )
    if args.target_recording is not None:
        if args.target_masks is None:
            raise ValueError("--target-masks is required with --target-recording")
        source_k = np.loadtxt(args.demo_recording / "cam_K.txt").reshape(3, 3)
        target_k = np.loadtxt(args.target_recording / "cam_K.txt").reshape(3, 3)
        if not np.allclose(source_k, target_k):
            raise ValueError("this benchmark currently requires matching source and target intrinsics")
        live_rgb = _image(
            (args.target_recording / "rgb" / f"{args.live_frame:06d}.jpg").read_bytes(),
            cv2.IMREAD_COLOR,
        )
        live_depth = _image(
            (args.target_recording / "depth" / f"{args.live_frame:06d}.png").read_bytes(),
            cv2.IMREAD_UNCHANGED,
        )
        live_mask = (
            _image(
                (args.target_masks / f"{args.live_frame:06d}.png").read_bytes(),
                cv2.IMREAD_GRAYSCALE,
            )
            > 0
        )
    else:
        if args.role is None:
            raise ValueError("--role is required with --live-root")
        table = pq.read_table(
            args.live_root / "data/chunk-000/file-000.parquet",
            columns=["observation.images.overhead", "observation.images.overhead_depth"],
        )
        live_rgb = _image(
            table["observation.images.overhead"][args.live_frame].as_py()["bytes"], cv2.IMREAD_COLOR
        )
        live_depth = _image(
            table["observation.images.overhead_depth"][args.live_frame].as_py()["bytes"],
            cv2.IMREAD_UNCHANGED,
        )
        live_mask = _live_masks(args.live_root, [args.live_frame], live_depth.shape, args.role)[
            args.live_frame
        ]
    k = np.loadtxt(args.demo_recording / "cam_K.txt").reshape(3, 3)
    source = cloud(demo_depth, demo_mask, k, args.edge_pixels)
    target = cloud(live_depth, live_mask, k, args.edge_pixels)
    hypotheses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    details = {}
    timings = {}
    stage_start = time.perf_counter()
    for index, transform in enumerate(pca_hypotheses(source, target)):
        hypotheses[f"pca_{index:02d}"] = transform
    timings["pca_hypotheses"] = time.perf_counter() - stage_start
    for method in ("sift", "orb"):
        stage_start = time.perf_counter()
        transform, info = feature_hypothesis(
            demo_rgb,
            demo_depth,
            demo_mask,
            live_rgb,
            live_depth,
            live_mask,
            k,
            method,
            trials=args.feature_trials,
            tolerance_m=args.feature_tolerance_m,
        )
        details[method] = info
        timings[method] = time.perf_counter() - stage_start
        if transform is not None:
            hypotheses[method] = transform
    if args.dino_model is not None:
        stage_start = time.perf_counter()
        transform, info = dino_hypothesis(
            demo_rgb,
            demo_depth,
            demo_mask,
            live_rgb,
            live_depth,
            live_mask,
            k,
            args.dino_model,
            args.dino_device,
            trials=args.feature_trials,
            tolerance_m=args.feature_tolerance_m,
        )
        details["dino"] = info
        timings["dino"] = time.perf_counter() - stage_start
        if transform is not None:
            hypotheses["dino"] = transform
    stage_start = time.perf_counter()
    try:
        geometric = geometry_hypotheses(source, target, args.voxel_m)
        for name, value in geometric.items():
            if name.endswith("_error"):
                details[name] = value
            else:
                hypotheses[name] = value
    except ImportError as error:
        details["open3d_error"] = str(error)
    timings["fpfh_global"] = time.perf_counter() - stage_start
    candidates = []
    stage_start = time.perf_counter()
    for name, (rotation, translation) in hypotheses.items():
        rotation, translation = trimmed_icp(
            source,
            target,
            rotation,
            translation,
            iterations=args.icp_iterations,
            trim_fraction=args.trim_fraction,
        )
        metrics = score(source, target, live_mask, live_depth, k, rotation, translation, args.proximity_m)
        # Balanced coverage prevents a small agreeing patch from winning.
        quality = min(metrics["source_supported_fraction"], metrics["target_supported_fraction"])
        result = {
            "method": name,
            "quality": quality,
            **metrics,
            "rotation": rotation.tolist(),
            "translation_m": translation.tolist(),
        }
        if args.landmark_xy is not None:
            x, y = args.landmark_xy
            depth = demo_depth[y, x] * 0.001
            if depth > 0:
                point = np.array([(x - k[0, 2]) * depth / k[0, 0], (y - k[1, 2]) * depth / k[1, 1], depth])
                result["landmark_projection_xy"] = _project(apply(point[None], rotation, translation), k)[
                    0
                ].tolist()
        candidates.append(result)
    timings["trimmed_icp_and_scoring"] = time.perf_counter() - stage_start
    candidates.sort(key=lambda candidate: candidate["quality"], reverse=True)
    if candidates:
        seed = candidates[0]
        stage_start = time.perf_counter()
        try:
            refinements, refinement_errors = surface_refinements(
                source,
                target,
                cloud_colors(demo_rgb, demo_depth, demo_mask, args.edge_pixels),
                cloud_colors(live_rgb, live_depth, live_mask, args.edge_pixels),
                np.asarray(seed["rotation"]),
                np.asarray(seed["translation_m"]),
                args.voxel_m,
            )
        except ImportError:
            refinements, refinement_errors = {}, {"open3d": "not installed"}
        details["surface_refinement_errors"] = refinement_errors
        timings["surface_refinements"] = time.perf_counter() - stage_start
        for name, (rotation, translation) in refinements.items():
            metrics = score(source, target, live_mask, live_depth, k, rotation, translation, args.proximity_m)
            quality = min(metrics["source_supported_fraction"], metrics["target_supported_fraction"])
            refined = {
                "method": name,
                "seed": seed["method"],
                "quality": quality,
                **metrics,
                "rotation": rotation.tolist(),
                "translation_m": translation.tolist(),
            }
            if args.landmark_xy is not None:
                x, y = args.landmark_xy
                depth = demo_depth[y, x] * 0.001
                if depth > 0:
                    point = np.array(
                        [(x - k[0, 2]) * depth / k[0, 0], (y - k[1, 2]) * depth / k[1, 1], depth]
                    )
                    refined["landmark_projection_xy"] = _project(
                        apply(point[None], rotation, translation), k
                    )[0].tolist()
            candidates.append(refined)
        candidates.sort(key=lambda candidate: candidate["quality"], reverse=True)
    args.output.mkdir(parents=True, exist_ok=True)
    overlay = live_rgb.copy()
    contours, _ = cv2.findContours(live_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    for index, candidate in enumerate(candidates[: args.show_top]):
        color = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 165, 255), (255, 255, 255)][index % 5]
        rotation, translation = np.asarray(candidate["rotation"]), np.asarray(candidate["translation_m"])
        _, pixels, supported = projection_evidence(
            apply(source, rotation, translation), live_mask, live_depth, k, args.proximity_m
        )
        for x, y in pixels[supported][::8]:
            if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
                cv2.circle(overlay, (int(x), int(y)), 1, color, -1)
        if "landmark_projection_xy" in candidate:
            x, y = np.rint(candidate["landmark_projection_xy"]).astype(int)
            cv2.circle(overlay, (x, y), 6, color, 2)
            cv2.putText(
                overlay, candidate["method"], (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
            )
    cv2.imwrite(str(args.output / "overlay.png"), overlay)
    report = {
        "demo_frame": args.demo_frame,
        "live_frame": args.live_frame,
        "target_kind": "recording" if args.target_recording is not None else "live_trial",
        "source_points": len(source),
        "target_points": len(target),
        "seconds": time.perf_counter() - start,
        "timings_s": timings,
        "settings": {
            "edge_pixels": args.edge_pixels,
            "voxel_m": args.voxel_m,
            "feature_trials": args.feature_trials,
            "feature_tolerance_m": args.feature_tolerance_m,
            "icp_iterations": args.icp_iterations,
            "trim_fraction": args.trim_fraction,
            "proximity_m": args.proximity_m,
        },
        "method_details": details,
        "candidates": candidates,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-recording", type=Path, required=True)
    parser.add_argument("--demo-masks", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--live-root", type=Path)
    target.add_argument("--target-recording", type=Path)
    parser.add_argument("--target-masks", type=Path)
    parser.add_argument("--role")
    parser.add_argument("--demo-frame", type=int, required=True)
    parser.add_argument("--live-frame", type=int, required=True)
    parser.add_argument("--landmark-xy", type=int, nargs=2)
    parser.add_argument("--edge-pixels", type=int, default=1)
    parser.add_argument("--voxel-m", type=float, default=0.003)
    parser.add_argument("--feature-trials", type=int, default=2000)
    parser.add_argument("--dino-model", type=Path)
    parser.add_argument("--dino-device", default="cuda")
    parser.add_argument("--feature-tolerance-m", type=float, default=0.008)
    parser.add_argument("--icp-iterations", type=int, default=40)
    parser.add_argument("--trim-fraction", type=float, default=0.8)
    parser.add_argument("--proximity-m", type=float, default=0.006)
    parser.add_argument("--show-top", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))
    for candidate in report["candidates"][:10]:
        print(candidate["method"], candidate["quality"], candidate.get("landmark_projection_xy"))
