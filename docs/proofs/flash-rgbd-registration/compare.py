"""Offline, causal adjacent-frame RGB-D registration on the recorded dowel.

All methods see identical SAM masks, measured aligned depth, and cropped RGB.
The recorded gripper motion is used only after estimation for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.surface_correspondence import (
    crop_pair,
    full_pixels,
    measured_pairs,
    rigid_fit as rigid_fit,
    robust_fit,
    ufm_pairs,
)

HERE = Path(__file__).resolve().parent
SURFACE = HERE.parent / "flash-mask-depth-tracker"
sys.path.insert(0, str(SURFACE))
from run_surface_tracker import move, observation  # noqa: E402

RECORDING = Path.home() / ".cache/huggingface/lerobot/demos/insertion/recording"
MASKS = Path.home() / ".cache/flash-mask-depth-experiment/sam3-full"
CONFIG = json.loads((SURFACE / "connected_surface_config.json").read_text())


def frame_data(index: int, intrinsics: np.ndarray) -> dict:
    rgb = cv2.cvtColor(cv2.imread(str(RECORDING / "rgb" / f"{index:06d}.jpg")), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(RECORDING / "depth" / f"{index:06d}.png"), -1).astype(np.float32) * 0.001
    mask = cv2.imread(str(MASKS / f"{index:06d}.png"), 0) > 0
    obs = observation(depth, mask, intrinsics, CONFIG)
    # The sampled, connected surface is the same one given to colored ICP.
    # A 3-pixel image dilation admits neighboring valid correspondence pixels.
    connected = np.zeros(mask.shape, np.uint8)
    px = obs["pixels"]
    connected[px[:, 1], px[:, 0]] = 1
    connected = cv2.dilate(connected, np.ones((7, 7), np.uint8)) > 0
    return {"rgb": rgb, "depth": depth, "mask": mask, "obs": obs, "usable": obs["valid_image"] & connected}


def colored_icp(a: dict, b: dict, weight: float) -> tuple[np.ndarray | None, int, float]:
    import open3d as o3d

    def cloud(frame: dict) -> o3d.geometry.PointCloud:
        obs = frame["obs"]
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(obs["points"])
        result.normals = o3d.utility.Vector3dVector(obs["normals"])
        p = obs["pixels"]
        result.colors = o3d.utility.Vector3dVector(frame["rgb"][p[:, 1], p[:, 0]] / 255.0)
        return result

    if min(len(a["obs"]["points"]), len(b["obs"]["points"])) < 6:
        return None, 0, float("nan")
    try:
        result = o3d.pipelines.registration.registration_colored_icp(
            cloud(a),
            cloud(b),
            0.015,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=weight),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )
    except RuntimeError:
        return None, 0, float("nan")
    if result.fitness == 0:
        return None, 0, float("nan")
    return (
        result.transformation,
        int(round(result.fitness * len(a["obs"]["points"]))),
        float(result.inlier_rmse),
    )


def roma_pairs(
    model, image_a: np.ndarray, image_b: np.ndarray, crop, a: dict, b: dict, intrinsics: np.ndarray
):
    from PIL import Image

    output = model.match(Image.fromarray(image_a), Image.fromarray(image_b))
    warp = output["warp_AB"][0].detach().float().cpu().numpy()
    confidence = output["overlap_AB"][0].detach().float().cpu().numpy()[..., 0]
    h, w = warp.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    # RoMa normalized coordinates refer to the 320x320 input crop.
    uv_a = np.column_stack(((xs.ravel() + 0.5) * 320 / w - 0.5, (ys.ravel() + 0.5) * 320 / h - 0.5))
    uv_b = (warp.reshape(-1, 2) + 1) * 160 - 0.5
    keep = confidence.ravel() >= 0.5
    return measured_pairs(a, b, full_pixels(uv_a[keep], crop), full_pixels(uv_b[keep], crop), intrinsics)


def reference() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(HERE.parent / "flash-tracker-comparison/pose_traces.npz")
    cam_from_base = np.linalg.inv(data["base_from_camera"])
    ee = np.einsum("ij,njk->nik", cam_from_base, data["demo_ee_pose_base"])
    trace = np.load(SURFACE / "results/connected280/trace.npz")
    landmarks = move(trace["landmarks_object"], trace["poses"][0])
    return ee, landmarks


def estimate_pair(model, method: str, previous: dict, current: dict, intrinsics: np.ndarray):
    """No observation means no measured delta, rather than an identity estimate."""
    if not previous["usable"].any() or not current["usable"].any():
        return None, 0, float("nan"), 0
    if method.startswith("colored-icp"):
        weight = {"colored-icp-968": 0.968, "colored-icp-700": 0.7, "colored-icp-300": 0.3}[method]
        transform, inliers, residual = colored_icp(previous, current, weight)
        return transform, inliers, residual, 0
    image_a, image_b, crop = crop_pair(previous, current)
    matcher = ufm_pairs if method == "ufm" else roma_pairs
    source, target = matcher(model, image_a, image_b, crop, previous, current, intrinsics)
    transform, inliers, residual = robust_fit(source, target)
    return transform, inliers, residual, len(source)


def gripper_metrics(ee, landmarks, accumulated, transform, first, frame):
    shaft = landmarks[1] - landmarks[0]
    shaft /= np.linalg.norm(shaft)
    expected_pair = ee[frame] @ np.linalg.inv(ee[frame - 1])
    expected = ee[frame] @ np.linalg.inv(ee[first])
    pair_error = (
        float(np.mean(np.linalg.norm(move(landmarks, transform) - move(landmarks, expected_pair), axis=1)))
        if transform is not None
        else float("nan")
    )
    drift = float(np.mean(np.linalg.norm(move(landmarks, accumulated) - move(landmarks, expected), axis=1)))
    cosine = float(np.clip((accumulated[:3, :3] @ shaft) @ (expected[:3, :3] @ shaft), -1, 1))
    center_error = np.linalg.norm(move(landmarks, accumulated).mean(0) - move(landmarks, expected).mean(0))
    return {
        "pair_error_vs_gripper_mm": pair_error * 1000,
        "drift_vs_gripper_mm": drift * 1000,
        "axis_error_vs_gripper_deg": float(np.degrees(np.arccos(cosine))),
        "center_error_vs_gripper_mm": float(center_error * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["colored-icp-968", "colored-icp-700", "colored-icp-300", "ufm", "roma-turbo", "roma-fast"],
        required=True,
    )
    parser.add_argument("--start", type=int, default=280)
    parser.add_argument("--stop", type=int, default=399)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pose-only", action="store_true", help="Omit the held-only gripper evaluation")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    intrinsics = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    ee, landmarks = (None, None) if args.pose_only else reference()
    model = None
    loading_start = time.perf_counter()
    if args.method == "ufm":
        import torch
        from uniflowmatch.models.ufm import UniFlowMatchConfidence

        # UFM's constructor downloads a DINOv2 backbone before replacing it
        # with the complete UFM checkpoint. Avoid the redundant 1.1 GB load;
        # the downloaded UFM safetensors contain encoder.model.* weights.
        original_hub_load = torch.hub.load

        def uninitialized_dino(repo, name, *positional, **keywords):
            if repo == "facebookresearch/dinov2" and name.startswith("dinov2_"):
                keywords["pretrained"] = False
            return original_hub_load(repo, name, *positional, **keywords)

        torch.hub.load = uninitialized_dino
        try:
            model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base-DINOv2L-init")
        finally:
            torch.hub.load = original_hub_load
        # The published checkpoint is bf16 while its uint8 convenience API
        # normalizes images to fp32. Use one dtype for both sides.
        model = model.cuda().float().eval()
    elif args.method.startswith("roma"):
        from romav2 import RoMaV2

        model = RoMaV2()
        model.apply_setting(args.method.split("-")[1])
    load_s = time.perf_counter() - loading_start
    rows = []
    poses = [np.eye(4)]
    prev = frame_data(args.start, intrinsics)
    for frame in range(args.start + 1, args.stop + 1):
        input_start = time.perf_counter()
        cur = frame_data(frame, intrinsics)
        input_ms = (time.perf_counter() - input_start) * 1000
        start = time.perf_counter()
        transform, inliers, residual, pair_count = estimate_pair(model, args.method, prev, cur, intrinsics)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if transform is None:
            poses.append(poses[-1].copy())
        else:
            poses.append(transform @ poses[-1])
        evaluation = (
            {} if args.pose_only else gripper_metrics(ee, landmarks, poses[-1], transform, args.start, frame)
        )
        rows.append(
            {
                "frame": frame,
                "valid": int(transform is not None),
                "candidate_pairs": pair_count,
                "inliers": inliers,
                "fit_residual_mm": residual * 1000,
                **evaluation,
                "input_ms": input_ms,
                "elapsed_ms": elapsed_ms,
            }
        )
        if (frame - args.start) % 20 == 0:
            print(
                f"{args.method} frame {frame}: valid {sum(r['valid'] for r in rows)}/{len(rows)}, "
                f"time {elapsed_ms:.1f} ms",
                flush=True,
            )
        prev = cur
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trace.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output / "trace.npz",
        frames=np.arange(args.start, args.stop + 1),
        poses=np.array(poses),
        **({} if args.pose_only else {"landmarks_camera_at_start": landmarks}),
    )
    summary = {
        "method": args.method,
        "start": args.start,
        "stop": args.stop,
        "model_load_s": load_s,
        "valid": sum(r["valid"] for r in rows),
        "pairs": len(rows),
        "median_runtime_ms": float(np.median([r["elapsed_ms"] for r in rows])),
        "p95_runtime_ms": float(np.percentile([r["elapsed_ms"] for r in rows], 95)),
        "median_input_ms": float(np.median([r["input_ms"] for r in rows])),
        "p95_input_ms": float(np.percentile([r["input_ms"] for r in rows], 95)),
    }
    if not args.pose_only:
        for name in (
            "pair_error_vs_gripper_mm",
            "drift_vs_gripper_mm",
            "axis_error_vs_gripper_deg",
            "center_error_vs_gripper_mm",
        ):
            summary["median_" + name] = float(np.nanmedian([r[name] for r in rows]))
        for name in ("drift_vs_gripper_mm", "axis_error_vs_gripper_deg"):
            summary["last_" + name] = rows[-1][name]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
