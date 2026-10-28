"""Compare rejected vision and gripper axes with visible RGB-D geometry.

The observed axis is the dominant principal direction of clean masked depth.
This is a useful independent check only when that direction is well separated
from the other two; it is not a full object-pose ground truth.
"""

import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from lerobot.policies.flash.geometry import interpolate, inverse
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.pose_disagreement import surface_support
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.surface_observation import depth_quality_mask
from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics


def axis_gap_deg(first, second):
    return float(np.degrees(np.arccos(np.clip(abs(first @ second), -1, 1))))


def load_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def measure(result_path: Path, sample_rows: list[int]) -> list[dict]:
    result = load_json(result_path)
    trial = Path(result["trial"])
    bundle = PreparedFlashSkill.load(result["prepared_bundle"])
    symmetry = next(item for item in bundle.axial_symmetries if item.role == result["held_role"])
    model = next(
        item.points.points_object_m
        for item in bundle.appearance_models
        if item.points.role == result["held_role"]
    )
    model_axis = symmetry.axis_object
    center = symmetry.center_object_m
    extent = (model - center) @ model_axis
    endpoints = np.stack((center + model_axis * extent.min(), center + model_axis * extent.max()))
    rows = {item["row"]: item for item in result["frames"]}
    with np.load(result_path.parent / "masks.npz") as masks:
        shape = tuple(int(value) for value in masks["shape"])
        packed_masks = {
            int(row): bits.copy() for row, bits in zip(masks["rows"], masks["packed"], strict=True)
        }
    trial_rows = [json.loads(line) for line in (trial / "flash/v1/episode_000000.jsonl").open()]
    table = pq.read_table(
        trial / "data/chunk-000/file-000.parquet",
        columns=["observation.state", "observation.images.overhead_depth"],
    )
    settings = json.loads((trial / "flash/v1/trial_config.json").read_text())
    calibration = json.loads(Path(settings["calibration_path"]).expanduser().read_text())
    full = make_calibrated_so107_kinematics(calibration, arm=settings["robot_arm"])
    motor_times = np.asarray([item["observed_at_s"] for item in trial_rows], dtype=float)
    motor_states = np.asarray(table.column("observation.state").to_pylist(), dtype=float)
    tips = np.stack([full.forward_kinematics(state) for state in motor_states])
    camera = next(
        item["camera"]
        for line in (trial / "flash/v1/tracker_episode_000000.jsonl").open()
        if (item := json.loads(line))["camera"] is not None
    )
    intrinsics = np.asarray(camera["intrinsics"], dtype=float)
    base_from_camera = np.asarray(camera["base_from_camera"], dtype=float)
    out = []
    for row in sample_rows:
        if row not in packed_masks:
            continue
        recorded = rows[row]
        mask = np.unpackbits(packed_masks[row], count=shape[0] * shape[1]).reshape(shape).astype(bool)
        saved = table.slice(row, 1).to_pylist()[0]
        depth = np.asarray(Image.open(io.BytesIO(saved["observation.images.overhead_depth"]["bytes"])))
        metric_depth = depth.astype(float) * 0.001
        clean = depth_quality_mask(metric_depth, mask, bundle.vision_pipeline.observation)
        frame = RGBDFrame(
            recorded["frame_id"],
            np.zeros((*depth.shape, 3), dtype=np.uint8),
            depth,
            intrinsics,
            0.001,
            base_from_camera,
            recorded["source_time_s"],
            recorded["source_time_s"],
            bundle.references[0].snippets[0].calibration_version,
        )
        y, x = np.nonzero(clean)
        if len(x) < 3:
            continue
        z = metric_depth[y, x]
        measured = np.column_stack(
            ((x - intrinsics[0, 2]) * z / intrinsics[0, 0], (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z)
        )
        eigenvalues, directions = np.linalg.eigh(np.cov(measured.T))
        measured_axis = directions[:, -1]
        spreads_mm = np.sqrt(eigenvalues) * 1000
        source_time_s = recorded["source_time_s"]
        right = int(np.searchsorted(motor_times, source_time_s))
        fraction = (source_time_s - motor_times[right - 1]) / (motor_times[right] - motor_times[right - 1])
        ee = interpolate(tips[right - 1], tips[right], fraction)
        observation_2d = np.stack(np.nonzero(mask)[::-1]).astype(float)
        _, axes_2d = np.linalg.eigh(np.cov(observation_2d))
        observed_2d_axis = axes_2d[:, -1]
        proposals = {}
        for name, pose in (
            ("vision_candidate", recorded["candidate_pose"]),
            ("gripper_fallback", ee @ np.asarray(recorded["attachment"])),
        ):
            if pose is None:
                continue
            camera_from_object = inverse(base_from_camera) @ np.asarray(pose)
            direction = camera_from_object[:3, :3] @ model_axis
            camera_endpoints = (camera_from_object[:3, :3] @ endpoints.T).T + camera_from_object[:3, 3]
            projected = (
                camera_endpoints[:, :2] / camera_endpoints[:, 2, None] @ intrinsics[:2, :2].T
                + intrinsics[:2, 2]
            )
            projected_axis = projected[1] - projected[0]
            projected_axis /= np.linalg.norm(projected_axis)
            support = surface_support(
                frame,
                mask,
                model,
                np.asarray(pose),
                depth_occlusion_tolerance_m=bundle.causal_tracking.fit_threshold_m,
                clean_depth_mask=clean,
            )
            proposals[name] = {
                "axis_gap_3d_deg": axis_gap_deg(direction, measured_axis),
                "axis_gap_mask_2d_deg": axis_gap_deg(projected_axis, observed_2d_axis),
                "projected_endpoints_xy": projected.tolist(),
                "mask_fraction": support.projected_point_mask_fraction,
                "clean_depth_residual_p90_mm": (
                    support.clean_depth_residual_p90_m * 1000
                    if support.clean_depth_residual_p90_m is not None
                    else None
                ),
            }
        out.append(
            {
                "row": row,
                "camera_frame_id": recorded["frame_id"],
                "source_time_s": source_time_s,
                "clean_depth_points": len(measured),
                "pca_spread_mm": spreads_mm.tolist(),
                "dominant_spread_excess_mm": float(spreads_mm[-1] - spreads_mm[-2]),
                "backend_failure": recorded["backend_failure"],
                "fit_metrics": recorded["fit_metrics"],
                "proposals": proposals,
            }
        )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--sample-row", action="append", type=int, default=[])
    parser.add_argument("--all-held", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.all_held and not args.sample_row:
        parser.error("choose --all-held or at least one --sample-row")
    if args.all_held:
        saved = load_json(args.result)
        sample_rows = [
            item["row"]
            for item in saved["frames"]
            if saved["grasp_time_s"] <= item["source_time_s"] < saved["release_time_s"]
            and item["candidate_pose"] is not None
            and item["attachment"] is not None
        ]
    else:
        sample_rows = args.sample_row
    measurements = measure(args.result, sample_rows)
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wt", encoding="utf-8") as stream:
            json.dump({"measurements": measurements}, stream)
    else:
        args.output.write_text(json.dumps({"measurements": measurements}, indent=2) + "\n")
