"""Re-run current FLASH perception on a saved SO-107 RGB-D trial.

This source-time counterfactual reads ordinary LeRobot data and the FLASH
event log. It does not command the robot or reproduce inference latency.
Recorded pose proposals are not used by the current vision backend.
"""

import argparse
import gzip
import io
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from lerobot.policies.flash.geometry import interpolate
from lerobot.policies.flash.kinematics import SelectedJointKinematics
from lerobot.policies.flash.perception import HeldAttachmentUpdater, RGBDFrame
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.runtime import AttachmentRuntimeLimits
from lerobot.policies.flash.vision_pipeline import VisionBackendFactory
from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trial", required=True, type=Path)
parser.add_argument("--bundle", required=True, type=Path)
parser.add_argument("--held-role", required=True)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--sample-row", action="append", type=int, default=[])
args = parser.parse_args()
root = args.trial.expanduser().resolve()
bundle_path = args.bundle.expanduser().resolve()
output_dir = args.output_dir.expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)
bundle = PreparedFlashSkill.load(bundle_path)
settings = json.loads((root / "flash/v1/trial_config.json").read_text())
role = args.held_role
model = next(m.points.points_object_m for m in bundle.appearance_models if m.points.role == role)
version = next(r.version for r in bundle.references if r.role == role)
rows = [json.loads(line) for line in (root / "flash/v1/episode_000000.jsonl").open()]
table = pq.read_table(
    root / "data/chunk-000/file-000.parquet",
    columns=["observation.state", "observation.images.overhead", "observation.images.overhead_depth"],
)
if table.num_rows != len(rows):
    raise ValueError("the recorded motor and RGB-D streams must have the same number of rows")
feature_names = json.loads((root / "meta/info.json").read_text())["features"]["observation.state"]["names"]
if feature_names != [f"{name}.pos" for name in MOTOR_NAMES]:
    raise ValueError("SO-107 joint order differs from the saved episode")
tracker_path = root / "flash/v1/tracker_episode_000000.jsonl"
camera = next(d["camera"] for line in tracker_path.open() if (d := json.loads(line))["camera"] is not None)
calib = json.loads(Path(settings["calibration_path"]).read_text())
full = make_calibrated_so107_kinematics(calib, arm=settings["robot_arm"])
names = tuple(name for name in MOTOR_NAMES if name != "gripper")
kin = SelectedJointKinematics(full, tuple(MOTOR_NAMES), names, {"gripper": 0.0})
states = table.column("observation.state").to_pylist()
joints = np.asarray([state[:6] for state in states], float)
motor_times = np.asarray([row["observed_at_s"] for row in rows], float)
tips = np.stack([kin.forward_kinematics(q) for q in joints])


def tip_at(at):
    right = int(np.searchsorted(motor_times, at))
    if right == 0:
        return tips[0]
    if right == len(tips):
        return tips[-1]
    alpha = (at - motor_times[right - 1]) / (motor_times[right] - motor_times[right - 1])
    return interpolate(tips[right - 1], tips[right], alpha)


grasp_idx = next(i for i, r in enumerate(rows) if r["event_id"] == "attach-held")
release_idx = next(i for i, r in enumerate(rows) if r["event_id"] == "detach-held")
grasp_time = motor_times[grasp_idx]
release_time = motor_times[release_idx]
limits = AttachmentRuntimeLimits.from_config(settings["attachment"])
updater = HeldAttachmentUpdater(
    role,
    version,
    kin,
    names,
    history_capacity=limits.history_capacity,
    max_motor_sample_gap_s=limits.max_motor_sample_gap_s,
    max_attachment_age_s=limits.max_attachment_age_s,
    max_update_shift_m=limits.max_update_shift_m,
    max_update_turn_rad=limits.max_update_turn_rad,
    object_center_object_m=model.mean(axis=0),
    object_points_object_m=model,
    min_visual_support_points=settings["causal_tracking"]["min_inliers"],
    depth_occlusion_tolerance_m=settings["causal_tracking"]["fit_threshold_m"],
    observation_quality=bundle.vision_pipeline.observation,
)
backend = VisionBackendFactory(
    bundle.appearance_models,
    bundle.prompts_by_role,
    f"offline-{root.name}",
    bundle.vision_pipeline,
    dino_model_id=bundle.descriptor_model_id,
    axial_symmetries=bundle.axial_symmetries,
)()
backend.initialize(bundle.references)
last_id = -1
last_direct = None
activated = False
seeded = False
result = []
mask_rows = []
mask_bits = []
mask_shape = None
start = time.perf_counter()
for i, row in enumerate(rows):
    fid = row["camera_frame_id"]
    ts = row["camera_source_time_s"]
    if fid is None or fid <= last_id:
        continue
    last_id = fid
    record = table.slice(i, 1).to_pylist()[0]
    rgb = np.asarray(Image.open(io.BytesIO(record["observation.images.overhead"]["bytes"])).convert("RGB"))
    depth = np.asarray(Image.open(io.BytesIO(record["observation.images.overhead_depth"]["bytes"])))
    frame = RGBDFrame(
        fid,
        rgb,
        depth,
        np.asarray(camera["intrinsics"], float),
        0.001,
        np.asarray(camera["base_from_camera"], float),
        ts,
        ts,
        bundle.references[0].snippets[0].calibration_version,
    )
    estimates = backend.update(frame)
    held = next((e for e in estimates if e.role == role), None)
    if held is not None:
        held = replace(held, available_time_s=ts)
    if not activated and ts >= grasp_time:
        seeded = updater.activate(grasp_time, last_direct, joints[grasp_idx])
        print(
            "activate",
            i,
            fid,
            "seeded",
            seeded,
            "last direct",
            last_direct.source_frame_id if last_direct else None,
            flush=True,
        )
        activated = True
    if held is not None and held.evidence.value == "observed" and held.pose is not None:
        last_direct = held
    evidence = None
    accepted = False
    if (
        activated
        and ts < release_time
        and held is not None
        and held.evidence.value == "observed"
        and held.pose is not None
    ):
        mask = backend.last_masks.get(role)
        if mask is not None:
            accepted = updater.on_pose_update(
                i, {role: held}, frame=frame, mask=mask, ee_at_capture=tip_at(ts)
            )
            evidence = updater.latest_visual_evidence()
    state = updater.state(ts) if activated and ts < release_time else None
    inner = getattr(backend, "inner", backend)
    mask = backend.last_masks.get(role)
    if mask is not None:
        mask_rows.append(i)
        mask_bits.append(np.packbits(mask.reshape(-1)))
        mask_shape = mask.shape
    failure = getattr(inner, "last_failure", {}).get(role)
    if i in args.sample_row:
        Image.fromarray(rgb).save(output_dir / f"rgb-{i}.png")
        if mask is not None:
            Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / f"mask-{i}.png")
    detail = getattr(inner, "last_details", {}).get(role, {})
    diagnostics = detail.get("diagnostics", {})
    tracked_fields = (
        "surface_p90_mm",
        "visible_lateral_spread_mm",
        "inliers",
        "converged",
        "reference_depth_pixels",
        "reference_depth_p90_mm",
        "rgb_rotation_deg",
        "surface_correction_deg",
        "final_rotation_deg",
    )
    metrics = {
        k: (float(diagnostics[k]) if isinstance(diagnostics[k], (int, float, np.number)) else diagnostics[k])
        for k in tracked_fields
        if k in diagnostics
    }
    candidate = getattr(inner, "last_candidate_poses", {}).get(role)
    result.append(
        {
            "fit_metrics": metrics,
            "candidate_pose": candidate.tolist() if candidate is not None else None,
            "mask_pixels": int(mask.sum()) if mask is not None else None,
            "backend_failure": str(failure) if failure else None,
            "row": i,
            "frame_id": fid,
            "source_time_s": ts,
            "held_evidence": held.evidence.value if held else None,
            "held_pose": held.pose.tolist() if held and held.pose is not None else None,
            "held_estimator": held.estimator if held else None,
            "accepted": accepted,
            "attachment_available": state is not None,
            "attachment": updater.last_accepted_gripper_from_object.tolist()
            if activated and ts < release_time and updater.last_accepted_gripper_from_object is not None
            else None,
            "visual_evidence": evidence,
        }
    )
    if len(result) % 50 == 0:
        print(
            "frame",
            len(result),
            "elapsed",
            round(time.perf_counter() - start, 1),
            "observed",
            sum(x["held_evidence"] == "observed" for x in result),
            "conflict",
            sum(
                not x["attachment_available"]
                for x in result
                if grasp_time <= x["source_time_s"] < release_time
            ),
            flush=True,
        )
backend.close()
output = output_dir / "result.json.gz"
if mask_bits:
    np.savez_compressed(
        output_dir / "masks.npz",
        rows=np.asarray(mask_rows, dtype=np.int32),
        packed=np.stack(mask_bits),
        shape=np.asarray(mask_shape, dtype=np.int32),
    )
with gzip.open(output, "wt", encoding="utf-8") as stream:
    json.dump(
        {
            "trial": str(root),
            "prepared_bundle": str(bundle_path),
            "held_role": role,
            "grasp_time_s": grasp_time,
            "release_time_s": release_time,
            "pregrasp_attachment_seeded": seeded,
            "source_time_replay_without_inference_latency": True,
            "frames": result,
        },
        stream,
    )
print("saved", output, "frames", len(result), flush=True)
